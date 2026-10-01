//! ChatGPT-subscription access through the Codex backend.
//!
//! **Unofficial for third-party use.** This reuses the login that the official
//! Codex CLI stores in `$CODEX_HOME/auth.json` and calls the endpoint that CLI
//! uses. OpenAI does not document it for other clients, it can change without
//! notice, and using it may conflict with OpenAI's terms. Intended for personal
//! experiments; read `docs/providers.md` first.
//!
//! Wire format follows the open-source Codex CLI and pi's `openai-codex`
//! provider: `POST {base}/codex/responses`, Responses-API body with
//! `stream: true` and `store: false`, server-sent events back.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde_json::{Value, json};

#[cfg(test)]
use super::SseFrames;
use super::{DeltaForwarder, error_for_status, http_client, read_sse, transport_error};
use crate::chat::{
    ChatDelta, ChatRequest, ChatResponse, FinishReason, Message, MessageRole, OutputFormat, Part,
    ToolCall, Usage, json_instruction,
};
use crate::{BoxFuture, LlmError, LlmProvider};

const PROVIDER: &str = "codex";
const DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api";
const DEFAULT_INSTRUCTIONS: &str = "You are a helpful assistant.";
/// Claim namespace OpenAI uses for ChatGPT account data inside its JWTs.
pub(super) const AUTH_CLAIM: &str = "https://api.openai.com/auth";

/// An [`LlmProvider`] billed to your ChatGPT plan. See the module docs for the
/// caveats.
///
/// Two ways to get credentials:
///
/// - [`from_login`](Self::from_login) (feature `codex-login`): worldfn's own
///   login, `worldfn login codex`. The access token is refreshed shortly
///   before it expires and written back to the token store.
/// - [`from_codex_home`](Self::from_codex_home): the login saved by the
///   official Codex CLI. Re-read from disk on every call, so a token the Codex
///   CLI refreshed is picked up; worldfn never refreshes these itself.
///
/// ```no_run
/// # use worldfn::{AgentWorld, providers::CodexLlm};
/// # fn f() -> Result<(), Box<dyn std::error::Error>> {
/// let mut world = AgentWorld::new();
/// world.provide_llm(CodexLlm::from_codex_home("gpt-5.5")?)?;
/// # Ok(()) }
/// ```
pub struct CodexLlm {
    http: reqwest::Client,
    source: Source,
    endpoint: String,
    model: String,
    instructions: String,
    reasoning_effort: Option<String>,
    native_json: bool,
}

impl CodexLlm {
    /// Use `$CODEX_HOME/auth.json`, defaulting to `~/.codex/auth.json`.
    /// Fails now if no usable ChatGPT login is found there.
    pub fn from_codex_home(model: impl Into<String>) -> Result<Self, LlmError> {
        Self::from_auth_file(default_auth_file()?, model)
    }

    /// Use an explicit `auth.json` written by the Codex CLI.
    pub fn from_auth_file(
        path: impl Into<PathBuf>,
        model: impl Into<String>,
    ) -> Result<Self, LlmError> {
        let auth_file = path.into();
        CodexCredentials::load(&auth_file)?;
        Ok(Self::with_source(Source::CodexCli(auth_file), model.into()))
    }

    /// Use worldfn's own login from the default [`TokenStore`] location
    /// (`~/.worldfn/auth.json`). Run `worldfn login codex` first.
    ///
    /// [`TokenStore`]: super::codex_login::TokenStore
    #[cfg(feature = "codex-login")]
    pub fn from_login(model: impl Into<String>) -> Result<Self, LlmError> {
        let store = super::codex_login::TokenStore::default_location()
            .map_err(|e| LlmError(format!("{PROVIDER}: {e}")))?;
        Self::from_token_store(store, model)
    }

    /// Use worldfn's own login from a specific [`TokenStore`].
    ///
    /// [`TokenStore`]: super::codex_login::TokenStore
    #[cfg(feature = "codex-login")]
    pub fn from_token_store(
        store: super::codex_login::TokenStore,
        model: impl Into<String>,
    ) -> Result<Self, LlmError> {
        let tokens = store
            .load()
            .map_err(|e| LlmError(format!("{PROVIDER}: {e}")))?
            .ok_or_else(|| {
                LlmError(format!(
                    "{PROVIDER}: not logged in ({} has no Codex login); run `worldfn login codex`",
                    store.path().display()
                ))
            })?;
        let login = login::LoginSource::new(store, tokens);
        Ok(Self::with_source(Source::Login(login), model.into()))
    }

    /// OAuth server used to refresh a [`from_login`](Self::from_login)
    /// token. Only for tests against a mock issuer.
    #[cfg(feature = "codex-login")]
    pub fn oauth_endpoints(mut self, endpoints: super::codex_login::OAuthEndpoints) -> Self {
        if let Source::Login(login) = &mut self.source {
            login.endpoints = endpoints;
        }
        self
    }

    fn with_source(source: Source, model: String) -> Self {
        Self {
            http: http_client(),
            source,
            endpoint: codex_endpoint(DEFAULT_BASE_URL),
            model,
            instructions: DEFAULT_INSTRUCTIONS.into(),
            reasoning_effort: None,
            native_json: false,
        }
    }

    /// System instructions sent with every request.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    /// Reasoning effort, e.g. `"low"`, `"medium"`, `"high"`, for models that
    /// support it.
    pub fn reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    /// Override the backend root (default `https://chatgpt.com/backend-api`).
    /// Mostly for tests.
    pub fn base_url(mut self, base_url: &str) -> Self {
        self.endpoint = codex_endpoint(base_url);
        self
    }

    /// Request JSON output with the Responses API's native `text.format`
    /// JSON Schema instead of only instructing the model. Off by default: the
    /// Codex backend's support for it is not documented for other clients.
    pub fn native_structured_output(mut self, enabled: bool) -> Self {
        self.native_json = enabled;
        self
    }

    /// Maps a [`ChatRequest`] to a Responses-API body. `max_output_tokens` is
    /// not sent: the Codex backend's handling of it is undocumented.
    pub(crate) fn request_body(&self, request: &ChatRequest) -> Result<Value, LlmError> {
        let mut instructions = request
            .system
            .clone()
            .unwrap_or_else(|| self.instructions.clone());
        let mut text_format = None;
        if let OutputFormat::Json { name, schema } = &request.output {
            instructions = format!("{instructions}\n\n{}", json_instruction(name, schema));
            if self.native_json {
                let schema: Value = serde_json::from_str(schema)
                    .map_err(|e| LlmError(format!("codex: output schema is not JSON: {e}")))?;
                text_format = Some(json!({
                    "format": { "type": "json_schema", "name": name, "schema": schema, "strict": false },
                }));
            }
        }

        let mut input = Vec::new();
        for message in &request.messages {
            encode_message(message, &mut input);
        }
        let mut body = json!({
            "model": self.model,
            "instructions": instructions,
            "input": input,
            "store": false,
            "stream": true,
        });
        if let Some(key) = &request.cache_key {
            body["prompt_cache_key"] = json!(key);
        }
        if !request.tools.is_empty() {
            let tools = request
                .tools
                .iter()
                .map(|t| {
                    let parameters: Value = serde_json::from_str(&t.parameters).map_err(|e| {
                        LlmError(format!(
                            "codex: tool `{}` parameters are not JSON: {e}",
                            t.name
                        ))
                    })?;
                    Ok(json!({
                        "type": "function",
                        "name": t.name,
                        "description": t.description,
                        "parameters": parameters,
                        "strict": false,
                    }))
                })
                .collect::<Result<Vec<_>, LlmError>>()?;
            body["tools"] = Value::Array(tools);
        }
        if let Some(text) = text_format {
            body["text"] = text;
        }
        if let Some(effort) = &self.reasoning_effort {
            body["reasoning"] = json!({ "effort": effort });
        }
        Ok(body)
    }
}

fn encode_message(message: &Message, out: &mut Vec<Value>) {
    match message.role {
        MessageRole::User => out.push(json!({
            "type": "message",
            "role": "user",
            "content": [{ "type": "input_text", "text": message.text() }],
        })),
        MessageRole::Assistant => {
            let text = message.text();
            if !text.is_empty() {
                out.push(json!({
                    "type": "message",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": text }],
                }));
            }
            for call in message.tool_calls() {
                out.push(json!({
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                }));
            }
        }
        MessageRole::Tool => {
            for part in &message.parts {
                if let Part::ToolResult(result) = part {
                    let output = if result.is_error {
                        format!("Error: {}", result.content)
                    } else {
                        result.content.clone()
                    };
                    out.push(json!({
                        "type": "function_call_output",
                        "call_id": result.call_id,
                        "output": output,
                    }));
                }
            }
        }
    }
}

enum Source {
    /// The official Codex CLI's `auth.json`, re-read on every call.
    CodexCli(PathBuf),
    /// worldfn's own login, refreshed as needed.
    #[cfg(feature = "codex-login")]
    Login(login::LoginSource),
}

impl Source {
    async fn credentials(&self) -> Result<CodexCredentials, LlmError> {
        match self {
            Source::CodexCli(path) => CodexCredentials::load(path),
            #[cfg(feature = "codex-login")]
            Source::Login(login) => login.credentials().await,
        }
    }

    fn relogin_hint(&self) -> &'static str {
        match self {
            Source::CodexCli(_) => "login rejected; run `codex login` again",
            #[cfg(feature = "codex-login")]
            Source::Login(_) => "login rejected; run `worldfn login codex` again",
        }
    }
}

#[cfg(feature = "codex-login")]
mod login {
    use super::super::codex_login::{self, CodexTokens, OAuthEndpoints, TokenStore};
    use super::{CodexCredentials, PROVIDER};
    use crate::LlmError;

    /// Refresh when the token has less than this many seconds left.
    const REFRESH_MARGIN_SECS: u64 = 5 * 60;

    pub(super) struct LoginSource {
        store: TokenStore,
        pub(super) endpoints: OAuthEndpoints,
        /// Held across a refresh, so concurrent calls refresh once: refresh
        /// tokens may be single-use.
        cached: tokio::sync::Mutex<CodexTokens>,
    }

    impl LoginSource {
        pub(super) fn new(store: TokenStore, tokens: CodexTokens) -> Self {
            Self {
                store,
                endpoints: OAuthEndpoints::openai(),
                cached: tokio::sync::Mutex::new(tokens),
            }
        }

        pub(super) async fn credentials(&self) -> Result<CodexCredentials, LlmError> {
            let fail = |e: codex_login::LoginError| LlmError(format!("{PROVIDER}: {e}"));
            let mut tokens = self.cached.lock().await;
            if tokens.expires_within(REFRESH_MARGIN_SECS) {
                // Another process may already have refreshed and saved.
                if let Some(saved) = self.store.load().map_err(fail)? {
                    *tokens = saved;
                }
            }
            if tokens.expires_within(REFRESH_MARGIN_SECS) {
                let fresh = codex_login::refresh(&self.endpoints, &tokens)
                    .await
                    .map_err(fail)?;
                self.store.save(&fresh).map_err(fail)?;
                *tokens = fresh;
            }
            Ok(CodexCredentials {
                access_token: tokens.access_token.clone(),
                account_id: tokens.account_id.clone(),
            })
        }
    }
}

fn codex_endpoint(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');
    if base.ends_with("/codex/responses") {
        base.to_owned()
    } else if base.ends_with("/codex") {
        format!("{base}/responses")
    } else {
        format!("{base}/codex/responses")
    }
}

fn default_auth_file() -> Result<PathBuf, LlmError> {
    if let Some(home) = std::env::var_os("CODEX_HOME").filter(|h| !h.is_empty()) {
        return Ok(PathBuf::from(home).join("auth.json"));
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .filter(|h| !h.is_empty())
        .ok_or_else(|| LlmError("codex: cannot locate home directory; set CODEX_HOME".into()))?;
    Ok(PathBuf::from(home).join(".codex").join("auth.json"))
}

/// The subset of the Codex CLI's `auth.json` this provider needs.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct CodexCredentials {
    pub access_token: String,
    pub account_id: String,
}

impl CodexCredentials {
    pub(crate) fn load(path: &Path) -> Result<Self, LlmError> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            LlmError(format!(
                "codex: cannot read {}: {e} (run `codex login` and choose ChatGPT; \
                 credentials must be stored in a file, not the OS keyring)",
                path.display()
            ))
        })?;
        Self::parse(&text, now_unix())
    }

    pub(crate) fn parse(auth_json: &str, now: u64) -> Result<Self, LlmError> {
        let root: Value = serde_json::from_str(auth_json)
            .map_err(|e| LlmError(format!("codex: auth.json is not valid JSON: {e}")))?;
        let tokens = root
            .get("tokens")
            .filter(|t| t.is_object())
            .ok_or_else(|| {
                LlmError(
                    "codex: auth.json has no ChatGPT tokens (logged in with an API key? \
                 run `codex login` and choose ChatGPT)"
                        .into(),
                )
            })?;
        let access_token = tokens
            .get("access_token")
            .and_then(Value::as_str)
            .filter(|t| !t.is_empty())
            .ok_or_else(|| LlmError("codex: auth.json has no access_token".into()))?
            .to_owned();

        let claims = jwt_claims(&access_token);
        if let Some(exp) = claims
            .as_ref()
            .and_then(|c| c.get("exp"))
            .and_then(Value::as_u64)
        {
            if exp <= now {
                return Err(LlmError(
                    "codex: saved access token has expired; run `codex` once so it \
                     refreshes the login, or `codex login`"
                        .into(),
                ));
            }
        }

        let account_id = tokens
            .get("account_id")
            .and_then(Value::as_str)
            .filter(|a| !a.is_empty())
            .map(str::to_owned)
            .or_else(|| {
                claims
                    .as_ref()?
                    .get(AUTH_CLAIM)?
                    .get("chatgpt_account_id")?
                    .as_str()
                    .map(str::to_owned)
            })
            .ok_or_else(|| LlmError("codex: cannot determine the ChatGPT account id".into()))?;

        Ok(Self {
            access_token,
            account_id,
        })
    }
}

pub(super) fn jwt_claims(token: &str) -> Option<Value> {
    let payload = token.split('.').nth(1)?;
    let bytes = URL_SAFE_NO_PAD.decode(payload.trim_end_matches('=')).ok()?;
    serde_json::from_slice(&bytes).ok()
}

pub(super) fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Incremental parser for the Codex server-sent event stream. Collects output
/// text until a terminal event.
#[derive(Default)]
pub(crate) struct CodexStream {
    #[cfg(test)]
    frames: SseFrames,
    text: String,
    calls: Vec<ToolCall>,
    usage: Option<Usage>,
    truncated: bool,
    done: bool,
}

impl CodexStream {
    /// Feed raw bytes as they arrive. Returns `Ok(true)` once the response is
    /// complete.
    #[cfg(test)]
    pub(crate) fn feed(&mut self, chunk: &str) -> Result<bool, LlmError> {
        for data in self.frames.push(chunk) {
            if self.handle_data(&data)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// One event's `data` payload. Returns `Ok(true)` once the response is
    /// complete.
    pub(crate) fn handle_data(&mut self, data: &str) -> Result<bool, LlmError> {
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            return Ok(false);
        }
        let event: Value = serde_json::from_str(data)
            .map_err(|e| LlmError(format!("codex: malformed event: {e}: {data}")))?;
        let done = self.handle(&event)?;
        self.done |= done;
        Ok(done)
    }

    fn handle(&mut self, event: &Value) -> Result<bool, LlmError> {
        match event.get("type").and_then(Value::as_str) {
            Some("response.output_text.delta") => {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    self.text.push_str(delta);
                }
                Ok(false)
            }
            Some("error") => Err(LlmError(format!(
                "codex: {}",
                event
                    .get("message")
                    .or_else(|| event.pointer("/error/message"))
                    .and_then(Value::as_str)
                    .unwrap_or("stream error")
            ))),
            Some("response.failed") => Err(LlmError(format!(
                "codex: response failed: {}",
                event
                    .pointer("/response/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown error")
            ))),
            Some("response.output_item.done") => {
                if let Some(call) = event.get("item").and_then(function_call) {
                    self.calls.push(call);
                }
                Ok(false)
            }
            Some(kind @ ("response.completed" | "response.done" | "response.incomplete")) => {
                let response = event.get("response").unwrap_or(&Value::Null);
                if self.text.is_empty() {
                    self.text = output_text(response);
                }
                // Some backends only report output items in the final event.
                if self.calls.is_empty() {
                    self.calls = response
                        .get("output")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .filter_map(function_call)
                        .collect();
                }
                self.usage = response.get("usage").map(responses_usage);
                self.truncated = kind == "response.incomplete"
                    || response
                        .pointer("/incomplete_details/reason")
                        .and_then(Value::as_str)
                        == Some("max_output_tokens");
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    pub(crate) fn finish(self) -> Result<ChatResponse, LlmError> {
        if self.done {
            let finish = if !self.calls.is_empty() {
                FinishReason::ToolCalls
            } else if self.truncated {
                FinishReason::Length
            } else {
                FinishReason::Stop
            };
            let mut parts = Vec::new();
            if !self.text.is_empty() {
                parts.push(Part::Text(self.text));
            }
            parts.extend(self.calls.into_iter().map(Part::ToolCall));
            Ok(ChatResponse {
                message: Message {
                    role: MessageRole::Assistant,
                    parts,
                },
                finish,
                usage: self.usage,
            })
        } else {
            Err(LlmError(
                "codex: stream ended before the response completed".into(),
            ))
        }
    }
}

/// Responses-API usage, including cached input and reasoning tokens.
fn responses_usage(u: &Value) -> Usage {
    let n = |pointer: &str| u.pointer(pointer).and_then(Value::as_u64).unwrap_or(0);
    Usage {
        input_tokens: n("/input_tokens"),
        cached_input_tokens: n("/input_tokens_details/cached_tokens"),
        output_tokens: n("/output_tokens"),
        reasoning_tokens: n("/output_tokens_details/reasoning_tokens"),
    }
}

/// A `function_call` output item as a tool call.
fn function_call(item: &Value) -> Option<ToolCall> {
    if item.get("type").and_then(Value::as_str) != Some("function_call") {
        return None;
    }
    let field = |key: &str| item.get(key).and_then(Value::as_str).map(str::to_owned);
    Some(ToolCall {
        id: field("call_id")?,
        name: field("name")?,
        arguments: field("arguments").unwrap_or_else(|| "{}".into()),
    })
}

/// Concatenate `output[*].content[*].text` of type `output_text`.
fn output_text(response: &Value) -> String {
    let mut text = String::new();
    for item in response
        .get("output")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        for part in item
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if part.get("type").and_then(Value::as_str) == Some("output_text") {
                if let Some(t) = part.get("text").and_then(Value::as_str) {
                    text.push_str(t);
                }
            }
        }
    }
    text
}

impl CodexLlm {
    fn call<'a>(
        &'a self,
        request: ChatRequest,
        on_delta: Option<&'a mut (dyn FnMut(ChatDelta) + Send)>,
    ) -> BoxFuture<'a, Result<ChatResponse, LlmError>> {
        let body = self.request_body(&request);
        Box::pin(async move {
            let body = body?;
            let credentials = self.source.credentials().await?;
            let response = self
                .http
                .post(&self.endpoint)
                .bearer_auth(&credentials.access_token)
                .header("chatgpt-account-id", &credentials.account_id)
                .header("OpenAI-Beta", "responses=experimental")
                .header("originator", "worldfn")
                .header("accept", "text/event-stream")
                .header("content-type", "application/json")
                .body(body.to_string())
                .send()
                .await
                .map_err(|e| transport_error(PROVIDER, e))?;
            let relogin = self.source.relogin_hint();
            let response = error_for_status(PROVIDER, response, |status| match status {
                401 | 403 => Some(relogin),
                429 => Some("rate or usage limit reached for this ChatGPT plan"),
                _ => None,
            })
            .await?;

            let mut stream = CodexStream::default();
            let mut forward = DeltaForwarder::new(on_delta);
            read_sse(PROVIDER, response, |data| {
                let done = stream.handle_data(data)?;
                // Includes the fallback text of a final event when the
                // backend sent no deltas.
                forward.forward(&stream.text);
                Ok(done)
            })
            .await?;
            stream.finish()
        })
    }
}

impl LlmProvider for CodexLlm {
    fn chat(&self, request: ChatRequest) -> BoxFuture<'_, Result<ChatResponse, LlmError>> {
        self.call(request, None)
    }

    fn chat_streaming<'a>(
        &'a self,
        request: ChatRequest,
        on_delta: &'a mut (dyn FnMut(ChatDelta) + Send),
    ) -> BoxFuture<'a, Result<ChatResponse, LlmError>> {
        self.call(request, Some(on_delta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jwt(claims: Value) -> String {
        format!(
            "e30.{}.sig",
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims).unwrap())
        )
    }

    #[test]
    fn parses_codex_cli_auth_json() {
        let token = jwt(json!({ "exp": 2_000, AUTH_CLAIM: { "chatgpt_account_id": "acct-jwt" } }));
        let auth = json!({
            "OPENAI_API_KEY": null,
            "tokens": { "id_token": "x", "access_token": token, "refresh_token": "r", "account_id": "acct-file" },
            "last_refresh": "2026-09-01T00:00:00Z"
        });
        let creds = CodexCredentials::parse(&auth.to_string(), 1_000).unwrap();
        assert_eq!(creds.account_id, "acct-file");
        assert_eq!(creds.access_token, token);

        // Falls back to the JWT claim when the file has no account id.
        let auth = json!({ "tokens": { "access_token": token } });
        let creds = CodexCredentials::parse(&auth.to_string(), 1_000).unwrap();
        assert_eq!(creds.account_id, "acct-jwt");
    }

    #[test]
    fn rejects_expired_or_api_key_only_logins() {
        let token = jwt(json!({ "exp": 1_000, AUTH_CLAIM: { "chatgpt_account_id": "a" } }));
        let auth = json!({ "tokens": { "access_token": token } }).to_string();
        let err = CodexCredentials::parse(&auth, 1_000).unwrap_err();
        assert!(err.0.contains("expired"), "{err}");

        let err = CodexCredentials::parse(r#"{"OPENAI_API_KEY":"sk-x"}"#, 0).unwrap_err();
        assert!(err.0.contains("no ChatGPT tokens"), "{err}");
    }

    #[test]
    fn stream_collects_deltas_across_split_frames() {
        let mut stream = CodexStream::default();
        let events = "event: response.created\ndata: {\"type\":\"response.created\"}\n\n\
                      data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hel\"}\n\n\
                      data: {\"type\":\"response.output_text.delta\",\"delta\":\"lo\"}\r\n\r\n\
                      data: {\"type\":\"response.completed\",\"response\":{}}\n\n";
        let (a, b) = events.split_at(40);
        assert!(!stream.feed(a).unwrap());
        assert!(stream.feed(b).unwrap());
        assert_eq!(stream.finish().unwrap().message.text(), "Hello");
    }

    #[test]
    fn stream_falls_back_to_completed_output_and_reports_failures() {
        let mut stream = CodexStream::default();
        let done = stream
            .feed(
                "data: {\"type\":\"response.completed\",\"response\":{\"output\":[\
                 {\"type\":\"reasoning\"},\
                 {\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hi\"}]}]}}\n\n",
            )
            .unwrap();
        assert!(done);
        assert_eq!(stream.finish().unwrap().message.text(), "Hi");

        let mut stream = CodexStream::default();
        let err = stream
            .feed("data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"usage limit\"}}}\n\n")
            .unwrap_err();
        assert_eq!(err.0, "codex: response failed: usage limit");

        let mut stream = CodexStream::default();
        stream
            .feed("data: {\"type\":\"response.output_text.delta\",\"delta\":\"x\"}\n\n")
            .unwrap();
        assert!(stream.finish().is_err());
    }

    #[test]
    fn endpoint_normalization() {
        assert_eq!(
            codex_endpoint("https://chatgpt.com/backend-api/"),
            "https://chatgpt.com/backend-api/codex/responses"
        );
        assert_eq!(codex_endpoint("http://x/codex"), "http://x/codex/responses");
        assert_eq!(
            codex_endpoint("http://x/codex/responses"),
            "http://x/codex/responses"
        );
    }
}
