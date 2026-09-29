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

use super::{error_for_status, http_client, transport_error};
use crate::{BoxFuture, LlmError, LlmProvider};

const PROVIDER: &str = "codex";
const DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api";
const DEFAULT_INSTRUCTIONS: &str = "You are a helpful assistant.";
/// Claim namespace OpenAI uses for ChatGPT account data inside its JWTs.
const AUTH_CLAIM: &str = "https://api.openai.com/auth";

/// An [`LlmProvider`] billed to your ChatGPT plan, using the login saved by
/// the official Codex CLI. See the module docs for the caveats.
///
/// Credentials are re-read from disk on every call, so a token refreshed by
/// the Codex CLI is picked up without restarting. This provider never
/// refreshes tokens itself: when the saved token expires, run `codex` (or
/// `codex login`) again.
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
    auth_file: PathBuf,
    endpoint: String,
    model: String,
    instructions: String,
    reasoning_effort: Option<String>,
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
        Ok(Self {
            http: http_client(),
            auth_file,
            endpoint: codex_endpoint(DEFAULT_BASE_URL),
            model: model.into(),
            instructions: DEFAULT_INSTRUCTIONS.into(),
            reasoning_effort: None,
        })
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

    pub(crate) fn request_body(&self, prompt: &str) -> Value {
        let mut body = json!({
            "model": self.model,
            "instructions": self.instructions,
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": prompt }],
            }],
            "store": false,
            "stream": true,
        });
        if let Some(effort) = &self.reasoning_effort {
            body["reasoning"] = json!({ "effort": effort });
        }
        body
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

fn jwt_claims(token: &str) -> Option<Value> {
    let payload = token.split('.').nth(1)?;
    let bytes = URL_SAFE_NO_PAD.decode(payload.trim_end_matches('=')).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Incremental parser for the Codex server-sent event stream. Collects output
/// text until a terminal event.
#[derive(Default)]
pub(crate) struct CodexStream {
    buffer: String,
    text: String,
    done: bool,
}

impl CodexStream {
    /// Feed raw bytes as they arrive. Returns `Ok(true)` once the response is
    /// complete.
    pub(crate) fn feed(&mut self, chunk: &str) -> Result<bool, LlmError> {
        self.buffer.push_str(&chunk.replace("\r\n", "\n"));
        while let Some(end) = self.buffer.find("\n\n") {
            let frame: String = self.buffer.drain(..end + 2).collect();
            let data: Vec<&str> = frame
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(str::trim)
                .collect();
            let data = data.join("\n");
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            let event: Value = serde_json::from_str(&data)
                .map_err(|e| LlmError(format!("codex: malformed event: {e}: {data}")))?;
            if self.handle(&event)? {
                self.done = true;
                return Ok(true);
            }
        }
        Ok(false)
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
            Some("response.completed" | "response.done" | "response.incomplete") => {
                if self.text.is_empty() {
                    self.text = output_text(event.get("response").unwrap_or(&Value::Null));
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    pub(crate) fn finish(self) -> Result<String, LlmError> {
        if self.done {
            Ok(self.text)
        } else {
            Err(LlmError(
                "codex: stream ended before the response completed".into(),
            ))
        }
    }
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

impl LlmProvider for CodexLlm {
    fn complete(&self, prompt: String) -> BoxFuture<'_, Result<String, LlmError>> {
        let body = self.request_body(&prompt);
        Box::pin(async move {
            let credentials = CodexCredentials::load(&self.auth_file)?;
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
            let mut response = error_for_status(PROVIDER, response, |status| match status {
                401 | 403 => Some("login rejected; run `codex login` again"),
                429 => Some("rate or usage limit reached for this ChatGPT plan"),
                _ => None,
            })
            .await?;

            let mut stream = CodexStream::default();
            let mut pending = Vec::new();
            while let Some(bytes) = response
                .chunk()
                .await
                .map_err(|e| transport_error(PROVIDER, e))?
            {
                pending.extend_from_slice(&bytes);
                // Only decode complete UTF-8; keep a split code point for later.
                let valid = match std::str::from_utf8(&pending) {
                    Ok(s) => s.len(),
                    Err(e) => e.valid_up_to(),
                };
                let text = std::str::from_utf8(&pending[..valid]).expect("validated above");
                let done = stream.feed(text)?;
                pending.drain(..valid);
                if done {
                    break;
                }
            }
            if !stream.done {
                // Treat EOF as the end of a final unterminated frame.
                stream.feed("\n\n")?;
            }
            stream.finish()
        })
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
        assert_eq!(stream.finish().unwrap(), "Hello");
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
        assert_eq!(stream.finish().unwrap(), "Hi");

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
