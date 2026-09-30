use serde_json::{Value, json};

use super::{DeltaForwarder, error_for_status, http_client, read_sse, transport_error};
use crate::chat::{
    ChatDelta, ChatRequest, ChatResponse, FinishReason, Message, MessageRole, OutputFormat, Part,
    ToolCall, Usage, json_instruction,
};
use crate::{BoxFuture, LlmError, LlmProvider};

/// How a JSON output format is requested, since support varies by server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonMode {
    /// `response_format: json_schema` (OpenAI and servers that copy it).
    Schema,
    /// `response_format: json_object` plus the schema in the system prompt
    /// (DeepSeek).
    Object,
    /// Only the schema in the system prompt; works everywhere.
    Instructions,
}

/// An OpenAI-compatible Chat Completions backend authenticated with an API
/// key: OpenAI itself, DeepSeek, or a local server (vLLM, Ollama, …).
///
/// Maps a [`ChatRequest`] to `/chat/completions` (messages, tool definitions,
/// tool calls and results, JSON output) without streaming, and maps
/// `choices[0]` back, including tool calls, finish reason and usage.
///
/// ```no_run
/// # use worldfn::{AgentWorld, providers::OpenAiCompatLlm};
/// # fn f() -> Result<(), Box<dyn std::error::Error>> {
/// let mut world = AgentWorld::new();
/// world.provide_llm(OpenAiCompatLlm::deepseek("deepseek-v4-flash")?)?;
/// # Ok(()) }
/// ```
pub struct OpenAiCompatLlm {
    http: reqwest::Client,
    endpoint: String,
    api_key: String,
    model: String,
    system: Option<String>,
    name: &'static str,
    json_mode: JsonMode,
}

impl OpenAiCompatLlm {
    /// `base_url` is the API root, e.g. `https://api.openai.com/v1`;
    /// `/chat/completions` is appended.
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        let base_url = base_url.into();
        Self {
            http: http_client(),
            endpoint: format!("{}/chat/completions", base_url.trim_end_matches('/')),
            api_key: api_key.into(),
            model: model.into(),
            system: None,
            name: "openai-compatible",
            json_mode: JsonMode::Instructions,
        }
    }

    /// DeepSeek, keyed by `DEEPSEEK_API_KEY`. Pass a current model name from
    /// DeepSeek's model list; names change, so none is assumed.
    pub fn deepseek(model: impl Into<String>) -> Result<Self, LlmError> {
        let key = env_key("DEEPSEEK_API_KEY")?;
        let mut llm = Self::new("https://api.deepseek.com", key, model);
        llm.name = "deepseek";
        llm.json_mode = JsonMode::Object;
        Ok(llm)
    }

    /// The OpenAI API, keyed by `OPENAI_API_KEY` (billed per token, separate
    /// from ChatGPT subscriptions).
    pub fn openai(model: impl Into<String>) -> Result<Self, LlmError> {
        let key = env_key("OPENAI_API_KEY")?;
        let mut llm = Self::new("https://api.openai.com/v1", key, model);
        llm.name = "openai";
        llm.json_mode = JsonMode::Schema;
        Ok(llm)
    }

    /// A default system prompt, used when a request has none.
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// How JSON output is requested. Defaults: `Schema` for [`openai`](Self::openai),
    /// `Object` for [`deepseek`](Self::deepseek), `Instructions` otherwise.
    pub fn json_mode(mut self, mode: JsonMode) -> Self {
        self.json_mode = mode;
        self
    }

    pub(crate) fn request_body(&self, request: &ChatRequest) -> Result<Value, LlmError> {
        let mut system = request.system.clone().or_else(|| self.system.clone());
        let mut response_format = None;
        if let OutputFormat::Json { name, schema } = &request.output {
            let schema_value: Value = serde_json::from_str(schema)
                .map_err(|e| LlmError(format!("{}: output schema is not JSON: {e}", self.name)))?;
            match self.json_mode {
                JsonMode::Schema => {
                    response_format = Some(json!({
                        "type": "json_schema",
                        "json_schema": { "name": name, "schema": schema_value },
                    }));
                }
                JsonMode::Object | JsonMode::Instructions => {
                    let instruction = json_instruction(name, schema);
                    system = Some(match system {
                        Some(s) => format!("{s}\n\n{instruction}"),
                        None => instruction,
                    });
                    if self.json_mode == JsonMode::Object {
                        response_format = Some(json!({ "type": "json_object" }));
                    }
                }
            }
        }

        let mut messages = Vec::new();
        if let Some(system) = system {
            messages.push(json!({ "role": "system", "content": system }));
        }
        for message in &request.messages {
            encode_message(message, &mut messages);
        }

        let mut body = json!({ "model": self.model, "messages": messages, "stream": false });
        if !request.tools.is_empty() {
            let tools = request
                .tools
                .iter()
                .map(|t| {
                    let parameters: Value = serde_json::from_str(&t.parameters).map_err(|e| {
                        LlmError(format!(
                            "{}: tool `{}` parameters are not JSON: {e}",
                            self.name, t.name
                        ))
                    })?;
                    Ok(json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": parameters,
                        },
                    }))
                })
                .collect::<Result<Vec<_>, LlmError>>()?;
            body["tools"] = Value::Array(tools);
        }
        if let Some(format) = response_format {
            body["response_format"] = format;
        }
        if let Some(max) = request.max_output_tokens {
            body["max_tokens"] = json!(max);
        }
        Ok(body)
    }
}

fn encode_message(message: &Message, out: &mut Vec<Value>) {
    match message.role {
        MessageRole::User => out.push(json!({ "role": "user", "content": message.text() })),
        MessageRole::Assistant => {
            let text = message.text();
            let mut encoded = json!({
                "role": "assistant",
                "content": if text.is_empty() { Value::Null } else { Value::String(text) },
            });
            let calls: Vec<Value> = message
                .tool_calls()
                .map(|c| {
                    json!({
                        "id": c.id,
                        "type": "function",
                        "function": { "name": c.name, "arguments": c.arguments },
                    })
                })
                .collect();
            if !calls.is_empty() {
                encoded["tool_calls"] = Value::Array(calls);
            }
            out.push(encoded);
        }
        MessageRole::Tool => {
            for part in &message.parts {
                if let Part::ToolResult(result) = part {
                    let content = if result.is_error {
                        format!("Error: {}", result.content)
                    } else {
                        result.content.clone()
                    };
                    out.push(json!({
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "content": content,
                    }));
                }
            }
        }
    }
}

fn env_key(var: &str) -> Result<String, LlmError> {
    std::env::var(var)
        .ok()
        .filter(|k| !k.trim().is_empty())
        .ok_or_else(|| LlmError(format!("{var} is not set")))
}

pub(crate) fn parse_response(provider: &str, body: &Value) -> Result<ChatResponse, LlmError> {
    let choice = body
        .pointer("/choices/0")
        .ok_or_else(|| LlmError(format!("{provider}: response has no choices: {body}")))?;
    let message = choice.get("message").unwrap_or(&Value::Null);
    let mut parts = Vec::new();
    if let Some(text) = message.get("content").and_then(Value::as_str) {
        if !text.is_empty() {
            parts.push(Part::Text(text.to_owned()));
        }
    }
    for call in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let field = |pointer: &str| {
            call.pointer(pointer)
                .and_then(Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| LlmError(format!("{provider}: malformed tool call: {call}")))
        };
        parts.push(Part::ToolCall(ToolCall {
            id: field("/id")?,
            name: field("/function/name")?,
            arguments: field("/function/arguments")?,
        }));
    }
    let finish = match choice.get("finish_reason").and_then(Value::as_str) {
        Some("stop") => FinishReason::Stop,
        Some("tool_calls") | Some("function_call") => FinishReason::ToolCalls,
        Some("length") => FinishReason::Length,
        Some("content_filter") => FinishReason::ContentFilter,
        _ => FinishReason::Other,
    };
    let usage = body.get("usage").map(|u| Usage {
        input_tokens: u.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0),
        output_tokens: u
            .get("completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    });
    Ok(ChatResponse {
        message: Message {
            role: MessageRole::Assistant,
            parts,
        },
        finish,
        usage,
    })
}

impl OpenAiCompatLlm {
    async fn send(&self, body: &Value) -> Result<reqwest::Response, LlmError> {
        let response = self
            .http
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .body(body.to_string())
            .send()
            .await
            .map_err(|e| transport_error(self.name, e))?;
        error_for_status(self.name, response, |status| {
            (status == 401).then_some("check the API key")
        })
        .await
    }
}

impl LlmProvider for OpenAiCompatLlm {
    fn chat(&self, request: ChatRequest) -> BoxFuture<'_, Result<ChatResponse, LlmError>> {
        let body = self.request_body(&request);
        Box::pin(async move {
            let response = self.send(&body?).await?;
            let text = response
                .text()
                .await
                .map_err(|e| transport_error(self.name, e))?;
            let json: Value = serde_json::from_str(&text)
                .map_err(|e| LlmError(format!("{}: invalid JSON response: {e}", self.name)))?;
            parse_response(self.name, &json)
        })
    }

    /// Sends `stream: true` with `stream_options.include_usage`, and rebuilds
    /// the same response `chat` would return from the chunks.
    fn chat_streaming<'a>(
        &'a self,
        request: ChatRequest,
        on_delta: &'a mut (dyn FnMut(ChatDelta) + Send),
    ) -> BoxFuture<'a, Result<ChatResponse, LlmError>> {
        let body = self.request_body(&request).map(|mut body| {
            body["stream"] = json!(true);
            body["stream_options"] = json!({ "include_usage": true });
            body
        });
        Box::pin(async move {
            let response = self.send(&body?).await?;
            let mut stream = ChunkStream::default();
            let mut forward = DeltaForwarder::new(Some(on_delta));
            read_sse(self.name, response, |data| {
                let done = stream.handle_data(self.name, data)?;
                forward.forward(&stream.text);
                Ok(done)
            })
            .await?;
            if stream.finish_reason.is_none() && !stream.done {
                return Err(LlmError(format!(
                    "{}: stream ended before the response completed",
                    self.name
                )));
            }
            parse_response(self.name, &stream.into_body())
        })
    }
}

/// Accumulates `chat.completion.chunk` events into a non-streaming body.
#[derive(Default)]
struct ChunkStream {
    text: String,
    /// `(id, name, arguments)` by the chunk's `index`.
    calls: Vec<(String, String, String)>,
    finish_reason: Option<String>,
    usage: Option<Value>,
    done: bool,
}

impl ChunkStream {
    fn handle_data(&mut self, provider: &str, data: &str) -> Result<bool, LlmError> {
        let data = data.trim();
        if data == "[DONE]" {
            self.done = true;
            return Ok(true);
        }
        if data.is_empty() {
            return Ok(false);
        }
        let chunk: Value = serde_json::from_str(data)
            .map_err(|e| LlmError(format!("{provider}: malformed stream chunk: {e}: {data}")))?;
        if let Some(error) = chunk.get("error") {
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .map_or_else(|| error.to_string(), str::to_owned);
            return Err(LlmError(format!("{provider}: {message}")));
        }
        if let Some(usage) = chunk.get("usage").filter(|u| !u.is_null()) {
            self.usage = Some(usage.clone());
        }
        let Some(choice) = chunk.pointer("/choices/0") else {
            return Ok(false);
        };
        if let Some(text) = choice.pointer("/delta/content").and_then(Value::as_str) {
            self.text.push_str(text);
        }
        for call in choice
            .pointer("/delta/tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let index = call
                .get("index")
                .and_then(Value::as_u64)
                .map_or(self.calls.len().saturating_sub(1), |i| i as usize);
            if self.calls.len() <= index {
                self.calls.resize_with(index + 1, Default::default);
            }
            let (id, name, arguments) = &mut self.calls[index];
            let text = |pointer: &str| call.pointer(pointer).and_then(Value::as_str);
            if let Some(v) = text("/id") {
                *id = v.to_owned();
            }
            if let Some(v) = text("/function/name") {
                name.push_str(v);
            }
            if let Some(v) = text("/function/arguments") {
                arguments.push_str(v);
            }
        }
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish_reason = Some(reason.to_owned());
        }
        Ok(false)
    }

    fn into_body(self) -> Value {
        let calls: Vec<Value> = self
            .calls
            .into_iter()
            .map(|(id, name, arguments)| {
                json!({ "id": id, "type": "function",
                        "function": { "name": name, "arguments": arguments } })
            })
            .collect();
        let mut message = json!({ "role": "assistant", "content": self.text });
        if !calls.is_empty() {
            message["tool_calls"] = json!(calls);
        }
        let mut body = json!({
            "choices": [{ "message": message, "finish_reason": self.finish_reason }],
        });
        if let Some(usage) = self.usage {
            body["usage"] = usage;
        }
        body
    }
}
