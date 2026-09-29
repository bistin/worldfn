use serde_json::{Value, json};

use super::{error_for_status, http_client, transport_error};
use crate::{BoxFuture, LlmError, LlmProvider};

/// An OpenAI-compatible Chat Completions backend authenticated with an API
/// key: OpenAI itself, DeepSeek, or a local server (vLLM, Ollama, …).
///
/// Sends one system message (optional) and one user message per
/// [`complete`](LlmProvider::complete) call, without streaming, and returns
/// `choices[0].message.content`.
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
        }
    }

    /// DeepSeek, keyed by `DEEPSEEK_API_KEY`. Pass a current model name from
    /// DeepSeek's model list; names change, so none is assumed.
    pub fn deepseek(model: impl Into<String>) -> Result<Self, LlmError> {
        let key = env_key("DEEPSEEK_API_KEY")?;
        let mut llm = Self::new("https://api.deepseek.com", key, model);
        llm.name = "deepseek";
        Ok(llm)
    }

    /// The OpenAI API, keyed by `OPENAI_API_KEY` (billed per token, separate
    /// from ChatGPT subscriptions).
    pub fn openai(model: impl Into<String>) -> Result<Self, LlmError> {
        let key = env_key("OPENAI_API_KEY")?;
        let mut llm = Self::new("https://api.openai.com/v1", key, model);
        llm.name = "openai";
        Ok(llm)
    }

    /// A system message sent before every prompt.
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub(crate) fn request_body(&self, prompt: &str) -> Value {
        let mut messages = Vec::new();
        if let Some(system) = &self.system {
            messages.push(json!({ "role": "system", "content": system }));
        }
        messages.push(json!({ "role": "user", "content": prompt }));
        json!({ "model": self.model, "messages": messages, "stream": false })
    }
}

fn env_key(var: &str) -> Result<String, LlmError> {
    std::env::var(var)
        .ok()
        .filter(|k| !k.trim().is_empty())
        .ok_or_else(|| LlmError(format!("{var} is not set")))
}

pub(crate) fn parse_completion(provider: &str, body: &Value) -> Result<String, LlmError> {
    body.pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            LlmError(format!(
                "{provider}: response has no choices[0].message.content: {body}"
            ))
        })
}

impl LlmProvider for OpenAiCompatLlm {
    fn complete(&self, prompt: String) -> BoxFuture<'_, Result<String, LlmError>> {
        let body = self.request_body(&prompt);
        Box::pin(async move {
            let response = self
                .http
                .post(&self.endpoint)
                .bearer_auth(&self.api_key)
                .header("content-type", "application/json")
                .body(body.to_string())
                .send()
                .await
                .map_err(|e| transport_error(self.name, e))?;
            let response = error_for_status(self.name, response, |status| {
                (status == 401).then_some("check the API key")
            })
            .await?;
            let text = response
                .text()
                .await
                .map_err(|e| transport_error(self.name, e))?;
            let json: Value = serde_json::from_str(&text)
                .map_err(|e| LlmError(format!("{}: invalid JSON response: {e}", self.name)))?;
            parse_completion(self.name, &json)
        })
    }
}
