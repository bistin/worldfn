//! Real [`LlmProvider`](crate::LlmProvider)s, each behind a cargo feature.
//!
//! | Feature | Provider | Auth |
//! |---|---|---|
//! | `codex` | [`CodexLlm`] | ChatGPT subscription, via the official Codex CLI's saved login |
//! | `openai-compat` | [`OpenAiCompatLlm`] | API key: OpenAI, DeepSeek, or any compatible server |
//!
//! Both are type-erased behind [`Llm`](crate::Llm), so agents never name a
//! backend: swapping providers, or a provider for `FakeLlm`, changes only how
//! the world is built.

#[cfg(feature = "codex")]
mod codex;
#[cfg(feature = "openai-compat")]
mod openai_compat;

#[cfg(feature = "codex")]
pub use codex::CodexLlm;
#[cfg(feature = "openai-compat")]
pub use openai_compat::{JsonMode, OpenAiCompatLlm};

use crate::LlmError;

const USER_AGENT: &str = concat!("worldfn/", env!("CARGO_PKG_VERSION"));

pub(crate) fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .build()
        .expect("static reqwest client configuration is valid")
}

/// Turn a non-success HTTP response into an `LlmError` carrying the status and
/// (truncated) body, which is where providers put the actual reason.
pub(crate) async fn error_for_status(
    provider: &str,
    response: reqwest::Response,
    hint: impl FnOnce(u16) -> Option<&'static str>,
) -> Result<reqwest::Response, LlmError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let mut body = response.text().await.unwrap_or_default();
    if body.len() > 500 {
        let cut = (0..=500)
            .rev()
            .find(|&i| body.is_char_boundary(i))
            .unwrap_or(0);
        body.truncate(cut);
        body.push('…');
    }
    let mut message = format!("{provider}: HTTP {status}: {body}");
    if let Some(hint) = hint(status.as_u16()) {
        message.push_str(" (");
        message.push_str(hint);
        message.push(')');
    }
    Err(LlmError(message))
}

pub(crate) fn transport_error(provider: &str, error: reqwest::Error) -> LlmError {
    LlmError(format!("{provider}: request failed: {error}"))
}
