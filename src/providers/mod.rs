//! Real [`LlmProvider`](crate::LlmProvider)s, each behind a cargo feature.
//!
//! | Feature | Provider | Auth |
//! |---|---|---|
//! | `codex` | [`CodexLlm`] | ChatGPT subscription, via the official Codex CLI's saved login |
//! | `codex-login` | [`CodexLlm::from_login`] + [`codex_login`] | ChatGPT subscription, via worldfn's own login (`worldfn login codex`) |
//! | `openai-compat` | [`OpenAiCompatLlm`] | API key: OpenAI, DeepSeek, or any compatible server |
//!
//! Both are type-erased behind [`Llm`](crate::Llm), so agents never name a
//! backend: swapping providers, or a provider for `FakeLlm`, changes only how
//! the world is built.

#[cfg(feature = "codex")]
mod codex;
#[cfg(feature = "codex-login")]
pub mod codex_login;
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

/// Splits a server-sent-event body into its `data` payloads.
#[derive(Default)]
pub(crate) struct SseFrames {
    buffer: String,
}

impl SseFrames {
    /// Feed decoded text as it arrives; returns the payloads of the frames it
    /// completed. `\r` is dropped, so CRLF line endings split across chunks
    /// are handled.
    pub(crate) fn push(&mut self, chunk: &str) -> Vec<String> {
        self.buffer.extend(chunk.chars().filter(|&c| c != '\r'));
        let mut payloads = Vec::new();
        while let Some(end) = self.buffer.find("\n\n") {
            let frame: String = self.buffer.drain(..end + 2).collect();
            let data: Vec<&str> = frame
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(|d| d.strip_prefix(' ').unwrap_or(d))
                .collect();
            if !data.is_empty() {
                payloads.push(data.join("\n"));
            }
        }
        payloads
    }
}

/// Read a server-sent-event response, passing each `data` payload to
/// `on_data` until it returns `Ok(true)` or the body ends. A final frame
/// without its closing blank line is still delivered.
pub(crate) async fn read_sse(
    provider: &str,
    mut response: reqwest::Response,
    mut on_data: impl FnMut(&str) -> Result<bool, LlmError>,
) -> Result<(), LlmError> {
    let mut frames = SseFrames::default();
    let mut pending = Vec::new();
    while let Some(bytes) = response
        .chunk()
        .await
        .map_err(|e| transport_error(provider, e))?
    {
        pending.extend_from_slice(&bytes);
        // Only decode complete UTF-8; keep a split code point for later.
        let valid = match std::str::from_utf8(&pending) {
            Ok(s) => s.len(),
            Err(e) => e.valid_up_to(),
        };
        let text = std::str::from_utf8(&pending[..valid]).expect("validated above");
        let payloads = frames.push(text);
        pending.drain(..valid);
        for data in payloads {
            if on_data(&data)? {
                return Ok(());
            }
        }
    }
    for data in frames.push("\n\n") {
        if on_data(&data)? {
            break;
        }
    }
    Ok(())
}

/// Forwards text that `current` gained since the last call, as one delta.
/// Providers accumulate the reply and call this after each event.
pub(crate) struct DeltaForwarder<'a> {
    on_delta: Option<&'a mut (dyn FnMut(crate::ChatDelta) + Send)>,
    sent: usize,
}

impl<'a> DeltaForwarder<'a> {
    pub(crate) fn new(on_delta: Option<&'a mut (dyn FnMut(crate::ChatDelta) + Send)>) -> Self {
        Self { on_delta, sent: 0 }
    }

    pub(crate) fn forward(&mut self, current: &str) {
        if let Some(on_delta) = &mut self.on_delta {
            if current.len() > self.sent {
                on_delta(crate::ChatDelta::Text(current[self.sent..].to_owned()));
                self.sent = current.len();
            }
        }
    }
}

/// An image as a `data:` URL, the form both supported APIs accept.
pub(crate) fn data_url(image: &crate::chat::Image) -> String {
    use base64::Engine as _;
    format!(
        "data:{};base64,{}",
        image.media_type(),
        base64::engine::general_purpose::STANDARD.encode(image.bytes())
    )
}

/// Images may only appear in user messages.
pub(crate) fn reject_misplaced_images(
    provider: &str,
    message: &crate::chat::Message,
) -> Result<(), LlmError> {
    if message.role != crate::chat::MessageRole::User && message.images().next().is_some() {
        return Err(LlmError(format!(
            "{provider}: images are only supported in user messages"
        )));
    }
    Ok(())
}
