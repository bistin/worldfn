//! Provider-neutral chat types: the contract between agents and LLM backends.
//!
//! Shaped after what current SDKs converge on (Vercel AI SDK, pi-ai, rig,
//! LangChain messages): a request is a system prompt, a list of messages made
//! of content parts, optional tool definitions, and an output format; a
//! response is one assistant message plus why it stopped and token usage.
//!
//! JSON payloads (tool arguments, schemas) are kept as JSON *text*, so the core
//! stays free of dependencies; the `structured` feature adds typed helpers on
//! top (see [`Llm::complete_as`](crate::Llm::complete_as)).

/// Who a message is from. The system prompt is not a message; it lives on
/// [`ChatRequest::system`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MessageRole {
    User,
    Assistant,
    /// Results of tool calls, sent back to the model.
    Tool,
}

/// One piece of a message.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Part {
    Text(String),
    /// The model asking for a tool to run (assistant messages only).
    ToolCall(ToolCall),
    /// The outcome of a tool call (tool messages only).
    ToolResult(ToolResult),
    /// An image for the model to look at (user messages only). Providers
    /// that cannot take images fail rather than drop it.
    Image(Image),
}

/// A validated image: PNG or JPEG bytes with their dimensions. The bytes are
/// shared, so cloning a request that carries screenshots stays cheap.
#[derive(Clone, PartialEq, Eq)]
pub struct Image {
    media_type: &'static str,
    bytes: std::sync::Arc<[u8]>,
    width: u32,
    height: u32,
}

/// Why bytes were not accepted as an [`Image`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageError(pub String);

impl std::fmt::Display for ImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid image: {}", self.0)
    }
}

impl std::error::Error for ImageError {}

impl Image {
    /// Accepts PNG (dimensions read from its header) or JPEG (dimensions
    /// from its first frame header). Anything else is refused.
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Result<Self, ImageError> {
        let bytes: Vec<u8> = bytes.into();
        let (media_type, (width, height)) = if let Some(size) = png_size(&bytes) {
            ("image/png", size)
        } else if let Some(size) = jpeg_size(&bytes) {
            ("image/jpeg", size)
        } else {
            return Err(ImageError("not a PNG or JPEG image".into()));
        };
        if width == 0 || height == 0 {
            return Err(ImageError(format!("empty image ({width}x{height})")));
        }
        Ok(Self {
            media_type,
            bytes: bytes.into(),
            width,
            height,
        })
    }

    /// `image/png` or `image/jpeg`.
    pub fn media_type(&self) -> &'static str {
        self.media_type
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

/// Never prints the pixels.
impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Image({}, {}x{}, {} bytes)",
            self.media_type,
            self.width,
            self.height,
            self.bytes.len()
        )
    }
}

fn png_size(bytes: &[u8]) -> Option<(u32, u32)> {
    const SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
    // Signature, then the IHDR chunk: length, "IHDR", width, height.
    if bytes.len() < 24 || !bytes.starts_with(SIGNATURE) || &bytes[12..16] != b"IHDR" {
        return None;
    }
    let be = |i: usize| u32::from_be_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
    Some((be(16), be(20)))
}

fn jpeg_size(bytes: &[u8]) -> Option<(u32, u32)> {
    if !bytes.starts_with(&[0xFF, 0xD8]) {
        return None;
    }
    let mut i = 2;
    while i + 4 <= bytes.len() {
        if bytes[i] != 0xFF {
            return None;
        }
        let marker = bytes[i + 1];
        let len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
        // Start-of-frame markers carry the size; skip DHT (C4), JPG (C8), DAC (CC).
        if matches!(marker, 0xC0..=0xCF) && !matches!(marker, 0xC4 | 0xC8 | 0xCC) {
            if i + 9 > bytes.len() {
                return None;
            }
            let h = u16::from_be_bytes([bytes[i + 5], bytes[i + 6]]) as u32;
            let w = u16::from_be_bytes([bytes[i + 7], bytes[i + 8]]) as u32;
            return Some((w, h));
        }
        i += 2 + len;
    }
    None
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Provider-assigned id, echoed back in the matching [`ToolResult`].
    pub id: String,
    pub name: String,
    /// Arguments as JSON text, exactly as the model produced them.
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResult {
    pub call_id: String,
    pub content: String,
    pub is_error: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: MessageRole,
    pub parts: Vec<Part>,
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            parts: vec![Part::Text(text.into())],
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            parts: vec![Part::Text(text.into())],
        }
    }

    /// A user message made of parts, e.g. text and images.
    pub fn user_parts(parts: impl IntoIterator<Item = Part>) -> Self {
        Self {
            role: MessageRole::User,
            parts: parts.into_iter().collect(),
        }
    }

    pub fn tool_result(result: ToolResult) -> Self {
        Self {
            role: MessageRole::Tool,
            parts: vec![Part::ToolResult(result)],
        }
    }

    /// All text parts joined.
    pub fn text(&self) -> String {
        self.parts
            .iter()
            .filter_map(|p| match p {
                Part::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    pub fn images(&self) -> impl Iterator<Item = &Image> {
        self.parts.iter().filter_map(|p| match p {
            Part::Image(image) => Some(image),
            _ => None,
        })
    }

    pub fn tool_calls(&self) -> impl Iterator<Item = &ToolCall> {
        self.parts.iter().filter_map(|p| match p {
            Part::ToolCall(call) => Some(call),
            _ => None,
        })
    }
}

/// A tool the model may call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema of the arguments, as JSON text.
    pub parameters: String,
}

/// What shape the reply should have.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum OutputFormat {
    #[default]
    Text,
    /// A JSON value matching `schema` (JSON Schema, as JSON text). Providers
    /// use native structured output where they have it and fall back to
    /// instructions otherwise; callers must still validate the result.
    Json { name: String, schema: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ChatRequest {
    pub system: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub output: OutputFormat,
    pub max_output_tokens: Option<u32>,
    /// Groups requests that share a prompt prefix, so a provider can route
    /// them to where that prefix is already cached (`prompt_cache_key`). Use
    /// one key per conversation or agent run. Never affects the reply.
    pub cache_key: Option<String>,
}

impl ChatRequest {
    pub fn new() -> Self {
        Self::default()
    }

    /// Shorthand for a single user message.
    pub fn prompt(text: impl Into<String>) -> Self {
        Self::new().user(text)
    }

    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn user(mut self, text: impl Into<String>) -> Self {
        self.messages.push(Message::user(text));
        self
    }

    pub fn assistant(mut self, text: impl Into<String>) -> Self {
        self.messages.push(Message::assistant(text));
        self
    }

    pub fn message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    pub fn tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn output(mut self, output: OutputFormat) -> Self {
        self.output = output;
        self
    }

    pub fn max_output_tokens(mut self, max: u32) -> Self {
        self.max_output_tokens = Some(max);
        self
    }

    /// See [`cache_key`](Self::cache_key).
    pub fn cache_key(mut self, key: impl Into<String>) -> Self {
        self.cache_key = Some(key.into());
        self
    }

    /// The text of the last user message, if any.
    pub fn last_user_text(&self) -> Option<String> {
        self.messages
            .iter()
            .rev()
            .find(|m| m.role == MessageRole::User)
            .map(Message::text)
    }
}

/// A piece of a reply, delivered while the model is still generating it. See
/// [`Llm::chat_streaming`](crate::Llm::chat_streaming).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChatDelta {
    /// More reply text. Concatenating every `Text` delta of one call gives
    /// the reply's text.
    Text(String),
}

/// Why the model stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// A complete answer.
    Stop,
    /// It wants tool calls run (see [`Message::tool_calls`]).
    ToolCalls,
    /// It hit the output token limit; the text may be cut off.
    Length,
    /// Blocked by the provider's content filter.
    ContentFilter,
    Other,
}

/// Tokens a call used, as the provider reported them.
///
/// `cached_input_tokens` is the part of `input_tokens` served from the
/// provider's prompt cache (a prefix it had seen recently); `reasoning_tokens`
/// is the part of `output_tokens` spent on hidden reasoning. Both are 0 when
/// the provider does not report them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Usage {
    pub input_tokens: u64,
    pub cached_input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: u64,
}

impl Usage {
    pub fn new(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
            ..Self::default()
        }
    }

    /// Fraction of input tokens served from the cache, 0.0 to 1.0.
    pub fn cache_hit_rate(&self) -> f64 {
        if self.input_tokens == 0 {
            0.0
        } else {
            self.cached_input_tokens as f64 / self.input_tokens as f64
        }
    }
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, other: Self) {
        self.input_tokens += other.input_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.output_tokens += other.output_tokens;
        self.reasoning_tokens += other.reasoning_tokens;
    }
}

impl std::ops::Add for Usage {
    type Output = Self;
    fn add(mut self, other: Self) -> Self {
        self += other;
        self
    }
}

/// `in 1200 (cached 1024, 85%) · out 80 (reasoning 32)`
impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "in {}", self.input_tokens)?;
        if self.cached_input_tokens > 0 {
            write!(
                f,
                " (cached {}, {:.0}%)",
                self.cached_input_tokens,
                self.cache_hit_rate() * 100.0
            )?;
        }
        write!(f, " · out {}", self.output_tokens)?;
        if self.reasoning_tokens > 0 {
            write!(f, " (reasoning {})", self.reasoning_tokens)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatResponse {
    /// Always an assistant message.
    pub message: Message,
    pub finish: FinishReason,
    /// `None` when the provider did not report usage.
    pub usage: Option<Usage>,
}

impl ChatResponse {
    /// A plain text answer that stopped normally.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            message: Message::assistant(text),
            finish: FinishReason::Stop,
            usage: None,
        }
    }

    /// An answer asking for tool calls.
    pub fn tool_calls(calls: impl IntoIterator<Item = ToolCall>) -> Self {
        Self {
            message: Message {
                role: MessageRole::Assistant,
                parts: calls.into_iter().map(Part::ToolCall).collect(),
            },
            finish: FinishReason::ToolCalls,
            usage: None,
        }
    }
}

/// An instruction appended to the system prompt when a provider has no native
/// structured output, or as a belt-and-braces hint when it does.
#[cfg_attr(not(feature = "http"), allow(dead_code))]
pub(crate) fn json_instruction(name: &str, schema: &str) -> String {
    format!(
        "Reply with only a JSON value (no prose, no code fence) named `{name}` \
         that matches this JSON Schema:\n{schema}"
    )
}
