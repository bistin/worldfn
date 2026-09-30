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
pub enum Part {
    Text(String),
    /// The model asking for a tool to run (assistant messages only).
    ToolCall(ToolCall),
    /// The outcome of a tool call (tool messages only).
    ToolResult(ToolResult),
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

    /// The text of the last user message, if any.
    pub fn last_user_text(&self) -> Option<String> {
        self.messages
            .iter()
            .rev()
            .find(|m| m.role == MessageRole::User)
            .map(Message::text)
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
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
