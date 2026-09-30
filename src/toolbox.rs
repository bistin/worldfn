//! Letting a model call tools: `Toolbox<(A, B, ..)>`.
//!
//! The tuple in the signature is the model's whole tool surface. The toolbox
//! turns each `ToolSpec` into a definition for the model (schema generated
//! from `Request`), and dispatches the model's calls back to the world's
//! handlers: arguments are deserialized into `Request` first, and a call to a
//! tool outside the tuple is refused. The model's arguments are untrusted
//! input, like any other request data.
//!
//! [`Toolbox::run`] is an explicit, bounded loop the agent calls itself: send,
//! run requested tools, send the results back, until the model answers or the
//! step limit is hit. Nothing loops behind the agent's back; agents that want
//! their own control flow use [`Toolbox::definitions`] and
//! [`Toolbox::dispatch`] directly.

use std::fmt;
use std::future::{Ready, ready};
use std::marker::PhantomData;

use schemars::JsonSchema;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::chat::{
    ChatDelta, ChatRequest, ChatResponse, Message, ToolCall, ToolDefinition, ToolResult, Usage,
};
use crate::{
    AgentParam, AgentWorld, BoxFuture, Llm, LlmError, ParamError, Requirement, Scope, Tool,
    ToolSpec, all_tuples,
};

/// A `ToolSpec` a model can call: its request has a schema and can be
/// deserialized, and its response can be serialized. Implemented
/// automatically.
pub trait ModelTool: ToolSpec
where
    Self::Request: DeserializeOwned + JsonSchema,
    Self::Response: Serialize,
{
}

impl<T> ModelTool for T
where
    T: ToolSpec,
    T::Request: DeserializeOwned + JsonSchema,
    T::Response: Serialize,
{
}

/// Tool arguments must be a JSON object; a non-object request type (e.g. a
/// bare `String`) is wrapped as `{"input": ...}`.
const WRAPPED_KEY: &str = "input";

fn request_schema<T: ToolSpec>() -> (Value, bool)
where
    T::Request: JsonSchema,
{
    let schema = serde_json::to_value(schemars::schema_for!(T::Request))
        .expect("a generated schema serializes");
    if schema.get("type").and_then(Value::as_str) == Some("object") {
        (schema, false)
    } else {
        let wrapped = json!({
            "type": "object",
            "properties": { WRAPPED_KEY: schema },
            "required": [WRAPPED_KEY],
        });
        (wrapped, true)
    }
}

fn definition<T>() -> ToolDefinition
where
    T: ToolSpec,
    T::Request: JsonSchema,
{
    ToolDefinition {
        name: T::NAME.to_owned(),
        description: T::DESCRIPTION.to_owned(),
        parameters: request_schema::<T>().0.to_string(),
    }
}

fn error_result(call: &ToolCall, message: String) -> ToolResult {
    ToolResult {
        call_id: call.id.clone(),
        content: message,
        is_error: true,
    }
}

async fn call_one<T>(tool: &Tool<T>, call: &ToolCall) -> ToolResult
where
    T: ToolSpec,
    T::Request: DeserializeOwned + JsonSchema,
    T::Response: Serialize,
{
    let arguments: Value = match serde_json::from_str(&call.arguments) {
        Ok(v) => v,
        Err(e) => return error_result(call, format!("invalid arguments: {e}")),
    };
    let arguments = if request_schema::<T>().1 {
        arguments.get(WRAPPED_KEY).cloned().unwrap_or(Value::Null)
    } else {
        arguments
    };
    let request: T::Request = match serde_json::from_value(arguments) {
        Ok(r) => r,
        Err(e) => return error_result(call, format!("invalid arguments: {e}")),
    };
    match tool.call(request).await {
        Ok(response) => match serde_json::to_string(&response) {
            Ok(content) => ToolResult {
                call_id: call.id.clone(),
                content,
                is_error: false,
            },
            Err(e) => error_result(call, format!("could not serialize the result: {e}")),
        },
        Err(e) => error_result(call, e.0),
    }
}

/// A tuple of model-callable tools. Implemented for tuples of 1 to 8
/// `ModelTool`s.
pub trait ToolSet: Send + 'static {
    /// One `Tool<T>` handle per tool.
    type Handles: Clone + Send + Sync + 'static;

    fn describe(out: &mut Vec<Requirement>);
    fn init(world: &AgentWorld) -> Result<Self::Handles, Vec<Requirement>>;
    fn definitions() -> Vec<ToolDefinition>;
    /// `None` if `call.name` is not in the set.
    fn dispatch<'a>(
        handles: &'a Self::Handles,
        call: &'a ToolCall,
    ) -> Option<BoxFuture<'a, ToolResult>>;
}

macro_rules! impl_tool_set {
    () => {};
    ($($T:ident),+) => {
        impl<$($T),+> ToolSet for ($($T,)+)
        where
            $($T: ToolSpec, $T::Request: DeserializeOwned + JsonSchema, $T::Response: Serialize,)+
        {
            type Handles = ($(Tool<$T>,)+);

            fn describe(out: &mut Vec<Requirement>) {
                $(<Tool<$T> as AgentParam>::describe(out);)+
            }

            #[allow(non_snake_case)]
            fn init(world: &AgentWorld) -> Result<Self::Handles, Vec<Requirement>> {
                let mut missing = Vec::new();
                $(
                    let $T = world.resource::<Tool<$T>>().cloned();
                    if $T.is_none() {
                        <Tool<$T> as AgentParam>::describe(&mut missing);
                    }
                )+
                if !missing.is_empty() {
                    return Err(missing);
                }
                Ok(($($T.unwrap(),)+))
            }

            fn definitions() -> Vec<ToolDefinition> {
                vec![$(definition::<$T>()),+]
            }

            #[allow(non_snake_case)]
            fn dispatch<'a>(
                handles: &'a Self::Handles,
                call: &'a ToolCall,
            ) -> Option<BoxFuture<'a, ToolResult>> {
                let ($($T,)+) = handles;
                $(
                    if call.name == $T::NAME {
                        return Some(Box::pin(call_one($T, call)));
                    }
                )+
                None
            }
        }
    };
}

all_tuples!(impl_tool_set; T0, T1, T2, T3, T4, T5, T6, T7);

/// Parameter: the tools a model may call during this invocation.
pub struct Toolbox<S: ToolSet> {
    handles: S::Handles,
    _set: PhantomData<fn() -> S>,
}

impl<S: ToolSet> Clone for Toolbox<S> {
    fn clone(&self) -> Self {
        Self {
            handles: self.handles.clone(),
            _set: PhantomData,
        }
    }
}

impl<S: ToolSet> fmt::Debug for Toolbox<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<_> = S::definitions().into_iter().map(|d| d.name).collect();
        f.debug_struct("Toolbox").field("tools", &names).finish()
    }
}

impl<S: ToolSet> AgentParam for Toolbox<S> {
    type State = S::Handles;
    type Future = Ready<Result<Self, ParamError>>;

    fn describe(out: &mut Vec<Requirement>) {
        S::describe(out);
    }

    fn init(world: &AgentWorld) -> Result<S::Handles, Vec<Requirement>> {
        S::init(world)
    }

    fn resolve(state: &mut S::Handles, _world: &AgentWorld, _scope: &Scope) -> Self::Future {
        ready(Ok(Toolbox {
            handles: state.clone(),
            _set: PhantomData,
        }))
    }
}

/// Something that happened inside [`Toolbox::run`], for progress events.
#[derive(Debug)]
#[non_exhaustive]
pub enum LoopEvent<'a> {
    /// Reply text as the model generates it, from any step. Text that comes
    /// before a tool call ("let me check…") is included.
    Text(&'a str),
    /// The model asked for a tool; it is about to run.
    ToolCall(&'a ToolCall),
    /// A tool finished (or was refused).
    ToolResult(&'a ToolCall, &'a ToolResult),
}

/// The outcome of [`Toolbox::run`].
#[derive(Debug, Clone)]
pub struct ToolRun {
    /// The model's final answer.
    pub response: ChatResponse,
    /// The request as it ended: the original messages plus every tool call
    /// and result, ready to continue the conversation.
    pub request: ChatRequest,
    /// Every call made, with its result, in order.
    pub calls: Vec<(ToolCall, ToolResult)>,
    /// Summed over all model calls that reported usage.
    pub usage: Usage,
    /// How many times the model was called.
    pub steps: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolLoopError {
    Llm(LlmError),
    /// The model was still calling tools after `max_steps` model calls.
    MaxSteps {
        max_steps: usize,
        calls: Vec<(ToolCall, ToolResult)>,
    },
}

impl fmt::Display for ToolLoopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolLoopError::Llm(e) => e.fmt(f),
            ToolLoopError::MaxSteps { max_steps, calls } => write!(
                f,
                "model still calling tools after {max_steps} step(s) ({} tool call(s))",
                calls.len()
            ),
        }
    }
}

impl std::error::Error for ToolLoopError {}

impl From<LlmError> for ToolLoopError {
    fn from(e: LlmError) -> Self {
        ToolLoopError::Llm(e)
    }
}

impl<S: ToolSet> Toolbox<S> {
    /// The definitions to put in a `ChatRequest`.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        S::definitions()
    }

    /// Run one call. Never fails: bad arguments, tool errors and calls to
    /// tools outside this toolbox all come back as error results for the
    /// model to see.
    pub async fn dispatch(&self, call: &ToolCall) -> ToolResult {
        match S::dispatch(&self.handles, call) {
            Some(result) => result.await,
            None => {
                let available: Vec<_> = S::definitions().into_iter().map(|d| d.name).collect();
                error_result(
                    call,
                    format!(
                        "unknown tool `{}`; available: {}",
                        call.name,
                        available.join(", ")
                    ),
                )
            }
        }
    }

    /// Call the model with these tools, run what it asks for, and repeat
    /// until it answers without tool calls, at most `max_steps` model calls.
    /// Calls within one model turn run in order. `observe` sees streamed reply
    /// text and each call and result as they happen (e.g. to forward them
    /// through an `Emit`).
    pub async fn run(
        &self,
        llm: &Llm,
        request: ChatRequest,
        max_steps: usize,
        mut observe: impl FnMut(LoopEvent<'_>) + Send,
    ) -> Result<ToolRun, ToolLoopError> {
        let mut request = request;
        for definition in self.definitions() {
            if !request.tools.iter().any(|t| t.name == definition.name) {
                request.tools.push(definition);
            }
        }
        let mut calls = Vec::new();
        let mut usage = Usage::default();
        for step in 1..=max_steps {
            let response = llm
                .chat_streaming(request.clone(), |delta| match delta {
                    ChatDelta::Text(text) => observe(LoopEvent::Text(&text)),
                })
                .await?;
            if let Some(u) = response.usage {
                usage.input_tokens += u.input_tokens;
                usage.output_tokens += u.output_tokens;
            }
            let requested: Vec<ToolCall> = response.message.tool_calls().cloned().collect();
            if requested.is_empty() {
                return Ok(ToolRun {
                    response,
                    request,
                    calls,
                    usage,
                    steps: step,
                });
            }
            request = request.message(response.message);
            for call in requested {
                observe(LoopEvent::ToolCall(&call));
                let result = self.dispatch(&call).await;
                observe(LoopEvent::ToolResult(&call, &result));
                request = request.message(Message::tool_result(result.clone()));
                calls.push((call, result));
            }
        }
        Err(ToolLoopError::MaxSteps { max_steps, calls })
    }
}
