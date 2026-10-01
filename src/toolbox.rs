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

use std::time::Duration;

use crate::cancel::{CancelToken, Deadline, StopReason};
use std::sync::Arc;

use crate::chat::{
    ChatDelta, ChatRequest, ChatResponse, FinishReason, Image, Message, Part, ToolCall,
    ToolDefinition, ToolResult, Usage,
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

/// Model-callable tools: a single `ModelTool`, or a tuple of 1 to 8 tool
/// sets. Tuples nest, so more than eight tools can be grouped, e.g.
/// `Toolbox<((A, B, C, D, E), (F, G, H, I))>`.
pub trait ToolSet: Send + 'static {
    /// The `Tool<T>` handles, shaped like the set.
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

impl<T> ToolSet for T
where
    T: ToolSpec,
    T::Request: DeserializeOwned + JsonSchema,
    T::Response: Serialize,
{
    type Handles = Tool<T>;

    fn describe(out: &mut Vec<Requirement>) {
        <Tool<T> as AgentParam>::describe(out);
    }

    fn init(world: &AgentWorld) -> Result<Tool<T>, Vec<Requirement>> {
        world.resource::<Tool<T>>().cloned().ok_or_else(|| {
            let mut missing = Vec::new();
            Self::describe(&mut missing);
            missing
        })
    }

    fn definitions() -> Vec<ToolDefinition> {
        vec![definition::<T>()]
    }

    fn dispatch<'a>(handle: &'a Tool<T>, call: &'a ToolCall) -> Option<BoxFuture<'a, ToolResult>> {
        (call.name == T::NAME).then(|| Box::pin(call_one(handle, call)) as BoxFuture<'a, _>)
    }
}

macro_rules! impl_tool_set {
    () => {};
    ($($S:ident),+) => {
        impl<$($S: ToolSet),+> ToolSet for ($($S,)+) {
            type Handles = ($($S::Handles,)+);

            fn describe(out: &mut Vec<Requirement>) {
                $($S::describe(out);)+
            }

            #[allow(non_snake_case)]
            fn init(world: &AgentWorld) -> Result<Self::Handles, Vec<Requirement>> {
                let mut missing = Vec::new();
                $(
                    let $S = match $S::init(world) {
                        Ok(handles) => Some(handles),
                        Err(unmet) => {
                            missing.extend(unmet);
                            None
                        }
                    };
                )+
                match ($($S,)+) {
                    ($(Some($S),)+) => Ok(($($S,)+)),
                    _ => Err(missing),
                }
            }

            fn definitions() -> Vec<ToolDefinition> {
                let mut all = Vec::new();
                $(all.extend($S::definitions());)+
                all
            }

            #[allow(non_snake_case)]
            fn dispatch<'a>(
                handles: &'a Self::Handles,
                call: &'a ToolCall,
            ) -> Option<BoxFuture<'a, ToolResult>> {
                let ($($S,)+) = handles;
                $(
                    if let Some(result) = $S::dispatch($S, call) {
                        return Some(result);
                    }
                )+
                None
            }
        }
    };
}

all_tuples!(impl_tool_set; S0, S1, S2, S3, S4, S5, S6, S7);

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
    /// Model call number `step` (from 1) returned. `usage` is `None` when
    /// the provider did not report it.
    ModelResponded {
        step: usize,
        usage: Option<&'a Usage>,
    },
    /// An [`Observer`] attached an image for the model, from this call.
    Observed(&'a ToolCall, &'a Observation),
}

/// The outcome of [`Toolbox::run`].
#[derive(Debug, Clone)]
pub struct ToolRun {
    /// The model's final answer.
    pub response: ChatResponse,
    /// The whole conversation: the original messages, every tool call and
    /// result, and the final answer. Append a user message to continue it.
    pub request: ChatRequest,
    /// Every call made, with its result, in order.
    pub calls: Vec<(ToolCall, ToolResult)>,
    /// Summed over all model calls that reported usage.
    pub usage: Usage,
    /// How many times the model was called.
    pub steps: usize,
}

/// An image produced by a tool, for the model to look at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observation {
    /// Shown to the model next to the image, e.g. `"screenshot ss-3 of the
    /// desktop"`.
    pub label: String,
    pub image: Image,
}

/// Turns successful tool results into images for the model, e.g. the PNG
/// behind a screenshot tool's artifact id.
///
/// Tool results are text, so a screenshot tool returns metadata (an id, a
/// size, a hash) and the observer, which the host trusts, looks the pixels up
/// in its own store. The model never names a file to read.
///
/// After all calls of one reply have run, their images are sent in a single
/// user message, so every call's result stays right after the call. Plain
/// closures `Fn(&ToolCall, &ToolResult) -> Vec<Observation>` implement this.
pub trait Observer: Send + Sync {
    fn observe<'a>(
        &'a self,
        call: &'a ToolCall,
        result: &'a ToolResult,
    ) -> BoxFuture<'a, Vec<Observation>>;
}

impl<F> Observer for F
where
    F: Fn(&ToolCall, &ToolResult) -> Vec<Observation> + Send + Sync,
{
    fn observe<'a>(
        &'a self,
        call: &'a ToolCall,
        result: &'a ToolResult,
    ) -> BoxFuture<'a, Vec<Observation>> {
        Box::pin(std::future::ready(self(call, result)))
    }
}

/// Limits for [`Toolbox::run_with`]. Only `max_steps` is required; every
/// other limit is off until set.
#[derive(Clone)]
#[non_exhaustive]
pub struct LoopOptions {
    /// Model calls, including the final one.
    pub max_steps: usize,
    /// Tool calls over the whole run, failed ones included.
    pub max_tool_calls: Option<usize>,
    /// Tool calls the model may request in one reply.
    pub max_calls_per_turn: Option<usize>,
    /// Failed tool calls (bad arguments, unknown tools, tool errors) over the
    /// whole run.
    pub max_failed_calls: Option<usize>,
    /// The same call (name and arguments) failing this many times in a row.
    pub max_repeated_failures: Option<usize>,
    /// Tool results longer than this many bytes are cut, with a note saying
    /// so, before the model sees them.
    pub max_result_bytes: Option<usize>,
    /// Time limit for the whole run, model calls and tools included.
    pub timeout: Option<Duration>,
    /// Stops the run when cancelled.
    pub cancel: Option<CancelToken>,
    /// Attaches images to tool results; see [`Observer`].
    pub observer: Option<Arc<dyn Observer>>,
    /// Most images kept in the conversation. When a step would exceed it, the
    /// oldest images are replaced by a short text note until half the limit
    /// remains. Pruning in large, rare steps keeps the prompt prefix stable
    /// for caching between prunes.
    pub max_images: Option<usize>,
}

impl fmt::Debug for LoopOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoopOptions")
            .field("max_steps", &self.max_steps)
            .field("max_tool_calls", &self.max_tool_calls)
            .field("max_calls_per_turn", &self.max_calls_per_turn)
            .field("max_failed_calls", &self.max_failed_calls)
            .field("max_repeated_failures", &self.max_repeated_failures)
            .field("max_result_bytes", &self.max_result_bytes)
            .field("timeout", &self.timeout)
            .field("cancel", &self.cancel)
            .field("observer", &self.observer.as_ref().map(|_| "Observer"))
            .field("max_images", &self.max_images)
            .finish()
    }
}

impl LoopOptions {
    pub fn new(max_steps: usize) -> Self {
        Self {
            max_steps,
            max_tool_calls: None,
            max_calls_per_turn: None,
            max_failed_calls: None,
            max_repeated_failures: None,
            max_result_bytes: None,
            timeout: None,
            cancel: None,
            observer: None,
            max_images: None,
        }
    }

    pub fn max_tool_calls(mut self, n: usize) -> Self {
        self.max_tool_calls = Some(n);
        self
    }

    pub fn max_calls_per_turn(mut self, n: usize) -> Self {
        self.max_calls_per_turn = Some(n);
        self
    }

    pub fn max_failed_calls(mut self, n: usize) -> Self {
        self.max_failed_calls = Some(n);
        self
    }

    pub fn max_repeated_failures(mut self, n: usize) -> Self {
        self.max_repeated_failures = Some(n);
        self
    }

    pub fn max_result_bytes(mut self, n: usize) -> Self {
        self.max_result_bytes = Some(n);
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn cancel(mut self, token: CancelToken) -> Self {
        self.cancel = Some(token);
        self
    }

    pub fn observer(mut self, observer: impl Observer + 'static) -> Self {
        self.observer = Some(Arc::new(observer));
        self
    }

    pub fn max_images(mut self, n: usize) -> Self {
        self.max_images = Some(n);
        self
    }
}

/// Which [`LoopOptions`] limit ended a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopLimit {
    ToolCalls,
    CallsPerTurn,
    FailedCalls,
    RepeatedFailures,
}

/// Every error carries the tool calls completed before it (empty for
/// [`InvalidToolSurface`](Self::InvalidToolSurface)); see
/// [`calls`](Self::calls). Those calls had their effects: the loop never
/// undoes or repeats them, and a caller deciding whether to retry must look
/// at them first.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolLoopError {
    /// A model call failed. Tools that already ran are in `calls`: they are
    /// not undone, and retrying the whole run would run them again.
    Llm {
        error: LlmError,
        calls: Vec<(ToolCall, ToolResult)>,
    },
    /// The model was still calling tools after `max_steps` model calls.
    MaxSteps {
        max_steps: usize,
        calls: Vec<(ToolCall, ToolResult)>,
    },
    /// The model stopped without a usable answer: cut off at the length
    /// limit, blocked by a content filter, empty, or claiming tool calls it
    /// did not make. A reply like that is never treated as the result.
    Unfinished {
        response: ChatResponse,
        calls: Vec<(ToolCall, ToolResult)>,
    },
    /// A reply that cannot be executed safely, e.g. tool calls with empty or
    /// duplicate ids. Nothing from that reply was run.
    InvalidTurn {
        reason: String,
        calls: Vec<(ToolCall, ToolResult)>,
    },
    LimitExceeded {
        limit: LoopLimit,
        calls: Vec<(ToolCall, ToolResult)>,
    },
    /// Cancelled or out of time. A tool that was running was abandoned at its
    /// next await point; its result is not in `calls`.
    Stopped {
        reason: StopReason,
        calls: Vec<(ToolCall, ToolResult)>,
    },
    /// Two tools in the toolbox share a name.
    InvalidToolSurface(String),
}

impl ToolLoopError {
    /// The tool calls completed before the error.
    pub fn calls(&self) -> &[(ToolCall, ToolResult)] {
        match self {
            ToolLoopError::Llm { calls, .. }
            | ToolLoopError::MaxSteps { calls, .. }
            | ToolLoopError::Unfinished { calls, .. }
            | ToolLoopError::InvalidTurn { calls, .. }
            | ToolLoopError::LimitExceeded { calls, .. }
            | ToolLoopError::Stopped { calls, .. } => calls,
            ToolLoopError::InvalidToolSurface(_) => &[],
        }
    }
}

impl fmt::Display for ToolLoopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.calls().len();
        match self {
            ToolLoopError::Llm { error, .. } if n == 0 => error.fmt(f),
            ToolLoopError::Llm { error, .. } => write!(f, "{error} (after {n} tool call(s))"),
            ToolLoopError::MaxSteps { max_steps, .. } => write!(
                f,
                "model still calling tools after {max_steps} step(s) ({n} tool call(s))"
            ),
            ToolLoopError::Unfinished { response, .. } => write!(
                f,
                "model stopped without a usable answer (finish: {:?}, {n} tool call(s))",
                response.finish
            ),
            ToolLoopError::InvalidTurn { reason, .. } => {
                write!(f, "invalid model reply: {reason} ({n} tool call(s))")
            }
            ToolLoopError::LimitExceeded { limit, .. } => {
                write!(f, "loop limit reached: {limit:?} ({n} tool call(s))")
            }
            ToolLoopError::Stopped { reason, .. } => {
                write!(f, "tool loop stopped: {reason} ({n} tool call(s))")
            }
            ToolLoopError::InvalidToolSurface(reason) => write!(f, "invalid toolbox: {reason}"),
        }
    }
}

impl std::error::Error for ToolLoopError {}

impl From<LlmError> for ToolLoopError {
    fn from(e: LlmError) -> Self {
        ToolLoopError::Llm {
            error: e,
            calls: Vec::new(),
        }
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
    /// Same as [`run_with`](Self::run_with) with only a step limit.
    pub async fn run(
        &self,
        llm: &Llm,
        request: ChatRequest,
        max_steps: usize,
        observe: impl FnMut(LoopEvent<'_>) + Send,
    ) -> Result<ToolRun, ToolLoopError> {
        self.run_with(llm, request, LoopOptions::new(max_steps), observe)
            .await
    }

    /// The tool loop. Each step sends the conversation with exactly this
    /// toolbox's tools (replacing any in `request`), then:
    ///
    /// - a reply with tool calls is checked as a whole (ids, call count)
    ///   before any call runs; calls run in order and their results are
    ///   appended for the next step;
    /// - a reply without tool calls ends the run if it is a complete answer
    ///   (`Stop`, or `Other` with text), and fails as
    ///   [`Unfinished`](ToolLoopError::Unfinished) otherwise.
    ///
    /// `observe` sees streamed text, each call and result, and each model
    /// call's usage, e.g. to forward them through an `Emit`.
    pub async fn run_with(
        &self,
        llm: &Llm,
        request: ChatRequest,
        options: LoopOptions,
        mut observe: impl FnMut(LoopEvent<'_>) + Send,
    ) -> Result<ToolRun, ToolLoopError> {
        let definitions = self.definitions();
        let mut names = std::collections::HashSet::new();
        for definition in &definitions {
            if !names.insert(definition.name.as_str()) {
                return Err(ToolLoopError::InvalidToolSurface(format!(
                    "tool name `{}` is used twice",
                    definition.name
                )));
            }
        }

        let mut request = request;
        // The model sees exactly what `dispatch` can reach.
        request.tools = definitions;
        // Every step shares one prompt prefix; let the provider route them
        // to the same cache.
        if request.cache_key.is_none() {
            request.cache_key = Some(run_cache_key());
        }

        let deadline_signal = CancelToken::new();
        let deadline = options
            .timeout
            .map(|t| Deadline::start(t, deadline_signal.clone()));
        let caller_signal = options.cancel.clone().unwrap_or_default();
        let signals = [&caller_signal, &deadline_signal];
        let stop_reason = || {
            caller_signal.reason().or_else(|| {
                deadline_signal.reason().or_else(|| {
                    deadline
                        .as_ref()
                        .filter(|d| d.passed())
                        .map(|_| StopReason::DeadlineExceeded)
                })
            })
        };

        let mut calls: Vec<(ToolCall, ToolResult)> = Vec::new();
        let mut usage = Usage::default();
        let mut failed = 0usize;
        let mut repeated: Option<(String, String, usize)> = None;

        for step in 1..=options.max_steps {
            if let Some(reason) = stop_reason() {
                return Err(ToolLoopError::Stopped { reason, calls });
            }
            if let Some(max) = options.max_images {
                prune_images(&mut request, max);
            }
            let reply = CancelToken::race(
                &signals,
                llm.chat_streaming(request.clone(), |delta| match delta {
                    ChatDelta::Text(text) => observe(LoopEvent::Text(&text)),
                }),
            )
            .await;
            let response = match reply {
                Ok(Ok(response)) => response,
                Ok(Err(error)) => return Err(ToolLoopError::Llm { error, calls }),
                Err(reason) => return Err(ToolLoopError::Stopped { reason, calls }),
            };
            observe(LoopEvent::ModelResponded {
                step,
                usage: response.usage.as_ref(),
            });
            if let Some(u) = response.usage {
                usage += u;
            }

            let requested: Vec<ToolCall> = response.message.tool_calls().cloned().collect();
            if requested.is_empty() {
                let has_text = !response.message.text().trim().is_empty();
                let complete = match response.finish {
                    FinishReason::Stop => has_text,
                    FinishReason::Other => has_text,
                    _ => false,
                };
                if !complete {
                    return Err(ToolLoopError::Unfinished { response, calls });
                }
                request = request.message(response.message.clone());
                return Ok(ToolRun {
                    response,
                    request,
                    calls,
                    usage,
                    steps: step,
                });
            }

            // A reply cut off at the length limit or filtered may hold calls
            // the model never meant to finish; run none of them.
            if matches!(
                response.finish,
                FinishReason::Length | FinishReason::ContentFilter
            ) {
                return Err(ToolLoopError::Unfinished { response, calls });
            }
            if let Some(reason) = invalid_turn(&requested) {
                return Err(ToolLoopError::InvalidTurn { reason, calls });
            }
            if options
                .max_calls_per_turn
                .is_some_and(|max| requested.len() > max)
            {
                return Err(ToolLoopError::LimitExceeded {
                    limit: LoopLimit::CallsPerTurn,
                    calls,
                });
            }

            request = request.message(response.message);
            // Full results of this reply, for the observer: the model may see
            // a truncated copy, but an artifact id must survive intact.
            let mut batch: Vec<(ToolCall, ToolResult)> = Vec::new();
            for call in requested {
                if options.max_tool_calls.is_some_and(|max| calls.len() >= max) {
                    return Err(ToolLoopError::LimitExceeded {
                        limit: LoopLimit::ToolCalls,
                        calls,
                    });
                }
                if let Some(reason) = stop_reason() {
                    return Err(ToolLoopError::Stopped { reason, calls });
                }
                observe(LoopEvent::ToolCall(&call));
                let full = match CancelToken::race(&signals, self.dispatch(&call)).await {
                    Ok(result) => result,
                    Err(reason) => return Err(ToolLoopError::Stopped { reason, calls }),
                };
                let mut result = full.clone();
                if let Some(max) = options.max_result_bytes {
                    truncate_result(&mut result.content, max);
                }
                if options.observer.is_some() && !full.is_error {
                    batch.push((call.clone(), full));
                }
                observe(LoopEvent::ToolResult(&call, &result));
                request = request.message(Message::tool_result(result.clone()));

                let limit = if result.is_error {
                    failed += 1;
                    let count = match &repeated {
                        Some((name, args, n)) if *name == call.name && *args == call.arguments => {
                            n + 1
                        }
                        _ => 1,
                    };
                    repeated = Some((call.name.clone(), call.arguments.clone(), count));
                    if options
                        .max_repeated_failures
                        .is_some_and(|max| count >= max)
                    {
                        Some(LoopLimit::RepeatedFailures)
                    } else if options.max_failed_calls.is_some_and(|max| failed > max) {
                        Some(LoopLimit::FailedCalls)
                    } else {
                        None
                    }
                } else {
                    repeated = None;
                    None
                };
                calls.push((call, result));
                if let Some(limit) = limit {
                    return Err(ToolLoopError::LimitExceeded { limit, calls });
                }
            }

            if let Some(observer) = &options.observer {
                let mut parts = Vec::new();
                for (call, result) in &batch {
                    let observed =
                        CancelToken::race(&signals, observer.observe(call, result)).await;
                    let observations = match observed {
                        Ok(observations) => observations,
                        Err(reason) => return Err(ToolLoopError::Stopped { reason, calls }),
                    };
                    for observation in observations {
                        observe(LoopEvent::Observed(call, &observation));
                        parts.push(Part::Text(format!(
                            "{} (from tool call {} `{}`):",
                            observation.label, call.id, call.name
                        )));
                        parts.push(Part::Image(observation.image));
                    }
                }
                if !parts.is_empty() {
                    request = request.message(Message::user_parts(parts));
                }
            }
        }
        Err(ToolLoopError::MaxSteps {
            max_steps: options.max_steps,
            calls,
        })
    }
}

/// Replace the oldest images with a note once there are more than `max`,
/// keeping the newest `max / 2` (at least one).
fn prune_images(request: &mut ChatRequest, max: usize) {
    let total: usize = request.messages.iter().map(|m| m.images().count()).sum();
    if total <= max {
        return;
    }
    let mut to_drop = total - (max / 2).max(1).min(total);
    for message in &mut request.messages {
        for part in &mut message.parts {
            if to_drop == 0 {
                return;
            }
            if let Part::Image(image) = part {
                *part = Part::Text(format!(
                    "[earlier image removed to save context: {} {}x{}]",
                    image.media_type(),
                    image.width(),
                    image.height()
                ));
                to_drop -= 1;
            }
        }
    }
}

/// Why a reply's tool calls cannot be run, if they cannot.
fn invalid_turn(calls: &[ToolCall]) -> Option<String> {
    let mut ids = std::collections::HashSet::new();
    for call in calls {
        if call.id.is_empty() {
            return Some(format!("tool call `{}` has no id", call.name));
        }
        // Results are matched to calls by id, so ids must be unique within a
        // reply. Providers differ on uniqueness across replies; that is not
        // required here.
        if !ids.insert(call.id.as_str()) {
            return Some(format!("tool call id `{}` is used twice", call.id));
        }
    }
    None
}

/// Cut `content` to at most `max` bytes (on a character boundary) and say
/// how much was dropped.
fn truncate_result(content: &mut String, max: usize) {
    if content.len() <= max {
        return;
    }
    let total = content.len();
    let cut = (0..=max)
        .rev()
        .find(|&i| content.is_char_boundary(i))
        .unwrap_or(0);
    content.truncate(cut);
    content.push_str(&format!("\n[truncated: showing {cut} of {total} bytes]"));
}

/// A key unique to this process and run; not a secret.
fn run_cache_key() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!(
        "worldfn-{:x}-{:x}-{}",
        std::process::id(),
        nanos,
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}
