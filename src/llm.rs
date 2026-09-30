use std::collections::VecDeque;
use std::fmt;
use std::future::{Ready, ready};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::chat::{ChatRequest, ChatResponse};
use crate::param::unmet;
use crate::{AgentParam, AgentWorld, BoxFuture, ParamError, Requirement, Scope};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmError(pub String);

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "llm error: {}", self.0)
    }
}

impl std::error::Error for LlmError {}

/// A language-model backend.
///
/// One method: a provider-neutral [`ChatRequest`] in, one assistant
/// [`ChatResponse`] out. Returns a [`BoxFuture`] rather than being an
/// `async fn` because providers are type-erased (`Arc<dyn LlmProvider>`) so
/// that `Llm` in a signature does not name a backend.
pub trait LlmProvider: Send + Sync + 'static {
    fn chat(&self, request: ChatRequest) -> BoxFuture<'_, Result<ChatResponse, LlmError>>;
}

/// Parameter: the world's LLM, behind a type-erased provider.
#[derive(Clone)]
pub struct Llm {
    provider: Arc<dyn LlmProvider>,
}

impl Llm {
    pub fn new(provider: impl LlmProvider) -> Self {
        Self {
            provider: Arc::new(provider),
        }
    }

    /// Send a full request: system prompt, history, tools, output format.
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.provider.chat(request).await
    }

    /// One user message in, the reply's text out.
    pub async fn complete(&self, prompt: impl Into<String>) -> Result<String, LlmError> {
        Ok(self.chat(ChatRequest::prompt(prompt)).await?.message.text())
    }
}

impl AgentParam for Llm {
    type State = Llm;
    type Future = Ready<Result<Self, ParamError>>;

    fn describe(out: &mut Vec<Requirement>) {
        out.push(Requirement::Llm);
    }

    fn init(world: &AgentWorld) -> Result<Llm, Vec<Requirement>> {
        world.resource::<Llm>().cloned().ok_or_else(unmet::<Self>)
    }

    fn resolve(state: &mut Llm, _world: &AgentWorld, _scope: &Scope) -> Self::Future {
        ready(Ok(state.clone()))
    }
}

type Responder = Box<dyn Fn(&ChatRequest) -> ChatResponse + Send>;

#[derive(Default)]
struct FakeLlmState {
    script: VecDeque<Result<ChatResponse, LlmError>>,
    responder: Option<Responder>,
    requests: Vec<ChatRequest>,
}

/// A deterministic LLM for tests.
///
/// Replies from its script first, then from an optional responder, and
/// otherwise fails the call as unexpected. Every request is recorded in full.
/// Clones share state, so keep a clone after `provide_llm` to assert on it.
#[derive(Clone, Default)]
pub struct FakeLlm {
    state: Arc<Mutex<FakeLlmState>>,
}

impl FakeLlm {
    /// No script, no responder: every call fails as unexpected.
    pub fn new() -> Self {
        Self::default()
    }

    /// Answer the first call with `answer`.
    pub fn with_answer(answer: impl Into<String>) -> Self {
        Self::new().then_answer(answer)
    }

    /// Answer calls with each item in order.
    pub fn scripted<I, S>(answers: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        answers.into_iter().fold(Self::new(), Self::then_answer)
    }

    /// Append a text answer to the script.
    pub fn then_answer(self, answer: impl Into<String>) -> Self {
        self.then_response(ChatResponse::text(answer))
    }

    /// Append any response (e.g. [`ChatResponse::tool_calls`]) to the script.
    pub fn then_response(self, response: ChatResponse) -> Self {
        self.lock().script.push_back(Ok(response));
        self
    }

    /// Append a failure to the script.
    pub fn then_error(self, message: impl Into<String>) -> Self {
        self.lock().script.push_back(Err(LlmError(message.into())));
        self
    }

    /// Once the script is exhausted, answer with text computed from the last
    /// user message.
    pub fn responding(f: impl Fn(&str) -> String + Send + 'static) -> Self {
        Self::responding_to(move |request| {
            ChatResponse::text(f(&request.last_user_text().unwrap_or_default()))
        })
    }

    /// Once the script is exhausted, compute whole responses from requests.
    pub fn responding_to(f: impl Fn(&ChatRequest) -> ChatResponse + Send + 'static) -> Self {
        let llm = Self::new();
        llm.lock().responder = Some(Box::new(f));
        llm
    }

    /// Echo the last user message back.
    pub fn echo() -> Self {
        Self::responding(str::to_owned)
    }

    /// Every request received so far, in order.
    pub fn requests(&self) -> Vec<ChatRequest> {
        self.lock().requests.clone()
    }

    /// The last user message of every request, in order.
    pub fn prompts(&self) -> Vec<String> {
        self.lock()
            .requests
            .iter()
            .map(|r| r.last_user_text().unwrap_or_default())
            .collect()
    }

    pub fn calls(&self) -> usize {
        self.lock().requests.len()
    }

    fn lock(&self) -> MutexGuard<'_, FakeLlmState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl LlmProvider for FakeLlm {
    fn chat(&self, request: ChatRequest) -> BoxFuture<'_, Result<ChatResponse, LlmError>> {
        let mut state = self.lock();
        let reply = match state.script.pop_front() {
            Some(reply) => reply,
            None => match &state.responder {
                Some(f) => Ok(f(&request)),
                None => Err(LlmError(format!(
                    "FakeLlm received an unexpected call #{}",
                    state.requests.len() + 1
                ))),
            },
        };
        state.requests.push(request);
        Box::pin(std::future::ready(reply))
    }
}
