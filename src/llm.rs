use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::{AgentParam, AgentWorld, BoxFuture, Requirement};

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
/// Returns a [`BoxFuture`] instead of being an `async fn`: native
/// `async fn` in traits is not object-safe (`dyn`-compatible), and the world
/// stores providers as `Arc<dyn LlmProvider>`.
pub trait LlmProvider: Send + Sync + 'static {
    fn complete(&self, prompt: String) -> BoxFuture<'_, Result<String, LlmError>>;
}

/// Parameter: access to the world's LLM provider.
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

    pub async fn complete(&self, prompt: impl Into<String>) -> Result<String, LlmError> {
        self.provider.complete(prompt.into()).await
    }
}

impl AgentParam for Llm {
    fn describe(out: &mut Vec<Requirement>) {
        out.push(Requirement::Llm);
    }

    fn fetch(world: &AgentWorld) -> Result<Self, Vec<Requirement>> {
        world
            .resource::<Llm>()
            .cloned()
            .ok_or_else(|| vec![Requirement::Llm])
    }
}

type Responder = Box<dyn Fn(&str) -> String + Send>;

#[derive(Default)]
struct FakeLlmState {
    script: VecDeque<String>,
    fallback: Option<Responder>,
    prompts: Vec<String>,
}

/// A deterministic LLM for tests.
///
/// Answers from a script first, then from an optional responder closure, and
/// otherwise errors. Clones share state, so a test can keep a handle after
/// inserting the fake into a world and inspect [`FakeLlm::prompts`].
#[derive(Clone, Default)]
pub struct FakeLlm {
    state: Arc<Mutex<FakeLlmState>>,
}

impl FakeLlm {
    /// Reply with each item in order; error once exhausted.
    pub fn scripted<I, S>(responses: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let llm = Self::default();
        llm.lock().script = responses.into_iter().map(Into::into).collect();
        llm
    }

    /// Compute every reply from the prompt.
    pub fn responding(f: impl Fn(&str) -> String + Send + 'static) -> Self {
        let llm = Self::default();
        llm.lock().fallback = Some(Box::new(f));
        llm
    }

    /// Echo the prompt back.
    pub fn echo() -> Self {
        Self::responding(str::to_owned)
    }

    /// Every prompt received so far, in order.
    pub fn prompts(&self) -> Vec<String> {
        self.lock().prompts.clone()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, FakeLlmState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl LlmProvider for FakeLlm {
    fn complete(&self, prompt: String) -> BoxFuture<'_, Result<String, LlmError>> {
        let mut state = self.lock();
        let reply = match state.script.pop_front() {
            Some(reply) => Ok(reply),
            None => match &state.fallback {
                Some(f) => Ok(f(&prompt)),
                None => Err(LlmError("FakeLlm script exhausted".into())),
            },
        };
        state.prompts.push(prompt);
        Box::pin(std::future::ready(reply))
    }
}
