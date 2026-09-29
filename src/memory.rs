use std::any::type_name;
use std::collections::HashSet;
use std::future::{Ready, ready};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::param::unmet;
use crate::{AgentParam, AgentWorld, BoxFuture, ContextError, ParamError, Requirement, Scope};

/// A long-term memory backend that can be searched by query.
///
/// Boxed future for the same reason as [`LlmProvider`](crate::LlmProvider):
/// backends are type-erased behind [`Memory`].
pub trait MemoryStore: Send + Sync + 'static {
    fn search<'a>(
        &'a self,
        query: &'a str,
        limit: usize,
    ) -> BoxFuture<'a, Result<Vec<String>, ContextError>>;
}

/// Parameter / service: the world's memory backend. Usually consumed through
/// [`RelevantMemory`](crate::RelevantMemory) rather than directly.
#[derive(Clone)]
pub struct Memory {
    store: Arc<dyn MemoryStore>,
}

impl Memory {
    pub fn new(store: impl MemoryStore) -> Self {
        Self {
            store: Arc::new(store),
        }
    }

    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<String>, ContextError> {
        self.store.search(query, limit).await
    }
}

impl AgentParam for Memory {
    type State = Memory;
    type Future = Ready<Result<Self, ParamError>>;

    fn describe(out: &mut Vec<Requirement>) {
        out.push(Requirement::Service {
            type_name: type_name::<Memory>(),
        });
    }

    fn init(world: &AgentWorld) -> Result<Memory, Vec<Requirement>> {
        world
            .resource::<Memory>()
            .cloned()
            .ok_or_else(unmet::<Self>)
    }

    fn resolve(state: &mut Memory, _world: &AgentWorld, _scope: &Scope) -> Self::Future {
        ready(Ok(state.clone()))
    }
}

#[derive(Default)]
struct FakeMemoryState {
    entries: Vec<String>,
    failure: Option<String>,
    queries: Vec<(String, usize)>,
}

/// A deterministic in-memory store for tests.
///
/// Ranks entries by the number of words (longer than two characters) they
/// share with the query, keeping insertion order among ties and dropping
/// entries that share none. This is a stand-in for real retrieval, not an
/// implementation of it. Every query is recorded; clones share state.
#[derive(Clone, Default)]
pub struct FakeMemory {
    state: Arc<Mutex<FakeMemoryState>>,
}

impl FakeMemory {
    pub fn new<I, S>(entries: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let memory = Self::default();
        memory.lock().entries = entries.into_iter().map(Into::into).collect();
        memory
    }

    /// A store whose every search fails with `message`.
    pub fn failing(message: impl Into<String>) -> Self {
        let memory = Self::default();
        memory.lock().failure = Some(message.into());
        memory
    }

    /// Every `(query, limit)` received so far, in order.
    pub fn queries(&self) -> Vec<(String, usize)> {
        self.lock().queries.clone()
    }

    fn lock(&self) -> MutexGuard<'_, FakeMemoryState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

pub(crate) fn words(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
        .map(str::to_lowercase)
        .collect()
}

impl MemoryStore for FakeMemory {
    fn search<'a>(
        &'a self,
        query: &'a str,
        limit: usize,
    ) -> BoxFuture<'a, Result<Vec<String>, ContextError>> {
        let mut state = self.lock();
        state.queries.push((query.to_owned(), limit));
        if let Some(message) = &state.failure {
            return Box::pin(ready(Err(ContextError(message.clone()))));
        }
        let query = words(query);
        let mut scored: Vec<(usize, &String)> = state
            .entries
            .iter()
            .map(|entry| (words(entry).intersection(&query).count(), entry))
            .filter(|(score, _)| *score > 0)
            .collect();
        // Stable sort keeps insertion order among equal scores.
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        let hits = scored
            .into_iter()
            .take(limit)
            .map(|(_, e)| e.clone())
            .collect();
        Box::pin(ready(Ok(hits)))
    }
}
