use std::any::type_name;
use std::collections::HashSet;
use std::ops::Deref;

use crate::{AgentParam, AgentWorld, Requirement, Res};

/// A kind of context that is *derived* from other parameters rather than
/// stored directly — e.g. "the memories relevant to the current task".
///
/// Its dependencies are themselves an [`AgentParam`] (usually a tuple), so
/// contexts compose, and their requirements show up nested in
/// [`Requirement::Context`].
pub trait ContextSource: Sized + Send + 'static {
    type Deps: AgentParam;
    fn build(deps: Self::Deps) -> Self;
}

/// Parameter: context of kind `C`, built fresh for each run.
#[derive(Debug, Clone)]
pub struct Context<C>(pub C);

impl<C> Context<C> {
    pub fn into_inner(self) -> C {
        self.0
    }
}

impl<C> Deref for Context<C> {
    type Target = C;
    fn deref(&self) -> &C {
        &self.0
    }
}

impl<C: ContextSource> AgentParam for Context<C> {
    fn describe(out: &mut Vec<Requirement>) {
        let mut needs = Vec::new();
        C::Deps::describe(&mut needs);
        out.push(Requirement::Context {
            type_name: type_name::<C>(),
            needs,
        });
    }

    fn fetch(world: &AgentWorld) -> Result<Self, Vec<Requirement>> {
        match C::Deps::fetch(world) {
            Ok(deps) => Ok(Context(C::build(deps))),
            Err(needs) => Err(vec![Requirement::Context {
                type_name: type_name::<C>(),
                needs,
            }]),
        }
    }
}

/// Resource: what the agent is currently being asked to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Task(pub String);

impl Task {
    pub fn new(task: impl Into<String>) -> Self {
        Self(task.into())
    }
}

/// Resource: a naive long-term memory.
#[derive(Debug, Clone, Default)]
pub struct MemoryStore {
    pub entries: Vec<String>,
}

impl MemoryStore {
    pub fn new<I, S>(entries: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            entries: entries.into_iter().map(Into::into).collect(),
        }
    }
}

/// Context: up to [`RelevantMemory::LIMIT`] memory entries that share a word
/// with the current [`Task`], most overlapping first. A stand-in for
/// embedding retrieval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelevantMemory {
    pub entries: Vec<String>,
}

impl RelevantMemory {
    pub const LIMIT: usize = 3;
}

fn words(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
        .map(str::to_lowercase)
        .collect()
}

impl ContextSource for RelevantMemory {
    type Deps = (Res<MemoryStore>, Res<Task>);

    fn build((store, task): Self::Deps) -> Self {
        let query = words(&task.0);
        let mut scored: Vec<(usize, &String)> = store
            .entries
            .iter()
            .map(|entry| (words(entry).intersection(&query).count(), entry))
            .filter(|(score, _)| *score > 0)
            .collect();
        // Stable sort keeps insertion order among equal scores.
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        let entries = scored
            .into_iter()
            .take(Self::LIMIT)
            .map(|(_, e)| e.clone())
            .collect();
        RelevantMemory { entries }
    }
}
