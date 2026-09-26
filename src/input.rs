use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;
use std::fmt;
use std::future::{Ready, ready};
use std::ops::Deref;
use std::sync::Arc;

use crate::{AgentParam, AgentWorld, ParamError, Requirement};

/// Per-invocation inputs, keyed by type. Separate from the world: the world
/// holds long-lived bindings, the scope holds what *this* call is about.
#[derive(Clone, Default)]
pub struct Scope {
    inputs: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
}

impl Scope {
    pub fn new() -> Self {
        Self::default()
    }

    /// A scope holding a single input.
    pub fn of<T: Send + Sync + 'static>(input: T) -> Self {
        Self::new().with(input)
    }

    /// Add (or replace) an input.
    pub fn with<T: Send + Sync + 'static>(mut self, input: T) -> Self {
        self.inputs.insert(TypeId::of::<T>(), Arc::new(input));
        self
    }

    pub fn get<T: Send + Sync + 'static>(&self) -> Option<Arc<T>> {
        self.inputs.get(&TypeId::of::<T>())?.clone().downcast().ok()
    }
}

impl fmt::Debug for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scope")
            .field("inputs", &self.inputs.len())
            .finish()
    }
}

/// Parameter: an input of type `T` from the invocation [`Scope`].
///
/// Cannot be checked at `prepare` time; a missing input fails when the agent
/// is started, still before the function body runs.
pub struct Input<T>(Arc<T>);

impl<T> Input<T> {
    pub fn into_inner(self) -> Arc<T> {
        self.0
    }
}

impl<T> Clone for Input<T> {
    fn clone(&self) -> Self {
        Input(self.0.clone())
    }
}

impl<T> Deref for Input<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: fmt::Debug> fmt::Debug for Input<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Input").field(&self.0).finish()
    }
}

impl<T: Send + Sync + 'static> AgentParam for Input<T> {
    type State = ();
    type Future = Ready<Result<Self, ParamError>>;

    fn describe(out: &mut Vec<Requirement>) {
        out.push(Requirement::Input {
            type_name: type_name::<T>(),
        });
    }

    fn init(_world: &AgentWorld) -> Result<(), Vec<Requirement>> {
        Ok(())
    }

    fn resolve(_state: &mut (), _world: &AgentWorld, scope: &Scope) -> Self::Future {
        ready(scope.get::<T>().map(Input).ok_or_else(|| ParamError {
            param: type_name::<Self>(),
            message: "not present in the invocation scope".into(),
        }))
    }
}

/// The canonical agent input: what this invocation is asked to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Task(pub String);

impl Task {
    pub fn new(task: impl Into<String>) -> Self {
        Self(task.into())
    }
}
