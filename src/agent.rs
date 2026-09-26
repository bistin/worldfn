use std::any::type_name;
use std::borrow::Cow;
use std::fmt;
use std::future::ready;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, PoisonError};

use crate::param::short_type_name;
use crate::world::WorldStamp;
use crate::{
    AgentFunction, AgentParam, AgentWorld, BoxFuture, Diagnostics, Requirement, RunError, Scope,
};

/// The future of a started agent: resolution (including any async context
/// materialization), then the function body. Boxed so that [`Agent`] stays
/// object-safe; `'static` because it owns everything it uses.
pub type AgentFuture<O> = BoxFuture<'static, Result<O, RunError>>;

/// Metadata derived from an agent's signature at conversion time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentMeta {
    /// Diagnostic name. Defaults to the function's short type name.
    pub name: Cow<'static, str>,
    /// One entry per declared requirement, in parameter order.
    pub params: Vec<Requirement>,
}

/// Renders the signature as a tree, with context needs nested:
///
/// ```text
/// researcher
/// ├── requires Llm
/// ├── can call Tool<web_search>
/// └── requires Context<RelevantMemory>
///     ├── requires Memory
///     └── reads Input<Task>
/// ```
impl fmt::Display for AgentMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        write_tree(f, &self.params, "")
    }
}

fn write_tree(f: &mut fmt::Formatter<'_>, reqs: &[Requirement], prefix: &str) -> fmt::Result {
    for (i, requirement) in reqs.iter().enumerate() {
        let last = i + 1 == reqs.len();
        let branch = if last { "└──" } else { "├──" };
        let (verb, needs) = match requirement {
            Requirement::Tool { .. } => ("can call", None),
            Requirement::Input { .. } => ("reads", None),
            Requirement::Optional(inner) => match &**inner {
                Requirement::Context { needs, .. } => ("may use", Some(needs)),
                _ => ("may use", None),
            },
            Requirement::Context { needs, .. } => ("requires", Some(needs)),
            _ => ("requires", None),
        };
        write!(f, "\n{prefix}{branch} {verb} {requirement}")?;
        if let Some(needs) = needs {
            let child = format!("{prefix}{}", if last { "    " } else { "│   " });
            write_tree(f, needs, &child)?;
        }
    }
    Ok(())
}

/// A runnable agent. Analogue of Bevy's `System`.
///
/// Object-safe, so heterogeneous agents with the same output can be stored as
/// `Box<dyn Agent<Output = O>>`. Usually driven through
/// [`AgentWorld::prepare`] and [`AgentWorld::run_prepared`].
pub trait Agent: Send + 'static {
    type Output: Send + 'static;

    fn meta(&self) -> &AgentMeta;

    /// Initialize parameter state against `world`, replacing any previous
    /// state. Fails, listing every unmet requirement, before any user code runs.
    fn initialize(&mut self, world: &AgentWorld) -> Result<(), RunError>;

    /// Start one invocation with the given inputs.
    ///
    /// Synchronously checks that the agent was prepared against this world at
    /// its current binding generation and starts resolving parameters; the
    /// returned future finishes resolution, then runs the body. It borrows
    /// neither the world nor the agent.
    fn start(&mut self, world: &AgentWorld, scope: Scope) -> AgentFuture<Self::Output>;
}

/// An [`Agent`] built from an [`AgentFunction`]. Analogue of Bevy's
/// `FunctionSystem`: holds the callable instance, its metadata, and — once
/// initialized — the parameter state plus the world stamp it is valid for.
///
/// The function sits behind `Arc<Mutex<_>>` because it is called only after
/// asynchronous resolution completes, inside the `'static` run future. The
/// lock is held just for the synchronous `call`, never across an `.await`.
pub struct FunctionAgent<Marker, F: AgentFunction<Marker>> {
    func: Arc<Mutex<F>>,
    meta: AgentMeta,
    state: Option<(<F::Param as AgentParam>::State, WorldStamp)>,
    // `fn() -> Marker` keeps the agent `Send` whatever the marker is.
    _marker: PhantomData<fn() -> Marker>,
}

impl<Marker, F: AgentFunction<Marker>> FunctionAgent<Marker, F> {
    /// Override the diagnostic name.
    pub fn with_name(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        self.meta.name = name.into();
        self
    }

    fn resolve(
        &mut self,
        world: &AgentWorld,
        scope: &Scope,
    ) -> Result<<F::Param as AgentParam>::Future, RunError> {
        let agent = || self.meta.name.clone();
        let Some((state, stamp)) = self.state.as_mut() else {
            return Err(RunError::NotPrepared { agent: agent() });
        };
        let current = world.stamp();
        if stamp.world != current.world {
            return Err(RunError::ForeignWorld { agent: agent() });
        }
        if stamp.generation != current.generation {
            return Err(RunError::Stale { agent: agent() });
        }
        Ok(F::Param::resolve(state, world, scope))
    }
}

impl<Marker: 'static, F: AgentFunction<Marker>> Agent for FunctionAgent<Marker, F> {
    type Output = F::Output;

    fn meta(&self) -> &AgentMeta {
        &self.meta
    }

    fn initialize(&mut self, world: &AgentWorld) -> Result<(), RunError> {
        self.state = None;
        let state = F::Param::init(world)
            .map_err(|missing| RunError::Unresolved(Diagnostics::new(&self.meta, &missing)))?;
        self.state = Some((state, world.stamp()));
        Ok(())
    }

    fn start(&mut self, world: &AgentWorld, scope: Scope) -> AgentFuture<Self::Output> {
        let resolving = match self.resolve(world, &scope) {
            Ok(resolving) => resolving,
            Err(error) => return Box::pin(ready(Err(error))),
        };
        let agent = self.meta.name.clone();
        let func = self.func.clone();
        Box::pin(async move {
            let param = resolving
                .await
                .map_err(|error| RunError::Param { agent, error })?;
            let body = func
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .call(param);
            Ok(body.await)
        })
    }
}

/// Conversion into an [`Agent`]. Analogue of Bevy's `IntoSystem`.
///
/// Conversion only builds the uninitialized object and its metadata; it
/// touches no world. `Marker` keeps the blanket impls from overlapping.
pub trait IntoAgent<Marker>: Sized {
    type Agent: Agent;
    fn into_agent(self) -> Self::Agent;
}

/// Marker for "this is a function"; see [`IntoAgent`].
#[doc(hidden)]
pub struct IsFunctionAgent;

/// Marker for "this is already an agent"; see [`IntoAgent`].
#[doc(hidden)]
pub struct IsAgent;

impl<Marker: 'static, F: AgentFunction<Marker>> IntoAgent<(IsFunctionAgent, Marker)> for F {
    type Agent = FunctionAgent<Marker, F>;

    fn into_agent(self) -> Self::Agent {
        let mut params = Vec::new();
        F::Param::describe(&mut params);
        FunctionAgent {
            func: Arc::new(Mutex::new(self)),
            meta: AgentMeta {
                name: Cow::Owned(short_type_name(type_name::<F>())),
                params,
            },
            state: None,
            _marker: PhantomData,
        }
    }
}

impl<A: Agent> IntoAgent<IsAgent> for A {
    type Agent = A;

    fn into_agent(self) -> A {
        self
    }
}

impl<O: Send + 'static> Agent for Box<dyn Agent<Output = O>> {
    type Output = O;

    fn meta(&self) -> &AgentMeta {
        (**self).meta()
    }
    fn initialize(&mut self, world: &AgentWorld) -> Result<(), RunError> {
        (**self).initialize(world)
    }
    fn start(&mut self, world: &AgentWorld, scope: Scope) -> AgentFuture<O> {
        (**self).start(world, scope)
    }
}
