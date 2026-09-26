use std::any::type_name;
use std::borrow::Cow;
use std::fmt;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;

use crate::param::short_type_name;
use crate::world::WorldStamp;
use crate::{AgentFunction, AgentParam, AgentWorld, Diagnostics, Requirement, RunError};

/// The future of a started agent. Boxed so that [`Agent`] stays object-safe;
/// `'static` because every parameter is resolved (and owned) before it exists.
pub type AgentFuture<O> = Pin<Box<dyn Future<Output = O> + Send + 'static>>;

/// Metadata derived from an agent's signature at conversion time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentMeta {
    /// Diagnostic name. Defaults to the function's short type name.
    pub name: Cow<'static, str>,
    /// One entry per declared requirement, in parameter order.
    pub params: Vec<Requirement>,
}

/// Renders the signature as a tree:
///
/// ```text
/// researcher
/// ├── requires Llm
/// ├── can call Tool<web_search>
/// └── requires Context<RelevantMemory>
/// ```
impl fmt::Display for AgentMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        for (i, requirement) in self.params.iter().enumerate() {
            let branch = if i + 1 == self.params.len() {
                "└──"
            } else {
                "├──"
            };
            let verb = match requirement {
                Requirement::Tool { .. } => "can call",
                Requirement::Optional(_) => "may use",
                _ => "requires",
            };
            write!(f, "\n{branch} {verb} {requirement}")?;
        }
        Ok(())
    }
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

    /// Resolve this invocation's parameters and call the function.
    ///
    /// Requires state initialized against this same `world` at its current
    /// binding generation. The returned future owns its inputs and does not
    /// borrow the world or the agent.
    fn start(&mut self, world: &AgentWorld) -> Result<AgentFuture<Self::Output>, RunError>;
}

/// An [`Agent`] built from an [`AgentFunction`]. Analogue of Bevy's
/// `FunctionSystem`: holds the callable instance, its metadata, and — once
/// initialized — the parameter state plus the world stamp it is valid for.
pub struct FunctionAgent<Marker, F: AgentFunction<Marker>> {
    func: F,
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

    fn start(&mut self, world: &AgentWorld) -> Result<AgentFuture<Self::Output>, RunError> {
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
        let param = F::Param::resolve(state, world).map_err(|error| RunError::Param {
            agent: agent(),
            error,
        })?;
        Ok(Box::pin(self.func.call(param)))
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
            func: self,
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
    fn start(&mut self, world: &AgentWorld) -> Result<AgentFuture<O>, RunError> {
        (**self).start(world)
    }
}
