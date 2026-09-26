use std::any::type_name;
use std::borrow::Cow;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;

use crate::{AgentFunction, AgentParam, AgentWorld, Requirement, ResolveError};

/// The future of a started agent. Boxed so that [`Agent`] stays object-safe;
/// `'static` because all parameters are resolved (and owned) before it exists.
pub type AgentFuture<O> = Pin<Box<dyn Future<Output = O> + Send + 'static>>;

/// A runnable agent. Analogue of Bevy's `System`.
///
/// Object-safe, so heterogeneous agents with the same output can be stored as
/// `Box<dyn Agent<Output = O>>`.
pub trait Agent: Send + Sync + 'static {
    type Output: Send + 'static;

    fn name(&self) -> Cow<'static, str>;

    /// Static, declarative description of everything this agent needs.
    fn requirements(&self) -> Vec<Requirement>;

    /// Check that `world` can satisfy every requirement, without running.
    fn validate(&self, world: &AgentWorld) -> Result<(), ResolveError>;

    /// Resolve parameters from `world` and start the agent.
    ///
    /// Resolution is synchronous and happens *now*; the returned future owns
    /// its inputs and no longer borrows the world.
    fn start(&self, world: &AgentWorld) -> Result<AgentFuture<Self::Output>, ResolveError>;
}

/// An [`Agent`] built from an [`AgentFunction`]. Analogue of Bevy's
/// `FunctionSystem`.
pub struct FunctionAgent<Marker, F> {
    func: F,
    name: Cow<'static, str>,
    // `fn() -> Marker` keeps the agent `Send + Sync` whatever the marker is.
    _marker: PhantomData<fn() -> Marker>,
}

impl<Marker, F> FunctionAgent<Marker, F> {
    /// Override the default name (the function's type name).
    pub fn with_name(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        self.name = name.into();
        self
    }
}

impl<Marker: 'static, F: AgentFunction<Marker>> Agent for FunctionAgent<Marker, F> {
    type Output = F::Output;

    fn name(&self) -> Cow<'static, str> {
        self.name.clone()
    }

    fn requirements(&self) -> Vec<Requirement> {
        let mut out = Vec::new();
        F::Param::describe(&mut out);
        out
    }

    fn validate(&self, world: &AgentWorld) -> Result<(), ResolveError> {
        self.resolve(world).map(drop)
    }

    fn start(&self, world: &AgentWorld) -> Result<AgentFuture<Self::Output>, ResolveError> {
        let param = self.resolve(world)?;
        Ok(Box::pin(self.func.call(param)))
    }
}

impl<Marker: 'static, F: AgentFunction<Marker>> FunctionAgent<Marker, F> {
    fn resolve(&self, world: &AgentWorld) -> Result<F::Param, ResolveError> {
        F::Param::fetch(world).map_err(|missing| ResolveError {
            agent: self.name.clone(),
            missing,
        })
    }
}

/// Conversion into an [`Agent`]. Analogue of Bevy's `IntoSystem`.
///
/// `Marker` exists only to keep the blanket impls below (and the per-arity
/// [`AgentFunction`] impls) from overlapping; callers never name it.
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
        FunctionAgent {
            func: self,
            name: Cow::Borrowed(type_name::<F>()),
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

    fn name(&self) -> Cow<'static, str> {
        (**self).name()
    }
    fn requirements(&self) -> Vec<Requirement> {
        (**self).requirements()
    }
    fn validate(&self, world: &AgentWorld) -> Result<(), ResolveError> {
        (**self).validate(world)
    }
    fn start(&self, world: &AgentWorld) -> Result<AgentFuture<O>, ResolveError> {
        (**self).start(world)
    }
}
