use std::any::type_name;
use std::fmt;
use std::future::{Future, Ready, ready};
use std::ops::Deref;
use std::sync::Arc;

use crate::{AgentWorld, BoxFuture, ParamError, Scope, all_tuples};

/// A declarative description of one thing an agent needs.
///
/// Produced by [`AgentParam::describe`] from the parameter *type* alone, and
/// reported back by [`AgentParam::init`] for the requirements that are unmet.
/// Type names come from [`std::any::type_name`] and are for diagnostics only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Requirement {
    /// An [`Llm`](crate::Llm) provider.
    Llm,
    /// A [`Tool<T>`](crate::Tool) handler for the tool spec `spec`.
    Tool {
        name: &'static str,
        spec: &'static str,
    },
    /// A type-erased service handle such as [`Memory`](crate::Memory).
    Service { type_name: &'static str },
    /// A plain [`Res<T>`] dependency.
    Resource { type_name: &'static str },
    /// An [`Input<T>`](crate::Input) from the invocation [`Scope`]. Only
    /// checked when the agent is started, since inputs vary per invocation.
    Input { type_name: &'static str },
    /// An [`Emit<E>`](crate::Emit) event sink from the invocation [`Scope`].
    /// Like inputs, only checked when the agent is started.
    Emit { type_name: &'static str },
    /// A [`Context<S>`](crate::Context) materialized per invocation, and what
    /// materializing it needs.
    Context {
        type_name: &'static str,
        needs: Vec<Requirement>,
    },
    /// An `Option<P>` parameter: never unmet, but still declared.
    Optional(Box<Requirement>),
}

impl Requirement {
    /// Supplied per invocation through the [`Scope`], so it can only be
    /// checked when an agent is started, not when it is prepared.
    pub fn is_per_invocation(&self) -> bool {
        matches!(self, Requirement::Input { .. } | Requirement::Emit { .. })
    }

    /// Whether `other` refers to the same declared requirement. Contexts are
    /// matched by type alone, because an unmet context carries only its unmet
    /// needs.
    pub fn same_as(&self, other: &Requirement) -> bool {
        match (self, other) {
            (
                Requirement::Context { type_name: a, .. },
                Requirement::Context { type_name: b, .. },
            ) => a == b,
            _ => self == other,
        }
    }
}

impl fmt::Display for Requirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Requirement::Llm => write!(f, "Llm"),
            Requirement::Tool { name, .. } => write!(f, "Tool<{name}>"),
            Requirement::Service { type_name } => write!(f, "{}", short_type_name(type_name)),
            Requirement::Resource { type_name } => write!(f, "Res<{}>", short_type_name(type_name)),
            Requirement::Input { type_name } => write!(f, "Input<{}>", short_type_name(type_name)),
            Requirement::Emit { type_name } => write!(f, "Emit<{}>", short_type_name(type_name)),
            Requirement::Context { type_name, .. } => {
                write!(f, "Context<{}>", short_type_name(type_name))
            }
            Requirement::Optional(inner) => write!(f, "Option<{inner}>"),
        }
    }
}

/// `a::b::Foo<c::Bar>` → `Foo<Bar>`; `m::f::{{closure}}::g` → `g`.
pub(crate) fn short_type_name(full: &str) -> String {
    let mut out = String::new();
    let mut segment = String::new();
    let mut chars = full.chars().peekable();
    while let Some(c) = chars.next() {
        if c == ':' && chars.peek() == Some(&':') {
            chars.next();
            segment.clear();
        } else if c.is_alphanumeric() || matches!(c, '_' | '{' | '}') {
            segment.push(c);
        } else {
            out.push_str(&segment);
            segment.clear();
            out.push(c);
        }
    }
    out.push_str(&segment);
    out
}

/// A value an agent function can take as a parameter. Analogue of Bevy's
/// `SystemParam`, in three phases:
///
/// 1. [`describe`](Self::describe): declare what is needed, from the type alone.
/// 2. [`init`](Self::init): once per prepared agent, look up bindings and
///    build persistent [`State`](Self::State); report *every* unmet requirement.
/// 3. [`resolve`](Self::resolve): once per invocation. The synchronous part
///    may read the world and the invocation [`Scope`] but only to copy owned
///    handles out; it returns a `'static` future that does any async work
///    (such as context retrieval) without borrowing either.
///
/// That split is what makes async resolution possible without Bevy's
/// borrowed `Item<'w, 's>`; see `DESIGN.md`. Resolved values are owned, so
/// implementations should be cheap handles (usually `Arc`).
pub trait AgentParam: Sized + Send + 'static {
    /// Persistent per-agent state, created by `init` and reused across runs.
    type State: Send + 'static;

    /// The owned future that completes resolution.
    type Future: Future<Output = Result<Self, ParamError>> + Send + 'static;

    /// Append the declarative requirements of this parameter to `out`.
    fn describe(out: &mut Vec<Requirement>);

    /// Validate bindings and build state. On failure, return every unmet
    /// requirement so they can all be reported at once.
    fn init(world: &AgentWorld) -> Result<Self::State, Vec<Requirement>>;

    /// Start producing this invocation's value.
    fn resolve(state: &mut Self::State, world: &AgentWorld, scope: &Scope) -> Self::Future;
}

/// Shorthand for "the requirements `P` declares", used as the error of `init`.
pub(crate) fn unmet<P: AgentParam>() -> Vec<Requirement> {
    let mut out = Vec::new();
    P::describe(&mut out);
    out
}

/// A shared, read-only dependency of type `T`, bound with
/// [`AgentWorld::provide`].
///
/// An owned `Arc<T>` handle rather than Bevy's borrowed `Res<'w, T>`; shared
/// mutable state must use interior mutability inside `T`.
pub struct Res<T>(Arc<T>);

impl<T> Res<T> {
    pub fn into_inner(self) -> Arc<T> {
        self.0
    }
}

impl<T> Clone for Res<T> {
    fn clone(&self) -> Self {
        Res(self.0.clone())
    }
}

impl<T> Deref for Res<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: fmt::Debug> fmt::Debug for Res<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<T: Send + Sync + 'static> AgentParam for Res<T> {
    type State = Arc<T>;
    type Future = Ready<Result<Self, ParamError>>;

    fn describe(out: &mut Vec<Requirement>) {
        out.push(Requirement::Resource {
            type_name: type_name::<T>(),
        });
    }

    fn init(world: &AgentWorld) -> Result<Arc<T>, Vec<Requirement>> {
        world.resource_arc::<T>().ok_or_else(unmet::<Self>)
    }

    fn resolve(state: &mut Arc<T>, _world: &AgentWorld, _scope: &Scope) -> Self::Future {
        ready(Ok(Res(state.clone())))
    }
}

/// An optional capability: `None` if the world cannot provide `P`.
///
/// Only *binding* absence degrades to `None`. Once bound, a resolution
/// failure (e.g. a failing retrieval) is still an error.
impl<P: AgentParam> AgentParam for Option<P> {
    type State = Option<P::State>;
    type Future = BoxFuture<'static, Result<Self, ParamError>>;

    fn describe(out: &mut Vec<Requirement>) {
        let inner = unmet::<P>();
        out.extend(
            inner
                .into_iter()
                .map(|r| Requirement::Optional(Box::new(r))),
        );
    }

    fn init(world: &AgentWorld) -> Result<Self::State, Vec<Requirement>> {
        Ok(P::init(world).ok())
    }

    fn resolve(state: &mut Self::State, world: &AgentWorld, scope: &Scope) -> Self::Future {
        match state {
            Some(state) => {
                let resolving = P::resolve(state, world, scope);
                Box::pin(async move { resolving.await.map(Some) })
            }
            None => Box::pin(ready(Ok(None))),
        }
    }
}

/// Tuples of params are params. This is what lets an `N`-ary function be
/// treated as a function of one `Param` tuple, and lets users nest tuples to
/// go past the generated arity limit.
///
/// Elements start resolving in order during the synchronous phase, then are
/// awaited in order. The combined future is boxed because a per-arity join
/// future would have to be hand-written or generated.
macro_rules! impl_param_tuple {
    ($($P:ident),*) => {
        impl<$($P: AgentParam),*> AgentParam for ($($P,)*) {
            type State = ($($P::State,)*);
            type Future = BoxFuture<'static, Result<Self, ParamError>>;

            #[allow(unused_variables)]
            fn describe(out: &mut Vec<Requirement>) {
                $($P::describe(out);)*
            }

            #[allow(non_snake_case, unused_variables, unused_mut)]
            fn init(world: &AgentWorld) -> Result<Self::State, Vec<Requirement>> {
                let mut missing = Vec::new();
                $(
                    let $P = match $P::init(world) {
                        Ok(state) => Some(state),
                        Err(unmet) => {
                            missing.extend(unmet);
                            None
                        }
                    };
                )*
                if !missing.is_empty() {
                    return Err(missing);
                }
                // Every slot is `Some` when nothing is missing.
                Ok(($($P.unwrap(),)*))
            }

            #[allow(non_snake_case, unused_variables)]
            fn resolve(state: &mut Self::State, world: &AgentWorld, scope: &Scope) -> Self::Future {
                let ($($P,)*) = state;
                $(let $P = $P::resolve($P, world, scope);)*
                Box::pin(async move { Ok(($($P.await?,)*)) })
            }
        }
    };
}

all_tuples!(impl_param_tuple; P0, P1, P2, P3, P4, P5, P6, P7);
