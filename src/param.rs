use std::any::type_name;
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use crate::{AgentWorld, ParamError, all_tuples};

/// A declarative description of one thing an agent needs from the world.
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
    /// A materialized [`Context<T>`](crate::Context) snapshot.
    Context { type_name: &'static str },
    /// A plain [`Res<T>`] dependency.
    Resource { type_name: &'static str },
    /// An `Option<P>` parameter: never unmet, but still declared.
    Optional(Box<Requirement>),
}

impl fmt::Display for Requirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Requirement::Llm => write!(f, "Llm"),
            Requirement::Tool { name, .. } => write!(f, "Tool<{name}>"),
            Requirement::Context { type_name } => {
                write!(f, "Context<{}>", short_type_name(type_name))
            }
            Requirement::Resource { type_name } => write!(f, "Res<{}>", short_type_name(type_name)),
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
/// `SystemParam`, with the same three responsibilities:
///
/// 1. [`describe`](Self::describe): declare what is needed, from the type alone.
/// 2. [`init`](Self::init): once per prepared agent, look up bindings and
///    build persistent [`State`](Self::State); report *every* unmet requirement.
/// 3. [`resolve`](Self::resolve): once per invocation, produce the value.
///
/// Unlike Bevy, the resolved value is `Self` and is **owned** (`'static`), not
/// a `Item<'w, 's>` borrowed from the world. An agent is an `async fn` whose
/// future outlives the call; see `DESIGN.md` for why borrowing is not viable
/// in M1. Implementations should therefore be cheap handles (usually `Arc`).
pub trait AgentParam: Sized + Send + 'static {
    /// Persistent per-agent state, created by `init` and reused across runs.
    type State: Send + 'static;

    /// Append the declarative requirements of this parameter to `out`.
    fn describe(out: &mut Vec<Requirement>);

    /// Validate bindings and build state. On failure, return every unmet
    /// requirement so they can all be reported at once.
    fn init(world: &AgentWorld) -> Result<Self::State, Vec<Requirement>>;

    /// Produce this invocation's value.
    fn resolve(state: &mut Self::State, world: &AgentWorld) -> Result<Self, ParamError>;
}

/// Shorthand for "the requirements `P` declares", used as the error of `init`.
pub(crate) fn unmet<P: AgentParam>() -> Vec<Requirement> {
    let mut out = Vec::new();
    P::describe(&mut out);
    out
}

/// A shared, read-only dependency of type `T`, provided with
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

    fn describe(out: &mut Vec<Requirement>) {
        out.push(Requirement::Resource {
            type_name: type_name::<T>(),
        });
    }

    fn init(world: &AgentWorld) -> Result<Arc<T>, Vec<Requirement>> {
        world.resource_arc::<T>().ok_or_else(unmet::<Self>)
    }

    fn resolve(state: &mut Arc<T>, _world: &AgentWorld) -> Result<Self, ParamError> {
        Ok(Res(state.clone()))
    }
}

/// An optional capability: `None` if the world cannot provide `P`.
impl<P: AgentParam> AgentParam for Option<P> {
    type State = Option<P::State>;

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

    fn resolve(state: &mut Self::State, world: &AgentWorld) -> Result<Self, ParamError> {
        state.as_mut().map(|s| P::resolve(s, world)).transpose()
    }
}

/// Tuples of params are params. This is what lets an `N`-ary function be
/// treated as a function of one `Param` tuple, and lets users nest tuples to
/// go past the generated arity limit.
macro_rules! impl_param_tuple {
    ($($P:ident),*) => {
        impl<$($P: AgentParam),*> AgentParam for ($($P,)*) {
            type State = ($($P::State,)*);

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
            fn resolve(state: &mut Self::State, world: &AgentWorld) -> Result<Self, ParamError> {
                let ($($P,)*) = state;
                Ok(($($P::resolve($P, world)?,)*))
            }
        }
    };
}

all_tuples!(impl_param_tuple; P0, P1, P2, P3, P4, P5, P6, P7);
