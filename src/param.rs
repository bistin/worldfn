use std::any::type_name;
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use crate::{AgentWorld, all_tuples};

/// A declarative description of one thing an agent needs from the world.
///
/// Produced by [`AgentParam::describe`] (static introspection, no world
/// required) and by [`AgentParam::fetch`] when resolution fails (only the
/// requirements that are *unmet*).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Requirement {
    /// An [`Llm`](crate::Llm) provider.
    Llm,
    /// A [`Tool<T>`](crate::Tool) for the tool spec `spec`.
    Tool {
        name: &'static str,
        spec: &'static str,
    },
    /// A plain [`Res<T>`] resource.
    Resource { type_name: &'static str },
    /// A derived [`Context<C>`](crate::Context) and what *it* needs.
    Context {
        type_name: &'static str,
        needs: Vec<Requirement>,
    },
    /// An `Option<P>` parameter: never unmet, but still describable.
    Optional(Box<Requirement>),
}

impl fmt::Display for Requirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Requirement::Llm => write!(f, "Llm"),
            Requirement::Tool { name, .. } => write!(f, "Tool<{name}>"),
            Requirement::Resource { type_name } => write!(f, "Res<{type_name}>"),
            Requirement::Context { type_name, needs } => {
                write!(f, "Context<{type_name}>")?;
                if !needs.is_empty() {
                    let needs: Vec<_> = needs.iter().map(ToString::to_string).collect();
                    write!(f, " (needs {})", needs.join(", "))?;
                }
                Ok(())
            }
            Requirement::Optional(inner) => write!(f, "Option<{inner}>"),
        }
    }
}

/// A value an agent function can take as a parameter, resolved by type.
///
/// This is the analogue of Bevy's `SystemParam`, with one big difference:
/// parameters are **owned** (`'static`), not borrowed from the world. Bevy
/// systems are synchronous, so `Res<'w, T>` can borrow the world for the
/// duration of the call. An agent is an `async fn` whose future outlives the
/// call and is held across `.await`s; borrowed parameters would make the
/// returned future's type depend on a higher-ranked lifetime, which a plain
/// `Fn(P) -> Fut` bound cannot express. See the README for details.
///
/// Implementations should therefore be cheap handles (usually an `Arc`).
pub trait AgentParam: Sized + Send + 'static {
    /// Append the declarative requirements of this parameter to `out`.
    fn describe(out: &mut Vec<Requirement>);

    /// Resolve this parameter from the world.
    ///
    /// On failure returns **every** unmet requirement (not just the first),
    /// so a missing capability set can be reported in one go.
    fn fetch(world: &AgentWorld) -> Result<Self, Vec<Requirement>>;
}

/// A shared, read-only resource of type `T`.
///
/// Owned handle (an `Arc<T>`) rather than Bevy's borrowed `Res<'w, T>`.
/// Shared mutable state must use interior mutability inside `T`.
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
    fn describe(out: &mut Vec<Requirement>) {
        out.push(Requirement::Resource {
            type_name: type_name::<T>(),
        });
    }

    fn fetch(world: &AgentWorld) -> Result<Self, Vec<Requirement>> {
        world.resource_arc::<T>().map(Res).ok_or_else(|| {
            let mut missing = Vec::new();
            Self::describe(&mut missing);
            missing
        })
    }
}

/// An optional capability: `None` if the world cannot provide `P`.
impl<P: AgentParam> AgentParam for Option<P> {
    fn describe(out: &mut Vec<Requirement>) {
        let mut inner = Vec::new();
        P::describe(&mut inner);
        out.extend(
            inner
                .into_iter()
                .map(|r| Requirement::Optional(Box::new(r))),
        );
    }

    fn fetch(world: &AgentWorld) -> Result<Self, Vec<Requirement>> {
        Ok(P::fetch(world).ok())
    }
}

/// Tuples of params are params. This is what lets an `N`-ary function be
/// treated as a function of one `Param` tuple, and also lets users nest
/// tuples to go past the generated arity limit.
macro_rules! impl_param_tuple {
    ($($P:ident),*) => {
        impl<$($P: AgentParam),*> AgentParam for ($($P,)*) {
            #[allow(unused_variables)]
            fn describe(out: &mut Vec<Requirement>) {
                $($P::describe(out);)*
            }

            #[allow(non_snake_case, unused_variables, unused_mut)]
            fn fetch(world: &AgentWorld) -> Result<Self, Vec<Requirement>> {
                let mut missing = Vec::new();
                $(
                    let $P = match $P::fetch(world) {
                        Ok(value) => Some(value),
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
        }
    };
}

all_tuples!(impl_param_tuple; P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11);
