use std::future::Future;

use crate::{AgentParam, all_tuples};

/// A function whose parameters can all be resolved from an
/// [`AgentWorld`](crate::AgentWorld).
///
/// Analogue of Bevy's `SystemParamFunction`. It is implemented (by macro) for
/// every `Fn(P0, .., Pn) -> Fut` where each `Pi: AgentParam` and `Fut` is a
/// `Send + 'static` future — which is exactly what an ordinary
/// `async fn(P0, .., Pn) -> Out` desugars to when its parameters are owned.
///
/// ## Why the `Marker` parameter
///
/// A single type could in principle implement `Fn(A)` *and* `Fn(A, B)`, so
/// blanket impls for each arity would overlap and be rejected by coherence.
/// The marker (`fn(P0, .., Pn) -> Fut`) makes each arity's impl a *different
/// trait*, and type inference picks the one that applies. This is the same
/// trick Bevy uses; the cost is that `Marker` leaks into
/// [`IntoAgent`](crate::IntoAgent) and friends.
pub trait AgentFunction<Marker>: Send + Sync + 'static {
    /// All parameters, as one tuple.
    type Param: AgentParam;
    /// What the agent returns.
    type Output: Send + 'static;
    /// The concrete future returned by the function. Nameable here because
    /// it is a generic parameter of the impl — no boxing needed at this layer.
    type Future: Future<Output = Self::Output> + Send + 'static;

    fn call(&self, param: Self::Param) -> Self::Future;
}

macro_rules! impl_agent_function {
    ($($P:ident),*) => {
        impl<Func, Fut, $($P: AgentParam),*> AgentFunction<fn($($P,)*) -> Fut> for Func
        where
            Func: Fn($($P),*) -> Fut + Send + Sync + 'static,
            Fut: Future + Send + 'static,
            Fut::Output: Send + 'static,
        {
            type Param = ($($P,)*);
            type Output = Fut::Output;
            type Future = Fut;

            #[allow(non_snake_case)]
            fn call(&self, ($($P,)*): Self::Param) -> Fut {
                self($($P),*)
            }
        }
    };
}

all_tuples!(impl_agent_function; P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11);
