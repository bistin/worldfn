use std::future::Future;

use crate::{AgentParam, all_tuples};

/// A function whose parameters can all be resolved from an
/// [`AgentWorld`](crate::AgentWorld). Analogue of Bevy's `SystemParamFunction`.
///
/// Implemented (by macro) for every `FnMut(P0, .., Pn) -> Fut` with
/// `n <= 8`, each `Pi: AgentParam`, and `Fut` a `Send + 'static` future. That
/// is exactly what an ordinary `async fn` with owned parameters is, so no
/// adapter or attribute is needed.
///
/// ## Why the `Marker` parameter
///
/// A single type could implement both `FnMut(A)` and `FnMut(A, B)`, so blanket
/// impls per arity would overlap under coherence. The marker
/// (`fn(P0, .., Pn) -> Fut`) makes each arity's impl a *different trait* and
/// also constrains `Fut`, which would otherwise be an unconstrained impl
/// parameter. Type inference picks the one that applies; users never write it.
pub trait AgentFunction<Marker>: Send + 'static {
    /// All parameters, as one tuple.
    type Param: AgentParam;
    /// What the function returns: kept as is, e.g. `Result<Answer, MyError>`.
    type Output: Send + 'static;
    /// The concrete future. No boxing at this layer.
    type Future: Future<Output = Self::Output> + Send + 'static;

    fn call(&mut self, param: Self::Param) -> Self::Future;
}

macro_rules! impl_agent_function {
    ($($P:ident),*) => {
        impl<Func, Fut, $($P: AgentParam),*> AgentFunction<fn($($P,)*) -> Fut> for Func
        where
            Func: FnMut($($P),*) -> Fut + Send + 'static,
            Fut: Future + Send + 'static,
            Fut::Output: Send + 'static,
        {
            type Param = ($($P,)*);
            type Output = Fut::Output;
            type Future = Fut;

            #[allow(non_snake_case)]
            fn call(&mut self, ($($P,)*): Self::Param) -> Fut {
                self($($P),*)
            }
        }
    };
}

all_tuples!(impl_agent_function; P0, P1, P2, P3, P4, P5, P6, P7);
