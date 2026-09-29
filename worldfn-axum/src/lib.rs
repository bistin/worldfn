//! axum adapter for [worldfn](https://github.com/bistin/worldfn).
//!
//! worldfn itself knows nothing about web frameworks. It meets them at three
//! points, and this crate maps each one to axum:
//!
//! | worldfn (framework-neutral) | here |
//! |---|---|
//! | `Scope`: per-invocation inputs | built by your handler from extractors |
//! | `EventStream<E>` + `SseEvent` | [`sse`]: an `axum::response::Sse` |
//! | `RunError::http_status` | [`AgentError`]: an `IntoResponse` |
//!
//! ```no_run
//! use std::sync::Arc;
//! use axum::{Router, extract::{Query, State}, routing::get};
//! use worldfn::{emit, prelude::*, SseEvent, SseFrame};
//!
//! enum Progress { Status(String) }
//! impl SseEvent for Progress {
//!     fn to_sse(&self) -> SseFrame {
//!         match self { Progress::Status(s) => SseFrame::new("status", s.clone()) }
//!     }
//! }
//!
//! async fn agent(task: Input<Task>, events: Emit<Progress>) {
//!     events.send(Progress::Status(format!("working on {}", task.0)));
//! }
//!
//! #[derive(serde::Deserialize)]
//! struct Q { q: String }
//!
//! async fn handler(State(world): State<Arc<AgentWorld>>, Query(Q { q }): Query<Q>)
//!     -> axum::response::Response
//! {
//!     let (emitter, events) = emit::channel::<Progress>();
//!     tokio::spawn(world.run_with(agent, Scope::of(Task::new(q)).with(emitter)));
//!     axum::response::IntoResponse::into_response(worldfn_axum::sse(events))
//! }
//!
//! let app: Router = Router::new()
//!     .route("/run", get(handler))
//!     .with_state(Arc::new(AgentWorld::new()));
//! ```

use std::convert::Infallible;
use std::pin::Pin;
use std::task::{Context, Poll};

use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, KeepAliveStream, Sse};
use axum::response::{IntoResponse, Response};
use worldfn::{EventStream, RunError, SseEvent, SseFrame};

/// Serve an agent's event stream as server-sent events, with keep-alive
/// comments so proxies do not close idle connections. The response ends when
/// the agent run (and every other emitter clone) finishes.
pub fn sse<E: SseEvent + Send + 'static>(
    events: EventStream<E>,
) -> Sse<KeepAliveStream<SseStream<E>>> {
    Sse::new(SseStream(events)).keep_alive(KeepAlive::default())
}

/// `EventStream<E>` as a `Stream` of axum SSE events. Returned by [`sse`].
pub struct SseStream<E>(EventStream<E>);

impl<E: SseEvent> futures_core::Stream for SseStream<E> {
    type Item = Result<Event, Infallible>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // `SseStream` holds no self-references, so it is `Unpin`.
        self.get_mut()
            .0
            .poll_next(cx)
            .map(|event| event.map(|e| Ok(to_axum(e.to_sse()))))
    }
}

fn to_axum(frame: SseFrame) -> Event {
    // Same field order as `SseFrame::encode`: id, event, then data. axum
    // writes fields in call order and splits multi-line data itself.
    let mut event = Event::default();
    if let Some(id) = frame.id {
        event = event.id(id);
    }
    if let Some(name) = frame.event {
        event = event.event(name);
    }
    event.data(frame.data)
}

/// A [`RunError`] as an HTTP response: 400 when the caller left out an input,
/// 502 when a parameter's backend failed, 500 for server setup problems. The
/// body is the error's plain-text message.
#[derive(Debug)]
pub struct AgentError(pub RunError);

impl From<RunError> for AgentError {
    fn from(error: RunError) -> Self {
        AgentError(error)
    }
}

impl IntoResponse for AgentError {
    fn into_response(self) -> Response {
        let status =
            StatusCode::from_u16(self.0.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (status, self.0.to_string()).into_response()
    }
}
