//! The adapter end to end through axum's router, without a network socket.

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::{Query, State};
use axum::http::{Request, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use http_body_util::BodyExt;
use tower::ServiceExt;
use worldfn::prelude::*;
use worldfn::{SseEvent, SseFrame, emit};
use worldfn_axum::AgentError;

#[derive(Debug)]
enum Step {
    Status(String),
    Answer(String),
}

impl SseEvent for Step {
    fn to_sse(&self) -> SseFrame {
        match self {
            Step::Status(s) => SseFrame::new("status", s.clone()),
            Step::Answer(s) => SseFrame::new("answer", s.clone()),
        }
    }
}

async fn answer(task: Input<Task>, llm: Llm, events: Emit<Step>) -> String {
    events.send(Step::Status("thinking\nhard".into()));
    let reply = llm.complete(task.0.clone()).await.unwrap();
    events.send(Step::Answer(reply.clone()));
    reply
}

#[derive(serde::Deserialize)]
struct Q {
    q: Option<String>,
}

async fn stream(State(world): State<Arc<AgentWorld>>, Query(Q { q }): Query<Q>) -> Response {
    let (emitter, events) = emit::channel::<Step>();
    let mut scope = Scope::new().with(emitter);
    if let Some(q) = q {
        scope = scope.with(Task::new(q));
    }
    tokio::spawn(world.run_with(answer, scope));
    worldfn_axum::sse(events).into_response()
}

/// Non-streaming variant: errors become proper HTTP statuses.
async fn once(
    State(world): State<Arc<AgentWorld>>,
    Query(Q { q }): Query<Q>,
) -> Result<String, AgentError> {
    let (emitter, _events) = emit::channel::<Step>();
    let mut scope = Scope::new().with(emitter);
    if let Some(q) = q {
        scope = scope.with(Task::new(q));
    }
    Ok(world.run_with(answer, scope).await?)
}

fn app(world: AgentWorld) -> Router {
    Router::new()
        .route("/stream", get(stream))
        .route("/once", get(once))
        .with_state(Arc::new(world))
}

fn echo_world() -> AgentWorld {
    let mut world = AgentWorld::new();
    world.provide_llm(FakeLlm::echo()).unwrap();
    world
}

async fn get_path(app: Router, uri: &str) -> (StatusCode, Option<String>, String) {
    let response = app
        .oneshot(Request::get(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .map(|v| v.to_str().unwrap().to_owned());
    let body = response.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        content_type,
        String::from_utf8(body.to_vec()).unwrap(),
    )
}

#[tokio::test]
async fn streams_agent_events_as_sse_and_ends_with_the_run() {
    let (status, content_type, body) = get_path(app(echo_world()), "/stream?q=hello").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type.as_deref(), Some("text/event-stream"));
    // Collecting the whole body terminates only because the stream ends
    // when the agent run drops its emitter.
    assert_eq!(
        body,
        "event: status\ndata: thinking\ndata: hard\n\nevent: answer\ndata: hello\n\n"
    );
}

#[tokio::test]
async fn run_errors_map_to_http_statuses() {
    let (status, _, body) = get_path(app(echo_world()), "/once?q=hi").await;
    assert_eq!((status, body.as_str()), (StatusCode::OK, "hi"));

    // Caller left out the input: 400.
    let (status, _, body) = get_path(app(echo_world()), "/once").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.contains("Input<Task>"), "{body}");

    // Server is missing a binding: 500 with the diagnostics.
    let (status, _, body) = get_path(app(AgentWorld::new()), "/once?q=hi").await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(body.contains("✗ Llm: no provider registered"), "{body}");
}
