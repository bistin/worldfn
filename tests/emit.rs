//! `Emit<E>`: streaming events out of a running agent, framework-free.

use worldfn::emit::{self, Emitter};
use worldfn::prelude::*;
use worldfn::{ParamErrorKind, SseEvent, SseFrame};

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[derive(Debug, Clone, PartialEq)]
enum Progress {
    Searching(String),
    Found(usize),
}

impl SseEvent for Progress {
    fn to_sse(&self) -> SseFrame {
        match self {
            Progress::Searching(q) => SseFrame::new("searching", q.clone()),
            Progress::Found(n) => SseFrame::new("found", n.to_string()),
        }
    }
}

async fn searcher(task: Input<Task>, memory: Memory, events: Emit<Progress>) -> usize {
    events.send(Progress::Searching(task.0.clone()));
    let hits = memory.search(&task.0, 10).await.unwrap_or_default();
    events.send(Progress::Found(hits.len()));
    hits.len()
}

fn world() -> AgentWorld {
    let mut world = AgentWorld::new();
    world
        .provide_memory(FakeMemory::new([
            "rust async notes",
            "rust traits notes",
            "cats",
        ]))
        .unwrap();
    world
}

#[tokio::test]
async fn events_stream_while_the_agent_runs_and_end_with_it() -> TestResult {
    let (emitter, mut events) = emit::channel::<Progress>();
    let run =
        tokio::spawn(world().run_with(searcher, Scope::of(Task::new("rust notes")).with(emitter)));

    let mut seen = Vec::new();
    while let Some(event) = events.recv().await {
        seen.push(event);
    }
    assert_eq!(
        seen,
        [Progress::Searching("rust notes".into()), Progress::Found(2)]
    );
    assert_eq!(run.await??, 2);

    let wire: String = seen.iter().map(|e| e.to_sse().encode()).collect();
    assert_eq!(
        wire,
        "event: searching\ndata: rust notes\n\nevent: found\ndata: 2\n\n"
    );
    Ok(())
}

#[tokio::test]
async fn missing_emitter_is_a_caller_error() {
    let err = world()
        .run_with(searcher, Scope::of(Task::new("x")))
        .await
        .unwrap_err();
    let RunError::Param { error, .. } = &err else {
        panic!("{err:?}")
    };
    assert_eq!(error.kind, ParamErrorKind::MissingFromScope);
    assert!(err.is_caller_error());
    assert_eq!(err.http_status(), 400);
    assert!(err.to_string().contains("Emit<Progress>"), "{err}");
}

#[test]
fn http_status_classification() {
    // Setup problems are the server's fault.
    let err = AgentWorld::new().prepare(searcher).err().unwrap();
    assert_eq!(err.http_status(), 500);
    assert!(!err.is_caller_error());
}

#[tokio::test]
async fn upstream_context_failure_is_a_bad_gateway() {
    async fn agent(_m: Context<RelevantMemory>) {}
    let mut world = AgentWorld::new();
    world
        .provide_memory(FakeMemory::failing("index down"))
        .unwrap();
    let err = world
        .run_with(agent, Scope::of(Task::new("x")))
        .await
        .unwrap_err();
    assert_eq!(err.http_status(), 502);
}

#[tokio::test]
async fn agents_see_a_disconnected_client() -> TestResult {
    async fn chatty(events: Emit<Progress>) -> (bool, bool) {
        let first = events.send(Progress::Found(1));
        // ...the client goes away here...
        tokio::task::yield_now().await;
        (first, events.is_open())
    }
    let (emitter, events): (Emitter<Progress>, _) = emit::channel();
    let run = AgentWorld::new().run_with(chatty, Scope::new().with(emitter));
    drop(events);
    assert_eq!(run.await?, (false, false));
    Ok(())
}

#[test]
fn metadata_and_prepare_report() {
    assert_eq!(
        searcher.into_agent().meta().to_string(),
        "searcher\n\
         ├── reads Input<Task>\n\
         ├── requires Memory\n\
         └── emits Emit<Progress>"
    );
    let err = AgentWorld::new().prepare(searcher).err().unwrap();
    assert_eq!(
        err.to_string(),
        "Cannot prepare searcher:\n  \
         · Input<Task>: checked when started\n  \
         ✗ Memory: no provider registered\n  \
         · Emit<Progress>: checked when started"
    );
}
