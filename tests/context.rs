//! M2 slice: invocation inputs and per-invocation, async context
//! materialization declared by the signature.

use std::sync::atomic::{AtomicUsize, Ordering};

use worldfn::prelude::*;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn memory() -> FakeMemory {
    FakeMemory::new([
        "user prefers tokio for async rust",
        "user deploys to kubernetes on fridays",
        "user's cat is called Ferris",
        "kubernetes cluster runs in eu-west",
    ])
}

async fn recall(memory: Context<RelevantMemory>) -> Vec<String> {
    memory.into_inner().entries
}

// Acceptance criterion: two tasks, same agent, different context.
#[tokio::test]
async fn different_tasks_materialize_different_context() -> TestResult {
    let store = memory();
    let mut world = AgentWorld::new();
    world.provide_memory(store.clone())?;
    let mut agent = world.prepare(recall)?;

    let about_async = world
        .run_prepared_with(
            &mut agent,
            Scope::of(Task::new("how should I write async rust?")),
        )
        .await?;
    let about_deploys = world
        .run_prepared_with(
            &mut agent,
            Scope::of(Task::new("when does the kubernetes deploy happen?")),
        )
        .await?;

    assert_eq!(about_async, ["user prefers tokio for async rust"]);
    assert_eq!(
        about_deploys,
        [
            "user deploys to kubernetes on fridays",
            "kubernetes cluster runs in eu-west"
        ]
    );
    // Retrieval ran once per invocation, with the task text and the limit
    // taken from the type.
    assert_eq!(
        store.queries(),
        [
            ("how should I write async rust?".to_string(), 5),
            ("when does the kubernetes deploy happen?".to_string(), 5),
        ]
    );
    Ok(())
}

#[tokio::test]
async fn limit_comes_from_the_type() -> TestResult {
    async fn top_one(memory: Context<RelevantMemory<1>>) -> usize {
        memory.entries.len()
    }
    let store = memory();
    let mut world = AgentWorld::new();
    world.provide_memory(store.clone())?;

    let n = world
        .run_with(top_one, Scope::of(Task::new("kubernetes")))
        .await?;
    assert_eq!(n, 1);
    assert_eq!(store.queries()[0].1, 1);
    assert_eq!(
        top_one.into_agent().meta().to_string(),
        "top_one\n\
         └── requires Context<RelevantMemory<1>>\n    \
             ├── requires Memory\n    \
             └── reads Input<Task>"
    );
    Ok(())
}

#[tokio::test]
async fn missing_input_fails_before_body_and_before_retrieval() -> TestResult {
    static CALLED: AtomicUsize = AtomicUsize::new(0);
    async fn agent(_memory: Context<RelevantMemory>) {
        CALLED.fetch_add(1, Ordering::SeqCst);
    }
    let store = memory();
    let mut world = AgentWorld::new();
    world.provide_memory(store.clone())?;

    // Inputs vary per invocation, so prepare cannot check them...
    let mut prepared = world.prepare(agent)?;
    // ...but start does, before the body and before any retrieval.
    let err = world.run_prepared(&mut prepared).await.unwrap_err();
    assert!(matches!(err, RunError::Param { .. }));
    assert_eq!(
        err.to_string(),
        "agent `agent`: parameter `Input<Task>` failed to resolve: \
         not present in the invocation scope"
    );
    assert_eq!(CALLED.load(Ordering::SeqCst), 0);
    assert!(store.queries().is_empty());
    Ok(())
}

#[tokio::test]
async fn retrieval_failure_is_a_runtime_error() -> TestResult {
    static CALLED: AtomicUsize = AtomicUsize::new(0);
    async fn agent(_memory: Context<RelevantMemory>) {
        CALLED.fetch_add(1, Ordering::SeqCst);
    }
    let mut world = AgentWorld::new();
    world.provide_memory(FakeMemory::failing("index offline"))?;

    let err = world
        .run_with(agent, Scope::of(Task::new("anything")))
        .await
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "agent `agent`: parameter `Context<RelevantMemory>` failed to resolve: index offline"
    );
    assert_eq!(CALLED.load(Ordering::SeqCst), 0);
    Ok(())
}

#[tokio::test]
async fn inputs_are_plain_params_too() -> TestResult {
    async fn planner(task: Input<Task>, llm: Llm) -> String {
        llm.complete(format!("plan: {}", task.0)).await.unwrap()
    }
    let mut world = AgentWorld::new();
    world.provide_llm(FakeLlm::echo())?;
    let plan = world
        .run_with(planner, Scope::of(Task::new("ship M2")))
        .await?;
    assert_eq!(plan, "plan: ship M2");
    Ok(())
}

/// A user-defined context that awaits another context and an input.
#[derive(Debug)]
struct Briefing(String);

impl ContextSource for Briefing {
    type Deps = (Input<Task>, Context<RelevantMemory<2>>);

    async fn materialize((task, memory): Self::Deps) -> Result<Self, ContextError> {
        // Real sources would await I/O here; the world is not borrowed.
        let summary = async { memory.entries.join("; ") }.await;
        if summary.is_empty() {
            return Err(ContextError("nothing relevant to brief on".into()));
        }
        Ok(Briefing(format!("{} | {summary}", task.0)))
    }
}

#[tokio::test]
async fn contexts_compose() -> TestResult {
    async fn briefed(brief: Context<Briefing>) -> String {
        brief.into_inner().0
    }
    let mut world = AgentWorld::new();
    world.provide_memory(memory())?;

    assert_eq!(
        world
            .run_with(briefed, Scope::of(Task::new("async rust")))
            .await?,
        "async rust | user prefers tokio for async rust"
    );
    let err = world
        .run_with(briefed, Scope::of(Task::new("zzz")))
        .await
        .unwrap_err();
    assert!(
        err.to_string().ends_with("nothing relevant to brief on"),
        "{err}"
    );

    assert_eq!(
        briefed.into_agent().meta().to_string(),
        "briefed\n\
         └── requires Context<Briefing>\n    \
             ├── reads Input<Task>\n    \
             └── requires Context<RelevantMemory<2>>\n        \
                 ├── requires Memory\n        \
                 └── reads Input<Task>"
    );

    // Inputs are not checked at prepare time, and the report says so.
    async fn with_input(_task: Input<Task>, _llm: Llm) {}
    let err = AgentWorld::new().prepare(with_input).err().unwrap();
    assert_eq!(
        err.to_string(),
        "Cannot prepare with_input:\n  \
         · Input<Task>: checked when started\n  \
         ✗ Llm: no provider registered"
    );

    // Unmet needs of nested contexts are reported through the outer one.
    let err = AgentWorld::new().prepare(briefed).err().unwrap();
    assert_eq!(
        err.to_string(),
        "Cannot prepare briefed:\n  ✗ Context<Briefing>: needs Context<RelevantMemory<2>>"
    );
    Ok(())
}

#[tokio::test]
async fn materialization_runs_after_run_returns_without_borrowing_world() -> TestResult {
    let mut world = AgentWorld::new();
    world.provide_memory(memory())?;
    let pending = world.run_with(recall, Scope::of(Task::new("async rust")));
    // The world can be changed or dropped while retrieval is still pending.
    world.replace(Memory::new(FakeMemory::default()));
    drop(world);
    assert_eq!(pending.await?, ["user prefers tokio for async rust"]);
    Ok(())
}
