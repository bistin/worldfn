//! Session / account memory: swappable stores, a cache layer, and agent
//! parameters bound to the caller.

use worldfn::prelude::*;
use worldfn::store::{
    AccountId, Cached, InMemoryAccountMemory, InMemoryCache, InMemorySessions, SessionId,
    SessionStore, Turn, conformance,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

// ---- Every store combination passes the same suite -------------------------

#[tokio::test]
async fn in_memory_stores_conform() {
    conformance::session_store(InMemorySessions::default()).await;
    conformance::account_memory_store(InMemoryAccountMemory::default()).await;
}

#[tokio::test]
async fn cached_stores_conform() {
    conformance::session_store(Cached::new(
        InMemorySessions::default(),
        InMemoryCache::default(),
    ))
    .await;
    // A small window forces both the cached and the pass-through paths.
    conformance::session_store(
        Cached::new(InMemorySessions::default(), InMemoryCache::default()).window(4),
    )
    .await;
    conformance::account_memory_store(Cached::new(
        InMemoryAccountMemory::default(),
        InMemoryCache::default(),
    ))
    .await;

    let cache = InMemoryCache::default();
    let toggle = cache.clone();
    conformance::cached_sessions(
        Cached::new(InMemorySessions::default(), cache),
        move |down| toggle.set_down(down),
    )
    .await;
}

#[tokio::test]
async fn cache_actually_serves_repeat_reads() -> TestResult {
    let cache = InMemoryCache::default();
    let store = Cached::new(InMemorySessions::default(), cache.clone());
    let (a, s) = (AccountId::new("a"), SessionId::new("s"));
    store.append(&a, &s, Turn::user("hi")).await?;
    for _ in 0..3 {
        store.recent(&a, &s, 5).await?;
    }
    assert_eq!(cache.stats(), (3, 2), "one miss fills the cache, then hits");
    Ok(())
}

// ---- Agents only reach the caller's own memory -----------------------------

fn world() -> AgentWorld {
    let mut world = AgentWorld::new();
    world
        .provide_sessions(Cached::new(
            InMemorySessions::default(),
            InMemoryCache::default(),
        ))
        .unwrap()
        .provide_account_memory(InMemoryAccountMemory::default())
        .unwrap();
    world
}

fn scope(account: &str, session: &str, task: &str) -> Scope {
    Scope::of(Principal::new(account, session)).with(Task::new(task))
}

/// Reads history and relevant memories, answers, records the exchange, and
/// remembers anything stated as "remember ...".
async fn assistant(
    task: Input<Task>,
    history: Context<Conversation<10>>,
    recall: Context<Recall<3>>,
    log: SessionLog,
    memory: AccountMemory,
) -> (usize, Vec<String>) {
    if let Some(fact) = task.0.strip_prefix("remember ") {
        memory.remember(fact).await.unwrap();
    }
    log.record_exchange(task.0.clone(), "ok").await.unwrap();
    (
        history.turns.len(),
        recall.texts().into_iter().map(String::from).collect(),
    )
}

#[tokio::test]
async fn history_and_memories_are_per_session_and_per_account() -> TestResult {
    let world = world();
    let mut agent = world.prepare(assistant)?;

    let (seen, _) = world
        .run_prepared_with(
            &mut agent,
            scope("alice", "s1", "remember I prefer tokio for async"),
        )
        .await?;
    assert_eq!(seen, 0, "a new session starts empty");

    let (seen, recalled) = world
        .run_prepared_with(&mut agent, scope("alice", "s1", "which async runtime?"))
        .await?;
    assert_eq!(seen, 2, "the session log has the first exchange");
    assert_eq!(recalled, ["I prefer tokio for async"]);

    // A new session of the same account: fresh history, same long-term memory.
    let (seen, recalled) = world
        .run_prepared_with(&mut agent, scope("alice", "s2", "async runtime again"))
        .await?;
    assert_eq!(seen, 0);
    assert_eq!(recalled, ["I prefer tokio for async"]);

    // Another account, even reusing alice's session id, sees nothing of hers.
    let (seen, recalled) = world
        .run_prepared_with(&mut agent, scope("bob", "s1", "which async runtime?"))
        .await?;
    assert_eq!((seen, recalled.len()), (0, 0));
    Ok(())
}

#[tokio::test]
async fn read_only_agents_get_snapshots_not_handles() -> TestResult {
    // This signature has no SessionLog/AccountMemory, so it cannot write.
    async fn reader(history: Context<Conversation<5>>) -> String {
        history.transcript()
    }
    let world = world();
    world
        .run_with(assistant, scope("carol", "s", "hello"))
        .await?;
    assert_eq!(
        world.run_with(reader, scope("carol", "s", "")).await?,
        "user: hello\nassistant: ok"
    );
    Ok(())
}

#[tokio::test]
async fn missing_principal_is_a_caller_error() {
    let err = world()
        .run_with(assistant, Scope::of(Task::new("hi")))
        .await
        .unwrap_err();
    assert_eq!(err.http_status(), 400);
    assert!(err.to_string().contains("no Principal"), "{err}");
}

#[test]
fn signature_shows_scoping_and_missing_stores() {
    assert_eq!(
        assistant.into_agent().meta().to_string(),
        "assistant\n\
         ├── reads Input<Task>\n\
         ├── requires Context<Conversation<10>>\n\
         │   ├── requires SessionStore\n\
         │   └── reads Input<Principal>\n\
         ├── requires Context<Recall<3>>\n\
         │   ├── requires AccountMemoryStore\n\
         │   ├── reads Input<Principal>\n\
         │   └── reads Input<Task>\n\
         ├── requires SessionStore\n\
         ├── reads Input<Principal>\n\
         ├── requires AccountMemoryStore\n\
         └── reads Input<Principal>"
    );
    let err = AgentWorld::new().prepare(assistant).err().unwrap();
    assert!(
        err.to_string()
            .contains("✗ Context<Conversation<10>>: needs SessionStore"),
        "{err}"
    );
}
