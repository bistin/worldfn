//! Tests are grouped by the "Required validation" list of the design brief.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use worldfn::prelude::*;
use worldfn::{AgentMeta, BindError, FunctionAgent, LlmError, ParamError, SearchHit, ToolError};

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[derive(Debug, Clone, PartialEq)]
struct Answer(String);

/// The target signature, verbatim.
async fn researcher(llm: Llm, web: Tool<WebSearch>, memory: Context<RelevantMemory>) -> Answer {
    let hits = web.call("rust async".into()).await.expect("search");
    let sources: Vec<_> = hits.iter().map(|h| h.title.as_str()).collect();
    let prompt = format!("sources: {sources:?}\nmemory: {:?}", memory.entries);
    Answer(llm.complete(prompt).await.expect("llm"))
}

fn hit(title: &str) -> SearchHit {
    SearchHit {
        title: title.into(),
        url: format!("https://example.com/{title}"),
        snippet: String::new(),
    }
}

fn task(task: &str) -> Scope {
    Scope::of(Task::new(task))
}

fn research_world(
    llm: FakeLlm,
    web: FakeTool<WebSearch>,
    memory: FakeMemory,
) -> Result<AgentWorld, BindError> {
    let mut world = AgentWorld::new();
    world
        .provide_llm(llm)?
        .provide_tool::<WebSearch>(web)?
        .provide_memory(memory)?;
    Ok(world)
}

// 1. The target researcher runs using fakes, no network or API key.
#[tokio::test]
async fn researcher_runs_on_fakes() -> TestResult {
    let llm = FakeLlm::with_answer("futures are lazy");
    let web = FakeTool::<WebSearch>::with_response(vec![hit("async-book")]);
    let world = research_world(
        llm.clone(),
        web.clone(),
        FakeMemory::new(["prefers tokio for async", "the cat is called Ferris"]),
    )?;

    let answer = world
        .run_with(researcher, task("explain async rust"))
        .await?;

    assert_eq!(answer, Answer("futures are lazy".into()));
    assert_eq!(web.requests(), vec!["rust async".to_string()]);
    let prompts = llm.prompts();
    assert_eq!(prompts.len(), 1);
    assert!(prompts[0].contains("async-book") && prompts[0].contains("prefers tokio"));
    assert!(
        !prompts[0].contains("Ferris"),
        "irrelevant memory: {}",
        prompts[0]
    );
    Ok(())
}

// 2 + 7. Dependencies are declared in the signature only, and the metadata
// matches it — no world needed.
#[test]
fn metadata_matches_signature() {
    let agent = researcher.into_agent();
    assert_eq!(
        agent.meta(),
        &AgentMeta {
            name: "researcher".into(),
            params: vec![
                Requirement::Llm,
                Requirement::Tool {
                    name: "web_search",
                    spec: std::any::type_name::<WebSearch>(),
                },
                Requirement::Context {
                    type_name: std::any::type_name::<RelevantMemory>(),
                    needs: vec![
                        Requirement::Service {
                            type_name: std::any::type_name::<Memory>(),
                        },
                        Requirement::Input {
                            type_name: std::any::type_name::<Task>(),
                        },
                    ],
                },
            ],
        }
    );
    assert_eq!(
        agent.meta().to_string(),
        "researcher\n\
         ├── requires Llm\n\
         ├── can call Tool<web_search>\n\
         └── requires Context<RelevantMemory>\n    \
             ├── requires Memory\n    \
             └── reads Input<Task>"
    );
}

// 3. Missing dependencies fail before the body executes, and all are reported.
#[tokio::test]
async fn missing_dependencies_fail_before_body() -> TestResult {
    static CALLED: AtomicBool = AtomicBool::new(false);
    async fn agent(_llm: Llm, _web: Tool<WebSearch>, _mem: Context<RelevantMemory>) {
        CALLED.store(true, Ordering::SeqCst);
    }

    let mut world = AgentWorld::new();
    world.provide_llm(FakeLlm::new())?;

    let err = world.run(agent).await.unwrap_err();
    assert!(!CALLED.load(Ordering::SeqCst));
    let RunError::Unresolved(diagnostics) = &err else {
        panic!("expected Unresolved, got {err:?}");
    };
    assert_eq!(diagnostics.missing().count(), 2);
    assert_eq!(
        err.to_string(),
        "Cannot prepare agent:\n  \
         ✓ Llm\n  \
         ✗ Tool<web_search>: no provider registered\n  \
         ✗ Context<RelevantMemory>: needs Memory"
    );
    assert_eq!(world.prepare(agent).err(), Some(err));
    Ok(())
}

// 4. Substituting fakes changes neither the signature nor the body, and
// separate worlds stay isolated.
#[tokio::test]
async fn same_agent_different_isolated_worlds() -> TestResult {
    let (llm_a, llm_b) = (FakeLlm::with_answer("A"), FakeLlm::echo());
    let (web_a, web_b) = (
        FakeTool::with_response(vec![hit("a")]),
        FakeTool::with_response(vec![]),
    );
    let world_a = research_world(llm_a.clone(), web_a.clone(), FakeMemory::default())?;
    let world_b = research_world(
        llm_b.clone(),
        web_b.clone(),
        FakeMemory::new(["rust notes"]),
    )?;

    assert_eq!(
        world_a.run_with(researcher, task("rust")).await?,
        Answer("A".into())
    );
    let echoed = world_b.run_with(researcher, task("rust")).await?;
    assert!(echoed.0.contains("[\"rust notes\"]"), "{echoed:?}");

    assert_eq!((llm_a.calls(), web_a.calls()), (1, 1));
    assert_eq!((llm_b.calls(), web_b.calls()), (1, 1));
    Ok(())
}

// 5. Typed outputs and domain errors are preserved, one layer below RunError;
// provider errors are not reported as missing dependencies.
#[derive(Debug, PartialEq)]
enum AgentError {
    Llm(LlmError),
    Tool(ToolError),
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AgentError {}

async fn fallible(llm: Llm, web: Tool<WebSearch>) -> Result<Answer, AgentError> {
    web.call("q".into()).await.map_err(AgentError::Tool)?;
    let text = llm.complete("p").await.map_err(AgentError::Llm)?;
    Ok(Answer(text))
}

#[tokio::test]
async fn domain_errors_are_not_runtime_errors() -> TestResult {
    let mut world = AgentWorld::new();
    world
        .provide_llm(FakeLlm::with_answer("ok"))?
        .provide_tool::<WebSearch>(FakeTool::with_response(vec![]))?;
    let answer: Answer = world.run(fallible).await??;
    assert_eq!(answer, Answer("ok".into()));

    let mut world = AgentWorld::new();
    world
        .provide_llm(FakeLlm::new().then_error("overloaded"))?
        .provide_tool::<WebSearch>(FakeTool::with_response(vec![]))?;
    assert_eq!(
        world.run(fallible).await?,
        Err(AgentError::Llm(LlmError("overloaded".into())))
    );

    let mut world = AgentWorld::new();
    world
        .provide_llm(FakeLlm::new())?
        .provide_tool::<WebSearch>(FakeTool::failing("rate limited"))?;
    assert_eq!(
        world.run(fallible).await?,
        Err(AgentError::Tool(ToolError("rate limited".into())))
    );
    Ok(())
}

#[tokio::test]
async fn fake_llm_rejects_unexpected_calls() -> TestResult {
    async fn twice(llm: Llm) -> (Result<String, LlmError>, Result<String, LlmError>) {
        (llm.complete("1").await, llm.complete("2").await)
    }
    let mut world = AgentWorld::new();
    world.provide_llm(FakeLlm::with_answer("one"))?;
    let (first, second) = world.run(twice).await?;
    assert_eq!(first, Ok("one".into()));
    assert_eq!(
        second,
        Err(LlmError("FakeLlm received an unexpected call #2".into()))
    );
    Ok(())
}

// 6. Zero, one, multiple, and the maximum supported (8) parameters.
#[tokio::test]
async fn arities_zero_one_three_and_max() -> TestResult {
    async fn zero() -> u8 {
        0
    }
    async fn one(_: Llm) -> u8 {
        1
    }
    async fn three(_: Llm, _: Tool<WebSearch>, _: Context<RelevantMemory>) -> u8 {
        3
    }
    #[allow(clippy::too_many_arguments)]
    async fn eight(
        _: Llm,
        _: Tool<WebSearch>,
        _: Context<RelevantMemory>,
        _: Res<u32>,
        _: Res<String>,
        _: Option<Res<bool>>,
        _: Llm,
        (_, _): (Res<u32>, Res<String>),
    ) -> u8 {
        8
    }

    let mut world = research_world(
        FakeLlm::new(),
        FakeTool::failing("x"),
        FakeMemory::default(),
    )?;
    world.provide(7u32)?.provide(String::from("s"))?;

    assert_eq!(world.run(zero).await?, 0);
    assert_eq!(world.run(one).await?, 1);
    assert_eq!(world.run_with(three, task("t")).await?, 3);
    assert_eq!(world.run_with(eight, task("t")).await?, 8);
    Ok(())
}

// 8. Prepared runs initialize state once and resolve per invocation.
struct Counted {
    inits: Arc<AtomicUsize>,
    resolves: Arc<AtomicUsize>,
}

impl AgentParam for Counted {
    type State = Arc<AtomicUsize>;
    type Future = std::future::Ready<Result<Self, ParamError>>;

    fn describe(out: &mut Vec<Requirement>) {
        out.push(Requirement::Resource {
            type_name: "Counters",
        });
    }

    fn init(world: &AgentWorld) -> Result<Self::State, Vec<Requirement>> {
        let inits = world
            .resource::<Counters>()
            .ok_or_else(|| {
                vec![Requirement::Resource {
                    type_name: "Counters",
                }]
            })?
            .inits
            .clone();
        inits.fetch_add(1, Ordering::SeqCst);
        Ok(inits)
    }

    fn resolve(state: &mut Self::State, world: &AgentWorld, _scope: &Scope) -> Self::Future {
        let resolves = world.resource::<Counters>().unwrap().resolves.clone();
        resolves.fetch_add(1, Ordering::SeqCst);
        std::future::ready(Ok(Counted {
            inits: state.clone(),
            resolves,
        }))
    }
}

#[derive(Default)]
struct Counters {
    inits: Arc<AtomicUsize>,
    resolves: Arc<AtomicUsize>,
}

async fn counted(c: Counted) -> (usize, usize) {
    (
        c.inits.load(Ordering::SeqCst),
        c.resolves.load(Ordering::SeqCst),
    )
}

#[tokio::test]
async fn prepared_agent_reuses_init_state() -> TestResult {
    let mut world = AgentWorld::new();
    world.provide(Counters::default())?;

    let mut agent = world.prepare(counted)?;
    assert_eq!(world.run_prepared(&mut agent).await?, (1, 1));
    assert_eq!(world.run_prepared(&mut agent).await?, (1, 2));

    // The one-shot path initializes every time.
    assert_eq!(world.run(counted).await?, (2, 3));
    Ok(())
}

#[tokio::test]
async fn stateful_closures_are_agents() -> TestResult {
    let mut world = AgentWorld::new();
    world.provide_llm(FakeLlm::echo())?;
    let mut n = 0;
    let mut agent = world.prepare(move |llm: Llm| {
        n += 1;
        let prompt = format!("call {n}");
        async move { llm.complete(prompt).await.unwrap() }
    })?;
    assert_eq!(world.run_prepared(&mut agent).await?, "call 1");
    assert_eq!(world.run_prepared(&mut agent).await?, "call 2");
    Ok(())
}

// 9. Stale or cross-world prepared state cannot silently use wrong bindings.
#[tokio::test]
async fn stale_and_foreign_prepared_state_is_rejected() -> TestResult {
    async fn ask(llm: Llm) -> String {
        llm.complete("q").await.unwrap()
    }
    let mut world = AgentWorld::new();
    world.provide_llm(FakeLlm::with_answer("old"))?;
    let mut agent = world.prepare(ask)?;

    let other = research_world(
        FakeLlm::echo(),
        FakeTool::failing("x"),
        FakeMemory::default(),
    )?;
    assert!(matches!(
        other.run_prepared(&mut agent).await,
        Err(RunError::ForeignWorld { .. })
    ));

    world.replace(Llm::new(FakeLlm::with_answer("new")));
    assert!(matches!(
        world.run_prepared(&mut agent).await,
        Err(RunError::Stale { .. })
    ));

    agent.initialize(&world)?;
    assert_eq!(world.run_prepared(&mut agent).await?, "new");

    let mut unprepared = ask.into_agent();
    assert!(matches!(
        world.run_prepared(&mut unprepared).await,
        Err(RunError::NotPrepared { .. })
    ));
    Ok(())
}

#[test]
fn duplicate_bindings_are_rejected() {
    let mut world = AgentWorld::new();
    world.provide_llm(FakeLlm::new()).unwrap();
    let err = world.provide_llm(FakeLlm::new()).unwrap_err();
    assert_eq!(err.type_name, std::any::type_name::<Llm>());
    assert!(err.to_string().contains("use `replace` to rebind"));
}

// Extras: optional capabilities, boxed agents, 'static run futures.
#[tokio::test]
async fn optional_params_degrade_explicitly() -> TestResult {
    async fn agent(llm: Option<Llm>) -> &'static str {
        if llm.is_some() { "llm" } else { "no llm" }
    }
    let mut world = AgentWorld::new();
    assert_eq!(world.run(agent).await?, "no llm");
    world.provide_llm(FakeLlm::new())?;
    assert_eq!(world.run(agent).await?, "llm");
    assert_eq!(
        agent.into_agent().meta().to_string(),
        "agent\n└── may use Option<Llm>"
    );
    Ok(())
}

#[tokio::test]
async fn boxed_heterogeneous_agents() -> TestResult {
    async fn a(llm: Llm) -> String {
        llm.complete("a").await.unwrap()
    }
    async fn b(n: Res<u32>) -> String {
        n.to_string()
    }
    let mut world = AgentWorld::new();
    world.provide_llm(FakeLlm::echo())?.provide(7u32)?;

    let mut agents: Vec<Box<dyn Agent<Output = String>>> = vec![
        Box::new(world.prepare(a)?),
        Box::new(world.prepare(FunctionAgent::with_name(b.into_agent(), "reader"))?),
    ];
    assert_eq!(agents[1].meta().name, "reader");

    let mut outputs = Vec::new();
    for agent in &mut agents {
        outputs.push(world.run_prepared(agent).await?);
    }
    assert_eq!(outputs, ["a", "7"]);
    Ok(())
}

#[tokio::test]
async fn run_future_is_static_and_spawnable() -> TestResult {
    let world = research_world(
        FakeLlm::with_answer("ok"),
        FakeTool::with_response(vec![]),
        FakeMemory::default(),
    )?;
    let handle = tokio::spawn(world.run_with(researcher, task("t")));
    drop(world);
    assert_eq!(handle.await??, Answer("ok".into()));
    Ok(())
}
