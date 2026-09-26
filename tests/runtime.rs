use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use worldfn::prelude::*;
use worldfn::{FunctionAgent, LlmError, SearchHit, ToolError};

#[derive(Debug, Clone, PartialEq)]
struct Answer(String);

/// The motivating example, verbatim signature.
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

fn research_world(llm: FakeLlm, web: FakeTool<WebSearch>) -> AgentWorld {
    let mut world = AgentWorld::new();
    world
        .insert_llm(llm)
        .insert_tool::<WebSearch>(web)
        .insert(MemoryStore::new([
            "user prefers tokio over async-std",
            "the cat is called Ferris",
            "async rust futures are lazy",
        ]))
        .insert(Task::new("explain async rust futures"));
    world
}

#[tokio::test]
async fn runs_ordinary_async_fn_with_resolved_params() {
    let llm = FakeLlm::scripted(["futures are lazy state machines"]);
    let web = FakeTool::<WebSearch>::new(|q| {
        assert_eq!(q, "rust async");
        Ok(vec![hit("async-book")])
    });
    let world = research_world(llm.clone(), web.clone());

    let answer = world.run(researcher).await.unwrap();

    assert_eq!(answer, Answer("futures are lazy state machines".into()));
    assert_eq!(web.calls(), 1);
    let prompts = llm.prompts();
    assert_eq!(prompts.len(), 1);
    assert!(prompts[0].contains("async-book"));
    assert!(prompts[0].contains("async rust futures are lazy"));
    assert!(
        !prompts[0].contains("Ferris"),
        "irrelevant memory leaked: {}",
        prompts[0]
    );
}

#[test]
fn parameter_types_are_a_declarative_description() {
    // No world needed: the signature alone describes the agent.
    let requirements = researcher.into_agent().requirements();
    assert_eq!(
        requirements,
        vec![
            Requirement::Llm,
            Requirement::Tool {
                name: "web_search",
                spec: std::any::type_name::<WebSearch>()
            },
            Requirement::Context {
                type_name: std::any::type_name::<RelevantMemory>(),
                needs: vec![
                    Requirement::Resource {
                        type_name: std::any::type_name::<MemoryStore>()
                    },
                    Requirement::Resource {
                        type_name: std::any::type_name::<Task>()
                    },
                ],
            },
        ]
    );
}

#[tokio::test]
async fn reports_every_missing_requirement_and_never_calls_the_fn() {
    static CALLED: AtomicBool = AtomicBool::new(false);
    async fn agent(_llm: Llm, _web: Tool<WebSearch>, _mem: Context<RelevantMemory>) {
        CALLED.store(true, Ordering::SeqCst);
    }

    let mut world = AgentWorld::new();
    world.insert(MemoryStore::default()); // Task still missing.

    let err = world.run(agent).await.unwrap_err();
    assert!(!CALLED.load(Ordering::SeqCst));
    assert!(err.agent.ends_with("agent"), "{}", err.agent);
    assert_eq!(
        err.missing,
        vec![
            Requirement::Llm,
            Requirement::Tool {
                name: "web_search",
                spec: std::any::type_name::<WebSearch>()
            },
            Requirement::Context {
                type_name: std::any::type_name::<RelevantMemory>(),
                needs: vec![Requirement::Resource {
                    type_name: std::any::type_name::<Task>()
                }],
            },
        ]
    );
    let msg = err.to_string();
    assert!(msg.contains("- Llm"), "{msg}");
    assert!(msg.contains("- Tool<web_search>"), "{msg}");
    assert!(msg.contains("(needs Res<worldfn::context::Task>)"), "{msg}");

    assert_eq!(world.validate(agent), Err(err));
}

#[tokio::test]
async fn zero_arity_and_closures() {
    async fn constant() -> u32 {
        7
    }
    let world = AgentWorld::new();
    assert_eq!(world.run(constant).await, Ok(7));

    // A closure returning an `async move` block is also an agent.
    let mut world = AgentWorld::new();
    world.insert(Task::new("hi"));
    let shout = |task: Res<Task>| async move { task.0.to_uppercase() };
    assert_eq!(world.run(shout).await.unwrap(), "HI");
}

#[tokio::test]
async fn optional_and_nested_tuple_params() {
    async fn agent(maybe_llm: Option<Llm>, (task, _mem): (Res<Task>, Res<MemoryStore>)) -> String {
        match maybe_llm {
            Some(llm) => llm.complete(task.0.clone()).await.unwrap(),
            None => format!("no llm for {}", task.0),
        }
    }

    let mut world = AgentWorld::new();
    world.insert(Task::new("t")).insert(MemoryStore::default());
    assert_eq!(world.run(agent).await.unwrap(), "no llm for t");

    world.insert_llm(FakeLlm::echo());
    assert_eq!(world.run(agent).await.unwrap(), "t");

    assert_eq!(
        agent.into_agent().requirements()[0],
        Requirement::Optional(Box::new(Requirement::Llm))
    );
}

#[tokio::test]
async fn user_defined_context_and_param() {
    /// A custom context composed from another context and a resource.
    struct Briefing(String);
    impl ContextSource for Briefing {
        type Deps = (Res<Task>, Context<RelevantMemory>);
        fn build((task, mem): Self::Deps) -> Self {
            Briefing(format!("{} | {}", task.0, mem.entries.join("; ")))
        }
    }

    /// A custom param implemented by hand.
    struct Budget(u32);
    impl AgentParam for Budget {
        fn describe(out: &mut Vec<Requirement>) {
            out.push(Requirement::Resource {
                type_name: "Budget",
            });
        }
        fn fetch(world: &AgentWorld) -> Result<Self, Vec<Requirement>> {
            let task = world.resource::<Task>().ok_or_else(|| {
                let mut v = Vec::new();
                Self::describe(&mut v);
                v
            })?;
            Ok(Budget(task.0.len() as u32 * 10))
        }
    }

    async fn agent(brief: Context<Briefing>, budget: Budget) -> (String, u32) {
        (brief.0.0, budget.0)
    }

    let mut world = AgentWorld::new();
    world
        .insert(Task::new("rust tips"))
        .insert(MemoryStore::new(["rust has traits", "cats"]));
    assert_eq!(
        world.run(agent).await.unwrap(),
        ("rust tips | rust has traits".to_string(), 90)
    );
}

#[tokio::test]
async fn boxed_heterogeneous_agents_run_concurrently() {
    async fn a(llm: Llm) -> String {
        llm.complete("a").await.unwrap()
    }
    async fn b(task: Res<Task>) -> String {
        task.0.clone()
    }

    let mut world = AgentWorld::new();
    world.insert_llm(FakeLlm::echo()).insert(Task::new("b"));

    let agents: Vec<Box<dyn Agent<Output = String>>> = vec![
        Box::new(a.into_agent()),
        Box::new(FunctionAgent::with_name(b.into_agent(), "task-reader")),
    ];
    assert_eq!(agents[1].name(), "task-reader");

    let (x, y) = tokio::join!(world.run_agent(&*agents[0]), world.run_agent(&*agents[1]));
    assert_eq!((x.unwrap(), y.unwrap()), ("a".to_string(), "b".to_string()));
}

#[tokio::test]
async fn run_future_is_static_and_spawnable() {
    let world = research_world(FakeLlm::scripted(["ok"]), FakeTool::new(|_| Ok(vec![])));

    // Parameters are resolved at `run` time; the future owns them.
    let handle = tokio::spawn(world.run(researcher));
    drop(world);
    assert_eq!(handle.await.unwrap().unwrap(), Answer("ok".into()));
}

#[tokio::test]
async fn fake_errors_surface_to_the_agent() {
    async fn agent(
        llm: Llm,
        web: Tool<WebSearch>,
    ) -> (Result<String, LlmError>, Result<usize, ToolError>) {
        let search = web.call("x".into()).await.map(|hits| hits.len());
        (llm.complete("p").await, search)
    }

    let mut world = AgentWorld::new();
    world
        .insert_llm(FakeLlm::scripted(Vec::<String>::new()))
        .insert_tool::<WebSearch>(FakeTool::new(|_| Err(ToolError("rate limited".into()))));

    let (llm, web) = world.run(agent).await.unwrap();
    assert_eq!(llm, Err(LlmError("FakeLlm script exhausted".into())));
    assert_eq!(web, Err(ToolError("rate limited".into())));
}

#[tokio::test]
async fn world_resources_are_shared_handles() {
    let mut world = AgentWorld::new();
    world.insert(Task::new("one"));
    let arc: Option<Arc<Task>> = world.resource_arc::<Task>();
    assert_eq!(arc.as_deref(), Some(&Task::new("one")));
    assert!(world.contains::<Task>());
    assert!(!world.contains::<MemoryStore>());
}
