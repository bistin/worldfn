//! `cargo run --example researcher`

use worldfn::SearchHit;
use worldfn::prelude::*;

struct Answer {
    text: String,
    sources: Vec<String>,
}

async fn researcher(
    task: Input<Task>,
    llm: Llm,
    web: Tool<WebSearch>,
    memory: Context<RelevantMemory<2>>,
) -> Answer {
    let hits = web.call(task.0.clone()).await.unwrap_or_default();
    let sources: Vec<String> = hits.iter().map(|h| h.url.clone()).collect();
    let prompt = format!(
        "Task: {}\nSources: {sources:?}\nKnown about the user: {:?}",
        task.0, memory.entries
    );
    let text = llm.complete(prompt).await.unwrap_or_else(|e| e.to_string());
    Answer { text, sources }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The signature is the spec: inspect it before any world exists.
    println!("{}\n", researcher.into_agent().meta());

    // A partial world reports every requirement, satisfied or not.
    let mut partial = AgentWorld::new();
    partial.provide_llm(FakeLlm::new())?;
    if let Err(err) = partial.prepare(researcher) {
        println!("{err}\n");
    }

    let web = FakeTool::<WebSearch>::new(|query| {
        Ok(vec![SearchHit {
            title: format!("Results for {query}"),
            url: "https://docs.rs/bevy_ecs".into(),
            snippet: "SystemParam ...".into(),
        }])
    });
    let mut world = AgentWorld::new();
    world
        .provide_llm(FakeLlm::responding(|prompt| {
            // Show which memory reached the prompt.
            format!("(fake) {}", prompt.lines().last().unwrap_or_default())
        }))?
        .provide_tool::<WebSearch>(web.clone())?
        .provide_memory(FakeMemory::new([
            "user is building an agent runtime in Rust",
            "user likes Bevy's system params",
            "user deploys with kubernetes",
            "user's cat is called Ferris",
        ]))?;

    // Prepare once; each run materializes context for its own task.
    let mut agent = world.prepare(researcher)?;
    for task in [
        "How do Bevy system params work?",
        "Should the Rust agent runtime use kubernetes?",
    ] {
        let answer = world
            .run_prepared_with(&mut agent, Scope::of(Task::new(task)))
            .await?;
        println!(
            "task:    {task}\nanswer:  {}\nsources: {:?}\n",
            answer.text, answer.sources
        );
    }
    println!("web_search requests: {:?}", web.requests());
    Ok(())
}
