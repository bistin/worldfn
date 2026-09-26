//! `cargo run --example researcher`

use worldfn::SearchHit;
use worldfn::prelude::*;

struct Answer {
    text: String,
    sources: Vec<String>,
}

async fn researcher(llm: Llm, web: Tool<WebSearch>, memory: Context<RelevantMemory>) -> Answer {
    let hits = web
        .call("bevy system params".into())
        .await
        .unwrap_or_default();
    let sources: Vec<String> = hits.iter().map(|h| h.url.clone()).collect();
    let prompt = format!(
        "Answer using these sources: {sources:?}\nKnown about the user: {:?}",
        memory.entries
    );
    let text = llm.complete(prompt).await.unwrap_or_else(|e| e.to_string());
    Answer { text, sources }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // The signature *is* the spec: inspect it before building any world.
    println!("researcher needs:");
    for requirement in researcher.into_agent().requirements() {
        println!("  - {requirement}");
    }

    // An incomplete world reports everything that is missing at once.
    let empty = AgentWorld::new();
    if let Err(err) = empty.validate(researcher) {
        println!("\n{err}");
    }

    let mut world = AgentWorld::new();
    world
        .insert_llm(FakeLlm::responding(|prompt| {
            format!("(fake llm saw {} chars of prompt)", prompt.len())
        }))
        .insert_tool::<WebSearch>(FakeTool::new(|query| {
            Ok(vec![SearchHit {
                title: format!("Results for {query}"),
                url: "https://docs.rs/bevy_ecs".into(),
                snippet: "SystemParam ...".into(),
            }])
        }))
        .insert(MemoryStore::new([
            "user is building an agent runtime in Rust",
            "user likes Bevy's system params",
            "user's cat is called Ferris",
        ]))
        .insert(Task::new("How do Bevy system params work in Rust?"));

    let answer = world
        .run(researcher)
        .await
        .expect("world satisfies researcher");
    println!("\nanswer:  {}\nsources: {:?}", answer.text, answer.sources);
}
