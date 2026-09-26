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
            format!("(fake llm saw {} chars of prompt)", prompt.len())
        }))?
        .provide_tool::<WebSearch>(web.clone())?
        .provide_context(RelevantMemory::new([
            "user is building an agent runtime in Rust",
            "user likes Bevy's system params",
        ]))?;

    // Prepare once, run twice: parameter state is initialized a single time.
    let mut agent = world.prepare(researcher)?;
    for _ in 0..2 {
        let answer = world.run_prepared(&mut agent).await?;
        println!("answer:  {}\nsources: {:?}", answer.text, answer.sources);
    }
    println!("web_search requests: {:?}", web.requests());
    Ok(())
}
