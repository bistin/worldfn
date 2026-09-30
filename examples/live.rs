//! Run a worldfn agent against a real model.
//!
//! ```sh
//! # ChatGPT subscription (unofficial use, see docs/providers.md). Sign in once:
//! cargo run --features codex-login --bin worldfn -- login codex
//! WORLDFN_PROVIDER=codex WORLDFN_MODEL=gpt-5.5 \
//!   cargo run --example live --features codex-login -- "What should I name my Rust crate?"
//! # (or reuse an official `codex login` with --features codex)
//!
//! # DeepSeek with an API key:
//! DEEPSEEK_API_KEY=sk-... WORLDFN_PROVIDER=deepseek WORLDFN_MODEL=deepseek-v4-flash \
//!   cargo run --example live --features openai-compat -- "..."
//! ```
//!
//! The agent is the same whichever provider is chosen; only the world differs.

use worldfn::prelude::*;

mod common;

/// A small assistant: the task comes from the command line, and memory
/// relevant to it is retrieved before the body runs.
async fn assistant(
    task: Input<Task>,
    llm: Llm,
    memory: Context<RelevantMemory<3>>,
) -> Result<String, worldfn::LlmError> {
    let prompt = format!(
        "Things you know about the user:\n{}\n\nRequest: {}",
        memory
            .entries
            .iter()
            .map(|m| format!("- {m}"))
            .collect::<Vec<_>>()
            .join("\n"),
        task.0
    );
    llm.complete(prompt).await
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let task = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let task = if task.is_empty() {
        "Suggest a name for my Rust agent runtime crate.".to_owned()
    } else {
        task
    };

    let mut world = AgentWorld::new();
    world
        .provide(
            common::llm_from_env("Answer in at most five sentences.")?
                .ok_or("set WORLDFN_PROVIDER (codex / deepseek / openai)")?,
        )?
        .provide_memory(FakeMemory::new([
            "the user is building worldfn, a typed agent runtime in Rust",
            "the user likes Bevy's system params",
            "the user prefers short, direct answers",
            "the user's crate names are usually one lowercase word",
        ]))?;

    println!("{}\n", assistant.into_agent().meta());
    let answer = world
        .run_with(assistant, Scope::of(Task::new(task)))
        .await??;
    println!("{answer}");
    Ok(())
}
