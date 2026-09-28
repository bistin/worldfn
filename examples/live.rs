//! Run a worldfn agent against a real model.
//!
//! ```sh
//! # ChatGPT subscription via the official Codex CLI login (unofficial use,
//! # see docs/providers.md):
//! codex login
//! WORLDFN_PROVIDER=codex WORLDFN_MODEL=gpt-5.5 \
//!   cargo run --example live --features codex -- "What should I name my Rust crate?"
//!
//! # DeepSeek with an API key:
//! DEEPSEEK_API_KEY=sk-... WORLDFN_PROVIDER=deepseek WORLDFN_MODEL=deepseek-v4-flash \
//!   cargo run --example live --features openai-compat -- "..."
//! ```
//!
//! The agent is the same whichever provider is chosen; only the world differs.

use worldfn::prelude::*;

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

fn llm_from_env() -> Result<Llm, Box<dyn std::error::Error>> {
    let provider = std::env::var("WORLDFN_PROVIDER").unwrap_or_else(|_| "codex".into());
    let model = std::env::var("WORLDFN_MODEL")
        .map_err(|_| "set WORLDFN_MODEL to a model your provider currently offers")?;
    let instructions = "Answer in at most five sentences.";
    let llm = match provider.as_str() {
        #[cfg(feature = "codex")]
        "codex" => Llm::new(
            worldfn::providers::CodexLlm::from_codex_home(model)?.instructions(instructions),
        ),
        #[cfg(feature = "openai-compat")]
        "deepseek" => {
            Llm::new(worldfn::providers::OpenAiCompatLlm::deepseek(model)?.system(instructions))
        }
        #[cfg(feature = "openai-compat")]
        "openai" => {
            Llm::new(worldfn::providers::OpenAiCompatLlm::openai(model)?.system(instructions))
        }
        other => {
            return Err(format!(
                "unknown or disabled provider `{other}`; enable its cargo feature \
                 (codex / openai-compat)"
            )
            .into());
        }
    };
    Ok(llm)
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
        .provide(llm_from_env()?)?
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
