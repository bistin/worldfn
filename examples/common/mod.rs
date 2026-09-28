//! Shared by the examples: pick a real provider from the environment.
//!
//! `WORLDFN_PROVIDER` = `codex` | `deepseek` | `openai`, `WORLDFN_MODEL` = a
//! model that provider currently offers. See docs/providers.md.

#![allow(dead_code)]

use worldfn::Llm;

/// `Ok(None)` when `WORLDFN_PROVIDER` is unset, so examples can fall back to a
/// fake. Errors if a provider is requested but misconfigured or not compiled in.
pub fn llm_from_env(instructions: &str) -> Result<Option<Llm>, Box<dyn std::error::Error>> {
    let Ok(provider) = std::env::var("WORLDFN_PROVIDER") else {
        return Ok(None);
    };
    let model = std::env::var("WORLDFN_MODEL")
        .map_err(|_| "set WORLDFN_MODEL to a model your provider currently offers")?;
    let _ = (&model, instructions);
    match provider.as_str() {
        #[cfg(feature = "codex")]
        "codex" => {
            return Ok(Some(Llm::new(
                worldfn::providers::CodexLlm::from_codex_home(model)?.instructions(instructions),
            )));
        }
        #[cfg(feature = "openai-compat")]
        "deepseek" => {
            return Ok(Some(Llm::new(
                worldfn::providers::OpenAiCompatLlm::deepseek(model)?.system(instructions),
            )));
        }
        #[cfg(feature = "openai-compat")]
        "openai" => {
            return Ok(Some(Llm::new(
                worldfn::providers::OpenAiCompatLlm::openai(model)?.system(instructions),
            )));
        }
        _ => {}
    }
    Err(format!(
        "unknown provider `{provider}`, or its cargo feature is not enabled \
         (--features codex / --features openai-compat)"
    )
    .into())
}
