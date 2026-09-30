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
            return Ok(Some(Llm::new(codex(model)?.instructions(instructions))));
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
         (--features codex-login / --features openai-compat)"
    )
    .into())
}

/// worldfn's own login (`worldfn login codex`) when it exists, otherwise the
/// official Codex CLI's.
#[cfg(feature = "codex")]
fn codex(model: String) -> Result<worldfn::providers::CodexLlm, Box<dyn std::error::Error>> {
    use worldfn::providers::CodexLlm;
    #[cfg(feature = "codex-login")]
    {
        let store = worldfn::providers::codex_login::TokenStore::default_location()?;
        if store.load()?.is_some() {
            return Ok(CodexLlm::from_token_store(store, model)?);
        }
    }
    CodexLlm::from_codex_home(model).map_err(|e| {
        let hint = if cfg!(feature = "codex-login") {
            "run `cargo run --features codex-login --bin worldfn -- login codex`"
        } else {
            "build with --features codex-login and run `worldfn login codex`, or `codex login`"
        };
        format!("{e}\n  hint: {hint}").into()
    })
}
