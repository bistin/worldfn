//! `worldfn login codex [--device]`, `worldfn logout codex`, `worldfn status`.
//!
//! Signs in to ChatGPT for [`CodexLlm::from_login`](worldfn::providers::CodexLlm::from_login)
//! and saves the tokens to `~/.worldfn/auth.json` (or `$WORLDFN_HOME/auth.json`).

use std::process::ExitCode;

use worldfn::providers::codex_login::{
    BrowserLogin, CodexTokens, DeviceLogin, LoginError, OAuthEndpoints, TokenStore,
};

const USAGE: &str = "\
usage:
  worldfn login codex [--device]   sign in with your ChatGPT plan
  worldfn logout codex             forget the saved login
  worldfn status                   show who is signed in (never prints tokens)";

const WARNING: &str = "\
Note: this signs in to ChatGPT the way OpenAI's Codex clients do. It is not a
public API for third-party software: it can break without notice, and using it
may conflict with OpenAI's terms. Personal experiments only; for anything else
use an API key. See docs/providers.md.
";

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let result = match args.as_slice() {
        ["login", "codex"] => login(false).await,
        ["login", "codex", "--device"] => login(true).await,
        ["logout", "codex"] => logout(),
        ["status"] => status(),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn login(device: bool) -> Result<(), LoginError> {
    let store = TokenStore::default_location()?;
    eprintln!("{WARNING}");
    let tokens = if device {
        device_login().await?
    } else {
        browser_login().await?
    };
    store.save(&tokens)?;
    println!(
        "Signed in (account {}). Saved to {}.",
        tokens.account_id,
        store.path().display()
    );
    println!("Use it with CodexLlm::from_login(model), or WORLDFN_PROVIDER=codex in the examples.");
    Ok(())
}

async fn browser_login() -> Result<CodexTokens, LoginError> {
    let login = BrowserLogin::start(OAuthEndpoints::openai()).await?;
    println!("Open this URL to sign in:\n\n  {}\n", login.url());
    let opened = open_browser(login.url());
    if !login.has_callback_server() {
        println!("Port 1455 is busy, so the browser cannot hand the login back directly.");
    }
    if opened && login.has_callback_server() {
        println!("Waiting for the browser…");
    }
    println!(
        "If the browser ends on a page that fails to load, copy that page's full URL \
         and paste it here, then press Enter."
    );

    let (tx, rx) = tokio::sync::oneshot::channel();
    // A plain thread, so a blocked stdin read never delays exit.
    // Blank lines are skipped; at EOF the sender is dropped and only the
    // browser callback can finish the login.
    std::thread::spawn(move || {
        for line in std::io::stdin().lines() {
            let Ok(line) = line else { return };
            if !line.trim().is_empty() {
                let _ = tx.send(line);
                return;
            }
        }
    });
    let code = tokio::select! {
        code = login.wait_for_code() => code?,
        Ok(line) = rx => login.code_from_input(&line)?,
    };
    login.exchange(&code).await
}

async fn device_login() -> Result<CodexTokens, LoginError> {
    let login = DeviceLogin::start(OAuthEndpoints::openai()).await?;
    println!(
        "On any device, open {}\nand enter this code: {}\n\nWaiting (up to 15 minutes)…",
        login.verification_url(),
        login.user_code()
    );
    login.wait().await
}

fn logout() -> Result<(), LoginError> {
    let store = TokenStore::default_location()?;
    if store.remove()? {
        println!("Removed the Codex login from {}.", store.path().display());
    } else {
        println!("No Codex login saved in {}.", store.path().display());
    }
    Ok(())
}

fn status() -> Result<(), LoginError> {
    let store = TokenStore::default_location()?;
    match store.load()? {
        None => println!("codex: not signed in (run `worldfn login codex`)"),
        Some(tokens) => {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
            let expiry = if tokens.expires_at > now {
                format!(
                    "access token valid for {} min",
                    (tokens.expires_at - now) / 60
                )
            } else {
                "access token expired; it is refreshed on the next call".to_owned()
            };
            println!(
                "codex: signed in, account {} ({expiry})\n       {}",
                tokens.account_id,
                store.path().display()
            );
        }
    }
    Ok(())
}

/// Best effort; the URL is printed either way.
fn open_browser(url: &str) -> bool {
    use std::process::{Command, Stdio};
    let mut command = if cfg!(target_os = "macos") {
        Command::new("open")
    } else if cfg!(windows) {
        let mut c = Command::new("rundll32");
        c.arg("url.dll,FileProtocolHandler");
        c
    } else {
        Command::new("xdg-open")
    };
    command
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .is_ok()
}
