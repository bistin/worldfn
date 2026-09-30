//! worldfn's own ChatGPT login against a local mock of the OAuth server and
//! the Codex backend: browser callback, device code, token store, refresh.
#![cfg(feature = "codex-login")]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use worldfn::prelude::*;
use worldfn::providers::CodexLlm;
use worldfn::providers::codex_login::{
    BrowserLogin, CodexTokens, DeviceLogin, OAuthEndpoints, TokenStore,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[derive(Debug, Clone)]
struct Request {
    path: String,
    headers: HashMap<String, String>,
    body: String,
}

impl Request {
    fn form(&self) -> HashMap<String, String> {
        reqwest::Url::parse(&format!("http://x/?{}", self.body))
            .unwrap()
            .query_pairs()
            .into_owned()
            .collect()
    }
}

type Handler = dyn Fn(&Request, usize) -> (u16, &'static str, String) + Send + Sync;

/// A mock server answering every request with `handler(request, nth call
/// to this path)`. Returns its base URL and the requests it has seen.
async fn mock(handler: Box<Handler>) -> (String, Arc<Mutex<Vec<Request>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::<Request>::new()));
    let log = seen.clone();
    tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut raw = Vec::new();
            let mut buf = [0u8; 4096];
            let header_end = loop {
                let n = socket.read(&mut buf).await.unwrap();
                raw.extend_from_slice(&buf[..n]);
                if let Some(i) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                    break i + 4;
                }
            };
            let head = String::from_utf8_lossy(&raw[..header_end]).to_string();
            let mut lines = head.lines();
            let path = lines.next().unwrap().split(' ').nth(1).unwrap().to_owned();
            let headers: HashMap<String, String> = lines
                .filter_map(|l| l.split_once(':'))
                .map(|(k, v)| (k.trim().to_lowercase(), v.trim().to_owned()))
                .collect();
            let len: usize = headers
                .get("content-length")
                .map_or(0, |l| l.parse().unwrap());
            while raw.len() < header_end + len {
                let n = socket.read(&mut buf).await.unwrap();
                raw.extend_from_slice(&buf[..n]);
            }
            let request = Request {
                path,
                headers,
                body: String::from_utf8_lossy(&raw[header_end..header_end + len]).to_string(),
            };
            let nth = {
                let mut log = log.lock().unwrap();
                let nth = log.iter().filter(|r| r.path == request.path).count();
                log.push(request.clone());
                nth
            };
            let (status, content_type, body) = handler(&request, nth);
            let response = format!(
                "HTTP/1.1 {status} X\r\ncontent-type: {content_type}\r\n\
                 content-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.shutdown().await.unwrap();
        }
    });
    (url, seen)
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn access_token(account: &str, tag: &str) -> String {
    let claims = json!({
        "exp": now() + 3600,
        "tag": tag,
        "https://api.openai.com/auth": { "chatgpt_account_id": account },
    });
    let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(claims.to_string());
    format!("e30.{payload}.sig")
}

fn token_reply(tag: &str, refresh: Option<&str>) -> String {
    let mut reply = json!({
        "access_token": access_token("acct-9", tag),
        "id_token": "unused",
        "expires_in": 3600,
    });
    if let Some(refresh) = refresh {
        reply["refresh_token"] = json!(refresh);
    }
    reply.to_string()
}

fn scratch_store(name: &str) -> TokenStore {
    let dir = std::env::temp_dir().join(format!("worldfn-login-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    TokenStore::at(dir.join("auth.json"))
}

fn query(url: &str) -> HashMap<String, String> {
    reqwest::Url::parse(url)
        .unwrap()
        .query_pairs()
        .into_owned()
        .collect()
}

#[tokio::test]
async fn browser_login_checks_state_and_exchanges_the_code() -> TestResult {
    let (issuer, seen) = mock(Box::new(|_, _| {
        (
            200,
            "application/json",
            token_reply("first", Some("refresh-1")),
        )
    }))
    .await;
    let login = BrowserLogin::start_on(OAuthEndpoints::with_issuer(&issuer), "127.0.0.1:0").await?;
    assert!(login.has_callback_server());

    let params = query(login.url());
    assert!(
        login
            .url()
            .starts_with(&format!("{issuer}/oauth/authorize?"))
    );
    assert_eq!(params["client_id"], "app_EMoamEEZ73f0CkXaXp7hrann");
    assert_eq!(params["response_type"], "code");
    assert_eq!(params["code_challenge_method"], "S256");
    assert_eq!(params["scope"], "openid profile email offline_access");
    assert_eq!(params["originator"], "worldfn");
    assert_eq!(params["code_challenge"].len(), 43);
    let redirect = params["redirect_uri"].clone();
    let state = params["state"].clone();
    assert_eq!(state.len(), 32);

    // Play the browser: a forged callback first, then the real one.
    let browser = tokio::spawn(async move {
        let http = reqwest::Client::new();
        let forged = http
            .get(format!("{redirect}?code=evil&state=wrong"))
            .send()
            .await
            .unwrap();
        let real = http
            .get(format!("{redirect}?code=good-code&state={state}"))
            .send()
            .await
            .unwrap();
        (forged.status().as_u16(), real.status().as_u16())
    });
    let code = login.wait_for_code().await?;
    assert_eq!(code, "good-code");
    assert_eq!(browser.await?, (400, 200));

    let tokens = login.exchange(&code).await?;
    assert_eq!(tokens.account_id, "acct-9");
    assert_eq!(tokens.refresh_token, "refresh-1");
    assert!(tokens.expires_at > now() + 3000);

    let requests = seen.lock().unwrap().clone();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].path, "/oauth/token");
    let form = requests[0].form();
    assert_eq!(form["grant_type"], "authorization_code");
    assert_eq!(form["code"], "good-code");
    assert_eq!(form["redirect_uri"], params["redirect_uri"]);
    assert_eq!(form["code_verifier"].len(), 43);
    Ok(())
}

#[tokio::test]
async fn pasted_redirects_must_match_the_login() -> TestResult {
    let login =
        BrowserLogin::start_on(OAuthEndpoints::with_issuer("http://unused"), "127.0.0.1:0").await?;
    let state = query(login.url())["state"].clone();
    let pasted = format!("http://localhost:1455/auth/callback?code=c1&state={state}\n");
    assert_eq!(login.code_from_input(&pasted)?, "c1");
    assert_eq!(login.code_from_input(&format!("c2#{state}"))?, "c2");
    assert_eq!(login.code_from_input("c3")?, "c3");
    let foreign = login.code_from_input("http://localhost:1455/auth/callback?code=c&state=other");
    assert!(foreign.unwrap_err().0.contains("state mismatch"));
    assert!(login.code_from_input("   ").is_err());
    Ok(())
}

#[tokio::test]
async fn device_login_polls_until_approved() -> TestResult {
    let (issuer, seen) = mock(Box::new(|request, nth| match request.path.as_str() {
        "/api/accounts/deviceauth/usercode" => (
            200,
            "application/json",
            json!({ "device_auth_id": "dev-1", "user_code": "ABCD-1234", "interval": "0" })
                .to_string(),
        ),
        "/api/accounts/deviceauth/token" if nth == 0 => (403, "application/json", "{}".into()),
        "/api/accounts/deviceauth/token" => (
            200,
            "application/json",
            json!({ "authorization_code": "dev-code", "code_verifier": "dev-verifier" })
                .to_string(),
        ),
        "/oauth/token" => (
            200,
            "application/json",
            token_reply("device", Some("r-dev")),
        ),
        _ => (500, "text/plain", "unexpected".into()),
    }))
    .await;

    let login = DeviceLogin::start(OAuthEndpoints::with_issuer(&issuer)).await?;
    assert_eq!(login.user_code(), "ABCD-1234");
    assert_eq!(login.verification_url(), format!("{issuer}/codex/device"));
    let tokens = login.wait().await?;
    assert_eq!(tokens.refresh_token, "r-dev");

    let requests = seen.lock().unwrap().clone();
    let paths: Vec<_> = requests.iter().map(|r| r.path.as_str()).collect();
    assert_eq!(
        paths,
        [
            "/api/accounts/deviceauth/usercode",
            "/api/accounts/deviceauth/token",
            "/api/accounts/deviceauth/token",
            "/oauth/token",
        ]
    );
    let usercode: Value = serde_json::from_str(&requests[0].body)?;
    assert_eq!(usercode["client_id"], "app_EMoamEEZ73f0CkXaXp7hrann");
    let poll: Value = serde_json::from_str(&requests[1].body)?;
    assert_eq!(
        poll,
        json!({ "device_auth_id": "dev-1", "user_code": "ABCD-1234" })
    );
    let form = requests[3].form();
    assert_eq!(form["code"], "dev-code");
    assert_eq!(form["code_verifier"], "dev-verifier");
    assert_eq!(
        form["redirect_uri"],
        format!("{issuer}/deviceauth/callback")
    );
    Ok(())
}

#[tokio::test]
async fn device_login_reports_when_it_is_not_enabled() {
    let (issuer, _) = mock(Box::new(|_, _| (404, "text/plain", "no".into()))).await;
    let err = DeviceLogin::start(OAuthEndpoints::with_issuer(&issuer))
        .await
        .err()
        .unwrap();
    assert!(err.0.contains("not enabled"), "{err}");
}

#[test]
fn token_store_round_trip_keeps_other_keys_and_is_private() -> TestResult {
    let store = scratch_store("store");
    assert_eq!(store.load()?, None);
    assert!(!store.remove()?);

    let parent = store.path().parent().unwrap();
    std::fs::create_dir_all(parent)?;
    std::fs::write(store.path(), r#"{"other":{"keep":true}}"#)?;
    let tokens = CodexTokens {
        access_token: "a".into(),
        refresh_token: "r".into(),
        expires_at: 42,
        account_id: "acct".into(),
    };
    store.save(&tokens)?;
    assert_eq!(store.load()?, Some(tokens));
    let saved: Value = serde_json::from_str(&std::fs::read_to_string(store.path())?)?;
    assert_eq!(saved["other"]["keep"], true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(store.path())?.permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    assert!(store.remove()?);
    assert_eq!(store.load()?, None);
    let saved: Value = serde_json::from_str(&std::fs::read_to_string(store.path())?)?;
    assert_eq!(saved, json!({ "other": { "keep": true } }));
    Ok(())
}

async fn ask(llm: Llm) -> Result<String, worldfn::LlmError> {
    llm.complete("hi").await
}

#[tokio::test]
async fn from_login_refreshes_an_expiring_token_once_and_saves_it() -> TestResult {
    let sse = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n\
               data: {\"type\":\"response.completed\",\"response\":{}}\n\n";
    let (server, seen) = mock(Box::new(move |request, _| match request.path.as_str() {
        // No refresh_token in the reply: the old one must be kept.
        "/oauth/token" => (200, "application/json", token_reply("fresh", None)),
        "/codex/responses" => (200, "text/event-stream", sse.into()),
        _ => (500, "text/plain", "unexpected".into()),
    }))
    .await;

    let store = scratch_store("refresh");
    store.save(&CodexTokens {
        access_token: access_token("acct-9", "stale"),
        refresh_token: "refresh-old".into(),
        expires_at: now() + 30, // inside the refresh margin
        account_id: "acct-9".into(),
    })?;

    let mut world = AgentWorld::new();
    world.provide_llm(
        CodexLlm::from_token_store(store.clone(), "gpt-test")?
            .oauth_endpoints(OAuthEndpoints::with_issuer(&server))
            .base_url(&server),
    )?;
    assert_eq!(world.run(ask).await??, "hello");
    assert_eq!(world.run(ask).await??, "hello");

    let requests = seen.lock().unwrap().clone();
    let paths: Vec<_> = requests.iter().map(|r| r.path.as_str()).collect();
    assert_eq!(
        paths,
        ["/oauth/token", "/codex/responses", "/codex/responses"]
    );
    let form = requests[0].form();
    assert_eq!(form["grant_type"], "refresh_token");
    assert_eq!(form["refresh_token"], "refresh-old");
    for call in &requests[1..] {
        assert_eq!(
            call.headers["authorization"],
            format!("Bearer {}", access_token("acct-9", "fresh"))
        );
        assert_eq!(call.headers["chatgpt-account-id"], "acct-9");
    }

    let saved = store.load()?.unwrap();
    assert_eq!(saved.refresh_token, "refresh-old");
    assert_eq!(saved.access_token, access_token("acct-9", "fresh"));
    Ok(())
}

#[test]
fn from_login_without_a_saved_login_says_how_to_log_in() {
    let err = CodexLlm::from_token_store(scratch_store("missing"), "gpt-test")
        .err()
        .unwrap();
    assert!(err.0.contains("worldfn login codex"), "{err}");
}
