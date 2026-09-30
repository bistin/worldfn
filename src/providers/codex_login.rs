//! worldfn's own ChatGPT login for [`CodexLlm`](super::CodexLlm), so the
//! official Codex CLI is not needed.
//!
//! **Unofficial for third-party use**, like the provider itself: this signs in
//! with the OAuth client that OpenAI's Codex clients use, the way pi's
//! `openai-codex` login does. It can stop working at any time, and using it may
//! conflict with OpenAI's terms. Personal experiments only; read
//! `docs/providers.md` first.
//!
//! Two flows, both ending in [`CodexTokens`] that you save in a
//! [`TokenStore`]:
//!
//! - [`BrowserLogin`]: PKCE authorization code flow. The browser redirects to
//!   `http://localhost:1455/auth/callback`, served here; if that port is busy,
//!   paste the redirect URL instead.
//! - [`DeviceLogin`]: for machines without a browser. Enter a code at
//!   `auth.openai.com/codex/device` on any device.
//!
//! [`CodexLlm::from_login`](super::CodexLlm::from_login) then refreshes the
//! access token shortly before it expires and writes the new one back.
//!
//! The `worldfn` binary wraps all of this: `worldfn login codex [--device]`.

use std::fmt;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use super::codex::{AUTH_CLAIM, jwt_claims, now_unix};
use super::http_client;

const DEFAULT_ISSUER: &str = "https://auth.openai.com";
/// The public OAuth client of OpenAI's Codex clients (also used by pi).
const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const SCOPE: &str = "openid profile email offline_access";
const CALLBACK_ADDR: &str = "127.0.0.1:1455";
const CALLBACK_PATH: &str = "/auth/callback";
const REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
const DEVICE_TIMEOUT: Duration = Duration::from_secs(15 * 60);
/// RFC 8628: default poll interval, and the step added on `slow_down`.
const DEVICE_DEFAULT_INTERVAL: Duration = Duration::from_secs(5);
const DEVICE_SLOW_DOWN_STEP: Duration = Duration::from_secs(5);
const DEVICE_MIN_INTERVAL: Duration = Duration::from_secs(1);
/// Key under which Codex tokens live in worldfn's `auth.json`.
const STORE_KEY: &str = "codex";

/// A failed login, token exchange, refresh, or token-store operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoginError(pub String);

impl fmt::Display for LoginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LoginError {}

fn err(message: impl Into<String>) -> LoginError {
    LoginError(message.into())
}

/// Where to sign in. [`OAuthEndpoints::openai`] is the real service; tests
/// point `issuer` at a local mock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthEndpoints {
    /// `https://auth.openai.com` by default.
    pub issuer: String,
    pub client_id: String,
}

impl OAuthEndpoints {
    pub fn openai() -> Self {
        Self {
            issuer: DEFAULT_ISSUER.into(),
            client_id: CLIENT_ID.into(),
        }
    }

    /// Same client, different issuer. For tests.
    pub fn with_issuer(issuer: &str) -> Self {
        Self {
            issuer: issuer.trim_end_matches('/').into(),
            ..Self::openai()
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.issuer)
    }
}

impl Default for OAuthEndpoints {
    fn default() -> Self {
        Self::openai()
    }
}

/// A signed-in ChatGPT account. `Debug` never prints the tokens.
#[derive(Clone, PartialEq, Eq)]
pub struct CodexTokens {
    pub access_token: String,
    pub refresh_token: String,
    /// Unix seconds.
    pub expires_at: u64,
    /// ChatGPT account id, sent as the `chatgpt-account-id` header.
    pub account_id: String,
}

impl fmt::Debug for CodexTokens {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CodexTokens")
            .field("access_token", &"<redacted>")
            .field("refresh_token", &"<redacted>")
            .field("expires_at", &self.expires_at)
            .field("account_id", &self.account_id)
            .finish()
    }
}

impl CodexTokens {
    /// True when the access token expires within `margin` seconds.
    pub fn expires_within(&self, margin: u64) -> bool {
        self.expires_at <= now_unix().saturating_add(margin)
    }

    /// Reads an OAuth token response. `previous_refresh` is kept when a
    /// refresh response does not rotate the refresh token.
    fn from_token_response(
        body: &Value,
        previous_refresh: Option<&str>,
    ) -> Result<Self, LoginError> {
        let text = |key: &str| {
            body.get(key)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
        };
        let access_token = text("access_token")
            .ok_or_else(|| err("token response has no access_token"))?
            .to_owned();
        let refresh_token = text("refresh_token")
            .or(previous_refresh)
            .ok_or_else(|| err("token response has no refresh_token"))?
            .to_owned();
        let claims = jwt_claims(&access_token);
        let expires_at = match body.get("expires_in").and_then(Value::as_u64) {
            Some(secs) => now_unix().saturating_add(secs),
            None => claims
                .as_ref()
                .and_then(|c| c.get("exp"))
                .and_then(Value::as_u64)
                .ok_or_else(|| err("token response has no expires_in"))?,
        };
        let account_id = [Some(access_token.as_str()), text("id_token")]
            .into_iter()
            .flatten()
            .find_map(|token| {
                jwt_claims(token)?
                    .get(AUTH_CLAIM)?
                    .get("chatgpt_account_id")?
                    .as_str()
                    .filter(|a| !a.is_empty())
                    .map(str::to_owned)
            })
            .ok_or_else(|| {
                err("the token carries no ChatGPT account id (is this a ChatGPT login?)")
            })?;
        Ok(Self {
            access_token,
            refresh_token,
            expires_at,
            account_id,
        })
    }

    fn to_json(&self) -> Value {
        json!({
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "account_id": self.account_id,
        })
    }

    fn from_json(value: &Value) -> Option<Self> {
        let text = |key: &str| value.get(key)?.as_str().map(str::to_owned);
        Some(Self {
            access_token: text("access_token")?,
            refresh_token: text("refresh_token")?,
            expires_at: value.get("expires_at")?.as_u64()?,
            account_id: text("account_id")?,
        })
    }
}

/// worldfn's credential file: `$WORLDFN_HOME/auth.json`, defaulting to
/// `~/.worldfn/auth.json`. Written with mode `0600` on Unix. Other keys in
/// the file are preserved, so future providers can share it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenStore {
    path: PathBuf,
}

impl TokenStore {
    /// The default location. Fails only if no home directory can be found.
    pub fn default_location() -> Result<Self, LoginError> {
        if let Some(home) = std::env::var_os("WORLDFN_HOME").filter(|h| !h.is_empty()) {
            return Ok(Self::at(PathBuf::from(home).join("auth.json")));
        }
        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .filter(|h| !h.is_empty())
            .ok_or_else(|| err("cannot locate home directory; set WORLDFN_HOME"))?;
        Ok(Self::at(
            PathBuf::from(home).join(".worldfn").join("auth.json"),
        ))
    }

    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// `Ok(None)` when the file or its Codex entry does not exist.
    pub fn load(&self) -> Result<Option<CodexTokens>, LoginError> {
        let Some(root) = self.read_root()? else {
            return Ok(None);
        };
        match root.get(STORE_KEY) {
            None | Some(Value::Null) => Ok(None),
            Some(entry) => CodexTokens::from_json(entry).map(Some).ok_or_else(|| {
                err(format!(
                    "{}: malformed `{STORE_KEY}` entry; run `worldfn login codex` again",
                    self.path.display()
                ))
            }),
        }
    }

    pub fn save(&self, tokens: &CodexTokens) -> Result<(), LoginError> {
        let mut root = self.read_root()?.unwrap_or_default();
        root.insert(STORE_KEY.into(), tokens.to_json());
        self.write_root(&root)
    }

    /// Forget the Codex login. Returns whether there was one.
    pub fn remove(&self) -> Result<bool, LoginError> {
        let Some(mut root) = self.read_root()? else {
            return Ok(false);
        };
        let existed = root.remove(STORE_KEY).is_some();
        if existed {
            self.write_root(&root)?;
        }
        Ok(existed)
    }

    fn read_root(&self) -> Result<Option<Map<String, Value>>, LoginError> {
        let text = match std::fs::read_to_string(&self.path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(err(format!("cannot read {}: {e}", self.path.display()))),
        };
        match serde_json::from_str(&text) {
            Ok(Value::Object(map)) => Ok(Some(map)),
            _ => Err(err(format!(
                "{} is not a JSON object; delete it and log in again",
                self.path.display()
            ))),
        }
    }

    /// Write to a private temporary file, then rename over the target, so a
    /// crash never leaves a half-written file and the tokens are never
    /// world-readable, even briefly.
    fn write_root(&self, root: &Map<String, Value>) -> Result<(), LoginError> {
        let io = |e: std::io::Error| err(format!("cannot write {}: {e}", self.path.display()));
        if let Some(dir) = self.path.parent().filter(|d| !d.as_os_str().is_empty()) {
            create_private_dir(dir).map_err(io)?;
        }
        let tmp = self
            .path
            .with_extension(format!("tmp{}", std::process::id()));
        let mut file = open_private(&tmp).map_err(io)?;
        let text = serde_json::to_string_pretty(&Value::Object(root.clone()))
            .expect("JSON values serialize");
        file.write_all(text.as_bytes()).map_err(io)?;
        file.sync_all().map_err(io)?;
        drop(file);
        std::fs::rename(&tmp, &self.path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            io(e)
        })
    }
}

#[cfg(unix)]
fn create_private_dir(dir: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir)
}

#[cfg(not(unix))]
fn create_private_dir(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)
}

#[cfg(unix)]
fn open_private(path: &Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)
}

#[cfg(not(unix))]
fn open_private(path: &Path) -> std::io::Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
}

fn random_bytes<const N: usize>() -> [u8; N] {
    let mut bytes = [0u8; N];
    getrandom::fill(&mut bytes).expect("the OS random source is available");
    bytes
}

fn pkce_challenge(verifier: &str) -> String {
    URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
}

/// Browser sign-in: PKCE authorization code flow with a local callback.
///
/// ```no_run
/// # use worldfn::providers::codex_login::*;
/// # async fn f() -> Result<(), LoginError> {
/// let login = BrowserLogin::start(OAuthEndpoints::openai()).await?;
/// println!("open {}", login.url());
/// let code = login.wait_for_code().await?;
/// let tokens = login.exchange(&code).await?;
/// TokenStore::default_location()?.save(&tokens)?;
/// # Ok(()) }
/// ```
pub struct BrowserLogin {
    endpoints: OAuthEndpoints,
    verifier: String,
    state: String,
    redirect_uri: String,
    url: String,
    listener: Option<TcpListener>,
}

impl BrowserLogin {
    /// Prepare the authorization URL and listen on `127.0.0.1:1455`, the
    /// redirect registered for this client. If the port is busy (for example
    /// the Codex CLI is logging in), there is no callback server and the user
    /// pastes the redirect URL into [`code_from_input`](Self::code_from_input).
    pub async fn start(endpoints: OAuthEndpoints) -> Result<Self, LoginError> {
        let listener = TcpListener::bind(CALLBACK_ADDR).await.ok();
        Ok(Self::new(endpoints, listener, REDIRECT_URI.into()))
    }

    /// Listen on another address, with a matching `localhost` redirect.
    /// Only useful against a mock issuer: the real one accepts port 1455 only.
    pub async fn start_on(endpoints: OAuthEndpoints, addr: &str) -> Result<Self, LoginError> {
        let listener = TcpListener::bind(addr)
            .await
            .map_err(|e| err(format!("cannot listen on {addr}: {e}")))?;
        let port = listener
            .local_addr()
            .map_err(|e| err(e.to_string()))?
            .port();
        let redirect = format!("http://localhost:{port}{CALLBACK_PATH}");
        Ok(Self::new(endpoints, Some(listener), redirect))
    }

    fn new(endpoints: OAuthEndpoints, listener: Option<TcpListener>, redirect_uri: String) -> Self {
        let verifier = URL_SAFE_NO_PAD.encode(random_bytes::<32>());
        let state: String = random_bytes::<16>()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let url = reqwest::Url::parse_with_params(
            &endpoints.url("/oauth/authorize"),
            [
                ("response_type", "code"),
                ("client_id", endpoints.client_id.as_str()),
                ("redirect_uri", redirect_uri.as_str()),
                ("scope", SCOPE),
                ("code_challenge", pkce_challenge(&verifier).as_str()),
                ("code_challenge_method", "S256"),
                ("state", state.as_str()),
                ("id_token_add_organizations", "true"),
                ("codex_cli_simplified_flow", "true"),
                ("originator", "worldfn"),
            ],
        )
        .expect("issuer URL is valid")
        .into();
        Self {
            endpoints,
            verifier,
            state,
            redirect_uri,
            url,
            listener,
        }
    }

    /// Open this in a browser.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Whether the local callback server is listening.
    pub fn has_callback_server(&self) -> bool {
        self.listener.is_some()
    }

    /// Wait for the browser to hit the callback with the right `state`, and
    /// return the authorization code. Requests with a wrong state or path get
    /// an error page and are otherwise ignored. Without a callback server this
    /// never completes: race it against pasted input.
    pub async fn wait_for_code(&self) -> Result<String, LoginError> {
        let Some(listener) = &self.listener else {
            return std::future::pending().await;
        };
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                continue;
            };
            let target = read_request_target(&mut socket).await;
            let outcome = target.as_deref().map(|t| self.check_callback(t));
            let (status, page) = match &outcome {
                Some(Ok(_)) => ("200 OK", "Signed in. You can close this window."),
                Some(Err((status, message))) => (*status, message.as_str()),
                None => ("400 Bad Request", "Malformed request."),
            };
            let body = format!(
                "<!doctype html><meta charset=utf-8><title>worldfn login</title>\
                 <p style=\"font:16px system-ui;margin:3em\">{page}</p>"
            );
            let response = format!(
                "HTTP/1.1 {status}\r\ncontent-type: text/html; charset=utf-8\r\n\
                 content-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.shutdown().await;
            match outcome {
                Some(Ok(code)) => return Ok(code),
                // The user declined or OpenAI reported an error for this login.
                Some(Err(("403 Forbidden", message))) => return Err(err(message)),
                _ => {}
            }
        }
    }

    fn check_callback(&self, target: &str) -> Result<String, (&'static str, String)> {
        let url = reqwest::Url::parse("http://localhost")
            .and_then(|base| base.join(target))
            .map_err(|_| ("400 Bad Request", "Malformed request.".to_owned()))?;
        if url.path() != CALLBACK_PATH {
            return Err(("404 Not Found", "Not found.".into()));
        }
        let param = |name: &str| {
            url.query_pairs()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.into_owned())
        };
        if param("state").as_deref() != Some(self.state.as_str()) {
            return Err(("400 Bad Request", "State mismatch; ignored.".into()));
        }
        if let Some(error) = param("error") {
            let detail = param("error_description").unwrap_or_default();
            return Err(("403 Forbidden", format!("login failed: {error} {detail}")));
        }
        param("code")
            .filter(|c| !c.is_empty())
            .ok_or(("400 Bad Request", "Missing authorization code.".into()))
    }

    /// Accepts what a user pastes when the callback cannot be reached: the
    /// full redirect URL, its query string, `code#state`, or the bare code.
    /// A state that is present must match.
    pub fn code_from_input(&self, input: &str) -> Result<String, LoginError> {
        let (code, state) = parse_authorization_input(input);
        if let Some(state) = state {
            if state != self.state {
                return Err(err("state mismatch: that URL belongs to a different login"));
            }
        }
        code.ok_or_else(|| err("no authorization code found in the input"))
    }

    /// Exchange the authorization code for tokens.
    pub async fn exchange(&self, code: &str) -> Result<CodexTokens, LoginError> {
        exchange_code(&self.endpoints, code, &self.verifier, &self.redirect_uri).await
    }
}

/// Reads the request line of an HTTP request and returns its target.
async fn read_request_target(socket: &mut tokio::net::TcpStream) -> Option<String> {
    let mut raw = Vec::new();
    let mut buf = [0u8; 2048];
    let read = async {
        while !raw.windows(2).any(|w| w == b"\r\n") && raw.len() < 16 * 1024 {
            let n = socket.read(&mut buf).await.ok()?;
            if n == 0 {
                return None;
            }
            raw.extend_from_slice(&buf[..n]);
        }
        Some(())
    };
    tokio::time::timeout(Duration::from_secs(10), read)
        .await
        .ok()??;
    let line = String::from_utf8_lossy(&raw);
    let mut parts = line.lines().next()?.split(' ');
    (parts.next()? == "GET").then_some(())?;
    parts.next().map(str::to_owned)
}

/// `(code, state)` from a pasted redirect URL, query string, `code#state`, or
/// bare code.
fn parse_authorization_input(input: &str) -> (Option<String>, Option<String>) {
    let input = input.trim();
    if input.is_empty() {
        return (None, None);
    }
    let from_query = |query: &str| {
        let pairs: Vec<(String, String)> = reqwest::Url::parse(&format!("http://x/?{query}"))
            .map(|u| u.query_pairs().into_owned().collect())
            .unwrap_or_default();
        let get = |name: &str| {
            pairs
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.clone())
        };
        (get("code"), get("state"))
    };
    if let Ok(url) = reqwest::Url::parse(input) {
        if url.has_host() {
            return from_query(url.query().unwrap_or(""));
        }
    }
    if input.contains("code=") {
        return from_query(input.trim_start_matches('?'));
    }
    if let Some((code, state)) = input.split_once('#') {
        return (Some(code.to_owned()), Some(state.to_owned()));
    }
    (Some(input.to_owned()), None)
}

async fn exchange_code(
    endpoints: &OAuthEndpoints,
    code: &str,
    verifier: &str,
    redirect_uri: &str,
) -> Result<CodexTokens, LoginError> {
    let body = token_request(
        endpoints,
        &[
            ("grant_type", "authorization_code"),
            ("client_id", &endpoints.client_id),
            ("code", code),
            ("code_verifier", verifier),
            ("redirect_uri", redirect_uri),
        ],
        "token exchange",
    )
    .await?;
    CodexTokens::from_token_response(&body, None)
}

/// Trade the refresh token for a new access token. The refresh token is
/// replaced if the server rotates it.
pub async fn refresh(
    endpoints: &OAuthEndpoints,
    tokens: &CodexTokens,
) -> Result<CodexTokens, LoginError> {
    let body = token_request(
        endpoints,
        &[
            ("grant_type", "refresh_token"),
            ("refresh_token", &tokens.refresh_token),
            ("client_id", &endpoints.client_id),
        ],
        "token refresh",
    )
    .await
    .map_err(|e| err(format!("{e}; run `worldfn login codex` again")))?;
    CodexTokens::from_token_response(&body, Some(&tokens.refresh_token))
}

async fn token_request(
    endpoints: &OAuthEndpoints,
    form: &[(&str, &str)],
    what: &str,
) -> Result<Value, LoginError> {
    let response = http_client()
        .post(endpoints.url("/oauth/token"))
        .form(form)
        .send()
        .await
        .map_err(|e| err(format!("{what} failed: {e}")))?;
    read_json(response, what).await
}

async fn read_json(response: reqwest::Response, what: &str) -> Result<Value, LoginError> {
    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    if !status.is_success() {
        let mut text = text;
        if text.len() > 300 {
            let cut = (0..=300)
                .rev()
                .find(|&i| text.is_char_boundary(i))
                .unwrap_or(0);
            text.truncate(cut);
        }
        return Err(err(format!("{what} failed: HTTP {status}: {text}")));
    }
    serde_json::from_str(&text).map_err(|e| err(format!("{what}: invalid JSON reply: {e}")))
}

/// Headless sign-in: show [`user_code`](Self::user_code), let the user enter
/// it at [`verification_url`](Self::verification_url), then
/// [`wait`](Self::wait).
pub struct DeviceLogin {
    endpoints: OAuthEndpoints,
    device_auth_id: String,
    user_code: String,
    interval: Duration,
}

/// One poll of the device token endpoint.
enum DevicePoll {
    Pending,
    SlowDown,
    Approved {
        authorization_code: String,
        code_verifier: String,
    },
}

impl DeviceLogin {
    pub async fn start(endpoints: OAuthEndpoints) -> Result<Self, LoginError> {
        let response = http_client()
            .post(endpoints.url("/api/accounts/deviceauth/usercode"))
            .header("content-type", "application/json")
            .body(json!({ "client_id": endpoints.client_id }).to_string())
            .send()
            .await
            .map_err(|e| err(format!("device login failed: {e}")))?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(err(
                "device code login is not enabled for this account or server; \
                 use the browser login",
            ));
        }
        let body = read_json(response, "device login").await?;
        let text = |key: &str| {
            body.get(key)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| err(format!("device login: reply has no {key}")))
        };
        // `interval` may be a number or a numeric string.
        let interval = match body.get("interval") {
            Some(Value::Number(n)) => n.as_f64(),
            Some(Value::String(s)) => s.trim().parse().ok(),
            _ => None,
        }
        .filter(|s| s.is_finite() && *s >= 0.0)
        .map_or(DEVICE_DEFAULT_INTERVAL, Duration::from_secs_f64);
        Ok(Self {
            device_auth_id: text("device_auth_id")?,
            user_code: text("user_code")?,
            interval: interval.max(DEVICE_MIN_INTERVAL),
            endpoints,
        })
    }

    /// The code the user types in.
    pub fn user_code(&self) -> &str {
        &self.user_code
    }

    /// Where the user types it.
    pub fn verification_url(&self) -> String {
        self.endpoints.url("/codex/device")
    }

    /// Poll until the user approves (up to 15 minutes), then exchange the
    /// resulting code for tokens.
    pub async fn wait(self) -> Result<CodexTokens, LoginError> {
        let deadline = Instant::now() + DEVICE_TIMEOUT;
        let mut interval = self.interval;
        let mut slowed_down = false;
        loop {
            match self.poll().await? {
                DevicePoll::Approved {
                    authorization_code,
                    code_verifier,
                } => {
                    let redirect = self.endpoints.url("/deviceauth/callback");
                    return exchange_code(
                        &self.endpoints,
                        &authorization_code,
                        &code_verifier,
                        &redirect,
                    )
                    .await;
                }
                DevicePoll::SlowDown => {
                    slowed_down = true;
                    interval += DEVICE_SLOW_DOWN_STEP;
                }
                DevicePoll::Pending => {}
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(err(if slowed_down {
                    "device login timed out after slow_down replies; check that the \
                     system clock is correct"
                } else {
                    "device login timed out"
                }));
            }
            tokio::time::sleep(interval.min(deadline - now)).await;
        }
    }

    async fn poll(&self) -> Result<DevicePoll, LoginError> {
        let response = http_client()
            .post(self.endpoints.url("/api/accounts/deviceauth/token"))
            .header("content-type", "application/json")
            .body(
                json!({
                    "device_auth_id": self.device_auth_id,
                    "user_code": self.user_code,
                })
                .to_string(),
            )
            .send()
            .await
            .map_err(|e| err(format!("device login poll failed: {e}")))?;
        let status = response.status();
        // The server answers 403/404 until the user approves.
        if status == reqwest::StatusCode::FORBIDDEN || status == reqwest::StatusCode::NOT_FOUND {
            return Ok(DevicePoll::Pending);
        }
        let text = response.text().await.unwrap_or_default();
        let body: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        if status.is_success() {
            let field = |key: &str| body.get(key).and_then(Value::as_str).map(str::to_owned);
            return match (field("authorization_code"), field("code_verifier")) {
                (Some(authorization_code), Some(code_verifier)) => Ok(DevicePoll::Approved {
                    authorization_code,
                    code_verifier,
                }),
                _ => Err(err("device login: approval reply is missing its code")),
            };
        }
        let code = match body.get("error") {
            Some(Value::Object(e)) => e.get("code").and_then(Value::as_str),
            Some(Value::String(e)) => Some(e.as_str()),
            _ => None,
        };
        match code {
            Some("deviceauth_authorization_pending") => Ok(DevicePoll::Pending),
            Some("slow_down") => Ok(DevicePoll::SlowDown),
            _ => Err(err(format!("device login failed: HTTP {status}: {text}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jwt(claims: Value) -> String {
        format!(
            "e30.{}.sig",
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims).unwrap())
        )
    }

    #[test]
    fn pkce_challenge_is_base64url_sha256() {
        // printf %s "$v" | openssl dgst -sha256 -binary | base64 | tr '+/' '-_' | tr -d =
        assert_eq!(
            pkce_challenge("dBjftJeZ4CVP-mJ0z0I5bIIl2S0t3kAC1VgNmBFaOuo"),
            "s8xk9odc11HhEejXqcOLllgNO1XlugDtZ8ZDi-wdWho"
        );
    }

    #[test]
    fn pasted_input_forms() {
        let parse = parse_authorization_input;
        assert_eq!(
            parse("http://localhost:1455/auth/callback?code=abc&state=s1"),
            (Some("abc".into()), Some("s1".into()))
        );
        assert_eq!(
            parse("code=abc&state=s1"),
            (Some("abc".into()), Some("s1".into()))
        );
        assert_eq!(parse("abc#s1"), (Some("abc".into()), Some("s1".into())));
        assert_eq!(parse("  abc \n"), (Some("abc".into()), None));
        assert_eq!(parse(""), (None, None));
    }

    #[test]
    fn token_responses() {
        let access = jwt(json!({ AUTH_CLAIM: { "chatgpt_account_id": "acct-1" } }));
        let tokens = CodexTokens::from_token_response(
            &json!({ "access_token": access, "refresh_token": "r1", "expires_in": 3600 }),
            None,
        )
        .unwrap();
        assert_eq!(tokens.account_id, "acct-1");
        assert!(!tokens.expires_within(60) && tokens.expires_within(7200));
        assert!(!format!("{tokens:?}").contains("r1"));

        // A refresh that does not rotate the refresh token keeps the old one;
        // the account id may come from the id token.
        let id_token = jwt(json!({ AUTH_CLAIM: { "chatgpt_account_id": "acct-2" } }));
        let refreshed = CodexTokens::from_token_response(
            &json!({ "access_token": "opaque", "id_token": id_token, "expires_in": 60 }),
            Some("r1"),
        )
        .unwrap();
        assert_eq!(
            (
                refreshed.refresh_token.as_str(),
                refreshed.account_id.as_str()
            ),
            ("r1", "acct-2")
        );

        let no_account = CodexTokens::from_token_response(
            &json!({ "access_token": "opaque", "refresh_token": "r", "expires_in": 60 }),
            None,
        );
        assert!(no_account.unwrap_err().0.contains("account id"));
    }
}
