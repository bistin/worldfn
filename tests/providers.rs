//! Real providers against a local mock server: checks the exact request each
//! provider sends and how it reads the reply, without network or credentials.
#![cfg(all(feature = "codex", feature = "openai-compat"))]

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use worldfn::prelude::*;
use worldfn::providers::{CodexLlm, OpenAiCompatLlm};

type TestResult = Result<(), Box<dyn std::error::Error>>;

struct Captured {
    path: String,
    headers: HashMap<String, String>,
    body: Value,
}

/// Serve exactly one request: capture it, answer with `status`,
/// `content_type`, and `chunks` written one by one, then close.
async fn serve_once(
    status: &'static str,
    content_type: &'static str,
    chunks: Vec<String>,
) -> (String, tokio::task::JoinHandle<Captured>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let handle = tokio::spawn(async move {
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
        let len: usize = headers["content-length"].parse().unwrap();
        while raw.len() < header_end + len {
            let n = socket.read(&mut buf).await.unwrap();
            raw.extend_from_slice(&buf[..n]);
        }
        let body = serde_json::from_slice(&raw[header_end..header_end + len]).unwrap();

        let head = format!(
            "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\nconnection: close\r\n\r\n"
        );
        socket.write_all(head.as_bytes()).await.unwrap();
        for chunk in chunks {
            socket.write_all(chunk.as_bytes()).await.unwrap();
            socket.flush().await.unwrap();
        }
        socket.shutdown().await.unwrap();
        Captured {
            path,
            headers,
            body,
        }
    });
    (url, handle)
}

fn write_auth_json(dir: &std::path::Path, exp_offset: i64) -> std::path::PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    let claims = json!({
        "exp": now + exp_offset,
        "https://api.openai.com/auth": { "chatgpt_account_id": "acct-123" },
    });
    let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(claims.to_string());
    let auth = json!({
        "OPENAI_API_KEY": null,
        "tokens": {
            "id_token": "unused",
            "access_token": format!("e30.{payload}.sig"),
            "refresh_token": "unused",
        },
        "last_refresh": "2026-09-28T00:00:00Z",
    });
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join("auth.json");
    std::fs::write(&path, auth.to_string()).unwrap();
    path
}

fn scratch_dir(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("worldfn-test-{name}-{}", std::process::id()))
}

async fn summarize(task: Input<Task>, llm: Llm) -> Result<String, worldfn::LlmError> {
    llm.complete(format!("Summarize: {}", task.0)).await
}

#[tokio::test]
async fn codex_request_shape_and_streamed_reply() -> TestResult {
    let events = [
        "data: {\"type\":\"response.created\"}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"typed \"}\n\n",
        // A frame split across writes, with a multi-byte character.
        "data: {\"type\":\"response.output_text.delta\",\"del",
        "ta\":\"agents ✓\"}\n\ndata: {\"type\":\"response.completed\",\"response\":{}}\n\n",
    ];
    let (url, server) = serve_once(
        "200 OK",
        "text/event-stream",
        events.map(String::from).to_vec(),
    )
    .await;
    let auth = write_auth_json(&scratch_dir("codex-ok"), 3600);

    let mut world = AgentWorld::new();
    world.provide_llm(
        CodexLlm::from_auth_file(&auth, "gpt-test")?
            .instructions("Be brief.")
            .reasoning_effort("low")
            .base_url(&url),
    )?;
    let reply = world
        .run_with(summarize, Scope::of(Task::new("worldfn")))
        .await??;
    assert_eq!(reply, "typed agents ✓");

    let request = server.await?;
    assert_eq!(request.path, "/codex/responses");
    assert!(request.headers["authorization"].starts_with("Bearer e30."));
    assert_eq!(request.headers["chatgpt-account-id"], "acct-123");
    assert_eq!(request.headers["openai-beta"], "responses=experimental");
    assert_eq!(request.headers["originator"], "worldfn");
    assert_eq!(request.headers["accept"], "text/event-stream");
    assert_eq!(
        request.body,
        json!({
            "model": "gpt-test",
            "instructions": "Be brief.",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": "Summarize: worldfn" }],
            }],
            "store": false,
            "stream": true,
            "reasoning": { "effort": "low" },
        })
    );
    Ok(())
}

#[tokio::test]
async fn codex_http_errors_carry_status_and_hint() -> TestResult {
    let (url, server) = serve_once(
        "401 Unauthorized",
        "application/json",
        vec!["{\"detail\":\"token revoked\"}".into()],
    )
    .await;
    let auth = write_auth_json(&scratch_dir("codex-401"), 3600);
    let llm = Llm::new(CodexLlm::from_auth_file(&auth, "gpt-test")?.base_url(&url));

    let err = llm.complete("hi").await.unwrap_err();
    assert!(err.0.contains("HTTP 401"), "{err}");
    assert!(err.0.contains("token revoked"), "{err}");
    assert!(err.0.contains("run `codex login` again"), "{err}");
    server.await?;
    Ok(())
}

#[test]
fn codex_refuses_expired_login_up_front() {
    let auth = write_auth_json(&scratch_dir("codex-expired"), -60);
    let err = CodexLlm::from_auth_file(&auth, "gpt-test").err().unwrap();
    assert!(err.0.contains("expired"), "{err}");

    let err = CodexLlm::from_auth_file(scratch_dir("missing").join("auth.json"), "m")
        .err()
        .unwrap();
    assert!(err.0.contains("run `codex login`"), "{err}");
}

#[tokio::test]
async fn openai_compatible_request_shape_and_reply() -> TestResult {
    let reply = json!({
        "id": "x",
        "choices": [{ "index": 0, "message": { "role": "assistant", "content": "done" } }],
    });
    let (url, server) = serve_once("200 OK", "application/json", vec![reply.to_string()]).await;

    let mut world = AgentWorld::new();
    world
        .provide_llm(OpenAiCompatLlm::new(&url, "sk-test", "deepseek-test").system("Be brief."))?;
    let answer = world
        .run_with(summarize, Scope::of(Task::new("worldfn")))
        .await??;
    assert_eq!(answer, "done");

    let request = server.await?;
    assert_eq!(request.path, "/chat/completions");
    assert_eq!(request.headers["authorization"], "Bearer sk-test");
    assert_eq!(
        request.body,
        json!({
            "model": "deepseek-test",
            "messages": [
                { "role": "system", "content": "Be brief." },
                { "role": "user", "content": "Summarize: worldfn" },
            ],
            "stream": false,
        })
    );
    Ok(())
}

#[tokio::test]
async fn openai_compatible_errors_are_llm_errors_not_run_errors() -> TestResult {
    let (url, server) = serve_once(
        "402 Payment Required",
        "application/json",
        vec!["{\"error\":{\"message\":\"Insufficient Balance\"}}".into()],
    )
    .await;
    let mut world = AgentWorld::new();
    world.provide_llm(OpenAiCompatLlm::new(&url, "sk-test", "m"))?;

    // The run itself succeeds; the provider failure is the agent's own value.
    let result = world.run_with(summarize, Scope::of(Task::new("x"))).await?;
    let err = result.unwrap_err();
    assert!(
        err.0.contains("HTTP 402") && err.0.contains("Insufficient Balance"),
        "{err}"
    );
    server.await?;
    Ok(())
}
