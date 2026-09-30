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

// ---- Full chat requests: history, tools, tool results, JSON output --------

use worldfn::chat::{
    ChatRequest, FinishReason, Message, MessageRole, OutputFormat, Part, ToolCall, ToolDefinition,
    ToolResult,
};
use worldfn::providers::JsonMode;

fn tool_round_trip_request() -> ChatRequest {
    ChatRequest::new()
        .system("You are a mentor.")
        .user("Price of 2330?")
        .message(Message {
            role: MessageRole::Assistant,
            parts: vec![Part::ToolCall(ToolCall {
                id: "call_1".into(),
                name: "quote".into(),
                arguments: r#"{"instrument":"TWSE:2330"}"#.into(),
            })],
        })
        .message(Message::tool_result(ToolResult {
            call_id: "call_1".into(),
            content: "provider unavailable".into(),
            is_error: true,
        }))
        .user("Then just summarize.")
        .tool(ToolDefinition {
            name: "quote".into(),
            description: "Delayed quote with source and time.".into(),
            parameters: r#"{"type":"object","properties":{"instrument":{"type":"string"}}}"#.into(),
        })
        .output(OutputFormat::Json {
            name: "Summary".into(),
            schema: r#"{"type":"object","properties":{"text":{"type":"string"}}}"#.into(),
        })
}

#[tokio::test]
async fn openai_compatible_maps_the_full_request_and_reply() -> TestResult {
    let reply = json!({
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_2",
                    "type": "function",
                    "function": { "name": "quote", "arguments": "{\"instrument\":\"TWSE:2317\"}" },
                }],
            },
        }],
        "usage": { "prompt_tokens": 120, "completion_tokens": 18 },
    });
    let (url, server) = serve_once("200 OK", "application/json", vec![reply.to_string()]).await;
    let llm = Llm::new(OpenAiCompatLlm::new(&url, "sk", "m").json_mode(JsonMode::Schema));

    let response = llm.chat(tool_round_trip_request()).await?;
    assert_eq!(response.finish, FinishReason::ToolCalls);
    assert_eq!(
        response.message.tool_calls().collect::<Vec<_>>(),
        [&ToolCall {
            id: "call_2".into(),
            name: "quote".into(),
            arguments: r#"{"instrument":"TWSE:2317"}"#.into(),
        }]
    );
    let usage = response.usage.unwrap();
    assert_eq!((usage.input_tokens, usage.output_tokens), (120, 18));

    let body = server.await?.body;
    assert_eq!(
        body["messages"],
        json!([
            { "role": "system", "content": "You are a mentor." },
            { "role": "user", "content": "Price of 2330?" },
            { "role": "assistant", "content": null, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": { "name": "quote", "arguments": "{\"instrument\":\"TWSE:2330\"}" },
            }] },
            { "role": "tool", "tool_call_id": "call_1", "content": "Error: provider unavailable" },
            { "role": "user", "content": "Then just summarize." },
        ])
    );
    assert_eq!(body["tools"][0]["function"]["name"], "quote");
    assert_eq!(
        body["tools"][0]["function"]["parameters"]["properties"]["instrument"]["type"],
        "string"
    );
    assert_eq!(body["response_format"]["type"], "json_schema");
    assert_eq!(body["response_format"]["json_schema"]["name"], "Summary");
    Ok(())
}

#[tokio::test]
async fn deepseek_style_json_mode_uses_json_object_and_instructions() -> TestResult {
    let reply = json!({ "choices": [{ "finish_reason": "stop",
        "message": { "role": "assistant", "content": "{\"text\":\"ok\"}" } }] });
    let (url, server) = serve_once("200 OK", "application/json", vec![reply.to_string()]).await;
    let llm = Llm::new(OpenAiCompatLlm::new(&url, "sk", "m").json_mode(JsonMode::Object));

    let response = llm.chat(tool_round_trip_request()).await?;
    assert_eq!(response.message.text(), "{\"text\":\"ok\"}");
    assert_eq!(response.finish, FinishReason::Stop);
    assert!(response.usage.is_none());

    let body = server.await?.body;
    assert_eq!(body["response_format"], json!({ "type": "json_object" }));
    let system = body["messages"][0]["content"].as_str().unwrap();
    assert!(
        system.starts_with("You are a mentor.\n\nReply with only a JSON value"),
        "{system}"
    );
    assert!(system.contains("\"text\""), "{system}");
    Ok(())
}

#[tokio::test]
async fn codex_maps_the_full_request_and_streamed_tool_calls() -> TestResult {
    let events = [
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\"}}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\
         \"call_id\":\"call_9\",\"name\":\"quote\",\"arguments\":\"{\\\"instrument\\\":\\\"TWSE:2317\\\"}\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"output\":[],\
         \"usage\":{\"input_tokens\":300,\"output_tokens\":12}}}\n\n",
    ];
    let (url, server) = serve_once(
        "200 OK",
        "text/event-stream",
        events.map(String::from).to_vec(),
    )
    .await;
    let auth = write_auth_json(&scratch_dir("codex-chat"), 3600);
    let llm = Llm::new(
        CodexLlm::from_auth_file(&auth, "gpt-test")?
            .base_url(&url)
            .native_structured_output(true),
    );

    let response = llm.chat(tool_round_trip_request()).await?;
    assert_eq!(response.finish, FinishReason::ToolCalls);
    assert_eq!(response.message.tool_calls().next().unwrap().id, "call_9");
    assert_eq!(response.usage.unwrap().input_tokens, 300);

    let body = server.await?.body;
    assert_eq!(
        body["input"],
        json!([
            { "type": "message", "role": "user",
              "content": [{ "type": "input_text", "text": "Price of 2330?" }] },
            { "type": "function_call", "call_id": "call_1", "name": "quote",
              "arguments": "{\"instrument\":\"TWSE:2330\"}" },
            { "type": "function_call_output", "call_id": "call_1",
              "output": "Error: provider unavailable" },
            { "type": "message", "role": "user",
              "content": [{ "type": "input_text", "text": "Then just summarize." }] },
        ])
    );
    let instructions = body["instructions"].as_str().unwrap();
    assert!(instructions.starts_with("You are a mentor.\n\nReply with only a JSON value"));
    assert_eq!(body["tools"][0]["type"], "function");
    assert_eq!(body["tools"][0]["name"], "quote");
    assert_eq!(body["text"]["format"]["type"], "json_schema");
    Ok(())
}

/// Collects streamed text deltas.
fn collector() -> (
    std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    impl FnMut(worldfn::ChatDelta) + Send,
) {
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = seen.clone();
    (seen, move |delta| {
        if let worldfn::ChatDelta::Text(text) = delta {
            sink.lock().unwrap().push(text);
        }
    })
}

#[tokio::test]
async fn codex_streams_text_deltas_as_they_arrive() -> TestResult {
    let events = [
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"typed \"}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"del",
        "ta\":\"agents ✓\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{}}\n\n",
    ];
    let (url, _server) = serve_once(
        "200 OK",
        "text/event-stream",
        events.map(String::from).to_vec(),
    )
    .await;
    let auth = write_auth_json(&scratch_dir("codex-stream"), 3600);
    let llm = Llm::new(CodexLlm::from_auth_file(&auth, "gpt-test")?.base_url(&url));

    let (seen, on_delta) = collector();
    let response = llm
        .chat_streaming(ChatRequest::prompt("hi"), on_delta)
        .await?;
    assert_eq!(*seen.lock().unwrap(), ["typed ", "agents ✓"]);
    assert_eq!(response.message.text(), "typed agents ✓");
    Ok(())
}

#[tokio::test]
async fn codex_streams_final_text_when_the_backend_sends_no_deltas() -> TestResult {
    let events = [
        "data: {\"type\":\"response.completed\",\"response\":{\"output\":[\
                   {\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"whole\"}]}]}}\n\n",
    ];
    let (url, _server) = serve_once(
        "200 OK",
        "text/event-stream",
        events.map(String::from).to_vec(),
    )
    .await;
    let auth = write_auth_json(&scratch_dir("codex-stream-final"), 3600);
    let llm = Llm::new(CodexLlm::from_auth_file(&auth, "gpt-test")?.base_url(&url));
    let (seen, on_delta) = collector();
    llm.chat_streaming(ChatRequest::prompt("hi"), on_delta)
        .await?;
    assert_eq!(*seen.lock().unwrap(), ["whole"]);
    Ok(())
}

#[tokio::test]
async fn openai_compatible_streams_text_and_tool_call_fragments() -> TestResult {
    let chunk = |delta: Value, finish: Value| {
        format!(
            "data: {}\n\n",
            json!({ "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] })
        )
    };
    let chunks = vec![
        chunk(json!({ "role": "assistant", "content": "" }), Value::Null),
        chunk(json!({ "content": "Let me " }), Value::Null),
        // CRLF endings, split mid-frame, as some servers send.
        chunk(json!({ "content": "check." }), Value::Null).replace("\n\n", "\r\n\r\n"),
        chunk(
            json!({ "tool_calls": [{ "index": 0, "id": "call_1", "type": "function",
                    "function": { "name": "quote", "arguments": "{\"instr" } }] }),
            Value::Null,
        ),
        chunk(
            json!({ "tool_calls": [{ "index": 0, "function": { "arguments": "ument\":\"TWSE:2330\"}" } }] }),
            Value::Null,
        ),
        chunk(json!({}), json!("tool_calls")),
        format!(
            "data: {}\n\n",
            json!({ "choices": [], "usage": { "prompt_tokens": 40, "completion_tokens": 9 } })
        ),
        "data: [DONE]\n\n".to_owned(),
    ];
    let (url, server) = serve_once("200 OK", "text/event-stream", chunks).await;
    let llm = Llm::new(OpenAiCompatLlm::new(&url, "sk", "m"));

    let (seen, on_delta) = collector();
    let response = llm
        .chat_streaming(ChatRequest::prompt("price of 2330?"), on_delta)
        .await?;
    assert_eq!(*seen.lock().unwrap(), ["Let me ", "check."]);
    assert_eq!(response.finish, FinishReason::ToolCalls);
    assert_eq!(response.message.text(), "Let me check.");
    let call = response.message.tool_calls().next().unwrap();
    assert_eq!(
        (
            call.id.as_str(),
            call.name.as_str(),
            call.arguments.as_str()
        ),
        ("call_1", "quote", r#"{"instrument":"TWSE:2330"}"#)
    );
    assert_eq!(response.usage.unwrap().output_tokens, 9);

    let body = server.await?.body;
    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"]["include_usage"], true);
    Ok(())
}

#[tokio::test]
async fn openai_compatible_stream_errors_and_truncation() -> TestResult {
    let (url, _server) = serve_once(
        "200 OK",
        "text/event-stream",
        vec!["data: {\"error\":{\"message\":\"Insufficient Balance\"}}\n\n".into()],
    )
    .await;
    let llm = Llm::new(OpenAiCompatLlm::new(&url, "sk", "m"));
    let err = llm
        .chat_streaming(ChatRequest::prompt("x"), |_| {})
        .await
        .unwrap_err();
    assert!(err.0.contains("Insufficient Balance"), "{err}");

    let (url, _server) = serve_once(
        "200 OK",
        "text/event-stream",
        vec!["data: {\"choices\":[{\"delta\":{\"content\":\"half\"}}]}\n\n".into()],
    )
    .await;
    let llm = Llm::new(OpenAiCompatLlm::new(&url, "sk", "m"));
    let err = llm
        .chat_streaming(ChatRequest::prompt("x"), |_| {})
        .await
        .unwrap_err();
    assert!(err.0.contains("ended before"), "{err}");
    Ok(())
}
