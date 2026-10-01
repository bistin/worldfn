//! Images: validation, observers in the tool loop, and pruning.

use worldfn::chat::{ChatRequest, ChatResponse, Image, Message, MessageRole, Part};
use worldfn::prelude::*;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Just enough PNG for header parsing: signature + IHDR with a size.
fn png(width: u32, height: u32) -> Vec<u8> {
    let mut bytes = b"\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR".to_vec();
    bytes.extend(width.to_be_bytes());
    bytes.extend(height.to_be_bytes());
    bytes.extend([8, 6, 0, 0, 0, 0, 0, 0, 0]);
    bytes
}

#[test]
fn images_are_validated_and_never_printed() {
    let image = Image::from_bytes(png(1280, 800)).unwrap();
    assert_eq!(
        (image.media_type(), image.width(), image.height()),
        ("image/png", 1280, 800)
    );
    assert_eq!(format!("{image:?}"), "Image(image/png, 1280x800, 33 bytes)");

    // JPEG: SOI, an APP0 segment, then SOF0 with height 600, width 900.
    let jpeg = [
        &[0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x04, 0x00, 0x00][..],
        &[0xFF, 0xC0, 0x00, 0x11, 0x08, 0x02, 0x58, 0x03, 0x84, 0x03][..],
    ]
    .concat();
    let image = Image::from_bytes(jpeg).unwrap();
    assert_eq!(
        (image.media_type(), image.width(), image.height()),
        ("image/jpeg", 900, 600)
    );

    assert!(Image::from_bytes(b"GIF89a....".to_vec()).is_err());
    assert!(Image::from_bytes(png(0, 10)).is_err());
    assert!(Image::from_bytes(Vec::new()).is_err());
}

#[test]
fn messages_keep_text_and_images_in_order() {
    let image = Image::from_bytes(png(2, 2)).unwrap();
    let message = Message::user_parts([
        Part::Text("before ".into()),
        Part::Image(image.clone()),
        Part::Text("after".into()),
    ]);
    assert_eq!(message.text(), "before after");
    assert_eq!(message.images().collect::<Vec<_>>(), [&image]);
}

#[cfg(feature = "structured")]
mod tool_loop {
    use super::*;
    use serde::{Deserialize, Serialize};
    use worldfn::chat::{ToolCall, ToolResult};
    use worldfn::{LoopEvent, LoopOptions, Observation, ToolLoopError, ToolRun, Toolbox};

    struct Screenshot;

    #[derive(Deserialize, schemars::JsonSchema)]
    struct Shot {}

    #[derive(Serialize)]
    struct ShotInfo {
        artifact: String,
    }

    impl ToolSpec for Screenshot {
        const NAME: &'static str = "screenshot";
        type Request = Shot;
        type Response = ShotInfo;
    }

    struct Noop;
    impl ToolSpec for Noop {
        const NAME: &'static str = "noop";
        type Request = Shot;
        type Response = ShotInfo;
    }

    /// The host's own store: artifact id → pixels. The model only ever sees
    /// the id.
    fn observer(call: &ToolCall, result: &ToolResult) -> Vec<Observation> {
        if call.name != "screenshot" {
            return Vec::new();
        }
        let info: serde_json::Value = serde_json::from_str(&result.content).unwrap();
        let artifact = info["artifact"].as_str().unwrap();
        vec![Observation {
            label: format!("screenshot {artifact}"),
            image: Image::from_bytes(png(640, 400)).unwrap(),
        }]
    }

    fn world(llm: FakeLlm) -> AgentWorld {
        let shots = std::sync::atomic::AtomicUsize::new(0);
        let mut world = AgentWorld::new();
        world
            .provide_llm(llm)
            .unwrap()
            .provide_tool::<Screenshot>(FakeTool::new(move |_: &Shot| {
                let n = shots.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(ShotInfo {
                    artifact: format!("ss-{n}"),
                })
            }))
            .unwrap()
            .provide_tool::<Noop>(FakeTool::new(|_: &Shot| {
                Ok(ShotInfo {
                    artifact: "none".into(),
                })
            }))
            .unwrap();
        world
    }

    fn call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: "{}".into(),
        }
    }

    async fn agent(
        llm: Llm,
        tools: Toolbox<(Screenshot, Noop)>,
        options: Input<LoopOptions>,
    ) -> Result<(ToolRun, usize), ToolLoopError> {
        let mut seen = 0;
        let run = tools
            .run_with(&llm, ChatRequest::prompt("look"), (*options).clone(), |e| {
                if let LoopEvent::Observed(..) = e {
                    seen += 1;
                }
            })
            .await?;
        Ok((run, seen))
    }

    #[tokio::test]
    async fn images_follow_the_whole_batch_of_results() -> TestResult {
        let llm = FakeLlm::new()
            .then_response(ChatResponse::tool_calls([
                call("a", "screenshot"),
                call("b", "noop"),
            ]))
            .then_answer("I see a 640x400 screen");
        let world = world(llm.clone());
        let (run, seen) = world
            .run_with(agent, Scope::of(LoopOptions::new(3).observer(observer)))
            .await??;
        assert_eq!(seen, 1);

        let second = &llm.requests()[1];
        let roles: Vec<_> = second.messages.iter().map(|m| m.role).collect();
        use MessageRole::*;
        // Both results come before the image message.
        assert_eq!(roles, [User, Assistant, Tool, Tool, User]);
        let observed = second.messages.last().unwrap();
        assert_eq!(
            observed.text(),
            "screenshot ss-0 (from tool call a `screenshot`):"
        );
        assert_eq!(observed.images().next().unwrap().width(), 640);
        // The model saw only the id in the tool result.
        assert_eq!(run.calls[0].1.content, r#"{"artifact":"ss-0"}"#);
        Ok(())
    }

    #[tokio::test]
    async fn old_images_are_pruned_in_large_steps() -> TestResult {
        let llm = FakeLlm::responding_to(|request| {
            let n = request
                .messages
                .iter()
                .filter(|m| m.role == MessageRole::Tool)
                .count();
            if n == 6 {
                ChatResponse::text("done")
            } else {
                ChatResponse::tool_calls([call(&format!("c{n}"), "screenshot")])
            }
        });
        let world = world(llm.clone());
        world
            .run_with(
                agent,
                Scope::of(LoopOptions::new(10).observer(observer).max_images(4)),
            )
            .await??;

        let counts: Vec<usize> = llm
            .requests()
            .iter()
            .map(|r| r.messages.iter().map(|m| m.images().count()).sum())
            .collect();
        // Grows to 4, then drops to 2 in one step, then grows again.
        assert_eq!(counts, [0, 1, 2, 3, 4, 2, 3]);
        let notes = llm.requests()[5]
            .messages
            .iter()
            .filter(|m| m.text().contains("earlier image removed"))
            .count();
        assert_eq!(notes, 3);
        Ok(())
    }
}
