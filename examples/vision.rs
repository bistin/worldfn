//! A tool returns a screenshot id; an observer the host trusts attaches the
//! pixels; the model looks at them and answers. The shape of the VM demo,
//! without a VM.
//!
//! ```sh
//! cargo run --example vision                                  # scripted fake model
//! WORLDFN_PROVIDER=codex WORLDFN_MODEL=gpt-5.5 \
//!   cargo run --example vision --features codex-login         # a real model looks
//! WORLDFN_PROVIDER=codex WORLDFN_MODEL=gpt-5.5 \
//!   cargo run --example vision --features codex-login -- ~/Desktop/shot.png
//! ```
//!
//! Without a path, the "screenshot" is drawn here: a red square, a blue
//! circle and a green bar on white, so you can check the answer.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use worldfn::chat::{ChatRequest, ChatResponse, ToolCall, ToolResult};
use worldfn::prelude::*;
use worldfn::{Image, LoopEvent, LoopOptions, Observation, ToolLoopError, Toolbox};

mod common;

struct Screenshot;

#[derive(Deserialize, schemars::JsonSchema)]
struct ScreenshotRequest {}

/// What the model sees in the tool result: an id and a size, never pixels.
#[derive(Serialize)]
struct ScreenshotInfo {
    artifact: String,
    width: u32,
    height: u32,
}

impl ToolSpec for Screenshot {
    const NAME: &'static str = "screenshot";
    const DESCRIPTION: &'static str =
        "Capture the screen. The image is attached to the next message.";
    type Request = ScreenshotRequest;
    type Response = ScreenshotInfo;
}

/// Host-side artifact store: id → image. Only the host writes to it.
#[derive(Clone, Default)]
struct Artifacts(Arc<Mutex<HashMap<String, Image>>>);

const TASK: &str = "Take a screenshot, then describe exactly which shapes you see, \
their colors, and where they are. Be brief.";

async fn looker(
    llm: Llm,
    tools: Toolbox<Screenshot>,
    artifacts: Res<Artifacts>,
) -> Result<String, ToolLoopError> {
    let store = artifacts.clone();
    let options =
        LoopOptions::new(4)
            .max_images(4)
            .observer(move |call: &ToolCall, result: &ToolResult| {
                // Resolve the id from the tool's own result against the host's store.
                let info: serde_json::Value =
                    serde_json::from_str(&result.content).unwrap_or_default();
                let id = info["artifact"].as_str().unwrap_or_default();
                let images = store.0.lock().unwrap();
                images
                    .get(id)
                    .map(|image| Observation {
                        label: format!("screenshot {id} ({} `{}`)", call.id, call.name),
                        image: image.clone(),
                    })
                    .into_iter()
                    .collect()
            });
    let run = tools
        .run_with(
            &llm,
            ChatRequest::prompt(TASK),
            options,
            |event| match event {
                LoopEvent::ToolCall(call) => println!("  → {}({})", call.name, call.arguments),
                LoopEvent::ToolResult(_, result) => println!("  ← {}", result.content),
                LoopEvent::Observed(_, observation) => {
                    println!("  ◳ attached {:?}", observation.image)
                }
                LoopEvent::Text(text) => {
                    print!("{text}");
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                }
                _ => {}
            },
        )
        .await?;
    println!("\n\n  ({} model call(s))", run.steps);
    if run.usage != Default::default() {
        println!("  [tokens] {}", run.usage);
    }
    Ok(run.response.message.text())
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let image = match std::env::args().nth(1) {
        Some(path) => Image::from_bytes(std::fs::read(&path)?)?,
        None => Image::from_bytes(demo_png())?,
    };
    println!("screen: {image:?}\n");

    let artifacts = Artifacts::default();
    let shots = artifacts.clone();
    let llm =
        common::llm_from_env("You can see images attached to messages.")?.unwrap_or_else(|| {
            println!("(WORLDFN_PROVIDER unset: using a scripted fake model)\n");
            Llm::new(
                FakeLlm::new()
                    .then_response(ChatResponse::tool_calls([ToolCall {
                        id: "call_1".into(),
                        name: "screenshot".into(),
                        arguments: "{}".into(),
                    }]))
                    .then_answer("(fake) I was sent an image but cannot see it."),
            )
        });

    let mut world = AgentWorld::new();
    world
        .provide(llm)?
        .provide(artifacts)?
        .provide_tool::<Screenshot>(FakeTool::new(move |_: &ScreenshotRequest| {
            let mut store = shots.0.lock().unwrap();
            let id = format!("ss-{}", store.len() + 1);
            store.insert(id.clone(), image.clone());
            Ok(ScreenshotInfo {
                artifact: id,
                width: image.width(),
                height: image.height(),
            })
        }))?;

    println!("{}\n", looker.into_agent().meta());
    world.run(looker).await??;
    Ok(())
}

/// 240x150 RGB: a red square (left), a blue circle (middle), a green bar
/// (bottom right), on white.
fn demo_png() -> Vec<u8> {
    let (w, h) = (240u32, 150u32);
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let pixel = if (20..80).contains(&x) && (30..90).contains(&y) {
                [220, 30, 30]
            } else if (x - 130).pow(2) + (y - 60).pow(2) <= 30 * 30 {
                [30, 60, 220]
            } else if (170..230).contains(&x) && (115..135).contains(&y) {
                [30, 170, 60]
            } else {
                [255, 255, 255]
            };
            rgb.extend(pixel);
        }
    }
    encode_png(w, h, &rgb)
}

/// Minimal PNG encoder: 8-bit RGB, no filtering, stored (uncompressed)
/// deflate blocks. Large files, but valid and dependency-free.
fn encode_png(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
    let row = width as usize * 3;
    let mut raw = Vec::with_capacity((row + 1) * height as usize);
    for line in rgb.chunks(row) {
        raw.push(0); // filter: none
        raw.extend(line);
    }

    let mut zlib = vec![0x78, 0x01];
    let blocks: Vec<&[u8]> = raw.chunks(65_535).collect();
    for (i, block) in blocks.iter().enumerate() {
        zlib.push(u8::from(i + 1 == blocks.len()));
        let len = block.len() as u16;
        zlib.extend(len.to_le_bytes());
        zlib.extend((!len).to_le_bytes());
        zlib.extend(*block);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in &raw {
        a = (a + byte as u32) % 65_521;
        b = (b + a) % 65_521;
    }
    zlib.extend(((b << 16) | a).to_be_bytes());

    let mut ihdr = Vec::new();
    ihdr.extend(width.to_be_bytes());
    ihdr.extend(height.to_be_bytes());
    ihdr.extend([8, 2, 0, 0, 0]); // 8-bit, RGB, deflate, no filter, no interlace

    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    for (kind, data) in [(b"IHDR", ihdr), (b"IDAT", zlib), (b"IEND", Vec::new())] {
        png.extend((data.len() as u32).to_be_bytes());
        let start = png.len();
        png.extend(kind);
        png.extend(&data);
        let crc = crc32(&png[start..]);
        png.extend(crc.to_be_bytes());
    }
    png
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in bytes {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}
