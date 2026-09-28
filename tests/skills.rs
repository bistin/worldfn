//! Agent Skills: loading SKILL.md folders and disclosing them per invocation.

use std::path::{Path, PathBuf};

use worldfn::prelude::*;
use worldfn::{AsQuery, Skill};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// A fresh directory per call: tests run in parallel.
fn scratch(name: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir =
        std::env::temp_dir().join(format!("worldfn-skills-{name}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_skill(root: &Path, folder: &str, name: &str, description: &str, body: &str) {
    let dir = root.join(folder);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("SKILL.md"),
        format!("---\nname: {name}\ndescription: {description}\n---\n{body}\n"),
    )
    .unwrap();
}

fn library_dir() -> PathBuf {
    let root = scratch("lib");
    write_skill(
        &root,
        "refund-policy",
        "refund-policy",
        "Answer refund and billing questions. Use when a customer asks about refunds or charges.",
        "Refunds take five business days. Never promise faster.",
    );
    write_skill(
        &root,
        "incident-response",
        "incident-response",
        "Handle outage and error reports. Use when a customer reports the app is down or crashing.",
        "Apologize, link the status page, collect the error message.",
    );
    write_skill(
        &root,
        "tone",
        "tone",
        "House writing style for every customer reply.",
        "Be brief and warm. No exclamation marks.",
    );
    std::fs::create_dir_all(root.join("refund-policy/references")).unwrap();
    std::fs::write(root.join("refund-policy/references/fees.md"), "fee table").unwrap();
    std::fs::create_dir_all(root.join("not-a-skill")).unwrap();
    root
}

#[test]
fn loads_a_skill_directory() -> TestResult {
    let library = SkillLibrary::from_dir(library_dir())?;
    let names: Vec<_> = library.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["incident-response", "refund-policy", "tone"]);

    let refunds = library.get("refund-policy").unwrap();
    assert_eq!(refunds.files, [PathBuf::from("references/fees.md")]);
    assert!(refunds.prompt().contains("- references/fees.md"));

    assert_eq!(
        library.catalog(),
        "- incident-response: Handle outage and error reports. Use when a customer reports the app is down or crashing.\n\
         - refund-policy: Answer refund and billing questions. Use when a customer asks about refunds or charges.\n\
         - tone: House writing style for every customer reply."
    );
    Ok(())
}

#[test]
fn invalid_or_mismatched_skills_fail_loudly() {
    let root = scratch("bad");
    write_skill(&root, "billing", "refunds", "Refunds.", "x");
    let err = SkillLibrary::from_dir(&root).unwrap_err();
    assert!(err.0.contains("does not match folder `billing`"), "{err}");

    let root = scratch("dup");
    let a = Skill::parse("---\nname: a\ndescription: d\n---\n").unwrap();
    let err = SkillLibrary::new([a.clone(), a]).unwrap_err();
    assert!(err.0.contains("duplicate skill `a`"), "{err}");
    drop(root);
}

async fn support(task: Input<Task>, skills: Context<RelevantSkills<2>>) -> (String, Vec<String>) {
    (
        task.0.clone(),
        skills.names().into_iter().map(String::from).collect(),
    )
}

#[tokio::test]
async fn relevant_skills_are_chosen_per_invocation() -> TestResult {
    let mut world = AgentWorld::new();
    world.provide_skills(SkillLibrary::from_dir(library_dir())?)?;
    let mut agent = world.prepare(support)?;

    let (_, refunds) = world
        .run_prepared_with(
            &mut agent,
            Scope::of(Task::new("customer asks about a refund")),
        )
        .await?;
    assert_eq!(refunds, ["refund-policy"]);

    let (_, outage) = world
        .run_prepared_with(
            &mut agent,
            Scope::of(Task::new("app is crashing with an error")),
        )
        .await?;
    assert_eq!(outage, ["incident-response"]);

    let (_, none) = world
        .run_prepared_with(&mut agent, Scope::of(Task::new("hello")))
        .await?;
    assert!(none.is_empty());
    Ok(())
}

#[derive(Debug)]
struct Question(String);

impl AsQuery for Question {
    fn query(&self) -> &str {
        &self.0
    }
}

#[tokio::test]
async fn skills_select_from_any_query_type_and_catalog_lists_all() -> TestResult {
    async fn answer(
        catalog: Context<SkillCatalog>,
        chosen: Context<RelevantSkills<1, Question>>,
    ) -> (usize, String) {
        (catalog.count, chosen.prompt())
    }
    let mut world = AgentWorld::new();
    world.provide_skills(SkillLibrary::from_dir(library_dir())?)?;

    let (count, prompt) = world
        .run_with(
            answer,
            Scope::of(Question("was I charged twice? refund please".into())),
        )
        .await?;
    assert_eq!(count, 3);
    assert!(
        prompt.starts_with("## Skill: refund-policy\n\nRefunds take five business days."),
        "{prompt}"
    );

    assert_eq!(
        answer.into_agent().meta().to_string(),
        "answer\n\
         ├── requires Context<SkillCatalog>\n\
         │   └── requires Skills\n\
         └── requires Context<RelevantSkills<1, Question>>\n    \
             ├── requires Skills\n    \
             └── reads Input<Question>"
    );
    Ok(())
}

#[test]
fn missing_library_is_reported_at_prepare() {
    let err = AgentWorld::new().prepare(support).err().unwrap();
    assert_eq!(
        err.to_string(),
        "Cannot prepare support:\n  \
         · Input<Task>: checked when started\n  \
         ✗ Context<RelevantSkills<2>>: needs Skills"
    );
}
