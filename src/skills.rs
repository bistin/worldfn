//! [Agent Skills](https://agentskills.io/specification) for worldfn agents.
//!
//! A skill is a folder with a `SKILL.md`: YAML frontmatter (`name`,
//! `description`) followed by Markdown instructions, plus optional files.
//! Skills use *progressive disclosure*: models see a short catalog, and a
//! skill's full instructions enter the prompt only when it is relevant.
//!
//! In worldfn the agent's signature says which disclosure level it uses:
//!
//! | Level | What the model sees | Parameter |
//! |---|---|---|
//! | 1 | every skill's name + description | `Context<SkillCatalog>` |
//! | 2 | full instructions of the skills relevant to this invocation | `Context<RelevantSkills<N, Q>>` |
//! | 3 | bundled files | listed by path in level 2; never loaded or executed by worldfn |
//!
//! Selection today is done by the runtime, from the invocation's query `Q`
//! (by default the `Task`). Once models can call tools, level 2 can also be a
//! tool the model calls itself; the library and the `Skill` type stay the same.
//!
//! Skill text becomes model *instructions*, so load skills only from sources
//! you trust, like any other prompt you ship.

use std::any::type_name;
use std::fmt;
use std::future::{Ready, ready};
use std::marker::PhantomData;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::memory::words;
use crate::param::unmet;
use crate::{
    AgentParam, AgentWorld, ContextError, ContextSource, Input, ParamError, Requirement, Scope,
    Task,
};

/// A skill file could not be loaded or is invalid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillError(pub String);

impl fmt::Display for SkillError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "skill error: {}", self.0)
    }
}

impl std::error::Error for SkillError {}

/// One parsed skill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    /// What the skill does and when to use it; the text selection matches on.
    pub description: String,
    /// The Markdown body of `SKILL.md`.
    pub instructions: String,
    /// Other files in the skill folder, relative to it (level 3). Listed for
    /// the model; worldfn neither reads nor executes them.
    pub files: Vec<PathBuf>,
}

impl Skill {
    /// Parse the contents of a `SKILL.md`.
    pub fn parse(skill_md: &str) -> Result<Self, SkillError> {
        let (frontmatter, body) = split_frontmatter(skill_md)?;
        let fields = parse_frontmatter(frontmatter);
        let get = |key: &str| {
            fields
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.trim().to_owned())
        };
        let name = get("name").ok_or_else(|| SkillError("missing `name`".into()))?;
        let description =
            get("description").ok_or_else(|| SkillError("missing `description`".into()))?;
        validate_name(&name)?;
        if description.is_empty() || description.chars().count() > 1024 {
            return Err(SkillError(format!(
                "`{name}`: description must be 1-1024 characters"
            )));
        }
        if description.contains(['<', '>']) {
            return Err(SkillError(format!(
                "`{name}`: description must not contain angle brackets"
            )));
        }
        Ok(Skill {
            name,
            description,
            instructions: body.trim().to_owned(),
            files: Vec::new(),
        })
    }

    /// Load `dir/SKILL.md`. The frontmatter `name` must match the folder name.
    pub fn load(dir: &Path) -> Result<Self, SkillError> {
        let path = dir.join("SKILL.md");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| SkillError(format!("cannot read {}: {e}", path.display())))?;
        let mut skill =
            Skill::parse(&text).map_err(|e| SkillError(format!("{}: {}", path.display(), e.0)))?;
        let folder = dir.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        if skill.name != folder {
            return Err(SkillError(format!(
                "{}: name `{}` does not match folder `{folder}`",
                path.display(),
                skill.name
            )));
        }
        skill.files = list_files(dir)?;
        Ok(skill)
    }

    /// The skill as a prompt section (level 2).
    pub fn prompt(&self) -> String {
        let mut out = format!("## Skill: {}\n\n{}\n", self.name, self.instructions);
        if !self.files.is_empty() {
            out.push_str("\nFiles bundled with this skill (not loaded):\n");
            for file in &self.files {
                out.push_str(&format!("- {}\n", file.display()));
            }
        }
        out
    }
}

fn validate_name(name: &str) -> Result<(), SkillError> {
    let ok = !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        && !name.starts_with('-')
        && !name.ends_with('-')
        && !name.contains("--");
    if ok {
        Ok(())
    } else {
        Err(SkillError(format!(
            "invalid name `{name}`: use 1-64 lowercase letters, digits and single \
             hyphens, not at either end"
        )))
    }
}

fn split_frontmatter(text: &str) -> Result<(&str, &str), SkillError> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let missing = || SkillError("SKILL.md must start with `---` frontmatter".into());
    let rest = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))
        .ok_or_else(missing)?;
    let mut offset = 0;
    for line in rest.split_inclusive('\n') {
        if line.trim_end() == "---" {
            return Ok((&rest[..offset], &rest[offset + line.len()..]));
        }
        offset += line.len();
    }
    Err(SkillError("unterminated frontmatter".into()))
}

/// Top-level `key: value` pairs from YAML frontmatter.
///
/// Deliberately small, not a YAML parser: supports plain and quoted scalars
/// and `|` / `>` block scalars. Nested mappings (e.g. `metadata:`) and lists
/// are skipped, since the spec lets runtimes ignore fields they don't use.
fn parse_frontmatter(frontmatter: &str) -> Vec<(String, String)> {
    let lines: Vec<&str> = frontmatter.lines().collect();
    let mut fields = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        i += 1;
        if line.trim().is_empty() || line.starts_with([' ', '\t', '#']) {
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        let mut block = Vec::new();
        while i < lines.len() && (lines[i].starts_with([' ', '\t']) || lines[i].trim().is_empty()) {
            block.push(lines[i].trim());
            i += 1;
        }
        let value = match value.chars().next() {
            Some('|') => block.join("\n").trim().to_owned(),
            Some('>') => block
                .iter()
                .copied()
                .filter(|l| !l.is_empty())
                .collect::<Vec<_>>()
                .join(" "),
            Some('"') | Some('\'') if value.len() >= 2 && value.ends_with(&value[..1]) => {
                value[1..value.len() - 1].to_owned()
            }
            // An empty value followed by indented lines is a nested mapping.
            None => continue,
            _ => value.to_owned(),
        };
        fields.push((key.trim().to_owned(), value));
    }
    fields
}

fn list_files(dir: &Path) -> Result<Vec<PathBuf>, SkillError> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<PathBuf>, depth: usize) -> std::io::Result<()> {
        if depth > 4 {
            return Ok(());
        }
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() {
                walk(root, &path, out, depth + 1)?;
            } else if let Ok(relative) = path.strip_prefix(root) {
                if relative != Path::new("SKILL.md") {
                    out.push(relative.to_owned());
                }
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(dir, dir, &mut files, 0)
        .map_err(|e| SkillError(format!("cannot list {}: {e}", dir.display())))?;
    files.sort();
    Ok(files)
}

/// A set of skills with unique names, sorted by name.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SkillLibrary {
    skills: Vec<Skill>,
}

impl SkillLibrary {
    pub fn new(skills: impl IntoIterator<Item = Skill>) -> Result<Self, SkillError> {
        let mut skills: Vec<Skill> = skills.into_iter().collect();
        skills.sort_by(|a, b| a.name.cmp(&b.name));
        if let Some(pair) = skills.windows(2).find(|w| w[0].name == w[1].name) {
            return Err(SkillError(format!("duplicate skill `{}`", pair[0].name)));
        }
        Ok(Self { skills })
    }

    /// Load every `<dir>/<name>/SKILL.md`. Folders without a `SKILL.md` are
    /// ignored; an invalid skill is an error, not silently skipped.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, SkillError> {
        let dir = dir.as_ref();
        let entries = std::fs::read_dir(dir)
            .map_err(|e| SkillError(format!("cannot read {}: {e}", dir.display())))?;
        let mut skills = Vec::new();
        for entry in entries {
            let path = entry
                .map_err(|e| SkillError(format!("cannot read {}: {e}", dir.display())))?
                .path();
            if path.join("SKILL.md").is_file() {
                skills.push(Skill::load(&path)?);
            }
        }
        Self::new(skills)
    }

    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.skills.iter().find(|s| s.name == name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Skill> {
        self.skills.iter()
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// Level 1: one `- name: description` line per skill.
    pub fn catalog(&self) -> String {
        self.skills
            .iter()
            .map(|s| format!("- {}: {}", s.name, s.description))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Up to `limit` skills whose name or description shares words with
    /// `query`, best match first. Words that appear in more than half of the
    /// skills (like "customer" or "use" in a support library) do not count,
    /// since they cannot tell skills apart.
    ///
    /// A deterministic stand-in for semantic matching, like `FakeMemory`.
    pub fn select(&self, query: &str, limit: usize) -> Vec<&Skill> {
        let texts: Vec<_> = self
            .skills
            .iter()
            .map(|s| words(&format!("{} {}", s.name.replace('-', " "), s.description)))
            .collect();
        let common = |word: &String| {
            let df = texts.iter().filter(|t| t.contains(word)).count();
            self.skills.len() >= 2 && df * 2 > self.skills.len()
        };
        let query: std::collections::HashSet<String> =
            words(query).into_iter().filter(|w| !common(w)).collect();
        let mut scored: Vec<(usize, &Skill)> = self
            .skills
            .iter()
            .zip(&texts)
            .map(|(skill, text)| (text.intersection(&query).count(), skill))
            .filter(|(score, _)| *score > 0)
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        scored.into_iter().take(limit).map(|(_, s)| s).collect()
    }
}

/// Parameter / service: the world's skill library, bound with
/// [`AgentWorld::provide_skills`].
#[derive(Clone)]
pub struct Skills(Arc<SkillLibrary>);

impl Skills {
    pub fn new(library: SkillLibrary) -> Self {
        Skills(Arc::new(library))
    }
}

impl Deref for Skills {
    type Target = SkillLibrary;
    fn deref(&self) -> &SkillLibrary {
        &self.0
    }
}

impl AgentParam for Skills {
    type State = Skills;
    type Future = Ready<Result<Self, ParamError>>;

    fn describe(out: &mut Vec<Requirement>) {
        out.push(Requirement::Service {
            type_name: type_name::<Skills>(),
        });
    }

    fn init(world: &AgentWorld) -> Result<Skills, Vec<Requirement>> {
        world
            .resource::<Skills>()
            .cloned()
            .ok_or_else(unmet::<Self>)
    }

    fn resolve(state: &mut Skills, _world: &AgentWorld, _scope: &Scope) -> Self::Future {
        ready(Ok(state.clone()))
    }
}

/// An input that can be used as a selection query.
pub trait AsQuery: Send + Sync + 'static {
    fn query(&self) -> &str;
}

impl AsQuery for Task {
    fn query(&self) -> &str {
        &self.0
    }
}

impl AsQuery for String {
    fn query(&self) -> &str {
        self
    }
}

/// Context (level 1): the catalog of every available skill.
#[derive(Debug, Clone)]
pub struct SkillCatalog {
    /// `- name: description` lines.
    pub text: String,
    pub count: usize,
}

impl ContextSource for SkillCatalog {
    type Deps = Skills;

    async fn materialize(skills: Skills) -> Result<Self, ContextError> {
        Ok(SkillCatalog {
            text: skills.catalog(),
            count: skills.len(),
        })
    }
}

/// Context (level 2): up to `N` skills relevant to this invocation's input
/// `Q` (by default the [`Task`]), with their full instructions.
pub struct RelevantSkills<const N: usize = 3, Q = Task> {
    pub skills: Vec<Skill>,
    _query: PhantomData<fn() -> Q>,
}

impl<const N: usize, Q> RelevantSkills<N, Q> {
    pub fn names(&self) -> Vec<&str> {
        self.skills.iter().map(|s| s.name.as_str()).collect()
    }

    /// The selected skills as one prompt section; empty if none matched.
    pub fn prompt(&self) -> String {
        self.skills
            .iter()
            .map(Skill::prompt)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl<const N: usize, Q> fmt::Debug for RelevantSkills<N, Q> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RelevantSkills")
            .field("skills", &self.names())
            .finish()
    }
}

impl<const N: usize, Q: AsQuery> ContextSource for RelevantSkills<N, Q> {
    type Deps = (Skills, Input<Q>);

    async fn materialize((skills, input): Self::Deps) -> Result<Self, ContextError> {
        Ok(RelevantSkills {
            skills: skills
                .select(input.query(), N)
                .into_iter()
                .cloned()
                .collect(),
            _query: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REFUNDS: &str = "---
name: refund-policy
description: >
  How to answer refund and billing questions.
  Use when a customer asks about refunds or charges.
license: Apache-2.0
metadata:
  author: support-team
  version: \"1.0\"
---

# Refunds

Refunds take five business days.
";

    #[test]
    fn parses_frontmatter_and_body() {
        let skill = Skill::parse(REFUNDS).unwrap();
        assert_eq!(skill.name, "refund-policy");
        assert_eq!(
            skill.description,
            "How to answer refund and billing questions. Use when a customer asks about refunds or charges."
        );
        assert_eq!(
            skill.instructions,
            "# Refunds\n\nRefunds take five business days."
        );
    }

    #[test]
    fn quoted_and_literal_scalars() {
        let skill = Skill::parse(
            "---\nname: \"tone\"\ndescription: |\n  Line one.\n  Line two.\n---\nBe kind.",
        )
        .unwrap();
        assert_eq!(skill.name, "tone");
        assert_eq!(skill.description, "Line one.\nLine two.");
    }

    #[test]
    fn rejects_invalid_skills() {
        for (text, expected) in [
            ("no frontmatter", "must start with `---`"),
            ("---\nname: a\n", "unterminated"),
            ("---\ndescription: d\n---\n", "missing `name`"),
            ("---\nname: a\n---\n", "missing `description`"),
            ("---\nname: Bad_Name\ndescription: d\n---\n", "invalid name"),
            ("---\nname: a--b\ndescription: d\n---\n", "invalid name"),
            ("---\nname: -a\ndescription: d\n---\n", "invalid name"),
            (
                "---\nname: a\ndescription: use <b>\n---\n",
                "angle brackets",
            ),
        ] {
            let err = Skill::parse(text).unwrap_err();
            assert!(err.0.contains(expected), "{text:?}: {err}");
        }
    }
}
