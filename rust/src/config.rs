//! configobj-compatible INI parsing.
//!
//! The Python side uses `configobj`, whose file format is **not** standard INI:
//! sections nest by bracket count (`[sites]` / `[[Site Name]]`), section names
//! may be quoted so they can contain commas, and a comma-separated value is
//! parsed into a native list rather than left as a string.
//!
//! Two behaviours matter enough to state explicitly, because getting either
//! wrong silently breaks `[override]` column matching downstream:
//!
//! 1. **No type coercion.** Every scalar comes back a string. `EXPOSURE = 0`
//!    is `"0"`, not `0`; `USEOBSDATE = False` is `"False"`, not a bool. Any
//!    interpretation happens later, exactly as in Python.
//! 2. **Comma implies list.** `SQM = AOCSKYQ, AOCSKYQU` is a two-element list;
//!    `SITE = SITENAME` is a scalar. Verified against configobj 5.0.8 on
//!    `golden_tests/golden_config.ini`.

use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;
use std::path::Path;

/// A parsed value: configobj yields either a scalar string or a list of them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Str(String),
    List(Vec<String>),
}

impl Value {
    /// Every value as a list — a scalar becomes a one-element list.
    ///
    /// This is what `[override]` consumes: `_normalize_overrides` in
    /// `engine/loader.py` accepts both shapes and always produces a list.
    pub fn as_list(&self) -> Vec<String> {
        match self {
            Value::Str(s) => vec![s.clone()],
            Value::List(v) => v.clone(),
        }
    }

    /// The scalar form. A list renders the way Python's `str()` would, which
    /// is what the pre-A6 code accidentally relied on; kept so that any
    /// remaining scalar-context read matches Python rather than panicking.
    pub fn as_str(&self) -> String {
        match self {
            Value::Str(s) => s.clone(),
            Value::List(v) => {
                let inner: Vec<String> = v.iter().map(|s| format!("'{s}'")).collect();
                format!("[{}]", inner.join(", "))
            }
        }
    }
}

/// One INI section: scalar entries plus nested subsections.
#[derive(Debug, Default, Clone)]
pub struct Section {
    pub values: BTreeMap<String, Value>,
    pub sections: BTreeMap<String, Section>,
    /// Insertion order of `values`, preserved because configobj is
    /// order-preserving and `[override]` application order decides which
    /// hardware key wins when several are present.
    pub order: Vec<String>,
}

impl Section {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.values.get(key)
    }

    /// Entries in file order.
    pub fn iter_ordered(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.order
            .iter()
            .filter_map(move |k| self.values.get_key_value(k))
    }

    fn insert(&mut self, key: String, value: Value) {
        if !self.values.contains_key(&key) {
            self.order.push(key.clone());
        }
        self.values.insert(key, value);
    }
}

/// The whole file: top-level sections only (configobj allows root-level keys,
/// but this program's config never has any).
#[derive(Debug, Default, Clone)]
pub struct ConfigFile {
    pub sections: BTreeMap<String, Section>,
}

impl ConfigFile {
    /// Case-insensitive section lookup.
    ///
    /// `ConfigLoader.load` lowercases every top-level section key before
    /// dispatching on it, so `[Defaults]` and `[defaults]` are the same
    /// section to the Python side.
    pub fn section(&self, name: &str) -> Option<&Section> {
        let want = name.to_ascii_lowercase();
        self.sections
            .iter()
            .find(|(k, _)| k.to_ascii_lowercase() == want)
            .map(|(_, v)| v)
    }

    pub fn parse_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config file {}", path.display()))?;
        Self::parse_str(&text)
    }

    pub fn parse_str(text: &str) -> Result<Self> {
        let mut root = ConfigFile::default();
        // Path of section names from the top down to the section currently
        // being filled. Depth equals bracket count.
        let mut path: Vec<String> = Vec::new();

        for (lineno, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
                continue;
            }

            if line.starts_with('[') {
                let (name, depth) = parse_header(line)
                    .with_context(|| format!("line {}: bad section header {raw:?}", lineno + 1))?;
                if depth > path.len() + 1 {
                    bail!(
                        "line {}: section {name:?} is nested {depth} deep but its parent is missing",
                        lineno + 1
                    );
                }
                path.truncate(depth - 1);
                path.push(name);
                // Create it now so an empty section still exists, as in configobj.
                section_at_mut(&mut root, &path);
                continue;
            }

            let Some(eq) = line.find('=') else {
                bail!("line {}: expected 'key = value', got {raw:?}", lineno + 1);
            };
            let key = line[..eq].trim().to_string();
            let value = parse_value(&line[eq + 1..]);

            if path.is_empty() {
                bail!("line {}: key {key:?} outside any section", lineno + 1);
            }
            section_at_mut(&mut root, &path).insert(key, value);
        }

        Ok(root)
    }
}

/// Walks (creating as needed) to the section named by `path`.
fn section_at_mut<'a>(root: &'a mut ConfigFile, path: &[String]) -> &'a mut Section {
    let mut cur = root.sections.entry(path[0].clone()).or_default();
    for name in &path[1..] {
        cur = cur.sections.entry(name.clone()).or_default();
    }
    cur
}

/// `[name]` -> (name, 1); `[[name]]` -> (name, 2); and so on.
///
/// The name is unquoted if it is wrapped in matching quotes — that is how a
/// site name containing commas survives, e.g.
/// `[["Norton Close, Papworth Everard, ... CB23 3XT, United Kingdom"]]`.
fn parse_header(line: &str) -> Result<(String, usize)> {
    let open = line.len() - line.trim_start_matches('[').len();
    if open == 0 {
        bail!("no opening bracket");
    }
    // Trailing `#` comment after the closing brackets is not part of the name.
    let body = &line[open..];
    let close_at = body
        .rfind(&"]".repeat(open))
        .ok_or_else(|| anyhow::anyhow!("unbalanced brackets"))?;
    let name = body[..close_at].trim();
    Ok((unquote(name).to_string(), open))
}

/// Strips one layer of matching single or double quotes.
fn unquote(s: &str) -> &str {
    let b = s.as_bytes();
    if b.len() >= 2 && (b[0] == b'"' || b[0] == b'\'') && b[b.len() - 1] == b[0] {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

/// Parses the right-hand side of `key = value`.
///
/// A bare (unquoted) `#` starts a comment. A comma at top level makes the
/// value a list. Each element is trimmed and unquoted.
fn parse_value(rhs: &str) -> Value {
    let text = strip_inline_comment(rhs);
    let parts = split_top_level_commas(text);
    if parts.len() > 1 {
        Value::List(
            parts
                .iter()
                .map(|p| unquote(p.trim()).to_string())
                .collect(),
        )
    } else {
        Value::Str(unquote(text.trim()).to_string())
    }
}

/// Removes a trailing `# comment`, ignoring `#` inside quotes.
fn strip_inline_comment(s: &str) -> &str {
    let mut quote: Option<char> = None;
    for (i, c) in s.char_indices() {
        match (quote, c) {
            (None, '"') | (None, '\'') => quote = Some(c),
            (Some(q), c) if c == q => quote = None,
            (None, '#') => return &s[..i],
            _ => {}
        }
    }
    s
}

/// Splits on commas that are not inside quotes.
fn split_top_level_commas(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut quote: Option<char> = None;
    let mut start = 0usize;
    for (i, c) in s.char_indices() {
        match (quote, c) {
            (None, '"') | (None, '\'') => quote = Some(c),
            (Some(q), c) if c == q => quote = None,
            (None, ',') => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&s[start..]);
    // A trailing comma yields an empty final element in configobj too, but
    // this program's configs never have one; keep the simple behaviour.
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalars_are_never_coerced() {
        // configobj returns strings for everything: '0', '21', 'False'.
        let c = ConfigFile::parse_str("[defaults]\nEXPOSURE = 0\nUSEOBSDATE = False\n").unwrap();
        let d = c.section("defaults").unwrap();
        assert_eq!(d.get("EXPOSURE"), Some(&Value::Str("0".into())));
        assert_eq!(d.get("USEOBSDATE"), Some(&Value::Str("False".into())));
    }

    #[test]
    fn comma_makes_a_list_single_value_does_not() {
        let c = ConfigFile::parse_str(
            "[override]\nSQM = AOCSKYQ, AOCSKYQU\nSITE = SITENAME\n",
        )
        .unwrap();
        let o = c.section("override").unwrap();
        assert_eq!(
            o.get("SQM"),
            Some(&Value::List(vec!["AOCSKYQ".into(), "AOCSKYQU".into()]))
        );
        assert_eq!(o.get("SITE"), Some(&Value::Str("SITENAME".into())));
    }

    #[test]
    fn nested_sections_by_bracket_count() {
        let c = ConfigFile::parse_str(
            "[sites]\n        [[Papworth Everard]]\n                latitude = 52.2484\n",
        )
        .unwrap();
        let site = &c.section("sites").unwrap().sections["Papworth Everard"];
        assert_eq!(site.get("latitude"), Some(&Value::Str("52.2484".into())));
    }

    #[test]
    fn quoted_section_name_keeps_its_commas() {
        let c = ConfigFile::parse_str(
            "[sites]\n [[\"Norton Close, Papworth Everard, CB23 3XT, United Kingdom\"]]\n  bortle = 4\n",
        )
        .unwrap();
        let sites = c.section("sites").unwrap();
        assert!(sites
            .sections
            .contains_key("Norton Close, Papworth Everard, CB23 3XT, United Kingdom"));
    }

    #[test]
    fn comments_are_dropped() {
        let c =
            ConfigFile::parse_str("[filters]\n#Filter     code\nHa = 4663 # inline\n").unwrap();
        let f = c.section("filters").unwrap();
        assert_eq!(f.get("Ha"), Some(&Value::Str("4663".into())));
        assert_eq!(f.values.len(), 1);
    }

    #[test]
    fn section_lookup_is_case_insensitive() {
        let c = ConfigFile::parse_str("[Defaults]\nSITE = x\n").unwrap();
        assert!(c.section("defaults").is_some());
    }

    #[test]
    fn value_order_is_preserved() {
        let c = ConfigFile::parse_str("[override]\nZ = 1\nA = 2\nM = 3\n").unwrap();
        let keys: Vec<&str> = c
            .section("override")
            .unwrap()
            .iter_ordered()
            .map(|(k, _)| k.as_str())
            .collect();
        assert_eq!(keys, vec!["Z", "A", "M"]);
    }
}
