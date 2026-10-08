//! The authorized tool catalogue and the subset of JSON Schema it may use.
//!
//! The catalogue is loaded once, at startup, from a file the application owns. A request names
//! tools from it; it cannot bring its own. Every schema keyword is either enforced by
//! [`crate::validate`] or rejected here: a constraint the runner silently ignored would read as a
//! guarantee it does not give. The grammar behind `--constrain` restricts only tool names and
//! argument keys, so this check is what enforces types, enums, ranges and required arguments.

use serde_json::{Map, Value};

/// Largest catalogue accepted, in bytes. The catalogue is pasted into every prompt.
pub const MAX_CATALOGUE_BYTES: usize = 16 * 1024;
pub const MAX_TOOLS: usize = 32;
pub const MAX_PARAMS: usize = 16;

#[derive(Debug, Clone, PartialEq)]
pub enum ParamType {
    String {
        enum_values: Option<Vec<String>>,
        max_length: Option<usize>,
    },
    Integer {
        min: Option<i64>,
        max: Option<i64>,
    },
    Number {
        min: Option<f64>,
        max: Option<f64>,
    },
    Boolean,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: ParamType,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tool {
    pub name: String,
    /// The name as the constrained grammar spells it (`needle_infer::tokenizer::to_snake_case`).
    pub snake_name: String,
    pub params: Vec<Param>,
    pub required: Vec<String>,
    /// The tool's definition as the catalogue file spells it, key order included: the prompt
    /// carries these bytes (compacted by the engine), not a re-serialisation. serde_json without
    /// `preserve_order` would sort the keys and silently change the prompt.
    pub json: String,
}

#[derive(Debug, Clone)]
pub struct Catalogue {
    pub tools: Vec<Tool>,
}

impl Catalogue {
    pub fn parse(text: &str) -> Result<Self, String> {
        if text.len() > MAX_CATALOGUE_BYTES {
            return Err(format!(
                "catalogue is {} bytes (limit {MAX_CATALOGUE_BYTES})",
                text.len()
            ));
        }
        let v: Value = serde_json::from_str(text).map_err(|e| format!("catalogue: {e}"))?;
        let arr = v
            .as_array()
            .ok_or("catalogue must be a JSON array of tools")?;
        if arr.is_empty() || arr.len() > MAX_TOOLS {
            return Err(format!("catalogue must hold 1..={MAX_TOOLS} tools"));
        }
        let spans = top_level_elements(text).ok_or("catalogue array is malformed")?;
        if spans.len() != arr.len() {
            return Err("catalogue array is malformed".into());
        }
        let mut tools: Vec<Tool> = Vec::with_capacity(arr.len());
        for (i, (t, raw)) in arr.iter().zip(spans).enumerate() {
            let tool = parse_tool(t, raw).map_err(|e| format!("tool {i}: {e}"))?;
            if tools
                .iter()
                .any(|o| o.name == tool.name || o.snake_name == tool.snake_name)
            {
                return Err(format!("tool {i}: duplicate name {:?}", tool.name));
            }
            tools.push(tool);
        }
        Ok(Self { tools })
    }

    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// Resolve a name the model emitted: exact first, then the grammar's snake-case spelling.
    pub fn resolve(&self, emitted: &str) -> Option<&Tool> {
        self.get(emitted)
            .or_else(|| self.tools.iter().find(|t| t.snake_name == emitted))
    }

    /// The prompt's tool list for a subset, in catalogue order (order is part of the prompt).
    pub fn tools_json(&self, names: Option<&[String]>) -> Result<String, String> {
        let mut parts = Vec::new();
        match names {
            None => parts.extend(self.tools.iter().map(|t| t.json.as_str())),
            Some(names) => {
                for n in names {
                    if self.get(n).is_none() {
                        return Err(format!("tool {n:?} is not in the catalogue"));
                    }
                }
                for t in &self.tools {
                    if names.iter().any(|n| n == &t.name) {
                        parts.push(t.json.as_str());
                    }
                }
            }
        }
        Ok(format!("[{}]", parts.join(",")))
    }
}

fn only_keys(m: &Map<String, Value>, allowed: &[&str], what: &str) -> Result<(), String> {
    for k in m.keys() {
        if !allowed.contains(&k.as_str()) {
            return Err(format!("{what}: unsupported keyword {k:?}"));
        }
    }
    Ok(())
}

/// The source text of each element of a top-level JSON array, string-aware. Called only on
/// text serde_json has already accepted, so it needs no error reporting beyond `None`.
fn top_level_elements(text: &str) -> Option<Vec<&str>> {
    let b = text.as_bytes();
    let open = b.iter().position(|&c| !c.is_ascii_whitespace())?;
    if b[open] != b'[' {
        return None;
    }
    let (mut depth, mut in_str, mut esc) = (0usize, false, false);
    let mut start = None;
    let mut out = Vec::new();
    for (i, &c) in b.iter().enumerate().skip(open + 1) {
        if in_str {
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
            continue;
        }
        match c {
            b'"' => {
                in_str = true;
                start.get_or_insert(i);
            }
            b'{' | b'[' => {
                start.get_or_insert(i);
                depth += 1;
            }
            b'}' | b']' if depth > 0 => depth -= 1,
            b',' | b']' if depth == 0 => {
                if let Some(s) = start.take() {
                    out.push(text[s..i].trim_end());
                }
                if c == b']' {
                    return Some(out);
                }
            }
            c if !c.is_ascii_whitespace() => {
                start.get_or_insert(i);
            }
            _ => {}
        }
    }
    None
}

fn parse_tool(t: &Value, raw: &str) -> Result<Tool, String> {
    let m = t.as_object().ok_or("not an object")?;
    only_keys(m, &["name", "description", "parameters"], "tool")?;
    let name = m
        .get("name")
        .and_then(Value::as_str)
        .ok_or("missing string name")?;
    if name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return Err(format!("name {name:?} must be 1..=64 of [A-Za-z0-9_-]"));
    }
    if let Some(d) = m.get("description") {
        d.as_str().ok_or("description must be a string")?;
    }
    let mut params = Vec::new();
    let mut required = Vec::new();
    if let Some(p) = m.get("parameters") {
        let pm = p.as_object().ok_or("parameters must be an object")?;
        only_keys(pm, &["type", "properties", "required"], "parameters")?;
        if pm.get("type").and_then(Value::as_str) != Some("object") {
            return Err("parameters.type must be \"object\"".into());
        }
        if let Some(props) = pm.get("properties") {
            let props = props.as_object().ok_or("properties must be an object")?;
            if props.len() > MAX_PARAMS {
                return Err(format!("more than {MAX_PARAMS} parameters"));
            }
            for (k, v) in props {
                params.push(Param {
                    name: k.clone(),
                    ty: parse_param(v).map_err(|e| format!("parameter {k:?}: {e}"))?,
                });
            }
        }
        if let Some(r) = pm.get("required") {
            for x in r.as_array().ok_or("required must be an array")? {
                let s = x.as_str().ok_or("required entries must be strings")?;
                if !params.iter().any(|p| p.name == s) {
                    return Err(format!("required {s:?} is not a declared property"));
                }
                if !required.iter().any(|q: &String| q == s) {
                    required.push(s.to_string());
                }
            }
        }
    }
    let json = raw.to_string();
    Ok(Tool {
        name: name.to_string(),
        snake_name: needle_infer::tokenizer::to_snake_case(name),
        params,
        required,
        json,
    })
}

fn parse_param(v: &Value) -> Result<ParamType, String> {
    let m = v.as_object().ok_or("not an object")?;
    let ty = m
        .get("type")
        .and_then(Value::as_str)
        .ok_or("missing string type")?;
    let int = |k: &str| -> Result<Option<i64>, String> {
        m.get(k)
            .map(|x| x.as_i64().ok_or(format!("{k} must be an integer")))
            .transpose()
    };
    let num = |k: &str| -> Result<Option<f64>, String> {
        m.get(k)
            .map(|x| x.as_f64().ok_or(format!("{k} must be a number")))
            .transpose()
    };
    match ty {
        "string" => {
            only_keys(m, &["type", "description", "enum", "maxLength"], "string")?;
            let enum_values = match m.get("enum") {
                None => None,
                Some(e) => {
                    let a = e.as_array().ok_or("enum must be an array")?;
                    if a.is_empty() {
                        return Err("enum is empty".into());
                    }
                    Some(
                        a.iter()
                            .map(|x| {
                                x.as_str()
                                    .map(str::to_string)
                                    .ok_or("enum values must be strings")
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                }
            };
            let max_length = int("maxLength")?
                .map(|n| usize::try_from(n).map_err(|_| "maxLength must be >= 0".to_string()))
                .transpose()?;
            Ok(ParamType::String {
                enum_values,
                max_length,
            })
        }
        "integer" => {
            only_keys(m, &["type", "description", "minimum", "maximum"], "integer")?;
            Ok(ParamType::Integer {
                min: int("minimum")?,
                max: int("maximum")?,
            })
        }
        "number" => {
            only_keys(m, &["type", "description", "minimum", "maximum"], "number")?;
            Ok(ParamType::Number {
                min: num("minimum")?,
                max: num("maximum")?,
            })
        }
        "boolean" => {
            only_keys(m, &["type", "description"], "boolean")?;
            Ok(ParamType::Boolean)
        }
        other => Err(format!(
            "type {other:?} is not supported (string, integer, number, boolean)"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CAT: &str = r#"[
      {"name":"start_timer","description":"Propose a timer.","parameters":{"type":"object",
        "properties":{"seconds":{"type":"integer","minimum":1,"maximum":86400}},"required":["seconds"]}},
      {"name":"play_animation","description":"Propose an animation.","parameters":{"type":"object",
        "properties":{"name":{"type":"string","enum":["dance","nod"]}},"required":["name"]}}
    ]"#;

    #[test]
    fn parses_the_fixture_catalogue() {
        let c = Catalogue::parse(CAT).unwrap();
        assert_eq!(c.tools.len(), 2);
        assert_eq!(c.tools[0].required, vec!["seconds"]);
        assert!(c
            .tools_json(None)
            .unwrap()
            .starts_with("[{\"name\":\"start_timer\""));
    }

    #[test]
    fn the_prompt_keeps_the_files_key_order() {
        let c = Catalogue::parse(
            r#"[ {"name":"z","description":"d \"q\" ]","parameters":{"type":"object"}} ]"#,
        )
        .unwrap();
        assert_eq!(
            c.tools[0].json,
            r#"{"name":"z","description":"d \"q\" ]","parameters":{"type":"object"}}"#
        );
    }

    #[test]
    fn a_subset_keeps_catalogue_order() {
        let c = Catalogue::parse(CAT).unwrap();
        let s = c
            .tools_json(Some(&["play_animation".into(), "start_timer".into()]))
            .unwrap();
        assert!(s.find("start_timer").unwrap() < s.find("play_animation").unwrap());
        assert!(c.tools_json(Some(&["rm_rf".into()])).is_err());
    }

    #[test]
    fn unsupported_schema_is_rejected_not_ignored() {
        for bad in [
            r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"string","pattern":"a+"}}}}]"#,
            r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"array"}}}}]"#,
            r#"[{"name":"t","parameters":{"type":"object","properties":{},"additionalProperties":true}}]"#,
            r#"[{"name":"t","parameters":{"type":"object","properties":{},"required":["y"]}}]"#,
            r#"[{"name":"a b"}]"#,
            r#"[{"name":"t"},{"name":"t"}]"#,
            r#"[]"#,
        ] {
            assert!(Catalogue::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn resolves_the_grammars_snake_case_spelling() {
        let c = Catalogue::parse(r#"[{"name":"getWeather"}]"#).unwrap();
        assert_eq!(c.resolve("get_weather").unwrap().name, "getWeather");
        assert_eq!(c.resolve("getWeather").unwrap().name, "getWeather");
        assert!(c.resolve("weather").is_none());
    }
}
