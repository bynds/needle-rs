//! Turning a completion into calls the application may consider, or an explicit reason not to.
//!
//! Independent of the engine's extractor, which hands back an unterminated payload when the
//! budget runs out ("better than nothing" for a demo, wrong for a robot). Here a payload counts
//! only if both markers closed it, the turn ended on its own, it parses as a bounded JSON array of
//! `{"name", "arguments"}` objects, and every call satisfies the catalogue's schema.

use crate::catalog::{Catalogue, ParamType, Tool};
use needle_infer::prompt::{TOOL_CALL_END, TOOL_CALL_START};
use serde_json::{Map, Value};

pub const MAX_PAYLOAD_BYTES: usize = 4096;
pub const MAX_CALLS: usize = 4;

/// One validated call.
#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    pub name: String,
    pub arguments: Map<String, Value>,
}

/// What a completed (non-truncated, marker-closed) generation amounts to.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// One or more calls, each valid against the catalogue.
    Calls(Vec<Call>),
    /// `[]`: the model decided no tool applies. A considered answer, not a failure.
    NoCall,
    /// No `<tool_call>` marker at all.
    NoMarker,
    /// A `<tool_call>` that never closed.
    Unterminated,
    /// The payload is not a JSON array of call objects (or is oversized).
    Malformed(String),
    /// A call names a tool outside the request's catalogue subset.
    Unsupported(String),
    /// A call omits a required argument: ask, do not invent.
    NeedsClarification(String),
    /// A call has an argument of the wrong type, out of range, outside its enum, or undeclared.
    Invalid(String),
}

/// The text between the first `<tool_call>` and its `</tool_call>`, strictly.
pub fn strict_payload(text: &str) -> Result<&str, Outcome> {
    let Some(open) = text.find(TOOL_CALL_START) else {
        return Err(Outcome::NoMarker);
    };
    let rest = &text[open + TOOL_CALL_START.len()..];
    match rest.find(TOOL_CALL_END) {
        Some(end) => Ok(rest[..end].trim()),
        None => Err(Outcome::Unterminated),
    }
}

/// Validate a completion against the catalogue subset the request allowed.
pub fn validate(text: &str, catalogue: &Catalogue, allowed: Option<&[String]>) -> Outcome {
    let payload = match strict_payload(text) {
        Ok(p) => p,
        Err(o) => return o,
    };
    if payload.len() > MAX_PAYLOAD_BYTES {
        return Outcome::Malformed(format!("payload is {} bytes", payload.len()));
    }
    // serde_json bounds nesting depth (128) itself; the size bound above bounds the rest.
    let v: Value = match serde_json::from_str(payload) {
        Ok(v) => v,
        Err(e) => return Outcome::Malformed(format!("payload is not JSON: {e}")),
    };
    let Some(items) = v.as_array() else {
        return Outcome::Malformed("payload is not an array".into());
    };
    if items.is_empty() {
        return Outcome::NoCall;
    }
    if items.len() > MAX_CALLS {
        return Outcome::Malformed(format!("{} calls (limit {MAX_CALLS})", items.len()));
    }
    let mut calls = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let Some(obj) = item.as_object() else {
            return Outcome::Malformed(format!("call {i} is not an object"));
        };
        if obj.keys().any(|k| k != "name" && k != "arguments") {
            return Outcome::Malformed(format!("call {i} has keys other than name/arguments"));
        }
        let Some(name) = obj.get("name").and_then(Value::as_str) else {
            return Outcome::Malformed(format!("call {i} has no string name"));
        };
        let tool = match catalogue.resolve(name) {
            Some(t) if allowed.is_none_or(|a| a.iter().any(|n| n == &t.name)) => t,
            _ => return Outcome::Unsupported(format!("call {i}: tool {name:?} was not offered")),
        };
        let empty = Map::new();
        let args = match obj.get("arguments") {
            None => &empty,
            Some(Value::Object(m)) => m,
            Some(_) => return Outcome::Malformed(format!("call {i}: arguments is not an object")),
        };
        if let Err(o) = check_arguments(i, tool, args) {
            return o;
        }
        calls.push(Call {
            name: tool.name.clone(),
            arguments: args.clone(),
        });
    }
    Outcome::Calls(calls)
}

fn check_arguments(i: usize, tool: &Tool, args: &Map<String, Value>) -> Result<(), Outcome> {
    for (k, v) in args {
        let Some(p) = tool.params.iter().find(|p| &p.name == k) else {
            return Err(Outcome::Invalid(format!(
                "call {i}: {}.{k} is not a declared argument",
                tool.name
            )));
        };
        let bad = |why: &str| Outcome::Invalid(format!("call {i}: {}.{k} {why}", tool.name));
        match &p.ty {
            ParamType::String {
                enum_values,
                max_length,
            } => {
                let s = v.as_str().ok_or_else(|| bad("must be a string"))?;
                if let Some(e) = enum_values {
                    if !e.iter().any(|x| x == s) {
                        return Err(bad("is not one of the allowed values"));
                    }
                }
                if max_length.is_some_and(|m| s.chars().count() > m) {
                    return Err(bad("is too long"));
                }
            }
            ParamType::Integer { min, max } => {
                // 5.0 is not an integer argument; neither is 1e3. Only JSON integers pass.
                let n = v.as_i64().ok_or_else(|| bad("must be an integer"))?;
                if min.is_some_and(|m| n < m) || max.is_some_and(|m| n > m) {
                    return Err(bad("is out of range"));
                }
            }
            ParamType::Number { min, max } => {
                let n = v.as_f64().ok_or_else(|| bad("must be a number"))?;
                if !n.is_finite() || min.is_some_and(|m| n < m) || max.is_some_and(|m| n > m) {
                    return Err(bad("is out of range"));
                }
            }
            ParamType::Boolean => {
                v.as_bool().ok_or_else(|| bad("must be a boolean"))?;
            }
        }
    }
    for r in &tool.required {
        if !args.contains_key(r) {
            return Err(Outcome::NeedsClarification(format!(
                "call {i}: {}.{r} is required and was not given",
                tool.name
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cat() -> Catalogue {
        Catalogue::parse(
            r#"[
          {"name":"start_timer","parameters":{"type":"object",
            "properties":{"seconds":{"type":"integer","minimum":1,"maximum":86400}},"required":["seconds"]}},
          {"name":"play_animation","parameters":{"type":"object",
            "properties":{"name":{"type":"string","enum":["dance","nod"]}},"required":["name"]}}
        ]"#,
        )
        .unwrap()
    }

    fn tc(p: &str) -> String {
        format!("<think>\nx\n</think>\n<tool_call>{p}</tool_call>")
    }

    #[test]
    fn a_valid_call_passes() {
        let o = validate(
            &tc(r#"[{"name":"start_timer","arguments":{"seconds":450}}]"#),
            &cat(),
            None,
        );
        let Outcome::Calls(c) = o else {
            panic!("{o:?}")
        };
        assert_eq!(c[0].name, "start_timer");
        assert_eq!(c[0].arguments["seconds"], 450);
    }

    #[test]
    fn each_failure_has_its_own_outcome() {
        let c = cat();
        let cases: &[(&str, fn(&Outcome) -> bool)] = &[
            ("I can't help", |o| *o == Outcome::NoMarker),
            ("<tool_call>[{\"name\":\"start_t", |o| *o == Outcome::Unterminated),
            ("<tool_call>[]</tool_call>", |o| *o == Outcome::NoCall),
            ("<tool_call>{\"name\":1}</tool_call>", |o| matches!(o, Outcome::Malformed(_))),
            ("<tool_call>[{\"name\":\"rm\",\"arguments\":{}}]</tool_call>", |o| matches!(o, Outcome::Unsupported(_))),
            ("<tool_call>[{\"name\":\"start_timer\",\"arguments\":{}}]</tool_call>", |o| matches!(o, Outcome::NeedsClarification(_))),
            ("<tool_call>[{\"name\":\"start_timer\",\"arguments\":{\"seconds\":0}}]</tool_call>", |o| matches!(o, Outcome::Invalid(_))),
            ("<tool_call>[{\"name\":\"start_timer\",\"arguments\":{\"seconds\":4.5}}]</tool_call>", |o| matches!(o, Outcome::Invalid(_))),
            ("<tool_call>[{\"name\":\"start_timer\",\"arguments\":{\"seconds\":\"9\"}}]</tool_call>", |o| matches!(o, Outcome::Invalid(_))),
            ("<tool_call>[{\"name\":\"play_animation\",\"arguments\":{\"name\":\"backflip\"}}]</tool_call>", |o| matches!(o, Outcome::Invalid(_))),
            ("<tool_call>[{\"name\":\"start_timer\",\"arguments\":{\"seconds\":5,\"x\":1}}]</tool_call>", |o| matches!(o, Outcome::Invalid(_))),
        ];
        for (text, ok) in cases {
            let o = validate(text, &c, None);
            assert!(ok(&o), "{text}: {o:?}");
        }
    }

    #[test]
    fn a_tool_outside_the_requests_subset_is_unsupported() {
        let o = validate(
            &tc(r#"[{"name":"play_animation","arguments":{"name":"nod"}}]"#),
            &cat(),
            Some(&["start_timer".to_string()]),
        );
        assert!(matches!(o, Outcome::Unsupported(_)), "{o:?}");
    }

    #[test]
    fn too_many_calls_or_bytes_are_malformed() {
        let one = r#"{"name":"play_animation","arguments":{"name":"nod"}}"#;
        let five = format!("[{}]", vec![one; 5].join(","));
        assert!(matches!(
            validate(&tc(&five), &cat(), None),
            Outcome::Malformed(_)
        ));
        let big = format!("[\"{}\"]", "a".repeat(MAX_PAYLOAD_BYTES));
        assert!(matches!(
            validate(&tc(&big), &cat(), None),
            Outcome::Malformed(_)
        ));
    }
}
