//! One loaded model answering bounded requests, with everything a caller needs to tell a
//! completed request from a merely plausible-looking one.
//!
//! Admission happens before compute: the request names tools from the authorized catalogue, its
//! prompt is tokenized and measured, and it is refused (`truncated`) rather than cut if prompt and
//! budget do not fit. Every response carries the model hash, depth, cache precision, grammar
//! setting, stop reason, token counts and phase timings. A response is `candidate` only when the
//! turn ended on its own, the markers closed and every call is valid against the schema; it is
//! still only a candidate: grounding and application policy are the caller's.

use crate::catalog::Catalogue;
use crate::validate::{validate, Outcome};
use needle_infer::v3_engine::{KvPrecision, StopReason, V3Engine, V3Options};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

/// Engineering limits from the port's handoff, to be validated on the robot.
#[derive(Debug, Clone)]
pub struct Limits {
    /// Prompt plus generated tokens, per request.
    pub max_total_tokens: usize,
    /// Generated tokens, per request; a request may ask for fewer.
    pub max_new_tokens: usize,
    /// A request whose prompt leaves less than this for generation is refused, not starved.
    pub min_new_tokens: usize,
    pub max_query_bytes: usize,
    /// Wall-clock limit from receipt, queue wait included. Checked between tokens; a prefill in
    /// flight completes first.
    pub deadline: Option<Duration>,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_total_tokens: 512,
            max_new_tokens: 256,
            min_new_tokens: 64,
            max_query_bytes: 2048,
            deadline: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Options {
    pub constrain: bool,
    pub kv_int8: bool,
    pub system: Option<String>,
    /// Score candidates with the confidence head (an extra forward pass; timed separately).
    pub confidence: bool,
    /// Include the raw completion in responses. Off by default: no transcript retention.
    pub debug_text: bool,
}

pub struct Service {
    pub engine: V3Engine,
    pub catalogue: Catalogue,
    pub limits: Limits,
    pub opts: Options,
    pub model_sha256: String,
    pub model_bytes: usize,
    pub depth: usize,
    pub load_ms: f64,
}

#[derive(Debug, Clone)]
pub struct Request {
    pub id: String,
    pub query: String,
    pub tools: Option<Vec<String>>,
    pub max_new_tokens: Option<usize>,
}

fn ms(d: Duration) -> f64 {
    (d.as_secs_f64() * 1e6).round() / 1e3
}

fn stop_name(s: StopReason) -> &'static str {
    match s {
        StopReason::Eos => "eos",
        StopReason::ImEnd => "im_end",
        StopReason::MaxTokens => "max_tokens",
        StopReason::MaxSeqLen => "max_seq_len",
        StopReason::Cancelled => "cancelled",
    }
}

/// A response that never reached the model.
pub fn refusal(id: &str, status: &str, detail: &str) -> Value {
    json!({"request_id": id, "status": status, "detail": detail})
}

/// A parsed request line.
#[derive(Debug, Clone)]
pub enum Line {
    Health,
    Request(Request),
}

/// Parse one request line. Unknown keys are refused: a typo in `max_new_tokens` must not
/// silently run with the default.
pub fn parse_line(line: &str, max_line: usize) -> Result<Line, Value> {
    if line.len() > max_line {
        return Err(refusal("", "invalid_request", "request line too long"));
    }
    let v: Value = serde_json::from_str(line)
        .map_err(|e| refusal("", "invalid_request", &format!("not JSON: {e}")))?;
    let m = v
        .as_object()
        .ok_or_else(|| refusal("", "invalid_request", "request is not an object"))?;
    let id = match m.get("request_id") {
        Some(Value::String(s)) if s.len() <= 128 => s.clone(),
        Some(_) => {
            return Err(refusal(
                "",
                "invalid_request",
                "request_id must be a string of <= 128 bytes",
            ))
        }
        None => String::new(),
    };
    let bad = |d: &str| refusal(&id, "invalid_request", d);
    if m.get("health") == Some(&Value::Bool(true)) {
        return Ok(Line::Health);
    }
    for k in m.keys() {
        if !matches!(
            k.as_str(),
            "request_id" | "query" | "tools" | "max_new_tokens"
        ) {
            return Err(bad(&format!("unknown key {k:?}")));
        }
    }
    let query = m
        .get("query")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("query must be a string"))?
        .to_string();
    let tools = match m.get("tools") {
        None | Some(Value::Null) => None,
        Some(Value::Array(a)) => Some(
            a.iter()
                .map(|x| x.as_str().map(str::to_string))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| bad("tools must be an array of catalogue names"))?,
        ),
        Some(_) => return Err(bad("tools must be an array of catalogue names")),
    };
    let max_new_tokens = match m.get("max_new_tokens") {
        None => None,
        Some(x) => Some(
            x.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| bad("max_new_tokens must be a non-negative integer"))?,
        ),
    };
    Ok(Line::Request(Request {
        id,
        query,
        tools,
        max_new_tokens,
    }))
}

impl Service {
    pub fn load(
        path: &std::path::Path,
        catalogue: Catalogue,
        depth: Option<usize>,
        limits: Limits,
        opts: Options,
    ) -> Result<Self, String> {
        let t = Instant::now();
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let model_sha256 = crate::sha256::hex(&bytes);
        let model_bytes = bytes.len();
        let engine = match depth {
            None => V3Engine::from_bytes(bytes),
            Some(d) => V3Engine::from_bytes_with_depth(bytes, d),
        }
        .map_err(|e| format!("{}: {e}", path.display()))?;
        let depth = engine.model.cfg.num_layers;
        if limits.max_total_tokens > engine.model.cfg.max_seq_len {
            return Err(format!(
                "max_total_tokens {} exceeds the model's context {}",
                limits.max_total_tokens, engine.model.cfg.max_seq_len
            ));
        }
        if limits.min_new_tokens > limits.max_new_tokens {
            return Err("min_new_tokens exceeds max_new_tokens".into());
        }
        Ok(Self {
            engine,
            catalogue,
            limits,
            opts,
            model_sha256,
            model_bytes,
            depth,
            load_ms: ms(t.elapsed()),
        })
    }

    pub fn kv_name(&self) -> &'static str {
        if self.opts.kv_int8 {
            "int8"
        } else {
            "f32"
        }
    }

    pub fn health(&self) -> Value {
        let c = &self.engine.model.cfg;
        json!({
            "status": "ok",
            "model_sha256": self.model_sha256,
            "model_bytes": self.model_bytes,
            "load_ms": self.load_ms,
            "depth": self.depth,
            "d_model": c.d_model,
            "vocab": c.vocab_size,
            "max_seq_len": c.max_seq_len,
            "kv_precision": self.kv_name(),
            "constrained": self.opts.constrain,
            "confidence_head": self.engine.confidence.is_some(),
            "tools": self.catalogue.tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            "limits": {
                "max_total_tokens": self.limits.max_total_tokens,
                "max_new_tokens": self.limits.max_new_tokens,
                "min_new_tokens": self.limits.min_new_tokens,
                "max_query_bytes": self.limits.max_query_bytes,
                "deadline_ms": self.limits.deadline.map(ms),
            },
            "parallel": cfg!(feature = "parallel"),
        })
    }

    /// Answer one request. `received` is when it arrived, so queue time counts against the
    /// deadline and shows in the timing.
    pub fn handle(&self, req: &Request, received: Instant) -> Value {
        let started = Instant::now();
        let queue = started.duration_since(received);
        let id = req.id.as_str();
        let meta = |mut v: Value| -> Value {
            let o = v.as_object_mut().expect("object");
            o.insert("model_sha256".into(), json!(self.model_sha256));
            o.insert("depth".into(), json!(self.depth));
            o.insert("kv_precision".into(), json!(self.kv_name()));
            o.insert("constrained".into(), json!(self.opts.constrain));
            v
        };
        if self.limits.deadline.is_some_and(|d| queue >= d) {
            return meta(json!({"request_id": id, "status": "timeout",
                "detail": "deadline passed while queued", "timing": {"queue_ms": ms(queue)}}));
        }
        if req.query.is_empty() || req.query.len() > self.limits.max_query_bytes {
            return meta(refusal(
                id,
                "invalid_request",
                &format!("query must be 1..={} bytes", self.limits.max_query_bytes),
            ));
        }
        if req.query.contains('\0') {
            return meta(refusal(id, "invalid_request", "query contains NUL"));
        }
        let tools_json = match self.catalogue.tools_json(req.tools.as_deref()) {
            Ok(t) => t,
            Err(e) => return meta(refusal(id, "invalid_request", &e)),
        };
        let want_new = req.max_new_tokens.unwrap_or(self.limits.max_new_tokens);
        if want_new == 0 || want_new > self.limits.max_new_tokens {
            return meta(refusal(
                id,
                "invalid_request",
                &format!("max_new_tokens must be 1..={}", self.limits.max_new_tokens),
            ));
        }

        // Admission on the real prompt length, before any model compute.
        let prompt_tokens = self
            .engine
            .prompt_ids(&req.query, &tools_json, self.opts.system.as_deref())
            .len();
        let room = self.limits.max_total_tokens.saturating_sub(prompt_tokens);
        if room < self.limits.min_new_tokens.min(want_new) {
            return meta(json!({"request_id": id, "status": "truncated",
                "detail": format!("prompt is {prompt_tokens} tokens; {} allowed with at least {} to generate",
                    self.limits.max_total_tokens, self.limits.min_new_tokens.min(want_new)),
                "prompt_truncated": false,
                "tokens": {"prompt": prompt_tokens}}));
        }
        let budget = want_new.min(room);

        let opts = V3Options {
            max_new_tokens: budget,
            temperature: 0.0,
            seed: 0,
            system: self.opts.system.clone(),
            constrain: self.opts.constrain,
            kv_precision: if self.opts.kv_int8 {
                KvPrecision::Int8
            } else {
                KvPrecision::F32
            },
        };
        let deadline = self.limits.deadline.map(|d| received + d);
        let mut first_token: Option<Duration> = None;
        let res = self.engine.generate_controlled(
            &req.query,
            &tools_json,
            &opts,
            |_, _| {
                first_token.get_or_insert_with(|| started.elapsed());
            },
            || deadline.is_none_or(|d| Instant::now() < d),
        );

        let (status, detail, calls): (&str, String, Option<Value>) = if res.prompt_truncated {
            ("truncated", "the engine truncated the prompt".into(), None)
        } else {
            match res.stop {
                StopReason::Cancelled => {
                    ("timeout", "deadline reached during generation".into(), None)
                }
                StopReason::MaxTokens | StopReason::MaxSeqLen => (
                    "incomplete",
                    format!("generation budget of {budget} tokens ran out"),
                    None,
                ),
                StopReason::Eos | StopReason::ImEnd => {
                    match validate(&res.text, &self.catalogue, req.tools.as_deref()) {
                        Outcome::Calls(c) => (
                            "candidate",
                            String::new(),
                            Some(Value::Array(
                                c.into_iter()
                                    .map(|c| json!({"name": c.name, "arguments": c.arguments}))
                                    .collect(),
                            )),
                        ),
                        Outcome::NoCall => {
                            ("no_call", "the model chose no tool".into(), Some(json!([])))
                        }
                        Outcome::NoMarker => (
                            "invalid_output",
                            "no <tool_call> in the completion".into(),
                            None,
                        ),
                        Outcome::Unterminated => {
                            ("invalid_output", "unterminated <tool_call>".into(), None)
                        }
                        Outcome::Malformed(d) | Outcome::Invalid(d) => ("invalid_output", d, None),
                        Outcome::Unsupported(d) => ("unsupported", d, None),
                        Outcome::NeedsClarification(d) => ("needs_clarification", d, None),
                    }
                }
            }
        };

        let (confidence, confidence_ms) = if self.opts.confidence && status == "candidate" {
            let t = Instant::now();
            let p = self
                .engine
                .confidence_for(&req.query, &tools_json, &res.text);
            (p.map(|p| json!(p)), Some(ms(t.elapsed())))
        } else {
            (None, None)
        };

        let mut out = json!({
            "request_id": id,
            "status": status,
            "prompt_truncated": res.prompt_truncated,
            "stop_reason": stop_name(res.stop),
            "schema_valid": matches!(status, "candidate" | "no_call"),
            // Not checked here: whether the arguments are grounded in what was said.
            "grounded": null,
            "confidence_raw": confidence,
            "tokens": {
                "prompt": res.prompt_tokens,
                "generated": res.tokens.len(),
                "budget": budget,
                "positions": res.positions,
            },
            "timing": {
                "queue_ms": ms(queue),
                "wall_ms": ms(started.elapsed()),
                "tokenize_ms": ms(res.timing.tokenize),
                "prefill_ms": ms(res.timing.prefill),
                "decode_ms": ms(res.timing.decode),
                "first_token_ms": first_token.map(ms),
                "confidence_ms": confidence_ms,
            },
        });
        let o = out.as_object_mut().expect("object");
        if !detail.is_empty() {
            o.insert("detail".into(), json!(detail));
        }
        if let Some(c) = calls {
            o.insert("calls".into(), c);
        }
        if self.opts.debug_text {
            o.insert("text".into(), json!(res.text));
        }
        meta(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_are_parsed_strictly() {
        let Line::Request(r) = parse_line(
            r#"{"request_id":"u1","query":"hi","tools":["a"],"max_new_tokens":8}"#,
            4096,
        )
        .unwrap() else {
            panic!()
        };
        assert_eq!((r.id.as_str(), r.max_new_tokens), ("u1", Some(8)));
        for bad in [
            r#"{"query":1}"#,
            r#"{"query":"x","max_new_token":3}"#,
            r#"{"query":"x","tools":"a"}"#,
            r#"{"query":"x","max_new_tokens":-1}"#,
            r#"["x"]"#,
            "not json",
        ] {
            let e = parse_line(bad, 4096).unwrap_err();
            assert_eq!(e["status"], "invalid_request", "{bad}");
        }
        assert!(parse_line(&"x".repeat(5000), 4096).is_err());
        assert!(matches!(
            parse_line(r#"{"health":true}"#, 4096),
            Ok(Line::Health)
        ));
    }
}
