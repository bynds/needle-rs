//! Is each argument of a call something the user actually said?
//!
//! Schema validation cannot catch a well-formed wrong answer: "how are you today?" proposing a
//! 30-minute timer, or "seven and a half minutes" becoming 950 seconds. This check reads the
//! query for the quantities it states and requires every numeric argument to be one of them, and
//! every free-text argument to appear in it. A call that fails is not executed: the runner
//! answers `needs_clarification` and names the argument.
//!
//! The rules are deliberately small and inspectable, not a second model:
//!
//! - A parameter whose name contains `second` takes a stated **duration**, converted to seconds:
//!   "90 seconds", "two minutes and 15 seconds", "seven and a half minutes", "an hour and a half",
//!   "half an hour", "quarter of an hour".
//! - In a tool with both, `hour` and `minute` take a stated **clock time**: "7:30 am",
//!   "4 pm", "seven thirty", "six o'clock", "half past seven", "quarter to eight", "noon",
//!   "midnight". A time with no am/pm grounds both readings.
//! - Any other integer or number parameter takes a **plain number** stated in the query (digits
//!   or English number words), or an alias: mute/silent/off/silence = 0, and
//!   max/maximum/loudest/highest/full = the schema's maximum, min/minimum/quietest/lowest = its
//!   minimum, when the schema declares them.
//! - A free-text string must occur in the query (case-insensitive). Enum strings are already
//!   restricted by the schema and are not checked, except in `strict` mode, where the value must
//!   start a word of the query ("wave" for "Wave hello!", but "dance" is refused for "Show me
//!   your moves"). Booleans are not checked.
//!
//! English only: a Spanish or French request with numbers is reported ungrounded, which costs a
//! clarification, not a wrong action. That trade-off and every rule above are measured on the
//! fixture suite (see the README), not assumed.

use crate::catalog::{ParamType, Tool};
use crate::validate::Call;
use serde_json::Value;

const ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [(&str, f64); 8] = [
    ("twenty", 20.0),
    ("thirty", 30.0),
    ("forty", 40.0),
    ("fifty", 50.0),
    ("sixty", 60.0),
    ("seventy", 70.0),
    ("eighty", 80.0),
    ("ninety", 90.0),
];

/// One token of the query: a number, or a lowercase word.
#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    /// "7:30" as written.
    Clock(f64, f64),
    Word(String),
}

fn word_value(w: &str) -> Option<f64> {
    if let Some(i) = ONES.iter().position(|o| *o == w) {
        return Some(i as f64);
    }
    TENS.iter().find(|(t, _)| *t == w).map(|(_, v)| *v)
}

fn tokenize(q: &str) -> Vec<Tok> {
    let lower = q.to_lowercase();
    let mut raw: Vec<String> = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = lower.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        let digit_join = (c == ':' || c == '.')
            && cur.chars().last().is_some_and(|p| p.is_ascii_digit())
            && chars.get(i + 1).is_some_and(|n| n.is_ascii_digit());
        if c.is_alphanumeric() || c == '\'' || digit_join {
            cur.push(c);
        } else {
            if !cur.is_empty() {
                raw.push(std::mem::take(&mut cur));
            }
            if c == '-' {
                raw.push("-".into());
            }
        }
    }
    if !cur.is_empty() {
        raw.push(cur);
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        let w = raw[i].as_str();
        if let Some((h, m)) = w.split_once(':') {
            if let (Ok(h), Ok(m)) = (h.parse::<f64>(), m.parse::<f64>()) {
                out.push(Tok::Clock(h, m));
                i += 1;
                continue;
            }
        }
        // "5pm", "7am"
        for suffix in ["am", "pm"] {
            if let Some(n) = w.strip_suffix(suffix).and_then(|n| n.parse::<f64>().ok()) {
                out.push(Tok::Num(n));
                out.push(Tok::Word(suffix.into()));
                i += 1;
                continue;
            }
        }
        if let Ok(n) = w.parse::<f64>() {
            out.push(Tok::Num(n));
            i += 1;
            continue;
        }
        if let Some(v) = word_value(w) {
            // twenty-five / twenty five
            let mut v = v;
            let mut j = i + 1;
            if v >= 20.0 {
                if raw.get(j).map(String::as_str) == Some("-") {
                    j += 1;
                }
                if let Some(o) = raw
                    .get(j)
                    .and_then(|x| word_value(x))
                    .filter(|o| *o < 10.0 && *o > 0.0)
                {
                    v += o;
                    j += 1;
                } else {
                    j = i + 1;
                }
            }
            out.push(Tok::Num(v));
            i = j;
            continue;
        }
        if w != "-" {
            out.push(Tok::Word(w.to_string()));
        }
        i += 1;
    }
    out
}

fn unit_seconds(w: &str) -> Option<f64> {
    match w {
        "second" | "seconds" | "sec" | "secs" => Some(1.0),
        "minute" | "minutes" | "min" | "mins" => Some(60.0),
        "hour" | "hours" | "hr" | "hrs" => Some(3600.0),
        _ => None,
    }
}

/// What a query states, read once.
#[derive(Debug, Default, Clone)]
pub struct Mentions {
    pub numbers: Vec<f64>,
    pub durations_s: Vec<f64>,
    /// (hour, minute) candidates in 24-hour time.
    pub clock: Vec<(f64, f64)>,
    pub zero_alias: bool,
    pub max_alias: bool,
    pub min_alias: bool,
    lower: String,
}

impl Mentions {
    pub fn read(q: &str) -> Self {
        let toks = tokenize(q);
        let mut m = Mentions {
            lower: q.to_lowercase(),
            ..Default::default()
        };
        let word = |i: usize| -> Option<&str> {
            match toks.get(i) {
                Some(Tok::Word(w)) => Some(w.as_str()),
                _ => None,
            }
        };
        for t in &toks {
            match t {
                Tok::Num(n) => m.numbers.push(*n),
                Tok::Clock(h, mi) => {
                    m.numbers.push(*h);
                    m.numbers.push(*mi);
                }
                Tok::Word(w) => match w.as_str() {
                    "mute" | "silent" | "silence" | "off" => m.zero_alias = true,
                    "max" | "maximum" | "loudest" | "highest" | "full" => m.max_alias = true,
                    "min" | "minimum" | "quietest" | "lowest" => m.min_alias = true,
                    _ => {}
                },
            }
        }

        // Durations: a quantity, optional "and a half", a unit; chained with "and".
        let mut i = 0;
        while i < toks.len() {
            let mut total = 0.0;
            let mut j = i;
            let mut found = false;
            loop {
                let (q, after) = match (&toks.get(j), word(j)) {
                    (Some(Tok::Num(n)), _) => (Some(*n), j + 1),
                    (_, Some("a" | "an")) => (Some(1.0), j + 1),
                    (_, Some("half")) if matches!(word(j + 1), Some("a" | "an")) => {
                        (Some(0.5), j + 2)
                    }
                    (_, Some("quarter")) => {
                        let k = if word(j + 1) == Some("of") {
                            j + 2
                        } else {
                            j + 1
                        };
                        if matches!(word(k), Some("a" | "an")) {
                            (Some(0.25), k + 1)
                        } else {
                            (None, j)
                        }
                    }
                    _ => (None, j),
                };
                let Some(mut q) = q else { break };
                let mut k = after;
                // "seven and a half minutes"
                if word(k) == Some("and") && word(k + 1) == Some("a") && word(k + 2) == Some("half")
                {
                    q += 0.5;
                    k += 3;
                }
                let Some(u) = word(k).and_then(unit_seconds) else {
                    break;
                };
                let mut v = q * u;
                k += 1;
                // "an hour and a half", "a minute and a half"
                if word(k) == Some("and") && word(k + 1) == Some("a") && word(k + 2) == Some("half")
                {
                    v += 0.5 * u;
                    k += 3;
                }
                total += v;
                found = true;
                j = k;
                if word(j) == Some("and") {
                    j += 1;
                } else {
                    break;
                }
            }
            if found {
                m.durations_s.push(total);
                i = j.max(i + 1);
            } else {
                i += 1;
            }
        }

        // Clock times.
        let meridiem = |k: usize| -> Option<bool> {
            match word(k) {
                Some("am" | "a.m") => Some(false),
                Some("pm" | "p.m") => Some(true),
                Some("in") if word(k + 1) == Some("the") => match word(k + 2) {
                    Some("morning") => Some(false),
                    Some("afternoon" | "evening") => Some(true),
                    _ => None,
                },
                Some("tonight") => Some(true),
                _ => None,
            }
        };
        let push = |m: &mut Mentions, h: f64, mi: f64, pm: Option<bool>| {
            if !(0.0..24.0).contains(&h) || !(0.0..60.0).contains(&mi) {
                return;
            }
            match pm {
                Some(true) if h < 12.0 => m.clock.push((h + 12.0, mi)),
                Some(false) if h == 12.0 => m.clock.push((0.0, mi)),
                Some(_) => m.clock.push((h, mi)),
                None => {
                    m.clock.push((h, mi));
                    if h < 12.0 {
                        m.clock.push((h + 12.0, mi));
                    }
                }
            }
        };
        for i in 0..toks.len() {
            match (&toks[i], word(i)) {
                (Tok::Clock(h, mi), _) => push(&mut m, *h, *mi, meridiem(i + 1)),
                (Tok::Num(h), _) if h.fract() == 0.0 && *h >= 1.0 && *h <= 12.0 => {
                    // "7 am", "six o'clock", "seven thirty"
                    if word(i + 1) == Some("o'clock") {
                        push(&mut m, *h, 0.0, meridiem(i + 2));
                    } else if let Some(Tok::Num(mi)) = toks.get(i + 1) {
                        if *mi >= 10.0 && *mi < 60.0 && mi.fract() == 0.0 {
                            push(&mut m, *h, *mi, meridiem(i + 2));
                        }
                    } else if meridiem(i + 1).is_some() {
                        push(&mut m, *h, 0.0, meridiem(i + 1));
                    }
                }
                (_, Some("half" | "quarter"))
                    if word(i + 1) == Some("past") || word(i + 1) == Some("to") =>
                {
                    if let Some(Tok::Num(h)) = toks.get(i + 2) {
                        let q = if word(i) == Some("half") { 30.0 } else { 15.0 };
                        let (hh, mi) = if word(i + 1) == Some("past") {
                            (*h, q)
                        } else {
                            (*h - 1.0, 60.0 - q)
                        };
                        push(&mut m, hh, mi, meridiem(i + 3));
                    }
                }
                (_, Some("noon" | "midday")) => m.clock.push((12.0, 0.0)),
                (_, Some("midnight")) => m.clock.push((0.0, 0.0)),
                _ => {}
            }
        }
        m
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

/// The arguments of `call` that the query does not state, as `tool.param=value` strings.
pub fn ungrounded(call: &Call, tool: &Tool, m: &Mentions, strict: bool) -> Vec<String> {
    let has = |p: &str| tool.params.iter().any(|x| x.name.contains(p));
    let clock_tool = has("hour") && has("minute");
    let mut out = Vec::new();
    for (k, v) in &call.arguments {
        let Some(p) = tool.params.iter().find(|p| &p.name == k) else {
            continue;
        };
        let name = p.name.to_lowercase();
        let ok = match (&p.ty, v) {
            (ParamType::Integer { .. } | ParamType::Number { .. }, _) if v.as_f64().is_some() => {
                let x = v.as_f64().unwrap_or(f64::NAN);
                let (min, max) = match &p.ty {
                    ParamType::Integer { min, max } => {
                        (min.map(|m| m as f64), max.map(|m| m as f64))
                    }
                    ParamType::Number { min, max } => (*min, *max),
                    _ => (None, None),
                };
                if name.contains("second") {
                    m.durations_s.iter().any(|d| close(*d, x))
                } else if clock_tool && name.contains("hour") {
                    m.clock.iter().any(|(h, _)| close(*h, x))
                } else if clock_tool && name.contains("minute") {
                    m.clock.iter().any(|(_, mi)| close(*mi, x))
                } else {
                    m.numbers.iter().any(|n| close(*n, x))
                        || (m.zero_alias && close(x, 0.0))
                        || (m.max_alias && max.is_some_and(|mx| close(mx, x)))
                        || (m.min_alias && min.is_some_and(|mn| close(mn, x)))
                }
            }
            (
                ParamType::String {
                    enum_values: None, ..
                },
                Value::String(s),
            ) => !s.trim().is_empty() && m.lower.contains(&s.to_lowercase()),
            (
                ParamType::String {
                    enum_values: Some(_),
                    ..
                },
                Value::String(s),
            ) if strict => {
                let lower = s.to_lowercase();
                // "dance" -> "danc" matches dancing; "wave" -> "wav" matches waving.
                let want = if lower.len() > 3 {
                    lower.trim_end_matches('e')
                } else {
                    &lower
                };
                !want.is_empty()
                    && m.lower
                        .split(|c: char| !c.is_alphanumeric())
                        .any(|w| w.starts_with(want))
            }
            _ => true,
        };
        if !ok {
            out.push(format!("{}.{}={}", tool.name, k, v));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Catalogue;

    fn cat() -> Catalogue {
        Catalogue::parse(include_str!("../../fixtures/suite-tools.json")).unwrap()
    }

    fn check(q: &str, name: &str, args: serde_json::Value) -> Vec<String> {
        let c = cat();
        let call = Call {
            name: name.into(),
            arguments: args.as_object().unwrap().clone(),
        };
        ungrounded(&call, c.get(name).unwrap(), &Mentions::read(q), false)
    }

    #[test]
    fn durations() {
        for (q, s) in [
            ("Set a timer for 5 minutes.", 300),
            ("Start a ninety second timer", 90),
            ("Time 2 minutes and 15 seconds.", 135),
            ("Set a timer for seven and a half minutes.", 450),
            ("Set a timer for a minute and a half.", 90),
            ("Set a timer for half an hour.", 1800),
            ("Timer for quarter of an hour.", 900),
            ("Set a timer for an hour and a half.", 5400),
            ("Set a timer for one hour.", 3600),
            ("Set a timer for twenty-five minutes", 1500),
            ("uh set a uh timer for ten minutes um", 600),
            ("Can you time 1.5 hours", 5400),
        ] {
            assert!(
                check(q, "start_timer", serde_json::json!({"seconds": s})).is_empty(),
                "{q} -> {s}"
            );
        }
    }

    #[test]
    fn the_models_known_mistakes_are_ungrounded() {
        for (q, s) in [
            ("Set a timer for seven and a half minutes.", 950),
            ("Set a timer for one hour.", 60),
            ("Can you time 2 minutes and 15 seconds?", 120),
            ("Hey Jibo, how are you today?", 1800),
            ("Do a backflip.", 1800),
            ("Set a timer.", 300),
            ("Set a timer for 5 minutes.", 5),
        ] {
            assert_eq!(
                check(q, "start_timer", serde_json::json!({"seconds": s})).len(),
                1,
                "{q} -> {s}"
            );
        }
    }

    #[test]
    fn clock_times() {
        for (q, h, mi) in [
            ("Wake me up at 7:30 am.", 7, 30),
            ("Set an alarm for 4:00 pm.", 16, 0),
            ("Set an alarm for six o'clock in the morning.", 6, 0),
            ("Set an alarm for half past seven in the morning.", 7, 30),
            ("wake me at seven thirty am", 7, 30),
            ("Set an alarm for quarter to eight", 7, 45),
            ("Set an alarm for noon.", 12, 0),
            ("Set an alarm for 5pm", 17, 0),
            ("Alarm at 7:15", 19, 15),
        ] {
            assert!(
                check(q, "set_alarm", serde_json::json!({"hour": h, "minute": mi})).is_empty(),
                "{q} -> {h}:{mi}"
            );
        }
        assert_eq!(
            check(
                "Wake me up at 7:30 am.",
                "set_alarm",
                serde_json::json!({"hour": 19, "minute": 30})
            )
            .len(),
            1
        );
        assert_eq!(
            check(
                "Set an alarm.",
                "set_alarm",
                serde_json::json!({"hour": 7, "minute": 0})
            )
            .len(),
            2
        );
    }

    #[test]
    fn plain_numbers_and_aliases() {
        assert!(check(
            "Set the volume to three.",
            "set_volume",
            serde_json::json!({"level": 3})
        )
        .is_empty());
        assert!(check(
            "volume to uh six",
            "set_volume",
            serde_json::json!({"level": 6})
        )
        .is_empty());
        assert!(check(
            "Mute yourself.",
            "set_volume",
            serde_json::json!({"level": 0})
        )
        .is_empty());
        assert!(check(
            "Turn it up to the max.",
            "set_volume",
            serde_json::json!({"level": 10})
        )
        .is_empty());
        assert_eq!(
            check(
                "Make it louder.",
                "set_volume",
                serde_json::json!({"level": 7})
            )
            .len(),
            1
        );
        assert_eq!(
            check(
                "Mute yourself.",
                "set_volume",
                serde_json::json!({"level": 10})
            )
            .len(),
            1
        );
    }

    #[test]
    fn free_text_must_be_said() {
        assert!(check(
            "whats the weather in boston",
            "get_weather",
            serde_json::json!({"city": "Boston"})
        )
        .is_empty());
        assert!(check(
            "Is it raining in São Paulo?",
            "get_weather",
            serde_json::json!({"city": "São Paulo"})
        )
        .is_empty());
        assert_eq!(
            check(
                "What's the weather like?",
                "get_weather",
                serde_json::json!({"city": "London"})
            )
            .len(),
            1
        );
        // Enum strings are the schema's business, unless strict.
        assert!(check(
            "Turn around in a circle.",
            "play_animation",
            serde_json::json!({"name": "spin"})
        )
        .is_empty());
        let c = cat();
        let call = |n: &str| Call {
            name: "play_animation".into(),
            arguments: serde_json::json!({ "name": n })
                .as_object()
                .unwrap()
                .clone(),
        };
        let t = c.get("play_animation").unwrap();
        assert!(ungrounded(&call("wave"), t, &Mentions::read("Wave hello!"), true).is_empty());
        assert!(ungrounded(&call("dance"), t, &Mentions::read("I love dancing"), true).is_empty());
        let m = Mentions::read("Will I need an umbrella?");
        assert_eq!(ungrounded(&call("wave"), t, &m, true).len(), 1);
    }
}
