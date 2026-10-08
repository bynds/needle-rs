//! Reference corpus for the C port of catalog.rs / validate.rs / grounding.rs.
//! Usage: runref OUT.jsonl
//! Records refer to catalogues by their index among the successfully parsed ones ("cat").
use needle_jibo::catalog::{Catalogue, ParamType};
use needle_jibo::grounding::{ungrounded, Mentions};
use needle_jibo::validate::{strict_payload, validate, Call, Outcome};
use serde_json::{json, Map, Value};
use std::io::Write;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
    fn chance(&mut self, p: u32) -> bool {
        self.below(100) < p as usize
    }
    fn pick<'a, T>(&mut self, v: &'a [T]) -> &'a T {
        &v[self.below(v.len())]
    }
}

fn bits(f: f64) -> String {
    format!("{:016x}", f.to_bits())
}

const POOL_SPECIAL: &[char] = &[
    'Σ',
    'σ',
    'ς',
    'Α',
    'İ',
    'ı',
    'ß',
    'ẞ',
    'é',
    'É',
    'ǅ',
    'ǈ',
    '\u{345}',
    '\u{300}',
    '\u{301}',
    '\u{200b}',
    '\u{200d}',
    '\u{feff}',
    '\u{3000}',
    '\u{a0}',
    '\u{85}',
    '\u{2028}',
    '\u{1680}',
    '\u{0}',
    '\u{1}',
    '\u{1f}',
    '\u{7f}',
    '\u{80}',
    '\u{9f}',
    '\u{ad}',
    '\u{e000}',
    '\u{fffd}',
    '\u{10ffff}',
    '😀',
    '👍',
    '\u{1f3fb}',
    '中',
    '文',
    '٣',
    '５',
    'Ⅻ',
    '½',
    '²',
    'Ω',
    'Ω',
    'K',
    'Å',
    'ﬁ',
    'ŉ',
    'ǰ',
    'ΐ',
    'Ⰰ',
    'Ꭰ',
    'ꭰ',
    '𐐀',
    '𞤀',
    '\'',
    '"',
    '\\',
    ':',
    '.',
    '-',
    ',',
    '\t',
    '\n',
    '\r',
    ' ',
    '\u{b}',
    '\u{c}',
    '\u{e0001}',
    '\u{e0100}',
    '\u{2066}',
    '\u{61c}',
    'ᾈ',
    'ᾼ',
    'ϴ',
];

fn rand_char(r: &mut Rng) -> char {
    match r.below(10) {
        0..=3 => (b' ' + r.below(95) as u8) as char,
        4 => (r.below(32) as u8) as char,
        5 | 6 => *r.pick(POOL_SPECIAL),
        7 => char::from_u32(0x80 + r.below(0x780) as u32).unwrap_or('x'),
        8 => char::from_u32(0x800 + r.below(0x3000) as u32).unwrap_or('y'),
        _ => loop {
            if let Some(c) = char::from_u32(r.below(0x110000) as u32) {
                break c;
            }
        },
    }
}

fn rand_string(r: &mut Rng, max: usize) -> String {
    let n = r.below(max + 1);
    (0..n).map(|_| rand_char(r)).collect()
}

/// A JSON string literal for an arbitrary Rust string, as serde_json writes it.
fn lit(s: &str) -> String {
    serde_json::to_string(s).unwrap()
}

fn mutate(r: &mut Rng, s: &str) -> String {
    let mut chars: Vec<char> = s.chars().collect();
    let n = 1 + r.below(3);
    const JS: &[char] = &[
        '{', '}', '[', ']', ',', ':', '"', '\\', '0', '1', '-', '+', '.', 'e', 'E', 'n', 't', 'f',
        ' ', '\n', 'u', 'x', '\u{1}', 'é', 'Σ',
    ];
    for _ in 0..n {
        if chars.is_empty() {
            chars.push(*r.pick(JS));
            continue;
        }
        let i = r.below(chars.len());
        match r.below(4) {
            0 => {
                chars.remove(i);
            }
            1 => chars.insert(i, *r.pick(JS)),
            2 => chars[i] = *r.pick(JS),
            _ => chars.truncate(i),
        }
    }
    chars.into_iter().collect()
}

fn param_dump(p: &needle_jibo::catalog::Param) -> Value {
    let mut o = json!({"name": p.name});
    let m = o.as_object_mut().unwrap();
    match &p.ty {
        ParamType::String {
            enum_values,
            max_length,
        } => {
            m.insert("kind".into(), json!("string"));
            m.insert("enum".into(), json!(enum_values));
            m.insert("max_length".into(), json!(max_length.map(|x| x as u64)));
        }
        ParamType::Integer { min, max } => {
            m.insert("kind".into(), json!("integer"));
            m.insert("imin".into(), json!(min));
            m.insert("imax".into(), json!(max));
        }
        ParamType::Number { min, max } => {
            m.insert("kind".into(), json!("number"));
            m.insert("fmin".into(), json!(min.map(bits)));
            m.insert("fmax".into(), json!(max.map(bits)));
        }
        ParamType::Boolean => {
            m.insert("kind".into(), json!("boolean"));
        }
    }
    o
}

fn cat_dump(c: &Catalogue) -> Value {
    Value::Array(
        c.tools
            .iter()
            .map(|t| {
                json!({"name": t.name, "snake": t.snake_name, "json": t.json,
                       "required": t.required,
                       "params": t.params.iter().map(param_dump).collect::<Vec<_>>()})
            })
            .collect(),
    )
}

struct Out {
    w: std::io::BufWriter<std::fs::File>,
    n: usize,
}
impl Out {
    fn rec(&mut self, v: Value) {
        writeln!(self.w, "{}", v).unwrap();
        self.n += 1;
    }
}

fn outcome_rec(o: &Outcome) -> (String, Option<String>, Option<String>) {
    match o {
        Outcome::Calls(c) => {
            let calls = Value::Array(
                c.iter()
                    .map(|c| json!({"name": c.name, "arguments": c.arguments}))
                    .collect(),
            );
            ("calls".into(), None, Some(calls.to_string()))
        }
        Outcome::NoCall => ("no_call".into(), None, None),
        Outcome::NoMarker => ("no_marker".into(), None, None),
        Outcome::Unterminated => ("unterminated".into(), None, None),
        Outcome::Malformed(d) => ("malformed".into(), Some(d.clone()), None),
        Outcome::Unsupported(d) => ("unsupported".into(), Some(d.clone()), None),
        Outcome::NeedsClarification(d) => ("needs_clarification".into(), Some(d.clone()), None),
        Outcome::Invalid(d) => ("invalid".into(), Some(d.clone()), None),
    }
}

fn val_rec(out: &mut Out, cats: &[Catalogue], ci: usize, allowed: Option<&[String]>, text: &str) {
    let o = validate(text, &cats[ci], allowed);
    let sp = match strict_payload(text) {
        Ok(p) => {
            let off = p.as_ptr() as usize - text.as_ptr() as usize;
            json!([off, p.len()])
        }
        Err(Outcome::NoMarker) => json!("no_marker"),
        Err(_) => json!("unterminated"),
    };
    let (kind, detail, calls) = outcome_rec(&o);
    out.rec(
        json!({"t": "val", "cat": ci, "allowed": allowed, "text": text, "sp": sp,
                   "kind": kind, "detail": detail, "calls": calls}),
    );
}

fn mentions_rec(out: &mut Out, q: &str) -> Mentions {
    let m = Mentions::read(q);
    out.rec(json!({"t": "men", "q": q,
        "numbers": m.numbers.iter().map(|x| bits(*x)).collect::<Vec<_>>(),
        "durations": m.durations_s.iter().map(|x| bits(*x)).collect::<Vec<_>>(),
        "clock": m.clock.iter().map(|(h, mi)| json!([bits(*h), bits(*mi)])).collect::<Vec<_>>(),
        "zero": m.zero_alias, "max": m.max_alias, "min": m.min_alias,
        "lower": q.to_lowercase()}));
    m
}

fn ug_rec(
    out: &mut Out,
    cats: &[Catalogue],
    ci: usize,
    tool: &str,
    args: &str,
    q: &str,
    m: &Mentions,
) {
    let a: Value = serde_json::from_str(args).unwrap();
    let call = Call {
        name: tool.into(),
        arguments: a.as_object().unwrap().clone(),
    };
    let t = cats[ci].get(tool).unwrap();
    for strict in [false, true] {
        let u = ungrounded(&call, t, m, strict);
        out.rec(
            json!({"t": "ug", "cat": ci, "tool": tool, "args": args, "q": q,
                       "strict": strict, "out": u}),
        );
    }
}

fn num_text(r: &mut Rng, x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 && !r.chance(15) {
        format!("{}", x as i64)
    } else {
        serde_json::to_string(&x).unwrap()
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("OUT");
    let mut out = Out {
        w: std::io::BufWriter::new(std::fs::File::create(&path).unwrap()),
        n: 0,
    };
    let mut r = Rng(0x9E3779B97F4A7C15);
    let fx = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");

    // ── Low-level: Debug, trim, to_lowercase, f64 parse, serde ──────────────────────────────
    let mut strs: Vec<String> = Vec::new();
    for c in POOL_SPECIAL {
        strs.push(c.to_string());
    }
    for _ in 0..4000 {
        strs.push(rand_string(&mut r, 12));
    }
    for _ in 0..1500 {
        let mut s = String::new();
        for _ in 0..r.below(8) {
            let pool = [
                'Σ', 'a', ' ', '\'', '.', '\u{345}', '\u{300}', 'Α', '1', 'ǅ', '\u{ad}', ':', 'é',
                '\u{200d}', '-', '·', 'Σ', 'ς', '\u{3000}', 'A', 'z',
            ];
            s.push(*r.pick(&pool));
        }
        strs.push(s);
    }
    for s in &strs {
        let t = s.trim();
        let a = t.as_ptr() as usize - s.as_ptr() as usize;
        let e = s.trim_end().len();
        out.rec(
            json!({"t": "str", "s": s, "dbg": format!("{:?}", s), "lc": s.to_lowercase(),
                       "trim": [a, a + t.len()], "trim_end": e}),
        );
    }
    let mut ftoks: Vec<String> = [
        "", "1", "+1", "-1", "1.", ".5", ".", "1e5", "1E5", "1e", "1e+", "1e-3", "e5", "inf",
        "INF", "Infinity", "-inf", "nan", "NaN", "-nan", "+nan", "infinit", "infinityy", "0x10",
        "1_000", "١", "1.5.3", "7:30", "00012", "1e309", "1e-400", "2.5e-324",
        "179769313486231580793728971405303415079934132710037826936173778980444968292764750946649017977587207096330286416692887910946555547851940402630657488671505820681908902000708383676273854845817711531623",
        "0.1", "0.30000000000000004", "123456789012345678901234567890", "4.9406564584124654e-324",
        "2.2250738585072011e-308", "  1", "1 ", "+", "-", "++1", "1e+-1", "9007199254740993",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for _ in 0..3000 {
        let mut s = String::new();
        for _ in 0..1 + r.below(12) {
            s.push(*r.pick(&[
                '0', '1', '5', '9', '.', 'e', 'E', '+', '-', 'i', 'n', 'f', 'a', 'x',
            ]));
        }
        ftoks.push(s);
    }
    for s in &ftoks {
        let v = s.parse::<f64>().ok().map(bits);
        out.rec(json!({"t": "f64", "s": s, "v": v}));
    }

    let mut sj: Vec<String> = vec![
        "".into(),
        " ".into(),
        "[".into(),
        "]".into(),
        "{".into(),
        "{\"a\"".into(),
        "{\"a\":".into(),
        "{\"a\":1,}".into(),
        "[1,]".into(),
        "[1 2]".into(),
        "{\"a\" 1}".into(),
        "{1:2}".into(),
        "{,}".into(),
        "[,]".into(),
        "nul".into(),
        "nulx".into(),
        "tru".into(),
        "falsey".into(),
        "-".into(),
        "-a".into(),
        "01".into(),
        "1.".into(),
        "1.e5".into(),
        "1e".into(),
        "1e+".into(),
        "1ex".into(),
        "\"\\u12\"".into(),
        "\"\\uzzzz\"".into(),
        "\"\\ud800\"".into(),
        "\"\\ud800\\\"".into(),
        "\"\\ud800\\u0041\"".into(),
        "\"\\udc00\"".into(),
        "\"\\ud800x\"".into(),
        "\"\\ud83d\\ude00\"".into(),
        "\"\\x\"".into(),
        "\"a\u{1}b\"".into(),
        "\"abc".into(),
        "\"\\".into(),
        "1e309".into(),
        "-1e309".into(),
        "1e-400".into(),
        "0e999999999999".into(),
        "1e999999999999".into(),
        "123456789012345678901234567890".into(),
        "123456789012345678901234567890e300".into(),
        "123456789012345678901234567890.5".into(),
        "12345678901234567890123.".into(),
        "[1]x".into(),
        "[1] x".into(),
        "\n\n  [1,\n 2,\n x]".into(),
        "{\"a\":1,\"a\":2}".into(),
        "{\"b\":1,\"a\":{\"z\":1,\"y\":[{\"q\":1,\"p\":2}]}}".into(),
        "-0".into(),
        "-0.0".into(),
        "5.0".into(),
        "1E3".into(),
        "18446744073709551615".into(),
        "18446744073709551616".into(),
        "-9223372036854775808".into(),
        "-9223372036854775809".into(),
        "0.1".into(),
        "1.7976931348623157e308".into(),
        "1.7976931348623159e308".into(),
        "4.9e-324".into(),
        "\"\\u0000\"".into(),
        "\"\u{feff}\"".into(),
        "\u{feff}[]".into(),
        "[\u{a0}]".into(),
        "\"é\\u00e9\"".into(),
        "[1e400]".into(),
        "{\"a\":1 \"b\":2}".into(),
    ];
    for d in [126usize, 127, 128, 129] {
        sj.push(format!("{}{}", "[".repeat(d), "]".repeat(d)));
        sj.push(format!("{}1{}", "{\"a\":".repeat(d), "}".repeat(d)));
        sj.push("[".repeat(d));
    }
    let seeds: Vec<String> = sj.clone();
    for _ in 0..3000 {
        let s = r.pick(&seeds).clone();
        sj.push(mutate(&mut r, &s));
    }
    for s in &sj {
        let v: Result<Value, _> = serde_json::from_str(s);
        match v {
            Ok(v) => out.rec(json!({"t": "sj", "text": s, "out": v.to_string()})),
            Err(e) => out.rec(json!({"t": "sj", "text": s, "err": e.to_string()})),
        }
    }

    // ── Catalogues ─────────────────────────────────────────────────────────────────────────
    let mut cat_texts: Vec<String> = Vec::new();
    for f in ["tools.json", "tools-extended.json", "suite-tools.json"] {
        cat_texts.push(std::fs::read_to_string(format!("{fx}/{f}")).unwrap());
    }
    cat_texts.push(r#"[
 {"name":"g_all","description":"x","parameters":{"type":"object","properties":{
   "Duration_Seconds":{"type":"integer","minimum":1,"maximum":86400},
   "hour":{"type":"integer","minimum":0,"maximum":23},
   "MINUTE":{"type":"integer"},
   "level":{"type":"number","minimum":-1.5,"maximum":10.25},
   "count":{"type":"integer","minimum":-5,"maximum":5},
   "label":{"type":"string","maxLength":5},
   "city":{"type":"string"},
   "mood":{"type":"string","enum":["dance","wave","ice","bee","Été","ΣΑΣ","nod","e","ee","see"]},
   "on":{"type":"boolean"},
   "ΣSecond":{"type":"number"}
 },"required":["hour"]}},
 {"name":"g_nums","parameters":{"type":"object","properties":{
   "level":{"type":"integer","minimum":0,"maximum":10},
   "temp":{"type":"number"},
   "seconds_total":{"type":"number","minimum":0}
 }}},
 {"name":"getWeather","parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["c","f"]}},"required":["city"]}},
 {"name":"aB-cD"}
]"#.into());
    let base_valid = cat_texts.len();
    let mut adv: Vec<String> = vec![
        "[]".into(), "{}".into(), "null".into(), "\"x\"".into(), "1".into(), "".into(),
        "[1]".into(), "[\"t\"]".into(), "[null]".into(), "[[]]".into(),
        r#"[{"name":"t"}]"#.into(),
        r#"  [ {"name":"t"} , {"name":"u"} ]  "#.into(),
        "[\n\t{\"name\":\"t\"}\n\t,\n{\"name\":\"u\"}\n]\n".into(),
        r#"[{"name":"t","name":"u"}]"#.into(),
        r#"[{"name":1,"name":"u"}]"#.into(),
        r#"[{"name":"u","name":1}]"#.into(),
        r#"[{"name":""}]"#.into(),
        r#"[{"name":"a b"}]"#.into(),
        r#"[{"name":"é"}]"#.into(),
        r#"[{"name":"\u0000x"}]"#.into(),
        r#"[{"name":"a\"b\\c\n\t\r\u0001\u007f\u00ad\u0300\u200b"}]"#.into(),
        format!(r#"[{{"name":"{}"}}]"#, "a".repeat(64)),
        format!(r#"[{{"name":"{}"}}]"#, "a".repeat(65)),
        r#"[{"name":"t"},{"name":"t"}]"#.into(),
        r#"[{"name":"getWeather"},{"name":"get_weather"}]"#.into(),
        r#"[{"name":"get_weather"},{"name":"getWeather"}]"#.into(),
        r#"[{"name":"GetWeather"},{"name":"getWeather"}]"#.into(),
        r#"[{"name":"t","x":1}]"#.into(),
        r#"[{"name":"t","zz":1,"aa":2}]"#.into(),
        r#"[{"name":"t","Σ":1}]"#.into(),
        r#"[{"name":"t","description":5}]"#.into(),
        r#"[{"name":"t","description":null}]"#.into(),
        r#"[{"name":"t","description":"a","description":"b"}]"#.into(),
        r#"[{"name":"t","parameters":5}]"#.into(),
        r#"[{"name":"t","parameters":{}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"array"}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","additionalProperties":true}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":[]}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":1}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":5}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"array"}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"Strïng"}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"string","pattern":"a+"}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"string","enum":"a"}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"string","enum":[]}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"string","enum":["a",1]}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"string","enum":["a","a","é"]}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"string","maxLength":-1}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"string","maxLength":1.5}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"string","maxLength":5.0}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"string","maxLength":"5"}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"string","maxLength":0}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"string","maxLength":4294967295}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"string","maxLength":4294967296}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"string","maxLength":9223372036854775807}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"string","maxLength":9223372036854775808}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"string","description":7}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"integer","minimum":1.0}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"integer","minimum":-9223372036854775808,"maximum":9223372036854775807}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"integer","maximum":9223372036854775808}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"integer","minimum":"a","maximum":"b"}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"integer","maximum":"b"}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"integer","exclusiveMinimum":1}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"number","minimum":1,"maximum":1e308}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"number","minimum":-0,"maximum":18446744073709551616}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"number","minimum":true}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"number","maximum":null}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"boolean","default":true}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"boolean"}},"required":["x","x"]}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{},"required":["y"]}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","required":["y"]}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","required":[]}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"boolean"}},"required":"x"}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"boolean"}},"required":[1]}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"boolean"}},"required":["x\u0000"]}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"boolean"},"x":{"type":"array"}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"x":{"type":"array"},"x":{"type":"boolean"}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"b":{"type":"array"},"a":{"type":"bad"}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","properties":{"é":{"type":"integer"},"\u0000":{"type":"string"},"":{"type":"boolean"}}}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"object","type":"array"}}]"#.into(),
        r#"[{"name":"t","parameters":{"type":"array","type":"object"}}]"#.into(),
        r#"[{"name":"z","description":"d \"q\" ]","parameters":{"type":"object"}} ]"#.into(),
        "[{\"name\":\"z\",\"description\":\"[{,]}\\\\\"} ,{\"name\":\"y\"}\u{3000}]".into(),
        "[{\"name\":\"z\"}\u{3000}]".into(),
        "\u{feff}[{\"name\":\"z\"}]".into(),
        "[{\"name\":\"z\"}]\u{3000}".into(),
        "[{\"name\":\"z\"}]\n".into(),
        "[{\"name\":\"z\"} \u{0c}]".into(),
        "[{\"name\":\"z\"}]]".into(),
        "[{\"name\":\"z\"},]".into(),
        "[{\"name\":\"z\" , }]".into(),
        "[{\"name\":\"t\",\"parameters\":{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"string\",\"enum\":[\"Σ\",\"é\"]}}}}]".into(),
    ];
    for n in [31usize, 32, 33] {
        let v: Vec<String> = (0..n).map(|i| format!(r#"{{"name":"t{i}"}}"#)).collect();
        adv.push(format!("[{}]", v.join(",")));
    }
    for n in [15usize, 16, 17] {
        let v: Vec<String> = (0..n)
            .map(|i| format!(r#""p{i:02}":{{"type":"boolean"}}"#))
            .collect();
        adv.push(format!(
            r#"[{{"name":"t","parameters":{{"type":"object","properties":{{{}}}}}}}]"#,
            v.join(",")
        ));
        let mut v2 = v.clone();
        v2.push(r#""p00":{"type":"integer"}"#.into());
        adv.push(format!(
            r#"[{{"name":"t","parameters":{{"type":"object","properties":{{{}}}}}}}]"#,
            v2.join(",")
        ));
    }
    for target in [16383usize, 16384, 16385, 20000] {
        let head = r#"[{"name":"t","description":""#;
        let tail = r#""}]"#;
        let pad = target - head.len() - tail.len();
        adv.push(format!("{head}{}{tail}", "x".repeat(pad)));
        let mut s = String::from(head);
        while s.len() + tail.len() + 2 <= target {
            s.push('é');
        }
        while s.len() + tail.len() < target {
            s.push('x');
        }
        s.push_str(tail);
        adv.push(s);
        adv.push(format!("[{{\"name\":\"t\"}}]{}", " ".repeat(target - 14)));
    }
    let keys_tool = ["name", "description", "parameters", "extra", "Name"];
    let types = [
        "string", "integer", "number", "boolean", "array", "object", "STRING", "null",
    ];
    let nums = [
        "0",
        "1",
        "-1",
        "5",
        "86400",
        "1.5",
        "-0",
        "1e3",
        "\"5\"",
        "null",
        "true",
        "9223372036854775807",
        "9223372036854775808",
        "-9223372036854775809",
        "2.5e-3",
        "18446744073709551615",
        "-7",
    ];
    let pkeys = [
        "seconds", "hour", "minute", "level", "city", "name", "x", "é", "Σ", "a b", "", "\\u0000",
        "label", "count",
    ];
    for _ in 0..2500 {
        let ntools = 1 + r.below(5);
        let mut tools = Vec::new();
        for ti in 0..ntools {
            let mut members = Vec::new();
            let name = match r.below(10) {
                0 => lit(&rand_string(&mut r, 6)),
                1 => format!("\"tool_{}\"", r.below(3)),
                2 => format!("\"Tool{}\"", r.below(3)),
                3 => "5".into(),
                _ => format!("\"t{}_{}\"", ti, r.below(1000)),
            };
            if !r.chance(5) {
                members.push(format!("\"name\":{name}"));
            }
            if r.chance(50) {
                let d = if r.chance(90) {
                    lit(&rand_string(&mut r, 20))
                } else {
                    "1".into()
                };
                members.push(format!("\"description\":{d}"));
            }
            if r.chance(5) {
                members.push(format!("\"{}\":1", r.pick(&keys_tool)));
            }
            if r.chance(85) {
                let mut pm = Vec::new();
                pm.push(format!(
                    "\"type\":\"{}\"",
                    if r.chance(95) { "object" } else { "array" }
                ));
                let np = r.below(6);
                let mut pnames = Vec::new();
                let mut props = Vec::new();
                for _ in 0..np {
                    let pn = r.pick(&pkeys).to_string();
                    pnames.push(pn.clone());
                    let ty = if r.chance(92) {
                        *r.pick(&types[..4])
                    } else {
                        *r.pick(&types)
                    };
                    let mut f = vec![format!("\"type\":\"{ty}\"")];
                    if r.chance(30) {
                        f.push(format!("\"description\":{}", lit(&rand_string(&mut r, 8))));
                    }
                    match ty {
                        "string" => {
                            if r.chance(40) {
                                let ne = r.below(4);
                                let ev: Vec<String> = (0..ne)
                                    .map(|_| {
                                        if r.chance(95) {
                                            lit(&rand_string(&mut r, 4))
                                        } else {
                                            "1".into()
                                        }
                                    })
                                    .collect();
                                f.push(format!("\"enum\":[{}]", ev.join(",")));
                            }
                            if r.chance(30) {
                                f.push(format!("\"maxLength\":{}", r.pick(&nums)));
                            }
                        }
                        "integer" | "number" => {
                            if r.chance(50) {
                                f.push(format!("\"minimum\":{}", r.pick(&nums)));
                            }
                            if r.chance(50) {
                                f.push(format!("\"maximum\":{}", r.pick(&nums)));
                            }
                        }
                        _ => {}
                    }
                    if r.chance(4) {
                        f.push("\"format\":\"date\"".into());
                    }
                    if r.chance(5) {
                        f.reverse();
                    }
                    props.push(format!("\"{pn}\":{{{}}}", f.join(",")));
                }
                pm.push(format!("\"properties\":{{{}}}", props.join(",")));
                if r.chance(60) {
                    let mut rq: Vec<String> = Vec::new();
                    for _ in 0..r.below(3) {
                        if !pnames.is_empty() && r.chance(85) {
                            rq.push(format!("\"{}\"", r.pick(&pnames)));
                        } else if r.chance(50) {
                            rq.push("\"missing\"".into());
                        } else {
                            rq.push("2".into());
                        }
                    }
                    pm.push(format!("\"required\":[{}]", rq.join(",")));
                }
                if r.chance(4) {
                    pm.push("\"additionalProperties\":false".into());
                }
                if r.chance(10) {
                    pm.reverse();
                }
                members.push(format!("\"parameters\":{{{}}}", pm.join(",")));
            }
            if r.chance(10) {
                members.reverse();
            }
            let ws = *r.pick(&["", " ", "\n  ", "\t"]);
            tools.push(format!("{ws}{{{}}}{ws}", members.join(&format!(",{ws}"))));
        }
        let s = format!("[{}]", tools.join(","));
        if r.chance(15) {
            let m = mutate(&mut r, &s);
            adv.push(m);
        }
        adv.push(s);
    }
    for _ in 0..1500 {
        let s = r.pick(&cat_texts[..base_valid]).clone();
        adv.push(mutate(&mut r, &s));
    }
    cat_texts.extend(adv);

    let mut cats: Vec<Catalogue> = Vec::new();
    for text in cat_texts.iter() {
        match Catalogue::parse(text) {
            Ok(c) => {
                out.rec(json!({"t": "cat", "text": text, "ok": cat_dump(&c)}));
                cats.push(c);
            }
            Err(e) => out.rec(json!({"t": "cat", "text": text, "err": e})),
        }
    }
    eprintln!("catalogues: {} ({} valid)", cat_texts.len(), cats.len());

    for k in 0..cats.len() {
        let c = &cats[k];
        let names: Vec<String> = c.tools.iter().map(|t| t.name.clone()).collect();
        let mut lists: Vec<Option<Vec<String>>> = vec![None, Some(vec![]), Some(names.clone())];
        let mut rev = names.clone();
        rev.reverse();
        lists.push(Some(rev));
        if names.len() > 1 {
            lists.push(Some(vec![
                names[names.len() - 1].clone(),
                names[0].clone(),
                names[0].clone(),
            ]));
        }
        lists.push(Some(vec![names[0].clone(), "nope".into()]));
        lists.push(Some(vec![rand_string(&mut r, 6)]));
        lists.push(Some(vec![c.tools[0].snake_name.clone()]));
        for l in lists {
            match c.tools_json(l.as_deref()) {
                Ok(s) => out.rec(json!({"t": "tj", "cat": k, "names": l, "ok": s})),
                Err(e) => out.rec(json!({"t": "tj", "cat": k, "names": l, "err": e})),
            }
        }
    }

    // ── Validation ─────────────────────────────────────────────────────────────────────────
    let wrap = |p: &str| format!("<think>\nx\n</think>\n<tool_call>{p}</tool_call>");
    let fixed: Vec<String> = vec![
        "".into(), "I can't help".into(), "<tool_call>".into(), "<tool_call>[]".into(),
        "<tool_call>[]</tool_call>".into(), "<tool_call> \n[]\t </tool_call>".into(),
        "<tool_call>\u{3000}[]\u{a0}</tool_call>".into(), "<tool_call>\u{feff}[]</tool_call>".into(),
        "</tool_call><tool_call>[]</tool_call>".into(), "<tool_call><tool_call>[]</tool_call>".into(),
        "<tool_call>[]</tool_call><tool_call>{}</tool_call>".into(), "<tool_call></tool_call>".into(),
        "<tool_call>  </tool_call>".into(), "<tool_call>{}</tool_call>".into(),
        "<tool_call>null</tool_call>".into(), "<tool_call>[1]</tool_call>".into(),
        "<tool_call>[[]]</tool_call>".into(), "<tool_call>[{}]</tool_call>".into(),
        "<tool_call>[{\"name\":1}]</tool_call>".into(),
        "<tool_call>[{\"name\":\"x\",\"extra\":1}]</tool_call>".into(),
        "<tool_call>[{\"arguments\":{}}]</tool_call>".into(),
        "<tool_call>[{\"name\":\"rm -rf \\\"/\\\" \\u0001é\"}]</tool_call>".into(),
        "<tool_call>[{\"name\":\"start_timer\",\"name\":\"nope\"}]</tool_call>".into(),
        "<tool_call>[{\"name\":\"nope\",\"name\":\"start_timer\",\"arguments\":{\"seconds\":5}}]</tool_call>".into(),
        "<tool_call>[{\"name\":\"start_timer\",\"arguments\":null}]</tool_call>".into(),
        "<tool_call>[{\"name\":\"start_timer\",\"arguments\":[]}]</tool_call>".into(),
        "<tool_call>[{\"name\":\"start_timer\"}]</tool_call>".into(),
        "<tool_call>[{\"name\":\"start_timer\",\"arguments\":{\"seconds\":5},\"arguments\":{}}]</tool_call>".into(),
        "<tool_call>[{\"name\":\"start_timer\",\"arguments\":{\"seconds\":\"x\",\"seconds\":5}}]</tool_call>".into(),
        "<tool_call>[{\"name\":\"start_timer\",\"arguments\":{\"seconds\":5,\"seconds\":\"x\"}}]</tool_call>".into(),
        "<tool_call>[{\"name\":\"start_timer\",\"arguments\":{\"seconds\":5}} {}]</tool_call>".into(),
        "<tool_call>[{\"name\":\"start_timer\",\"arguments\":{\"seconds\":5}}]".into(),
        "<tool_call>[{\"name\":\"start_timer\",\"arguments\":{\"seconds\":5}}]</tool_call >".into(),
        "<TOOL_CALL>[]</TOOL_CALL>".into(),
        format!("<tool_call>[\"{}\"]</tool_call>", "a".repeat(4092)),
        format!("<tool_call>[\"{}\"]</tool_call>", "a".repeat(4093)),
        format!("<tool_call>  [\"{}\"]\n</tool_call>", "a".repeat(4092)),
        format!("<tool_call>[\"{}\"]</tool_call>", "é".repeat(2046)),
        format!("<tool_call>[\"{}\"]</tool_call>", "é".repeat(2047)),
        format!("<tool_call>{}{}</tool_call>", "[".repeat(127), "]".repeat(127)),
        format!("<tool_call>{}{}</tool_call>", "[".repeat(128), "]".repeat(128)),
        format!("<tool_call>[{{\"name\":\"start_timer\",\"arguments\":{{\"seconds\":5,\"x\":{}{}}}}}]</tool_call>", "[".repeat(124), "]".repeat(124)),
        format!("<tool_call>[{{\"name\":\"start_timer\",\"arguments\":{{\"seconds\":5,\"x\":{}{}}}}}]</tool_call>", "[".repeat(125), "]".repeat(125)),
    ];
    for ci in 0..cats.len().min(4) {
        for t in &fixed {
            val_rec(&mut out, &cats, ci, None, t);
        }
    }
    let int_vals = [
        "0",
        "1",
        "-1",
        "5",
        "9",
        "10",
        "11",
        "23",
        "24",
        "59",
        "60",
        "86400",
        "86401",
        "450",
        "4.5",
        "5.0",
        "1e3",
        "-0",
        "-0.0",
        "\"9\"",
        "true",
        "null",
        "[]",
        "{}",
        "9223372036854775807",
        "9223372036854775808",
        "-9223372036854775808",
        "-9223372036854775809",
        "18446744073709551615",
        "18446744073709551616",
        "1e400",
        "-5",
        "-6",
        "1.0e1",
        "0.5",
        "10.25",
        "10.250000000000002",
        "-1.5",
        "-1.5000000000000002",
        "2.5e-324",
        "1.7976931348623157e308",
    ];
    let str_vals = [
        "\"dance\"",
        "\"nod\"",
        "\"wave\"",
        "\"spin\"",
        "\"Dance\"",
        "\"dance \"",
        "\"\"",
        "\"Boston\"",
        "\"São Paulo\"",
        "\"Été\"",
        "\"ΣΑΣ\"",
        "\"ice\"",
        "\"e\"",
        "\"ee\"",
        "\"see\"",
        "\"bee\"",
        "\"c\"",
        "\"f\"",
        "\"abcde\"",
        "\"abcdef\"",
        "\"ééééé\"",
        "\"éééééé\"",
        "\"😀😀😀😀😀\"",
        "\"😀😀😀😀😀😀\"",
        "\"a\\u0000b\"",
        "\"\\ud83d\\ude00\"",
        "5",
        "true",
        "null",
        "[\"dance\"]",
        "{\"a\":1}",
    ];
    let mut payloads: Vec<(usize, String)> = Vec::new();
    for ci in 0..cats.len() {
        let c = &cats[ci];
        let reps = if ci < 4 { 400 } else { 4 };
        for _ in 0..reps {
            let ncalls = match r.below(20) {
                0 => 0,
                1 => 5,
                2 => 6,
                3..=6 => 2,
                7 => 3,
                8 => 4,
                _ => 1,
            };
            let mut calls = Vec::new();
            for _ in 0..ncalls {
                let t = r.pick(&c.tools);
                let name = match r.below(20) {
                    0 => "\"nope\"".to_string(),
                    1 => lit(&t.snake_name),
                    2 => lit(&t.name.to_uppercase()),
                    3 => "7".to_string(),
                    _ => lit(&t.name),
                };
                let mut args = Vec::new();
                for p in &t.params {
                    if r.chance(15) {
                        continue;
                    }
                    let v = match &p.ty {
                        ParamType::String {
                            enum_values: Some(e),
                            ..
                        } if r.chance(70) => lit(r.pick::<String>(e)),
                        ParamType::String { .. } => {
                            if r.chance(80) {
                                r.pick(&str_vals).to_string()
                            } else {
                                lit(&rand_string(&mut r, 6))
                            }
                        }
                        ParamType::Boolean => r
                            .pick(&["true", "false", "1", "\"true\"", "null"])
                            .to_string(),
                        _ => r.pick(&int_vals).to_string(),
                    };
                    args.push(format!("{}:{}", lit(&p.name), v));
                }
                if r.chance(8) {
                    args.push(format!(
                        "{}:1",
                        lit(*r.pick(&["extra", "seconds", "zzz", "é", "a"]))
                    ));
                }
                if r.chance(5) && !args.is_empty() {
                    let d = args[0].clone();
                    args.push(d);
                }
                if r.chance(20) {
                    for i in (1..args.len()).rev() {
                        let j = r.below(i + 1);
                        args.swap(i, j);
                    }
                }
                let a = format!("{{{}}}", args.join(","));
                let obj = match r.below(25) {
                    0 => format!("{{\"name\":{name}}}"),
                    1 => format!("{{\"arguments\":{a},\"name\":{name}}}"),
                    2 => format!("{{\"name\":{name},\"arguments\":{a},\"x\":1}}"),
                    3 => format!("{{\"name\":{name},\"arguments\":\"{{}}\"}}"),
                    _ => format!("{{\"name\":{name},\"arguments\":{a}}}"),
                };
                calls.push(obj);
            }
            let mut p = format!("[{}]", calls.join(*r.pick(&[",", ", ", ",\n"])));
            if r.chance(10) {
                p = mutate(&mut r, &p);
            }
            payloads.push((ci, p));
        }
    }
    for (ci, p) in &payloads {
        let text = match r.below(10) {
            0 => format!("<tool_call>{p}</tool_call>"),
            1 => format!("prefix <tool_call>\n{p}\n</tool_call> suffix <tool_call>[]</tool_call>"),
            2 => format!("<tool_call>\u{2028}{p}\u{3000}</tool_call>"),
            3 => format!("<tool_call>{p}"),
            _ => wrap(p),
        };
        let allowed: Option<Vec<String>> = match r.below(6) {
            0 => Some(vec![]),
            1 => Some(
                cats[*ci]
                    .tools
                    .iter()
                    .take(1)
                    .map(|t| t.name.clone())
                    .collect(),
            ),
            2 => Some(vec![cats[*ci].tools[0].snake_name.clone(), "x".into()]),
            _ => None,
        };
        val_rec(&mut out, &cats, *ci, allowed.as_deref(), &text);
    }

    // One-parameter perturbations of an otherwise valid call, for every parameter.
    for ci in 0..cats.len().min(4) {
        let c = &cats[ci];
        for t in &c.tools {
            let valid = |p: &needle_jibo::catalog::Param| -> String {
                match &p.ty {
                    ParamType::String {
                        enum_values: Some(e),
                        ..
                    } => lit(&e[0]),
                    ParamType::String { .. } => "\"x\"".into(),
                    ParamType::Integer { min, max } => format!("{}", min.or(*max).unwrap_or(0)),
                    ParamType::Number { min, max } => {
                        serde_json::to_string(&min.or(*max).unwrap_or(0.0)).unwrap()
                    }
                    ParamType::Boolean => "true".into(),
                }
            };
            for (pi, _p) in t.params.iter().enumerate() {
                let mut vals: Vec<String> = int_vals
                    .iter()
                    .chain(str_vals.iter())
                    .map(|s| s.to_string())
                    .collect();
                vals.extend([
                    "false".to_string(),
                    "\"abcd\"".into(),
                    "\"ééé\"".into(),
                    "\"éééé\"".into(),
                ]);
                for v in vals {
                    let mut args = Vec::new();
                    for (qi, q) in t.params.iter().enumerate() {
                        if qi == pi {
                            args.push(format!("{}:{}", lit(&q.name), v));
                        } else if t.required.contains(&q.name) || r.chance(50) {
                            args.push(format!("{}:{}", lit(&q.name), valid(q)));
                        }
                    }
                    let text = wrap(&format!(
                        "[{{\"name\":{},\"arguments\":{{{}}}}}]",
                        lit(&t.name),
                        args.join(",")
                    ));
                    val_rec(&mut out, &cats, ci, None, &text);
                }
            }
        }
    }

    // ── Grounding ──────────────────────────────────────────────────────────────────────────
    let mut queries: Vec<String> = Vec::new();
    for f in ["requests.jsonl", "suite-dev.jsonl", "suite-heldout.jsonl"] {
        for l in std::fs::read_to_string(format!("{fx}/{f}"))
            .unwrap()
            .lines()
        {
            if let Ok(v) = serde_json::from_str::<Value>(l) {
                if let Some(q) = v["query"].as_str() {
                    queries.push(q.to_string());
                }
            }
        }
    }
    let adv_q = [
        "",
        " ",
        "Set a timer for 5 minutes.",
        "Start a ninety second timer",
        "Time 2 minutes and 15 seconds.",
        "Set a timer for seven and a half minutes.",
        "Set a timer for a minute and a half.",
        "Set a timer for half an hour.",
        "Timer for quarter of an hour.",
        "Set a timer for an hour and a half.",
        "Set a timer for twenty-five minutes",
        "twenty - five minutes",
        "twenty -five",
        "twenty--five",
        "twenty five seconds",
        "twenty zero",
        "twenty ten",
        "ninety nine",
        "Can you time 1.5 hours",
        "Wake me up at 7:30 am.",
        "Set an alarm for 4:00 pm.",
        "six o'clock in the morning",
        "half past seven in the morning",
        "seven thirty am",
        "quarter to eight",
        "quarter to one",
        "half past twelve",
        "half to 0",
        "noon",
        "midday",
        "midnight",
        "5pm",
        "5PM",
        "7am",
        "12am",
        "12pm",
        "12 am",
        "12 pm",
        "0:30",
        "24:00",
        "23:59",
        "7:60",
        "7:15",
        "1:2:3",
        "7.5:30",
        "12:00 pm",
        "12:30 am",
        "7 in the evening",
        "7 in the afternoon",
        "7 in the night",
        "7 tonight",
        "13 pm",
        "seven tonight",
        "nan",
        "inf",
        "infinity",
        "-inf",
        "nanpm",
        "infam",
        "5pm 6",
        "5pm six thirty",
        "am pm",
        "1e5 seconds",
        "1e309 seconds",
        "nan seconds",
        "inf minutes",
        "2.5.3 minutes",
        "3.14",
        "3.",
        ".5",
        "1,000",
        "1 000",
        "10'000",
        "Mute yourself.",
        "Turn it up to the max.",
        "maximum",
        "loudest",
        "quietest",
        "lowest",
        "off",
        "full",
        "silence please",
        "MAX",
        "Max!",
        "minimum volume",
        "min",
        "mins",
        "Is it raining in São Paulo?",
        "SÃO PAULO",
        "ΣΑΣ",
        "ΌΣΟΣ ΣΑΣ.",
        "İstanbul",
        "ISTANBUL",
        "Straße",
        "STRASSE",
        "ǅemal",
        "١٢٣ minutes",
        "５ minutes",
        "Ⅻ o'clock",
        "½ an hour",
        "a quarter of an hour",
        "quarter of a hour",
        "quarter of an",
        "half an",
        "half a minute",
        "an and a half minutes",
        "a and a half hours",
        "5 and a half and 3 minutes",
        "5 minutes and 3 seconds and 2 hours",
        "5 minutes and",
        "and 5 minutes",
        "5 minutes and and 3 seconds",
        "one hour and a half and ten minutes",
        "10 secs 5 mins 2 hrs",
        "1 hr",
        "0 seconds",
        "-5 minutes",
        "5 -minutes",
        "five-minutes",
        "o'clock",
        "7 o'clock pm",
        "7 o'clock in the morning",
        "seven o'clock tonight",
        "3 10",
        "3 9",
        "3 60",
        "3 59 pm",
        "12 45 am",
        "Wave hello!",
        "I love dancing",
        "Show me your moves",
        "Will I need an umbrella?",
        "ice ice baby",
        "bee careful",
        "see",
        "e",
        "Été",
        "été",
        "ÉTÉ indien",
        "nod nod",
        "N.O.D",
        "dance-off",
        "danc",
        "waves",
        "Ωmega",
        "ΣSecond 5",
        "level -1.5",
        "level 10.25",
        "count -5",
        "the volume to eleven",
        "Set the volume to three.",
        "volume to uh six",
        "seventeen",
        "nineteen eighty four",
        "zero",
        "twelve thirty",
        "twelve fifteen pm",
        "1 2 3 4 5",
        "\u{3000}5\u{3000}minutes",
        "5\u{a0}minutes",
        "5\tminutes\n",
        "5:30:00",
        "5.30 pm",
        "it's 7",
        "it’s 7",
        "rock'n'roll",
        "'5' minutes",
        "5' minutes",
        "Σ",
        "aΣ",
        "aΣ b",
        "aΣb",
        "a'Σ",
        "ΑΣ'",
        "A\u{345}Σ",
        "\u{345}Σ",
    ];
    queries.extend(adv_q.iter().map(|s| s.to_string()));
    let vocab = [
        "a",
        "an",
        "and",
        "half",
        "quarter",
        "of",
        "past",
        "to",
        "o'clock",
        "am",
        "pm",
        "in",
        "the",
        "morning",
        "evening",
        "afternoon",
        "tonight",
        "noon",
        "midnight",
        "midday",
        "second",
        "seconds",
        "sec",
        "secs",
        "minute",
        "minutes",
        "min",
        "mins",
        "hour",
        "hours",
        "hr",
        "hrs",
        "one",
        "two",
        "five",
        "seven",
        "ten",
        "twelve",
        "fifteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "ninety",
        "-",
        "1",
        "5",
        "7",
        "12",
        "13",
        "30",
        "45",
        "59",
        "60",
        "1.5",
        "7:30",
        "12:00",
        "4:15",
        "5pm",
        "7am",
        "nan",
        "inf",
        "1e2",
        "mute",
        "max",
        "min",
        "full",
        "off",
        "lowest",
        "timer",
        "set",
        "volume",
        "Boston",
        "São",
        "Paulo",
        "dance",
        "dancing",
        "wave",
        "Σ",
        "ΣΑΣ",
        "É",
        ",",
        ".",
        "!",
        "?",
        "'",
        "\u{3000}",
        "\u{345}",
    ];
    for _ in 0..6000 {
        let n = 1 + r.below(9);
        let mut q = String::new();
        for k in 0..n {
            if k > 0 {
                q.push_str(*r.pick(&[" ", " ", " ", "", "-", ", ", "  ", "\u{a0}"]));
            }
            let w = *r.pick(&vocab);
            if r.chance(10) {
                q.push_str(&w.to_uppercase());
            } else {
                q.push_str(w);
            }
        }
        if r.chance(5) {
            q.push(rand_char(&mut r));
        }
        queries.push(q);
    }
    for _ in 0..300 {
        queries.push(rand_string(&mut r, 20));
    }

    let gcats = [2usize, 3, 1];
    for q in &queries {
        let m = mentions_rec(&mut out, q);
        let mut cands: Vec<f64> = vec![
            0.0, 1.0, 5.0, 10.0, -1.5, 10.25, 7.0, 19.0, 30.0, 300.0, 450.0, 1800.0,
        ];
        cands.extend(m.numbers.iter().filter(|x| x.is_finite()));
        cands.extend(m.durations_s.iter().filter(|x| x.is_finite()));
        for (h, mi) in &m.clock {
            cands.push(*h);
            cands.push(*mi);
        }
        let reps = if r.chance(30) { 3 } else { 1 };
        for _ in 0..reps {
            let ci = *r.pick(&gcats);
            let c = &cats[ci];
            let t = r.pick(&c.tools);
            let mut args = Map::new();
            for p in &t.params {
                if r.chance(20) {
                    continue;
                }
                let v: Value = match &p.ty {
                    ParamType::Integer { .. } | ParamType::Number { .. } => {
                        let x = *r.pick(&cands);
                        let x = if r.chance(10) {
                            x + 1e-10
                        } else if r.chance(5) {
                            x + 1e-8
                        } else {
                            x
                        };
                        if r.chance(5) {
                            json!(x.to_string())
                        } else {
                            serde_json::from_str(&num_text(&mut r, x)).unwrap()
                        }
                    }
                    ParamType::String { enum_values, .. } => {
                        let s = match (enum_values, r.below(4)) {
                            (Some(e), 0 | 1) => r.pick(e).clone(),
                            (_, 0) if !q.is_empty() => {
                                let cs: Vec<char> = q.chars().collect();
                                let a = r.below(cs.len());
                                let b = a + r.below(cs.len() - a + 1);
                                let s: String = cs[a..b].iter().collect();
                                if r.chance(30) {
                                    s.to_uppercase()
                                } else {
                                    s
                                }
                            }
                            (_, 1) => r.pick(&vocab).to_string(),
                            (_, 2) => r
                                .pick(&["Boston", "São Paulo", " ", "", "dance", "ΣΑΣ", "Été", "x"])
                                .to_string(),
                            _ => rand_string(&mut r, 4),
                        };
                        if r.chance(4) {
                            json!(5)
                        } else {
                            json!(s)
                        }
                    }
                    ParamType::Boolean => json!(r.chance(50)),
                };
                args.insert(p.name.clone(), v);
            }
            if r.chance(5) {
                args.insert("undeclared".into(), json!(1));
            }
            let a = Value::Object(args).to_string();
            ug_rec(&mut out, &cats, ci, &t.name, &a, q, &m);
        }
    }
    for (f, ef) in [
        ("suite-dev.jsonl", "suite-expected.jsonl"),
        ("suite-heldout.jsonl", "suite-expected.jsonl"),
        ("requests.jsonl", "expected.jsonl"),
    ] {
        let mut exp = std::collections::HashMap::new();
        for l in std::fs::read_to_string(format!("{fx}/{ef}"))
            .unwrap()
            .lines()
        {
            if let Ok(v) = serde_json::from_str::<Value>(l) {
                exp.insert(
                    v["request_id"].as_str().unwrap_or("").to_string(),
                    v["expected"].clone(),
                );
            }
        }
        for l in std::fs::read_to_string(format!("{fx}/{f}"))
            .unwrap()
            .lines()
        {
            let Ok(v) = serde_json::from_str::<Value>(l) else {
                continue;
            };
            let q = v["query"].as_str().unwrap_or("").to_string();
            let id = v["request_id"].as_str().unwrap_or("");
            if let Some(e) = exp.get(id) {
                if !e.is_array() {
                    continue;
                }
                for ci in [0usize, 1, 2] {
                    let text = wrap(&e.to_string());
                    val_rec(&mut out, &cats, ci, None, &text);
                    if let Outcome::Calls(cs) = validate(&text, &cats[ci], None) {
                        let m = Mentions::read(&q);
                        for c in &cs {
                            let a = Value::Object(c.arguments.clone()).to_string();
                            ug_rec(&mut out, &cats, ci, &c.name, &a, &q, &m);
                        }
                    }
                }
            }
        }
    }
    out.w.flush().unwrap();
    eprintln!("{} records", out.n);
}
