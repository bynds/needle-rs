// Reference generator for ports/jibo/c/tests/test_grammar.c (throwaway).
use needle_infer::cact::CactV3;
use needle_infer::constrained::{byte_table, ConstrainedDecoder, ToolDef};
use needle_infer::sp_tokenizer::SpTokenizer;
use std::io::Write;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn chance(&mut self, a: u64, b: u64) -> bool {
        self.below(b) < a
    }
}

fn hex(b: &[u8]) -> String {
    if b.is_empty() {
        return "-".to_string();
    }
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

fn mask_line(m: &[f32]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut allowed = Vec::new();
    for (i, &x) in m.iter().enumerate() {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        if x == 0.0 {
            allowed.push(i);
        }
        assert!(x == 0.0 || x == -1e9);
    }
    let mut s = format!("M {:016x} {}", h, allowed.len());
    if allowed.len() == m.len() {
        s.push_str(" ALL");
    } else if allowed.len() <= 512 {
        for a in &allowed {
            s.push_str(&format!(" {}", a));
        }
    }
    s
}

fn tools_line(defs: &[ToolDef]) -> String {
    let mut s = format!("T {}", defs.len());
    for d in defs {
        s.push_str(&format!(
            " {} {} {}",
            hex(d.name.as_bytes()),
            hex(d.snake_name.as_bytes()),
            d.param_keys.len()
        ));
        for k in &d.param_keys {
            s.push_str(&format!(" {}", hex(k.as_bytes())));
        }
    }
    s
}

fn odd_tool_inputs(r: &mut Rng) -> Vec<String> {
    let mut v: Vec<String> = [
        "",
        "   ",
        "[]",
        "{}",
        "[{}]",
        "null",
        "]",
        "[{\"name\":\"a\"}]",
        "{\"name\":\"a\"}",
        "{\"name\":\"a\"},{\"name\":\"b\"}",
        "[{\"name\": \"spaced\"}]",
        "[{\"name\":\"x\",\"parameters\":{}}]",
        "[{\"name\":\"x\",\"parameters\":null}]",
        "[{\"name\":\"x\",\"parameters\": \t{\"a\":{},\"b\":1,\"c\":{\"d\":{}}}}]",
        "[{\"name\":\"x\",\"parameters\":\u{a0}\u{2003}{\"a\":{}}}]",
        "[{\"name\":\"x\",\"parameters\":\u{0b}{\"a\":{}}}]",
        "[{\"name\":\"x\",\"parameters\":\u{85}{\"a\":{}}}]",
        "[{\"name\":\"x\",\"parameters\":\u{feff}{\"a\":{}}}]",
        "[{\"name\":\"x\",\"parameters\":{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"},\"b\":{}},\"required\":[\"a\"]}}]",
        "[{\"name\":\"x\",\"parameters\":{\"type\":\"object\",\"properties\":\"oops\",\"q\":{}}}]",
        "[{\"name\":\"x\",\"parameters\":{\"properties\":{}}}]",
        "[{\"name\":\"x\",\"parameters\":{\"nested\":{\"properties\":{\"inner\":{}}},\"top\":{}}}]",
        "[{\"parameters\":{\"name\":\"inner\",\"k\":{}},\"name\":\"outer\"}]",
        "[{\"description\":\"has \\\"name\\\":\\\"fake\\\" inside\",\"name\":\"real\"}]",
        "[{\"description\":\"a } brace and { another\",\"name\":\"braces\",\"parameters\":{\"p\":{}}}]",
        "[{\"name\":\"esc\\\"aped\",\"parameters\":{\"k\\\"q\":{},\"k\\\\\":{}}}]",
        "[{\"name\":\"getWeather\"},{\"name\":\"HTTPServerError\"},{\"name\":\"get_URL_v2\"},{\"name\":\"__x__y__\"},{\"name\":\"a-b c.d\"},{\"name\":\"ABC\"},{\"name\":\"aBC\"},{\"name\":\"ABc\"},{\"name\":\"a1B2\"},{\"name\":\"1A\"},{\"name\":\"!!!\"},{\"name\":\"\"},{\"name\":\"_\"}]",
        "[{\"name\":\"ÉtatCivil\"},{\"name\":\"straße\"},{\"name\":\"ΣίσυφοςΑ\"},{\"name\":\"日本語Name\"},{\"name\":\"ǅungla\"},{\"name\":\"ⅫRoman\"},{\"name\":\"x²y\"},{\"name\":\"İstanbul\"},{\"name\":\"ⒶⓑⒸ\"},{\"name\":\"a\u{300}B\"},{\"name\":\"٣Arabic\"},{\"name\":\"😀Smile\"}]",
        "[{\"name\":\"dup\",\"parameters\":{\"a\":{}}},{\"name\":\"dup\",\"parameters\":{\"b\":{}}},{\"name\":\"Dup\",\"parameters\":{\"c\":{}}}]",
        "[{\"name\":\"unterminated\",\"parameters\":{\"a\":{}",
        "[{\"name\":\"unterminated2",
        "[{\"name\":\"t\",\"parameters\":{\"a\":{},\"a\":{},\"\":{}}}]",
        "[{\"name\":\"t\",\"parameters\":{\"type\":\"object\",\"properties\":{\"a\":1,\"b\":[1,{\"c\":2}],\"d\":\"s\",\"\":{},\"e\":{\"f\":{}}}}}]",
        "[{\"name\":\"t\",\"parameters\":{\"type\":\"object\",\"properties\":{\"a\"   :  {} , \"b\"\n:\n{}}}}]",
        "[{\"name\":\"t\",\"parameters\":{\"type\":\"object\",\"properties\":{a:{},\"b\":{}}}}]",
        "[{\"name\":\"t\",\"parameters\":{\"type\":\"object\",\"properties\":{\"a\":{}\"b\":{}}}}]",
        "[{\"name\":\"t\",\"parameters\":{\"type\":\"object\",\"properties\":{\"a\":]}}}]",
        "[{\"name\":\"t\",\"parameters\":{\"x\":\"no close",
        "[1,2,{\"name\":\"after_numbers\"},\"str\",{\"name\":\"z\"}]",
        "garbage before [{\"name\":\"g\"}] garbage after {\"name\":\"h\"}",
        "{\"name\":\"obj_root\",\"parameters\":{\"k\":{}}}{\"name\":\"second\"}",
        "[{\"name\":\"t\"}}}}]{\"name\":\"u\"}",
        "[{\"name\":\"t\",\"parameters\":{\"type\":\"object\",\"properties\":{\"p\\u0041\":{},\"é\":{},\"日本\":{}}}}]",
        "[\n  {\n    \"name\":\"nl\",\n    \"parameters\":\n{\"type\":\"object\",\"properties\":\n  {\"k\":{}}}}\n]",
        "[{\"name\":\"t\",\"parameters\":{\"type\":\"object\",\"properties\":{\"k\":{\"properties\":{\"deep\":{}}}}}}]",
        "[{\"name\":\"t\",\"parameters\":{\"properties\":{\"k\":{}}, \"type\":\"object\"}}]",
        "[{\"name\":\"t\",\"parameters\":{\"b\":{},\"type\":\"object\",\"properties\":{\"k\":{}}}}]",
        "[{\"name\":\"t\",\"parameters\":{\"\\\"properties\\\":\":{\"x\":{}}}}]",
        "[{\"name\":\"t\",\"parameters\":{\"properties\":    {\"x\":{}}}}]",
        "[{\"name\":\"t\",\"x\":{\"parameters\":{\"wrong\":{}}},\"parameters\":{\"right\":{}}}]",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    // random tool names exercising to_snake_case
    let pool: Vec<char> = "aAbBzZ09_- .$/\\ÉéßΣσςǅǄǆⅫⅻ²³٣İıﬁⒶⓑ😀日本\u{300}\u{345}ªºˢᵃΩω"
        .chars()
        .collect();
    for _ in 0..400 {
        let mut s = String::from("[");
        for t in 0..1 + r.below(4) {
            if t > 0 {
                s.push(',');
            }
            let mut name = String::new();
            for _ in 0..r.below(12) {
                name.push(pool[r.below(pool.len() as u64) as usize]);
            }
            let name = name.replace('\\', "\\\\");
            s.push_str(&format!(
                "{{\"name\":\"{}\",\"parameters\":{{\"k{}\":{{}}}}}}",
                name, t
            ));
        }
        s.push(']');
        v.push(s);
    }
    // random mutations of a real catalogue
    let base = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/tools-extended.json"
    ))
    .unwrap();
    let base: serde_json::Value = serde_json::from_str(&base).unwrap();
    let parts: Vec<String> = base
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.to_string())
        .collect();
    let mini = format!("[{}]", parts.join(","));
    let pool = b"{}[],:\"\\ abn";
    for _ in 0..600 {
        let mut b: Vec<u8> = mini.as_bytes().to_vec();
        for _ in 0..1 + r.below(4) {
            let i = r.below(b.len() as u64 + 1) as usize;
            match r.below(3) {
                0 if i < b.len() => {
                    b.remove(i);
                }
                1 => b.insert(i, pool[r.below(pool.len() as u64) as usize]),
                _ if i < b.len() => b.truncate(i),
                _ => {}
            }
        }
        if let Ok(s) = String::from_utf8(b) {
            v.push(s);
        }
    }
    v
}

fn catalogues() -> Vec<String> {
    let mut v = Vec::new();
    for f in ["tools.json", "tools-extended.json", "suite-tools.json"] {
        let raw =
            std::fs::read_to_string(format!("{}/../../fixtures/{f}", env!("CARGO_MANIFEST_DIR")))
                .unwrap();
        let val: serde_json::Value = serde_json::from_str(&raw).unwrap();
        // the runner's form: each tool's Value::to_string, joined
        let parts: Vec<String> = val
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.to_string())
            .collect();
        v.push(format!("[{}]", parts.join(",")));
        v.push(raw);
    }
    v.push(
        "[{\"name\":\"getWeather\",\"parameters\":{\"location\":{\"type\":\"string\"},\"unit\":{\"type\":\"string\"},\"loc\":{}}},\
          {\"name\":\"get_time\",\"parameters\":{\"type\":\"object\",\"properties\":{\"zone\":{},\"z\":{},\"zone_name\":{}}}},\
          {\"name\":\"get\",\"parameters\":{}},{\"name\":\"t\",\"parameters\":{\"alpha\":{},\"beta\":{}}},\
          {\"name\":\"\",\"parameters\":{\"e\":{}}},{\"name\":\"dup\",\"parameters\":{\"a\":{}}},{\"name\":\"dup\",\"parameters\":{\"b\":{},\"c\":{}}}]"
            .to_string(),
    );
    v.push("[{\"name\":\"t\",\"parameters\":{\"\u{fffd}\":{},\"x\u{fffd}y\":{}}},{\"name\":\"\u{fffd}\",\"parameters\":{\"k\":{}}}]".to_string());
    v
}

fn arg_value(r: &mut Rng) -> String {
    match r.below(9) {
        0 => format!("{}", r.below(100000)),
        1 => format!("-{}", r.below(1000)),
        2 => format!("{}.{}", r.below(100), r.below(100)),
        3 => "\"Berlin\"".to_string(),
        4 => "\"S\u{e3}o Paulo\"".to_string(),
        5 => "\"with \\\"quote\\\" and , comma\"".to_string(),
        6 => "{\"inner\":\"x\",\"n\":[1,2]}".to_string(),
        7 => "[\"a\",{\"b\":\"}\"}]".to_string(),
        _ => ["true", "false", "null", "\"\"", "\"dance\"", "\"nod\""][r.below(6) as usize]
            .to_string(),
    }
}

fn payloads(r: &mut Rng, defs: &[ToolDef]) -> Vec<String> {
    let mut v = Vec::new();
    v.push("[{\"name\":\"start_timer\",\"arguments\":{\"seconds\":450}}]".to_string());
    v.push("[{\"name\":\"play_animation\",\"arguments\":{\"name\":\"dance\"}}]".to_string());
    v.push("[{\"name\":\"set_alarm\",\"arguments\":{\"hour\":7,\"minute\":30}},{\"name\":\"start_timer\",\"arguments\":{\"seconds\":90}}]".to_string());
    v.push("[{\"name\":\"start_timer\",\"arguments\":{\"seconds\":1,\"seconds\":2}}]".to_string());
    v.push("[{\"name\":\"unknown_tool\",\"arguments\":{\"x\":1}}]".to_string());
    v.push(
        "<tool_call>[{\"name\":\"start_timer\",\"arguments\":{\"seconds\":450}}]</tool_call>"
            .to_string(),
    );
    v.push("[{\"name\": \"start_timer\", \"arguments\": {\"seconds\": 450}}]".to_string());
    v.push("[]".to_string());
    for d in defs {
        for _ in 0..3 {
            let mut s = format!("[{{\"name\":\"{}\",\"arguments\":{{", d.snake_name);
            let mut keys = d.param_keys.clone();
            if r.chance(1, 3) && !keys.is_empty() {
                let k = keys[r.below(keys.len() as u64) as usize].clone();
                keys.push(k);
            }
            if r.chance(1, 5) {
                keys.push("bogus_key".to_string());
            }
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&format!("\"{}\":{}", k, arg_value(r)));
            }
            s.push_str("}}");
            if r.chance(1, 3) {
                let d2 = &defs[r.below(defs.len() as u64) as usize];
                s.push_str(&format!(
                    ",{{\"name\":\"{}\",\"arguments\":{{}}}}",
                    d2.snake_name
                ));
            }
            s.push(']');
            v.push(s);
        }
    }
    v
}

struct Out {
    f: std::io::BufWriter<std::fs::File>,
    steps: u64,
}

fn emit_mask(o: &mut Out, dec: &ConstrainedDecoder, vocab: usize) -> Vec<f32> {
    let m = dec.logit_mask(vocab);
    let used: Vec<String> = dec.used_keys().iter().map(|k| hex(k.as_bytes())).collect();
    writeln!(o.f, "{} K {}", mask_line(&m), used.join(",")).unwrap();
    o.steps += 1;
    m
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let outdir = &a[1];
    let seed: u64 = a[2].parse().unwrap();
    let walks: usize = a[3].parse().unwrap();
    let mut r = Rng(seed);
    let cact = CactV3::load(std::env::var("NEEDLE_JIBO_CACT").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../../weights/needle3.cact"
        )
        .into()
    }))
    .unwrap();
    let tok = SpTokenizer::from_blob(cact.tokenizer_blob().unwrap()).unwrap();
    let n = tok.vocab_size();
    eprintln!("vocab {}", n);

    // pieces + types, and the expected byte table
    {
        let mut f =
            std::io::BufWriter::new(std::fs::File::create(format!("{outdir}/pieces.bin")).unwrap());
        f.write_all(&(n as u32).to_le_bytes()).unwrap();
        for id in 0..n as u32 {
            let p = tok.piece(id).unwrap().as_bytes();
            f.write_all(&[tok.piece_type(id).unwrap()]).unwrap();
            f.write_all(&(p.len() as u32).to_le_bytes()).unwrap();
            f.write_all(p).unwrap();
        }
        let tbl = byte_table(&tok);
        let mut f =
            std::io::BufWriter::new(std::fs::File::create(format!("{outdir}/table.bin")).unwrap());
        f.write_all(&(tbl.len() as u32).to_le_bytes()).unwrap();
        for (id, b) in &tbl {
            f.write_all(&id.to_le_bytes()).unwrap();
            f.write_all(&(b.len() as u32).to_le_bytes()).unwrap();
            f.write_all(b).unwrap();
        }
    }

    // parse_tools_json on odd inputs
    {
        let inputs = odd_tool_inputs(&mut r);
        let mut cf = std::io::BufWriter::new(
            std::fs::File::create(format!("{outdir}/tools_in.bin")).unwrap(),
        );
        let mut ef = std::io::BufWriter::new(
            std::fs::File::create(format!("{outdir}/tools_out.txt")).unwrap(),
        );
        let mut all = inputs.clone();
        all.extend(catalogues());
        for s in &all {
            cf.write_all(&(s.len() as u32).to_le_bytes()).unwrap();
            cf.write_all(s.as_bytes()).unwrap();
            writeln!(ef, "{}", tools_line(&ToolDef::from_json(s))).unwrap();
        }
        eprintln!("tool inputs: {}", all.len());
    }

    // decoder scenarios
    let cats = catalogues();
    {
        let mut cf =
            std::io::BufWriter::new(std::fs::File::create(format!("{outdir}/cats.bin")).unwrap());
        for s in &cats {
            cf.write_all(&(s.len() as u32).to_le_bytes()).unwrap();
            cf.write_all(s.as_bytes()).unwrap();
        }
    }
    let table = byte_table(&tok);
    // "structural" tokens to steer random walks through Free state
    let texts: Vec<Vec<u8>> = table.iter().map(|(_, b)| b.clone()).collect();
    let structural: Vec<u32> = (0..n as u32)
        .filter(|&i| {
            let t = &texts[i as usize];
            !t.is_empty()
                && t.len() <= 5
                && t.iter()
                    .all(|c| b"{}[]\":,0123456789_ \\".contains(c) || c.is_ascii_lowercase())
                && t.iter().any(|c| b"{}[]\":,".contains(c))
        })
        .collect();
    eprintln!("structural tokens: {}", structural.len());
    let mut o = Out {
        f: std::io::BufWriter::new(std::fs::File::create(format!("{outdir}/scen.txt")).unwrap()),
        steps: 0,
    };
    let mut nscen = 0;
    for (ci, cat) in cats.iter().enumerate() {
        let defs = ToolDef::from_json(cat);
        let vocab_choices = [n, n + 7, n - 5, 100];
        // 1) real payloads tokenized
        for (pi, p) in payloads(&mut r, &defs).iter().enumerate() {
            let unique = pi % 4 != 3;
            let vocab = vocab_choices[if pi % 9 == 8 { 1 + pi / 9 % 3 } else { 0 }];
            let ids = tok.encode(p);
            let mut dec = ConstrainedDecoder::new(&defs, table.clone());
            if unique {
                dec = dec.with_unique_arg_keys();
            }
            writeln!(o.f, "S {} {} {}", ci, unique as u8, vocab).unwrap();
            nscen += 1;
            emit_mask(&mut o, &dec, vocab);
            for id in ids {
                dec.update(id);
                writeln!(o.f, "U {}", id).unwrap();
                emit_mask(&mut o, &dec, vocab);
            }
            writeln!(o.f, "E").unwrap();
            // same payload fed as raw bytes in random chunks
            let mut dec = ConstrainedDecoder::new(&defs, table.clone());
            if unique {
                dec = dec.with_unique_arg_keys();
            }
            writeln!(o.f, "S {} {} {}", ci, unique as u8, vocab).unwrap();
            nscen += 1;
            let b = p.as_bytes();
            let mut i = 0;
            while i < b.len() {
                let k = (1 + r.below(6) as usize).min(b.len() - i);
                dec.feed_bytes(&b[i..i + k]);
                writeln!(o.f, "B {}", hex(&b[i..i + k])).unwrap();
                emit_mask(&mut o, &dec, vocab);
                i += k;
            }
            writeln!(o.f, "E").unwrap();
        }
        // 2) random walks picking allowed tokens
        let prefixes: Vec<String> = {
            let mut v = vec![String::new(), "[{\"name\":\"".to_string()];
            for d in defs.iter().take(6) {
                v.push(format!(
                    "[{{\"name\":\"{}\",\"arguments\":{{\"",
                    d.snake_name
                ));
            }
            v
        };
        for w in 0..walks {
            let unique = w % 5 != 4;
            let vocab = if w % 11 == 10 { n + 3 } else { n };
            let mut dec = ConstrainedDecoder::new(&defs, table.clone());
            if unique {
                dec = dec.with_unique_arg_keys();
            }
            writeln!(o.f, "S {} {} {}", ci, unique as u8, vocab).unwrap();
            nscen += 1;
            let pre = &prefixes[r.below(prefixes.len() as u64) as usize];
            if !pre.is_empty() {
                dec.feed_bytes(pre.as_bytes());
                writeln!(o.f, "B {}", hex(pre.as_bytes())).unwrap();
            }
            let mut m = emit_mask(&mut o, &dec, vocab);
            let len = 10 + r.below(60);
            for _ in 0..len {
                let allowed: Vec<u32> = m
                    .iter()
                    .enumerate()
                    .filter(|(_, &x)| x == 0.0)
                    .map(|(i, _)| i as u32)
                    .collect();
                let id = if allowed.len() == m.len() {
                    // unconstrained: mostly structural tokens, sometimes anything (or out of range)
                    match r.below(10) {
                        0..=6 => structural[r.below(structural.len() as u64) as usize],
                        7 | 8 => r.below(vocab as u64) as u32,
                        _ => {
                            // a whole trigger as raw bytes
                            let trig = [
                                "\"name\":\"",
                                "\"arguments\":{\"",
                                ",\"",
                                "}",
                                "}]",
                                "\":\"",
                                "\\\"",
                                "{\"",
                                "\"arguments\":{",
                            ];
                            let t = trig[r.below(trig.len() as u64) as usize];
                            dec.feed_bytes(t.as_bytes());
                            writeln!(o.f, "B {}", hex(t.as_bytes())).unwrap();
                            m = emit_mask(&mut o, &dec, vocab);
                            continue;
                        }
                    }
                } else {
                    allowed[r.below(allowed.len() as u64) as usize]
                };
                dec.update(id);
                writeln!(o.f, "U {}", id).unwrap();
                m = emit_mask(&mut o, &dec, vocab);
            }
            writeln!(o.f, "E").unwrap();
        }
        // 3) adversarial byte streams
        for w in 0..walks / 2 {
            let unique = w % 3 != 2;
            let mut dec = ConstrainedDecoder::new(&defs, table.clone());
            if unique {
                dec = dec.with_unique_arg_keys();
            }
            writeln!(o.f, "S {} {} {}", ci, unique as u8, n).unwrap();
            nscen += 1;
            emit_mask(&mut o, &dec, n);
            for _ in 0..20 + r.below(40) {
                let pieces: [&[u8]; 22] = [
                    b"\"name\":\"",
                    b"\"arguments\":{",
                    b"\"arguments\":{\"",
                    b"{\"",
                    b",\"",
                    b"\"",
                    b"}",
                    b"]",
                    b"{",
                    b"[",
                    b":\"",
                    b"\\",
                    b"\\\"",
                    b"\xff",
                    b"\xc3",
                    b"\xa9",
                    b"\xed\xa0\x80",
                    b"\xef\xbf\xbd",
                    b"start_timer",
                    b"seconds",
                    b"t",
                    b"a",
                ];
                let b: Vec<u8> = if r.chance(1, 4) {
                    (0..1 + r.below(4)).map(|_| r.below(256) as u8).collect()
                } else if r.chance(1, 6) {
                    // a declared key or tool name
                    if !defs.is_empty() {
                        let d = &defs[r.below(defs.len() as u64) as usize];
                        if !d.param_keys.is_empty() && r.chance(1, 2) {
                            d.param_keys[r.below(d.param_keys.len() as u64) as usize]
                                .as_bytes()
                                .to_vec()
                        } else {
                            d.snake_name.as_bytes().to_vec()
                        }
                    } else {
                        b"x".to_vec()
                    }
                } else {
                    pieces[r.below(22) as usize].to_vec()
                };
                if r.chance(1, 5) {
                    let id = if r.chance(1, 4) {
                        n as u32 + r.below(10) as u32
                    } else {
                        r.below(n as u64) as u32
                    };
                    dec.update(id);
                    writeln!(o.f, "U {}", id).unwrap();
                } else {
                    dec.feed_bytes(&b);
                    writeln!(o.f, "B {}", hex(&b)).unwrap();
                }
                emit_mask(&mut o, &dec, n);
            }
            writeln!(o.f, "E").unwrap();
        }
    }
    // 4) synthetic byte-per-token vocabularies (mirrors the unit tests) + empty/duplicate tables
    {
        let mut tf =
            std::io::BufWriter::new(std::fs::File::create(format!("{outdir}/synth.txt")).unwrap());
        let defs = ToolDef::from_json(&cats[cats.len() - 2]);
        writeln!(tf, "C {}", cats.len() - 2).unwrap();
        let tables: Vec<Vec<(u32, Vec<u8>)>> = vec![
            (0..128u32).map(|i| (i, vec![i as u8])).collect(),
            vec![],
            vec![
                (5, b"get".to_vec()),
                (2, b"\"".to_vec()),
                (5, b"t\"".to_vec()),
                (9, vec![]),
                (0, b"loc".to_vec()),
            ],
            (0..300u32)
                .map(|i| (i % 200, vec![(i % 128) as u8, b'"']))
                .collect(),
        ];
        for (ti, tb) in tables.iter().enumerate() {
            write!(tf, "TB {}", tb.len()).unwrap();
            for (id, b) in tb {
                write!(tf, " {}:{}", id, hex(b)).unwrap();
            }
            writeln!(tf).unwrap();
            for w in 0..60 {
                let mut dec = ConstrainedDecoder::new(&defs, tb.clone());
                if w % 2 == 0 {
                    dec = dec.with_unique_arg_keys();
                }
                let vocab = [128usize, 0, 1, 10, 300, 3][w % 6];
                writeln!(tf, "S {} {} {}", ti, (w % 2 == 0) as u8, vocab).unwrap();
                let m = dec.logit_mask(vocab);
                writeln!(tf, "{}", mask_line(&m)).unwrap();
                for _ in 0..40 {
                    if r.chance(1, 2) {
                        let id = r.below(310) as u32;
                        dec.update(id);
                        writeln!(tf, "U {}", id).unwrap();
                    } else {
                        let opts: [&[u8]; 10] = [
                            b"[{\"name\":\"",
                            b"get",
                            b"Weather",
                            b"_weather\",\"arguments\":{\"",
                            b"location\":1,\"",
                            b"unit\"",
                            b"\":\"x\",\"",
                            b"}}]",
                            b"t\",\"arguments\":{\"alpha\":\"x\",\"",
                            b"dup\",\"arguments\":{\"",
                        ];
                        let b = opts[r.below(10) as usize];
                        dec.feed_bytes(b);
                        writeln!(tf, "B {}", hex(b)).unwrap();
                    }
                    let m = dec.logit_mask(vocab);
                    writeln!(tf, "{}", mask_line(&m)).unwrap();
                }
                writeln!(tf, "E").unwrap();
            }
        }
    }
    eprintln!("scenarios {} mask steps {}", nscen, o.steps);
}
