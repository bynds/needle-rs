// Reference dumper for the C port of the Needle 3 tokenizer / prompt assembly.
// Usage: tokref <container.cact> <repo root> <out.bin>
use needle_infer::cact::CactV3;
use needle_infer::prompt::{build_prompt, compact_json};
use needle_infer::sp_tokenizer::{SpTokenizer, TokenizerError, TK_BYTE};
use needle_infer::tokenizer::to_snake_case;
use std::io::Write;
use std::path::Path;

struct W(Vec<u8>);
impl W {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes(&mut self, b: &[u8]) {
        self.u32(b.len() as u32);
        self.0.extend_from_slice(b);
    }
    fn ids(&mut self, ids: &[u32]) {
        self.u32(ids.len() as u32);
        for &i in ids {
            self.u32(i);
        }
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn walk(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let mut ents: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    ents.sort_by_key(|e| e.path());
    for e in ents {
        let p = e.path();
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if p.is_dir() {
            if name == "target" || name == ".git" || name.starts_with('.') {
                continue;
            }
            walk(&p, out);
        } else if name.ends_with(".md") || name.ends_with(".rs") || name.ends_with(".py") {
            out.push(p);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let c = CactV3::load(&args[1]).expect("load cact");
    let blob = c.tokenizer_blob().expect("no tokenizer");
    let tok = SpTokenizer::from_blob(blob).expect("tokenizer");
    let root = Path::new(&args[2]);
    let fixtures = root.join("ports/jibo/fixtures");
    let mut w = W(Vec::new());
    w.0.extend_from_slice(b"NDTOKREF1\0");

    let n = tok.vocab_size() as u32;
    w.u8(1);
    for v in [n, tok.pad_id, tok.eos_id, tok.bos_id, tok.unk_id] {
        w.u32(v);
    }
    w.u8(tok.add_dummy_prefix as u8);
    w.u8(tok.byte_fallback as u8);

    let mut byte_id = [u32::MAX; 256];
    let tb = tok.token_bytes();
    for id in 0..n {
        let p = tok.piece(id).unwrap();
        w.u8(2);
        w.u32(id);
        w.bytes(p.as_bytes());
        w.u8(tok.piece_type(id).unwrap());
        w.u32(tok.score(id).unwrap().to_bits());
        w.bytes(&tb[id as usize]);
        if tok.piece_type(id) == Some(TK_BYTE) {
            byte_id[tb[id as usize][0] as usize] = id;
        }
        // id_of of every surface (later duplicates win)
        w.u8(8);
        w.bytes(p.as_bytes());
        w.i64(tok.id_of(p).map_or(-1, |v| v as i64));
        // single-id decode
        w.u8(4);
        w.ids(&[id]);
        w.bytes(tok.decode(&[id]).as_bytes());
    }
    for s in [
        "",
        "nope-not-a-piece",
        "\u{2581}\u{2581}\u{2581}\u{2581}\u{2581}\u{2581}\u{2581}\u{2581}x",
        "<tool_call",
        "\0",
    ] {
        w.u8(8);
        w.bytes(s.as_bytes());
        w.i64(tok.id_of(s).map_or(-1, |v| v as i64));
    }

    let mut n_enc = 0usize;
    let mut enc = |w: &mut W, s: &str| {
        let ids = tok.encode(s);
        w.u8(3);
        w.bytes(s.as_bytes());
        w.ids(&ids);
        w.bytes(tok.decode(&ids).as_bytes());
        n_enc += 1;
    };

    // Queries.
    let mut queries: Vec<String> = Vec::new();
    let mut jl: Vec<_> = std::fs::read_dir(&fixtures)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |e| e == "jsonl"))
        .collect();
    jl.sort();
    for p in &jl {
        let txt = std::fs::read_to_string(p).unwrap();
        for line in txt.lines() {
            enc(&mut w, line);
            if line.trim().is_empty() {
                continue;
            }
            // c-parity-edge.jsonl holds deliberately malformed lines; they are encoded above.
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            if let Some(q) = v.get("query").and_then(|q| q.as_str()) {
                queries.push(q.to_string());
            }
        }
    }
    eprintln!("queries: {}", queries.len());
    for q in &queries {
        enc(&mut w, q);
    }

    // Prompts.
    let tools_files = ["tools.json", "tools-extended.json", "suite-tools.json"];
    let systems: [Option<&str>; 3] = [
        None,
        Some("You are Jibo, a friendly social robot. Call a tool only when the user asks for it."),
        Some(""),
    ];
    let mut n_prompt = 0;
    for tf in tools_files {
        let tools = std::fs::read_to_string(fixtures.join(tf)).unwrap();
        enc(&mut w, &tools);
        w.u8(6);
        w.bytes(tools.as_bytes());
        w.bytes(compact_json(&tools).as_bytes());
        enc(&mut w, &compact_json(&tools));
        // tool names -> snake case
        let v: serde_json::Value = serde_json::from_str(&tools).unwrap();
        if let Some(a) = v.as_array() {
            for t in a {
                if let Some(nm) = t.get("name").and_then(|x| x.as_str()) {
                    w.u8(7);
                    w.bytes(nm.as_bytes());
                    w.bytes(to_snake_case(nm).as_bytes());
                }
            }
        }
        for (si, sys) in systems.iter().enumerate() {
            for (qi, q) in queries.iter().enumerate() {
                // The empty system prompt only on a subset, to bound run time.
                if si == 2 && qi % 8 != 0 {
                    continue;
                }
                let p = build_prompt(q, &tools, *sys);
                w.u8(5);
                w.bytes(q.as_bytes());
                w.bytes(tools.as_bytes());
                w.u8(sys.is_some() as u8);
                w.bytes(sys.unwrap_or("").as_bytes());
                w.bytes(p.as_bytes());
                enc(&mut w, &p);
                n_prompt += 1;
            }
        }
    }
    eprintln!("prompts: {n_prompt}");

    // Repo lines.
    let mut files = Vec::new();
    walk(root, &mut files);
    let mut words: std::collections::BTreeSet<String> = Default::default();
    let mut n_lines = 0;
    for f in &files {
        let Ok(txt) = std::fs::read_to_string(f) else {
            continue;
        };
        for line in txt.lines() {
            enc(&mut w, line);
            n_lines += 1;
            for wd in line
                .split(|c: char| c.is_whitespace() || c == '(' || c == ')' || c == ',' || c == ';')
            {
                if !wd.is_empty() && wd.len() < 64 {
                    words.insert(wd.to_string());
                }
            }
        }
        // Some multi-line chunks too.
        let ls: Vec<&str> = txt.lines().collect();
        for ch in ls.chunks(7).take(20) {
            enc(&mut w, &ch.join("\n"));
        }
    }
    eprintln!(
        "files: {} lines: {n_lines} words: {}",
        files.len(),
        words.len()
    );
    for wd in &words {
        w.u8(7);
        w.bytes(wd.as_bytes());
        w.bytes(to_snake_case(wd).as_bytes());
    }

    // Adversarial strings.
    let mut adv: Vec<String> = vec![
        "".into(),
        " ".into(),
        "  ".into(),
        "\u{2581}".into(),
        " \u{2581} ".into(),
        "a".into(),
        "<tool_call><tool_call>".into(),
        "<tool_call></tool_call><tool_call>".into(),
        "text <|im_end|> inside".into(),
        "<|im_end|><|im_start|>".into(),
        "<|im_start".into(),
        "<<tool_call>>".into(),
        "</tool_call>x</tool_call".into(),
        "<think></think><tools></tools>".into(),
        "<tool_result>{}</tool_result>".into(),
        "<s></s><pad><unk>".into(),
        "<0x41><0xFF>".into(),
        "Hello 👋 world 🌍🚀👨‍👩‍👧‍👦 🏳️‍🌈".into(),
        "e\u{301}e\u{301}\u{302}\u{303}a\u{20DD}".into(),
        "Z̤͔ͧ̑̓ä͖̭̈̇lͮ̒ͫǧ̗͚̚o̙̔ͮ̇͐̇".into(),
        "日本語のテキストと中文字符和한국어".into(),
        "مرحبا بالعالم שלום עולם".into(),
        "\u{202E}abc\u{202C}\u{200F}\u{200E}".into(),
        "\t\n\r\x01\x02\x07\x1b[0m\x7f".into(),
        "\u{feff}BOM\u{200b}zw\u{2060}".into(),
        "\u{10FFFF}\u{E000}\u{FFFD}\u{FFFE}\u{1F600}".into(),
        "a\u{0}b".into(),
        "set_timer(seconds=90)".into(),
        "{\"name\":\"start_timer\",\"arguments\":{\"seconds\":450}}".into(),
        "   leading and trailing   ".into(),
        "multiple    spaces\there".into(),
    ];
    for k in [100usize, 1000, 3000] {
        adv.push("a".repeat(k));
        adv.push("ab".repeat(k / 2));
        adv.push("supercalifragilistic".repeat(k / 20));
        adv.push("😀".repeat(k / 4));
        adv.push(" ".repeat(k / 4));
        adv.push("<tool_call>".repeat(k / 10));
        adv.push("日本".repeat(k / 8));
    }
    for s in &adv {
        enc(&mut w, s);
        w.u8(7);
        w.bytes(s.as_bytes());
        w.bytes(to_snake_case(s).as_bytes());
        w.u8(6);
        w.bytes(s.as_bytes());
        w.bytes(compact_json(s).as_bytes());
    }
    for s in [
        "{ \"a\" : \"x y\\\" z\" , \"b\":[1, 2 ,\t3]\n}",
        "\"unterminated  string",
        "\\\"  x  \"",
        "  \"\\\\\" a b ",
        "{\"k\":\"\\u0020 \\n\"}  \r\n",
        "  ",
        "",
        "\"é ü\"  ü é",
    ] {
        w.u8(6);
        w.bytes(s.as_bytes());
        w.bytes(compact_json(s).as_bytes());
    }
    for s in [
        "getWeather",
        "GetWeather",
        "get-weather",
        "get.weather",
        "HTMLParser",
        "already_snake",
        "__x__",
        "A",
        "AB",
        "ABc",
        "aB1C",
        "x1Y",
        "_",
        "ÉtatCivil",
        "naïveÀB",
        "ΣίσυφοςΑΒγ",
        "ǅungla",
        "Ⅻroman",
        "x²y",
        "١٢٣Arabic",
        "ﬃLigature",
        "ⓐⒶb",
    ] {
        w.u8(7);
        w.bytes(s.as_bytes());
        w.bytes(to_snake_case(s).as_bytes());
    }

    // Every Unicode scalar, in chunks, through encode and through to_snake_case.
    let mut all: Vec<char> = (0u32..=0x10FFFF).filter_map(char::from_u32).collect();
    let mut chunk = String::new();
    let mut snake = String::new();
    for (i, ch) in all.iter().enumerate() {
        chunk.push(*ch);
        if i % 3 == 0 {
            chunk.push(' ');
        }
        for part in [
            ch.to_string(),
            "Ab".into(),
            ch.to_string(),
            "a".into(),
            ch.to_string(),
            "Aa".into(),
            ch.to_string(),
            "B-".into(),
            ch.to_string(),
            "_9".into(),
            ch.to_string(),
            "C".into(),
        ] {
            snake.push_str(&part);
        }
        if chunk.chars().count() >= 200 {
            enc(&mut w, &chunk);
            chunk.clear();
        }
        if (i + 1) % 32 == 0 {
            w.u8(7);
            w.bytes(snake.as_bytes());
            w.bytes(to_snake_case(&snake).as_bytes());
            snake.clear();
        }
    }
    enc(&mut w, &chunk);
    w.u8(7);
    w.bytes(snake.as_bytes());
    w.bytes(to_snake_case(&snake).as_bytes());
    all.clear();

    // Random strings drawn from the vocabulary's own pieces, spaces and markers.
    let mut rng = Rng(0x9E3779B97F4A7C15);
    for _ in 0..3000 {
        let mut s = String::new();
        let k = 1 + rng.below(30);
        for _ in 0..k {
            match rng.below(6) {
                0 => s.push(' '),
                1 => s.push_str(
                    [
                        "<tool_call>",
                        "</tool_call>",
                        "<|im_end|>",
                        "<tools>",
                        "<",
                        ">",
                        "|",
                    ][rng.below(7) as usize],
                ),
                _ => {
                    let id = rng.below(n as u64) as u32;
                    s.push_str(tok.piece(id).unwrap());
                }
            }
        }
        enc(&mut w, &s);
    }

    // Random id sequences.
    let mut n_dec = 0;
    let mut dec = |w: &mut W, ids: &[u32]| {
        w.u8(4);
        w.ids(ids);
        w.bytes(tok.decode(ids).as_bytes());
        n_dec += 1;
    };
    for _ in 0..30000 {
        let k = 1 + rng.below(40) as usize;
        let mut ids = Vec::with_capacity(k);
        for _ in 0..k {
            let r = rng.below(100);
            let id = if r < 45 {
                byte_id[rng.below(256) as usize]
            } else if r < 60 {
                byte_id[0x80 + rng.below(0x80) as usize]
            } else if r < 95 {
                rng.below(n as u64) as u32
            } else if r < 98 {
                n + rng.below(10) as u32
            } else {
                u32::MAX - rng.below(3) as u32
            };
            ids.push(id);
        }
        dec(&mut w, &ids);
    }
    // Exhaustive two-byte sequences, and systematic three/four-byte ones.
    for a in 0..256 {
        for b in 0..256 {
            dec(&mut w, &[byte_id[a], byte_id[b]]);
        }
    }
    let leads = [
        0x41u8, 0x80, 0xBF, 0xC0, 0xC1, 0xC2, 0xDF, 0xE0, 0xE1, 0xEC, 0xED, 0xEE, 0xEF, 0xF0, 0xF1,
        0xF3, 0xF4, 0xF5, 0xF8, 0xFF,
    ];
    let conts = [
        0x00u8, 0x41, 0x7F, 0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xBF, 0xC0, 0xC2, 0xE0, 0xF0, 0xFF,
    ];
    for &a in &leads {
        for b in 0..256usize {
            for &c3 in &conts {
                dec(
                    &mut w,
                    &[byte_id[a as usize], byte_id[b], byte_id[c3 as usize]],
                );
                for &d in &conts {
                    dec(
                        &mut w,
                        &[
                            byte_id[a as usize],
                            byte_id[b],
                            byte_id[c3 as usize],
                            byte_id[d as usize],
                        ],
                    );
                }
            }
        }
    }
    // Meta-space edge cases: a byte-assembled U+2581 and the dummy-prefix strip.
    let sp = tok.id_of("\u{2581}").unwrap_or(0);
    for ids in [
        vec![byte_id[0xE2], byte_id[0x96], byte_id[0x81]],
        vec![byte_id[0xE2], byte_id[0x96]],
        vec![byte_id[0x20], byte_id[0x20]],
        vec![sp, sp],
        vec![byte_id[0xE2], byte_id[0x96], byte_id[0x81], sp],
        vec![tok.bos_id, sp, byte_id[0x41]],
        vec![],
        vec![byte_id[0xFF], sp],
    ] {
        dec(&mut w, &ids);
    }
    eprintln!("encode records: {n_enc} decode records: {n_dec}");

    // ---- Mutated and crafted blobs -------------------------------------------------------
    let mut sample: Vec<String> = queries.iter().take(40).cloned().collect();
    sample.extend(adv.iter().filter(|s| s.len() < 400).cloned());
    sample.extend(
        [
            "abc abc",
            "ab",
            "a b",
            "zz\u{20AC}",
            "<tool_call>x<tool>y",
            " a",
            "a ",
            "\u{2581}a",
            "abcabc",
            "ba",
            "ccc",
        ]
        .iter()
        .map(|s| s.to_string()),
    );
    let blob_owned = blob.to_vec();
    let mut n_blob = 0;
    // flags: (add_dummy, byte_fallback)
    let mut muts: Vec<(u32, Vec<(u32, u8)>, usize)> = Vec::new();
    for (d, b) in [(1u8, 1u8), (1, 0), (0, 0)] {
        muts.push((blob_owned.len() as u32, vec![(20, d), (21, b)], 400));
    }
    for t in [
        0usize,
        1,
        4,
        20,
        23,
        24,
        25,
        30,
        31,
        32,
        100,
        1000,
        blob_owned.len() / 2,
        blob_owned.len() - 2,
        blob_owned.len() - 1,
    ] {
        muts.push((t as u32, vec![], 20));
    }
    // n_pieces edits, special-id edits.
    for v in [
        0u32,
        1,
        2,
        3,
        4,
        100,
        8191,
        8193,
        15589,
        15590,
        0x7FFF_FFFF,
        u32::MAX,
    ] {
        let b = v.to_le_bytes();
        muts.push((
            blob_owned.len() as u32,
            (0..4).map(|k| (k as u32, b[k])).collect(),
            20,
        ));
    }
    for (field, v) in [
        (4u32, 8191u32),
        (4, 8192),
        (8, 9000),
        (12, u32::MAX),
        (16, 8192),
        (16, 8191),
    ] {
        let b = v.to_le_bytes();
        muts.push((
            blob_owned.len() as u32,
            (0..4).map(|k| (field + k as u32, b[k])).collect(),
            20,
        ));
    }
    for _ in 0..300 {
        let np = 1 + rng.below(3) as usize;
        let mut ps = Vec::new();
        for _ in 0..np {
            let off = if rng.below(2) == 0 {
                rng.below(3000)
            } else {
                rng.below(blob_owned.len() as u64)
            } as u32;
            ps.push((off, rng.below(256) as u8));
        }
        let t = if rng.below(4) == 0 {
            rng.below(blob_owned.len() as u64 + 1) as u32
        } else {
            blob_owned.len() as u32
        };
        muts.push((t, ps, 20));
    }
    fn sub(w: &mut W, t: &SpTokenizer, sample: &[String], rng: &mut Rng, ndec: usize) {
        let n = t.vocab_size() as u64;
        let loops = (0..n as u32).any(|i| t.piece_type(i) == Some(3) && t.piece(i) == Some(""));
        let nenc = if loops { 0 } else { sample.len() };
        w.u32((nenc + ndec) as u32);
        for s in sample.iter().take(nenc) {
            let ids = t.encode(s);
            w.u8(3);
            w.bytes(s.as_bytes());
            w.ids(&ids);
            w.bytes(t.decode(&ids).as_bytes());
        }
        for _ in 0..ndec {
            let k = 1 + rng.below(12) as usize;
            let ids: Vec<u32> = (0..k).map(|_| rng.below(n + 3) as u32).collect();
            w.u8(4);
            w.ids(&ids);
            w.bytes(t.decode(&ids).as_bytes());
        }
    }
    fn status(r: &Result<SpTokenizer, TokenizerError>) -> (u8, String) {
        match r {
            Ok(_) => (0, String::new()),
            Err(e) => (
                match e {
                    TokenizerError::Truncated { .. } => 1,
                    TokenizerError::InvalidUtf8 { .. } => 2,
                    TokenizerError::Empty => 3,
                    TokenizerError::SpecialIdOutOfRange { .. } => 4,
                    TokenizerError::MalformedBytePiece { .. } => 5,
                },
                e.to_string(),
            ),
        }
    }
    let mut n_ok = 0;
    for (t, ps, ndec) in &muts {
        let mut b = blob_owned.clone();
        for &(o, v) in ps {
            if (o as usize) < b.len() {
                b[o as usize] = v;
            }
        }
        b.truncate(*t as usize);
        let r = SpTokenizer::from_blob(&b);
        let (st, msg) = status(&r);
        w.u8(9);
        w.u32(*t);
        w.u32(ps.len() as u32);
        for &(o, v) in ps {
            w.u32(o);
            w.u8(v);
        }
        w.u8(st);
        w.bytes(msg.as_bytes());
        if let Ok(tk) = &r {
            n_ok += 1;
            sub(&mut w, tk, &sample, &mut rng, *ndec);
        }
        n_blob += 1;
    }
    eprintln!("mutated blobs: {n_blob} (accepted {n_ok})");

    // Crafted blobs.
    fn r(v: &[(String, f32, u8)]) -> Vec<(&str, f32, u8)> {
        v.iter().map(|(a, b, c)| (a.as_str(), *b, *c)).collect()
    }
    fn mk(
        pieces: &[(&str, f32, u8)],
        add_dummy: bool,
        bf: bool,
        sp: (u32, u32, u32, u32),
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(pieces.len() as u32).to_le_bytes());
        for v in [sp.0, sp.1, sp.2, sp.3] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.push(add_dummy as u8);
        out.push(bf as u8);
        out.extend_from_slice(&[0, 0]);
        for (s, sc, ty) in pieces {
            out.extend_from_slice(&sc.to_le_bytes());
            out.push(*ty);
            out.extend_from_slice(&(s.len() as u16).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        out
    }
    let hex: Vec<String> = (0..256).map(|b| format!("<0x{b:02X}>")).collect();
    let base: Vec<(&str, f32, u8)> = vec![
        ("<pad>", 0.0, 2),
        ("</s>", 0.0, 2),
        ("<s>", 0.0, 2),
        ("<unk>", 0.0, 1),
        ("<tool_call>", 0.0, 3),
        ("<tool>", 0.0, 3),
        ("\u{2581}", -1.0, 0),
        ("a", -2.0, 0),
        ("b", -3.0, 0),
        ("c", -4.0, 0),
        ("ab", -0.5, 0),
        ("abc", -0.25, 0),
        ("\u{2581}a", -0.75, 0),
    ];
    let mut crafted: Vec<Vec<u8>> = Vec::new();
    let with_bytes =
        |mut p: Vec<(&'static str, f32, u8)>, hex: &[String]| -> Vec<(String, f32, u8)> {
            let mut v: Vec<(String, f32, u8)> =
                p.drain(..).map(|(a, b, c)| (a.to_string(), b, c)).collect();
            for h in hex {
                v.push((h.clone(), -50.0, 4));
            }
            v
        };
    let conv = |v: &Vec<(String, f32, u8)>| -> Vec<(String, f32, u8)> { v.clone() };
    let toy = with_bytes(base.clone(), &hex);
    for (d, bf) in [(true, true), (true, false), (false, true), (false, false)] {
        crafted.push(mk(&r(&toy), d, bf, (0, 1, 2, 3)));
    }
    // Ties, NaN, -inf, duplicates (later wins), equal-length markers, prefix markers.
    let mut v = conv(&toy);
    v.push(("ab".into(), -0.25, 0)); // duplicate surface: later id wins
    v.push(("bc".into(), -0.5, 0));
    v.push(("ca".into(), -0.5, 0));
    v.push(("aa".into(), f32::NAN, 0));
    v.push(("bb".into(), f32::NEG_INFINITY, 0));
    v.push(("cc".into(), -0.5, 0));
    v.push(("<0x41>".into(), -60.0, 4));
    v.push(("<ab>".into(), 0.0, 3));
    v.push(("<ba>".into(), 0.0, 3));
    v.push(("<a".into(), 0.0, 3));
    v.push(("<tool_call>".into(), 0.0, 3));
    v.push(("\u{2581}\u{2581}".into(), -0.1, 0));
    v.push(("\u{e9}".into(), -1.0, 0));
    v.push(("\u{e9}\u{e9}".into(), -0.1, 0));
    v.push(("a".into(), 5.0, 2)); // a control piece shadowing "a"
    v.push(("<0x+F>".into(), 0.0, 4));
    v.push(("<0xab>".into(), 0.0, 4));
    v.push(("xxx+a".into(), 0.0, 4));
    v.push(("<0x0F".into(), 0.0, 4));
    crafted.push(mk(&r(&v), true, true, (0, 1, 2, 3)));
    crafted.push(mk(&r(&v), false, true, (0, 1, 2, 3)));
    // Errors.
    for bad in [
        "<0x-1>",
        "<0x",
        "<0xG0>",
        "<0x\u{e9}>",
        "<0x+>",
        "<0x 1>",
        "<0x1\u{e9}",
    ] {
        let mut v = conv(&toy);
        v.push((bad.into(), 0.0, 4));
        crafted.push(mk(&r(&v), true, true, (0, 1, 2, 3)));
    }
    crafted.push(mk(&r(&toy), true, true, (0, 1, 2, 9999)));
    crafted.push(mk(&r(&toy), true, true, (269, 1, 2, 3)));
    crafted.push(mk(&[], true, true, (0, 0, 0, 0)));
    crafted.push(mk(&[("a", 0.0, 0)], true, false, (0, 0, 0, 0)));
    crafted.push(mk(&[("", 0.0, 0), ("", 1.0, 1)], true, false, (1, 1, 1, 1)));
    {
        let mut b = mk(&[("a", 0.0, 0)], true, false, (0, 0, 0, 0));
        let l = b.len();
        b[l - 1] = 0xFF;
        crafted.push(b);
    }
    {
        let mut b = mk(&[("ab", 0.0, 0)], true, false, (0, 0, 0, 0));
        let l = b.len();
        b[l - 1] = 0xC3;
        crafted.push(b.clone());
        b[l - 2] = 0xC3;
        b[l - 1] = 0xA9;
        crafted.push(b);
    }
    {
        let b = mk(&[("a", 0.0, 0)], true, false, (0, 0, 0, 0));
        crafted.push(b[..b.len() - 1].to_vec());
        crafted.push(b[..10].to_vec());
    }
    {
        let mut b = mk(&[("a", 0.0, 0)], true, false, (0, 0, 0, 0));
        b[0] = 2;
        crafted.push(b);
    }
    {
        let mut b = mk(&[("a", 0.0, 0), ("b", 0.0, 0)], true, false, (0, 0, 0, 0));
        b.extend_from_slice(&[1, 2, 3]);
        crafted.push(b);
    }
    for b in &crafted {
        let r = SpTokenizer::from_blob(b);
        let (st, msg) = status(&r);
        w.u8(10);
        w.bytes(b);
        w.u8(st);
        w.bytes(msg.as_bytes());
        if let Ok(tk) = &r {
            sub(&mut w, tk, &sample, &mut rng, 400);
        }
    }
    eprintln!("crafted blobs: {}", crafted.len());

    w.u8(0);
    let mut f = std::fs::File::create(&args[3]).unwrap();
    f.write_all(&w.0).unwrap();
    eprintln!("wrote {} bytes", w.0.len());
}
