// Reference generator for ports/jibo/c/tests/test_json.c (throwaway; serde_json 1.0.149).
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};
use std::fmt;
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
    fn chance(&mut self, num: u64, den: u64) -> bool {
        self.below(den) < num
    }
}

fn fnv(s: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in s {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ─── Ordered DOM via a custom visitor (keeps order and duplicates) ───

enum Ord_ {
    Null,
    Bool(bool),
    U(u64),
    I(i64),
    F(f64),
    S(String),
    A(Vec<Ord_>),
    O(Vec<(String, Ord_)>),
}

impl<'de> Deserialize<'de> for Ord_ {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Ord_;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("json")
            }
            fn visit_unit<E>(self) -> Result<Ord_, E> {
                Ok(Ord_::Null)
            }
            fn visit_bool<E>(self, b: bool) -> Result<Ord_, E> {
                Ok(Ord_::Bool(b))
            }
            fn visit_u64<E>(self, v: u64) -> Result<Ord_, E> {
                Ok(Ord_::U(v))
            }
            fn visit_i64<E>(self, v: i64) -> Result<Ord_, E> {
                Ok(Ord_::I(v))
            }
            fn visit_f64<E>(self, v: f64) -> Result<Ord_, E> {
                Ok(Ord_::F(v))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<Ord_, E> {
                Ok(Ord_::S(v.to_string()))
            }
            fn visit_string<E>(self, v: String) -> Result<Ord_, E> {
                Ok(Ord_::S(v))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Ord_, A::Error> {
                let mut v = Vec::new();
                while let Some(x) = a.next_element()? {
                    v.push(x);
                }
                Ok(Ord_::A(v))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Ord_, A::Error> {
                let mut v = Vec::new();
                while let Some(k) = a.next_key::<String>()? {
                    let x = a.next_value()?;
                    v.push((k, x));
                }
                Ok(Ord_::O(v))
            }
        }
        d.deserialize_any(V)
    }
}

impl Serialize for Ord_ {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Ord_::Null => s.serialize_unit(),
            Ord_::Bool(b) => s.serialize_bool(*b),
            Ord_::U(u) => s.serialize_u64(*u),
            Ord_::I(i) => s.serialize_i64(*i),
            Ord_::F(f) => s.serialize_f64(*f),
            Ord_::S(x) => s.serialize_str(x),
            Ord_::A(v) => {
                let mut q = s.serialize_seq(Some(v.len()))?;
                for x in v {
                    q.serialize_element(x)?;
                }
                q.end()
            }
            Ord_::O(v) => {
                let mut m = s.serialize_map(Some(v.len()))?;
                for (k, x) in v {
                    m.serialize_entry(k, x)?;
                }
                m.end()
            }
        }
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

fn dump(v: &Ord_, out: &mut String) {
    match v {
        Ord_::Null => out.push('n'),
        Ord_::Bool(b) => out.push(if *b { 't' } else { 'f' }),
        Ord_::U(u) => out.push_str(&format!("U{}", u)),
        Ord_::I(i) => out.push_str(&format!("I{}", i)),
        Ord_::F(f) => out.push_str(&format!("F{:016x}", f.to_bits())),
        Ord_::S(s) => {
            out.push('s');
            out.push_str(&hex(s.as_bytes()));
        }
        Ord_::A(v) => {
            out.push('[');
            for (i, x) in v.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                dump(x, out);
            }
            out.push(']');
        }
        Ord_::O(v) => {
            out.push('{');
            for (i, (k, x)) in v.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&hex(k.as_bytes()));
                out.push(':');
                dump(x, out);
            }
            out.push('}');
        }
    }
}

// ─── Document generator ───

const SPECIAL_NUMS: &[&str] = &[
    "18446744073709551615",
    "18446744073709551616",
    "18446744073709551614",
    "-9223372036854775808",
    "-9223372036854775809",
    "9223372036854775807",
    "9223372036854775808",
    "1e308",
    "1.7976931348623157e308",
    "1.7976931348623158e308",
    "1.7976931348623159e308",
    "1e309",
    "-1e309",
    "1e400",
    "1e-400",
    "-1e-400",
    "0e999999999999",
    "0e-999999999999",
    "1e-999999999999",
    "1e999999999999",
    "0.000000000000000000000000000000000000001e-300",
    "4.9e-324",
    "2.5e-324",
    "2.4e-324",
    "2.2250738585072014e-308",
    "2.2250738585072011e-308",
    "123456789012345678901234567890",
    "-123456789012345678901234567890",
    "123456789012345678901234567890.123456789e-10",
    "0.1",
    "0.2",
    "0.3",
    "1.0000000000000002",
    "9007199254740993",
    "-9007199254740993.0",
    "3.141592653589793238462643383279",
    "1e2147483647",
    "1e2147483648",
    "1e-2147483648",
    "1e-2147483649",
    "0.1e2147483647",
    "10e2147483647",
    "1.5e-2147483647",
    "12345678901234567890e-2147483647",
    "123456789012345678901234567890e-2147483647",
    "123456789012345678901234567890e2147483647",
    "0.12345678901234567890123e-2147483640",
    "-0e0",
    "0E-0",
    "1e00000000000000000000000000000000000001",
    "18446744073709551615.5",
    "18446744073709551616e-1",
    "1844674407370955161.5e1",
    "0.18446744073709551616",
];

fn gen_number(r: &mut Rng) -> String {
    let mut s = String::new();
    match r.below(12) {
        0 => {
            return [
                "0", "-0", "1", "-1", "0.0", "-0.0", "1e3", "5.0", "1E3", "1e+3", "1e-3",
            ][r.below(11) as usize]
                .to_string()
        }
        1 => return format!("{}", r.next()),
        2 => return format!("{}", r.next() as i64),
        3 => return SPECIAL_NUMS[r.below(SPECIAL_NUMS.len() as u64) as usize].to_string(),
        4 => {
            let n = 290 + r.below(40) as usize;
            let mut s = String::from(if r.chance(1, 2) { "1" } else { "9" });
            s.push_str(&"0".repeat(n));
            if r.chance(1, 2) {
                return format!("0.{}1", "0".repeat(n));
            }
            return s;
        }
        _ => {}
    }
    if r.chance(1, 3) {
        s.push('-');
    }
    let lim = if r.chance(1, 4) { 30 } else { 8 };
    let nint = 1 + r.below(lim);
    if r.chance(1, 5) {
        s.push('0');
    } else {
        s.push((b'1' + r.below(9) as u8) as char);
        for _ in 1..nint {
            s.push((b'0' + r.below(10) as u8) as char);
        }
    }
    if r.chance(1, 2) {
        s.push('.');
        let lim = if r.chance(1, 4) { 30 } else { 8 };
        let nf = 1 + r.below(lim);
        for _ in 0..nf {
            s.push((b'0' + r.below(10) as u8) as char);
        }
    }
    if r.chance(1, 2) {
        s.push(if r.chance(1, 2) { 'e' } else { 'E' });
        match r.below(3) {
            0 => s.push('+'),
            1 => s.push('-'),
            _ => {}
        }
        let lim = if r.chance(1, 8) { 12 } else { 3 };
        let ne = 1 + r.below(lim);
        for _ in 0..ne {
            s.push((b'0' + r.below(10) as u8) as char);
        }
    }
    s
}

fn gen_string(r: &mut Rng) -> String {
    let mut s = String::from("\"");
    let lim = if r.chance(1, 8) { 40 } else { 8 };
    let n = r.below(lim);
    for _ in 0..n {
        match r.below(14) {
            0 => {
                let e = ["\\\"", "\\\\", "\\/", "\\b", "\\f", "\\n", "\\r", "\\t"];
                s.push_str(e[r.below(8) as usize]);
            }
            1 => {
                let cp = r.below(0x10000) as u32;
                if (0xD800..0xE000).contains(&cp) && r.chance(3, 4) {
                    let hi = 0xD800 + r.below(0x400) as u32;
                    let lo = 0xDC00 + r.below(0x400) as u32;
                    if r.chance(1, 2) {
                        s.push_str(&format!("\\u{:04x}\\u{:04X}", hi, lo));
                    } else {
                        s.push_str(&format!("\\u{:04X}\\u{:04x}", hi, lo));
                    }
                } else {
                    s.push_str(&format!("\\u{:04x}", cp));
                }
            }
            2 => s.push_str(&format!("\\u{:04x}", r.below(0x20))),
            3 => s.push(char::from_u32(0x80 + r.below(0x700) as u32).unwrap_or('x')),
            4 => s.push(char::from_u32(0x10000 + r.below(0x100000) as u32).unwrap_or('y')),
            5 => s.push(char::from_u32(0x800 + r.below(0xD000) as u32).unwrap_or('z')),
            6 => s.push('\u{7f}'),
            _ => {
                let c = (b' ' + r.below(95) as u8) as char;
                match c {
                    '"' => s.push_str("\\\""),
                    '\\' => s.push_str("\\\\"),
                    _ => s.push(c),
                }
            }
        }
    }
    s.push('"');
    s
}

fn ws(r: &mut Rng, s: &mut String) {
    if r.chance(1, 4) {
        let w = [" ", "\n", "\t", "\r", "  ", " \n "];
        s.push_str(w[r.below(6) as usize]);
    }
}

fn gen_value(r: &mut Rng, depth: u32, s: &mut String, keys: &[String]) {
    let k = if depth > 6 { r.below(6) } else { r.below(9) };
    match k {
        0 => s.push_str("null"),
        1 => s.push_str(if r.chance(1, 2) { "true" } else { "false" }),
        2 | 3 => s.push_str(&gen_number(r)),
        4 | 5 => s.push_str(&gen_string(r)),
        6 | 7 => {
            s.push('[');
            ws(r, s);
            let n = r.below(5);
            for i in 0..n {
                if i > 0 {
                    s.push(',');
                    ws(r, s);
                }
                gen_value(r, depth + 1, s, keys);
                ws(r, s);
            }
            s.push(']');
        }
        _ => {
            s.push('{');
            ws(r, s);
            let n = r.below(6);
            for i in 0..n {
                if i > 0 {
                    s.push(',');
                    ws(r, s);
                }
                if r.chance(1, 2) {
                    s.push_str(&keys[r.below(keys.len() as u64) as usize]);
                } else {
                    s.push_str(&gen_string(r));
                }
                ws(r, s);
                s.push(':');
                ws(r, s);
                gen_value(r, depth + 1, s, keys);
                ws(r, s);
            }
            s.push('}');
        }
    }
}

fn mutate(r: &mut Rng, d: &[u8]) -> Vec<u8> {
    let mut v = d.to_vec();
    let edits = 1 + r.below(3);
    for _ in 0..edits {
        let len = v.len();
        match r.below(7) {
            0 if len > 0 => {
                let i = r.below(len as u64) as usize;
                v.remove(i);
            }
            1 => {
                let i = r.below(len as u64 + 1) as usize;
                let pool = b"{}[],:\"\\ 0123456789.eE+-tfnul\x00\x1f\x7f\xff\xc3\xa9\xed\xa0\x80";
                v.insert(i, pool[r.below(pool.len() as u64) as usize]);
            }
            2 if len > 0 => {
                let i = r.below(len as u64) as usize;
                v.truncate(i);
            }
            3 if len > 0 => {
                let i = r.below(len as u64) as usize;
                v[i] ^= 1 << r.below(8);
            }
            4 => v.push(b' '),
            5 => v.extend_from_slice(b",1"),
            _ if len > 0 => {
                let i = r.below(len as u64) as usize;
                v[i] = b"\"\\u{}[],"[r.below(8) as usize];
            }
            _ => {}
        }
    }
    v
}

fn handcrafted() -> Vec<Vec<u8>> {
    let mut v: Vec<Vec<u8>> = [
        "",
        " ",
        "null",
        "nul",
        "nulll",
        "true",
        "false",
        "True",
        "\"\"",
        "\"",
        "[]",
        "{}",
        "[",
        "]",
        "{",
        "}",
        "[,]",
        "[1,]",
        "[1,,2]",
        "{\"a\":1,}",
        "{,}",
        "{\"a\"}",
        "{\"a\":}",
        "{\"a\" 1}",
        "{1:2}",
        "{\"a\":1 \"b\":2}",
        "[1 2]",
        "01",
        "-",
        "-01",
        "1.",
        ".5",
        "+1",
        "1e",
        "1e+",
        "1E-",
        "0x10",
        "NaN",
        "Infinity",
        "-Infinity",
        "1.5e",
        "--1",
        "1..2",
        "\"\\x\"",
        "\"\\u12\"",
        "\"\\u12G4\"",
        "\"\\ud800\"",
        "\"\\udc00\"",
        "\"\\ud800\\u0041\"",
        "\"\\ud800\\ud800\"",
        "\"\\ud800\\udc00\"",
        "\"\\uDBFF\\uDFFF\"",
        "\"\\ud800x\"",
        "\"\\ud800\\n\"",
        "\"\\u0000\"",
        "\"a\\u0000b\"",
        "\"\u{7f}\"",
        "\"\t\"",
        "\"\n\"",
        "\"\\/\"",
        "\u{feff}1",
        "1 2",
        "[] []",
        "{} x",
        "\"a\"\"b\"",
        " \t\n\r1\r\n\t ",
        "\u{a0}1",
        "1\u{0b}",
        "\u{0c}1",
        "[\"\u{e9}\u{1F600}\"]",
        "{\"a\":1,\"a\":2,\"b\":3,\"a\":4}",
        "{\"\":0}",
        "{\"b\":1,\"a\":2,\"ab\":3,\"aa\":4,\"\u{e9}\":5,\"Z\":6}",
        "[-0,-0.0,0e0,-0e-0]",
        "[1e3,5.0,1E3,10]",
        "123456789012345678901234567890",
        "[1e400]",
        "[-1e400]",
        "[1e-400]",
        "\"\\uD834\\uDD1E\"",
        "\"\\u00e9\\u00E9\"",
        "[true,false,null]",
        "{\"a\":[{\"b\":{}}]}",
        "/* c */ 1",
        "1 // c",
        "'a'",
        "[\"a\",]",
        "{\"a\":1}}",
        "[[]]]",
        "\"\\\"",
        "tru",
        "fals",
        "{\"a\":{\"x\":1},\"a\":{\"y\":2}}",
        "[{\"k\":1,\"k\":[1,{\"k\":2,\"k\":3}]}]",
        "\"\\u001f\\u0001\\u0008\\u000b\\u007f\\u0080\\u00ff\\u2028\"",
        "\"</script>\"",
        "[0.1,0.2,0.30000000000000004,1e21,1e15,1e16,1e-5,1e-6,1e-7,450.0,450,-450]",
        "[1.0e+2,1.0E-2,1.5e0,100e-2]",
        "-9223372036854775808",
        "-9223372036854775809",
        "18446744073709551615",
        "18446744073709551616",
    ]
    .iter()
    .map(|s| s.as_bytes().to_vec())
    .collect();
    for b in [
        &b"\"\xff\""[..],
        b"\"\xc3\"",
        b"\"\xc3\xa9\"",
        b"\"\xed\xa0\x80\"",
        b"\"\xed\x9f\xbf\"",
        b"\"\xf4\x90\x80\x80\"",
        b"\"\xf4\x8f\xbf\xbf\"",
        b"\"\xf0\x9f\x98\x80\"",
        b"\"\xf0\x8f\xbf\xbf\"",
        b"\"\xe0\x80\x80\"",
        b"\"\xe0\xa0\x80\"",
        b"\"\xc0\xaf\"",
        b"\"\xc1\xbf\"",
        b"\"\xc2\x80\"",
        b"\"\xf5\x80\x80\x80\"",
        b"\"\x80\"",
        b"\xff",
        b"[\"\xc3\"\xa9]",
        b"\"\xc3\\u00a9\"",
        b"{\"\xff\":1}",
        b"\"a\x00b\"",
    ] {
        v.push(b.to_vec());
    }
    for d in [1usize, 63, 64, 65, 126, 127, 128, 129, 300] {
        let mut s = "[".repeat(d);
        s.push_str(&"]".repeat(d));
        v.push(s.into_bytes());
        let mut s = "{\"a\":".repeat(d);
        s.push('1');
        s.push_str(&"}".repeat(d));
        v.push(s.into_bytes());
        let mut s = String::new();
        for i in 0..d {
            s.push_str(if i % 2 == 0 { "[" } else { "{\"k\":" });
        }
        s.push_str("null");
        for i in (0..d).rev() {
            s.push_str(if i % 2 == 0 { "]" } else { "}" });
        }
        v.push(s.into_bytes());
        let mut s = "[".repeat(d);
        s.push_str(&"]".repeat(d - 1));
        v.push(s.into_bytes());
    }
    let mut big = String::from("[");
    for i in 0..5000 {
        if i > 0 {
            big.push(',');
        }
        big.push_str(&format!("{{\"k{}\":[{},{}.5,\"s{}\"]}}", i % 37, i, i, i));
    }
    big.push(']');
    v.push(big.into_bytes());
    v
}

fn docs(corpus: &str, expected: &str, seed: u64, n: usize) {
    let mut r = Rng(seed);
    let keys: Vec<String> = [
        "\"a\"",
        "\"b\"",
        "\"name\"",
        "\"arguments\"",
        "\"\"",
        "\"\\u0061\"",
        "\"\u{e9}\"",
        "\"A\"",
        "\"aa\"",
        "\"Z\"",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let mut docs = handcrafted();
    for i in 0..n {
        let mut s = String::new();
        ws(&mut r, &mut s);
        if i % 5 == 0 {
            s.push_str(&gen_number(&mut r));
        } else {
            gen_value(&mut r, 0, &mut s, &keys);
        }
        ws(&mut r, &mut s);
        let b = s.into_bytes();
        if r.chance(1, 3) {
            docs.push(mutate(&mut r, &b));
        } else {
            docs.push(b);
        }
    }
    let mut cf = std::io::BufWriter::new(std::fs::File::create(corpus).unwrap());
    let mut ef = std::io::BufWriter::new(std::fs::File::create(expected).unwrap());
    let (mut ok, mut bad) = (0, 0);
    for d in &docs {
        cf.write_all(&(d.len() as u32).to_le_bytes()).unwrap();
        cf.write_all(d).unwrap();
        let v: Result<serde_json::Value, _> = serde_json::from_slice(d);
        let o: Result<Ord_, _> = serde_json::from_slice(d);
        assert_eq!(v.is_ok(), o.is_ok());
        match (v, o) {
            (Ok(v), Ok(o)) => {
                ok += 1;
                let mut s = String::new();
                dump(&o, &mut s);
                let sorted = serde_json::to_string(&v).unwrap();
                let ins = serde_json::to_string(&o).unwrap();
                writeln!(
                    ef,
                    "OK {} {} {}",
                    s,
                    hex(sorted.as_bytes()),
                    hex(ins.as_bytes())
                )
                .unwrap();
            }
            (Err(e), _) => {
                bad += 1;
                let msg = e.to_string();
                if let Ok(st) = std::str::from_utf8(d) {
                    let e2 = serde_json::from_str::<serde_json::Value>(st).unwrap_err();
                    assert_eq!(msg, e2.to_string());
                }
                assert!(!msg.contains('\n'));
                writeln!(ef, "ERR {}", msg).unwrap();
            }
            _ => unreachable!(),
        }
    }
    eprintln!("docs: {} total, {} ok, {} rejected", docs.len(), ok, bad);
}

fn float_bits64(r: &mut Rng, i: u64) -> u64 {
    match i % 8 {
        0 | 1 | 2 => r.next(),
        3 => {
            let e = r.below(2048);
            (r.next() & 0x800F_FFFF_FFFF_FFFF) | (e << 52)
        }
        4 => (r.below(2_000_001) as f64 - 1_000_000.0).to_bits(),
        5 => {
            let k = r.below(100000) as f64;
            let p = 10f64.powi(r.below(40) as i32 - 20);
            (k * p).to_bits()
        }
        6 => {
            let e = r.below(2048);
            let m = (r.next() & 0xF) << r.below(49);
            (e << 52) | m | ((r.next() & 1) << 63)
        }
        _ => {
            let base = [
                0u64,
                0x0010_0000_0000_0000,
                0x7FEF_FFFF_FFFF_FFFF,
                0x3FF0_0000_0000_0000,
            ][r.below(4) as usize];
            (base.wrapping_add(r.below(2048)).wrapping_sub(1024) & 0x7FFF_FFFF_FFFF_FFFF)
                | ((r.next() & 1) << 63)
        }
    }
}

fn float_bits32(r: &mut Rng, i: u64) -> u32 {
    match i % 6 {
        0 | 1 | 2 => r.next() as u32,
        3 => (r.below(200001) as f32 - 100000.0).to_bits(),
        4 => {
            let k = r.below(100000) as f32;
            (k / 10f32.powi(r.below(8) as i32)).to_bits()
        }
        _ => {
            let e = r.below(256) as u32;
            ((r.next() as u32) & 0x8000_000F) | (e << 23)
        }
    }
}

fn floats(path: &str, seed: u64, n: u64, is64: bool) {
    let mut r = Rng(seed);
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    for i in 0..n {
        let (bits, s, sv) = if is64 {
            let b = float_bits64(&mut r, i);
            let x = f64::from_bits(b);
            (
                b,
                serde_json::to_string(&x).unwrap(),
                serde_json::to_string(&serde_json::Value::from(x)).unwrap(),
            )
        } else {
            let b = float_bits32(&mut r, i);
            let x = f32::from_bits(b);
            (
                b as u64,
                serde_json::to_string(&x).unwrap(),
                serde_json::to_string(&serde_json::Value::from(x)).unwrap(),
            )
        };
        f.write_all(&bits.to_le_bytes()).unwrap();
        f.write_all(&fnv(s.as_bytes()).to_le_bytes()).unwrap();
        f.write_all(&fnv(sv.as_bytes()).to_le_bytes()).unwrap();
    }
}

fn f32all(path: &str) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    let mut buf = Vec::with_capacity(64);
    for block in 0u64..256 {
        let mut h: u64 = 0xcbf29ce484222325;
        for lo in 0u64..(1 << 24) {
            let bits = ((block << 24) | lo) as u32;
            buf.clear();
            serde_json::to_writer(&mut buf, &f32::from_bits(bits)).unwrap();
            for &b in &buf {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h ^= 0x0a;
            h = h.wrapping_mul(0x100000001b3);
        }
        writeln!(f, "{:016x}", h).unwrap();
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a[1].as_str() {
        "docs" => docs(&a[2], &a[3], a[4].parse().unwrap(), a[5].parse().unwrap()),
        "floats64" => floats(&a[2], a[3].parse().unwrap(), a[4].parse().unwrap(), true),
        "floats32" => floats(&a[2], a[3].parse().unwrap(), a[4].parse().unwrap(), false),
        "f32all" => f32all(&a[2]),
        "show" => {
            for x in &a[2..] {
                let v: f64 = x.parse().unwrap();
                println!(
                    "{} {} {}",
                    serde_json::to_string(&v).unwrap(),
                    serde_json::to_string(&(v as f32)).unwrap(),
                    serde_json::to_string(&serde_json::Value::from(v as f32)).unwrap()
                );
            }
        }
        "showbits" => {
            for x in &a[2..] {
                let b = u64::from_str_radix(x, 16).unwrap();
                println!("{}", serde_json::to_string(&f64::from_bits(b)).unwrap());
            }
        }
        _ => panic!("mode"),
    }
}
