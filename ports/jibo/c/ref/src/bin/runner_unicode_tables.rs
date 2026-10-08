use std::fmt::Write;

fn ranges(name: &str, f: impl Fn(char) -> bool, out: &mut String) {
    let mut v: Vec<(u32, u32)> = Vec::new();
    let mut start: Option<u32> = None;
    for cp in 0..=0x110000u32 {
        let p = char::from_u32(cp).map(&f).unwrap_or(false);
        match (p, start) {
            (true, None) => start = Some(cp),
            (false, Some(s)) => {
                v.push((s, cp - 1));
                start = None;
            }
            _ => {}
        }
    }
    writeln!(out, "#define {}_N {}u", name.to_uppercase(), v.len()).unwrap();
    writeln!(
        out,
        "static const uint32_t {}[{}_N][2] = {{",
        name,
        name.to_uppercase()
    )
    .unwrap();
    let mut line = String::new();
    for (a, b) in &v {
        let item = format!("{{0x{:X}, 0x{:X}}}, ", a, b);
        if line.len() + item.len() > 96 {
            writeln!(out, "  {}", line.trim_end()).unwrap();
            line.clear();
        }
        line.push_str(&item);
    }
    if !line.is_empty() {
        writeln!(out, "  {}", line.trim_end()).unwrap();
    }
    writeln!(out, "}};").unwrap();
}

fn ends_final(s: &str) -> bool {
    s.to_lowercase().ends_with('ς')
}

fn main() {
    let which = std::env::args().nth(1).unwrap();
    let mut out = String::new();
    match which.as_str() {
        "debug" => {
            ranges(
                "nd_dbg_printable",
                |c| (c as u32) >= 0x80 && format!("{:?}", c.to_string()) == format!("\"{}\"", c),
                &mut out,
            );
            ranges("nd_white_space", |c| c.is_whitespace(), &mut out);
        }
        "ground" => {
            ranges("nd_g_alnum", |c| c.is_alphanumeric(), &mut out);
            ranges("nd_g_white", |c| c.is_whitespace(), &mut out);
            // Case_Ignorable / Cased via the Final_Sigma probes
            ranges(
                "nd_g_case_ignorable",
                |c| {
                    let p1 = ends_final(&format!("\u{3000}{}Σ", c));
                    let p2 = ends_final(&format!("Α{}Σ", c));
                    !p1 && p2
                },
                &mut out,
            );
            ranges(
                "nd_g_cased",
                |c| ends_final(&format!("\u{3000}{}Σ", c)),
                &mut out,
            );
            // lowercase mappings
            let mut v = Vec::new();
            for cp in 0..=0x10FFFFu32 {
                if let Some(c) = char::from_u32(cp) {
                    let l: Vec<char> = c.to_lowercase().collect();
                    if l != vec![c] {
                        v.push((cp, l));
                    }
                }
            }
            writeln!(out, "#define ND_G_LOWER_N {}u", v.len()).unwrap();
            writeln!(
                out,
                "static const uint32_t nd_g_lower[ND_G_LOWER_N][4] = {{"
            )
            .unwrap();
            let mut line = String::new();
            for (cp, l) in &v {
                let mut a = [0u32; 3];
                for (i, c) in l.iter().enumerate() {
                    a[i] = *c as u32;
                }
                let item = format!("{{0x{:X}, 0x{:X}, 0x{:X}, 0x{:X}}}, ", cp, a[0], a[1], a[2]);
                if line.len() + item.len() > 96 {
                    writeln!(out, "  {}", line.trim_end()).unwrap();
                    line.clear();
                }
                line.push_str(&item);
            }
            if !line.is_empty() {
                writeln!(out, "  {}", line.trim_end()).unwrap();
            }
            writeln!(out, "}};").unwrap();
        }
        _ => panic!(),
    }
    print!("{out}");
}
