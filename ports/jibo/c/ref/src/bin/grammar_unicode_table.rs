fn flags(c: char) -> u8 {
    (c.is_alphanumeric() as u8) | ((c.is_uppercase() as u8) << 1) | ((c.is_lowercase() as u8) << 2)
}
fn main() {
    let mut ranges: Vec<(u32, u32, u8)> = Vec::new();
    for cp in 0x80u32..=0x10FFFF {
        let f = match char::from_u32(cp) {
            Some(c) => flags(c),
            None => 0,
        };
        if f == 0 {
            continue;
        }
        if let Some(last) = ranges.last_mut() {
            if last.1 + 1 == cp && last.2 == f {
                last.1 = cp;
                continue;
            }
        }
        ranges.push((cp, cp, f));
    }
    // sanity: ASCII handled separately; verify ASCII matches simple rules
    for cp in 0u32..0x80 {
        let c = char::from_u32(cp).unwrap();
        let want = (c.is_ascii_alphanumeric() as u8)
            | ((c.is_ascii_uppercase() as u8) << 1)
            | ((c.is_ascii_lowercase() as u8) << 2);
        assert_eq!(flags(c), want);
    }
    println!("/* {} ranges, generated from Rust 1.87 char::is_alphanumeric / is_uppercase / is_lowercase */", ranges.len());
    let mut line = String::new();
    for (a, b, f) in &ranges {
        let item = format!("{{0x{:X},0x{:X},{}}},", a, b, f);
        if line.len() + item.len() > 96 {
            println!("    {}", line.trim_end());
            line.clear();
        }
        line.push_str(&item);
    }
    if !line.is_empty() {
        println!("    {}", line.trim_end());
    }
}
