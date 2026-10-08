// Emits the C range table of char::is_alphanumeric / is_uppercase / is_lowercase for this rustc.
fn class(c: u32) -> u8 {
    match char::from_u32(c) {
        None => 0,
        Some(ch) => {
            (ch.is_alphanumeric() as u8)
                | ((ch.is_uppercase() as u8) << 1)
                | ((ch.is_lowercase() as u8) << 2)
        }
    }
}
fn main() {
    let mut ranges: Vec<(u32, u8)> = Vec::new();
    let mut prev = 255u8;
    for c in 0u32..=0x10FFFF {
        // Surrogates are not chars; give them the class of their neighbours' run (never queried).
        let k = if (0xD800..0xE000).contains(&c) {
            prev
        } else {
            class(c)
        };
        if k != prev {
            ranges.push((c, k));
            prev = k;
        }
    }
    println!("/* Generated with rustc 1.87.0 (Unicode {:?}): char::is_alphanumeric (bit 0), is_uppercase (bit 1),", char::UNICODE_VERSION);
    println!(
        " * is_lowercase (bit 2): run starts and classes; a run lasts until the next start. */"
    );
    println!("#define ND_UCLASS_RUNS {}", ranges.len());
    println!("static const uint32_t nd_uclass_start[ND_UCLASS_RUNS] = {{");
    for ch in ranges.chunks(8) {
        let v: Vec<String> = ch.iter().map(|r| format!("0x{:X}", r.0)).collect();
        println!("  {},", v.join(", "));
    }
    println!("}};");
    println!("static const uint8_t nd_uclass_bits[ND_UCLASS_RUNS] = {{");
    for ch in ranges.chunks(24) {
        let v: Vec<String> = ch.iter().map(|r| format!("{}", r.1)).collect();
        println!("  {},", v.join(","));
    }
    println!("}};");
}
