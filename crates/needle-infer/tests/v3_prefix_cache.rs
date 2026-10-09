//! The tool-prefix cache must not change a single bit.
//!
//! `V3Engine::enable_prefix_cache` keeps the cache state after a prompt's tool prefix and starts
//! later requests from a copy, stepping only their own tokens. That is only sound if a prefill of
//! the prefix followed by decode steps over the rest leaves exactly the cache and logits one
//! prefill of the whole prompt would. These tests hold it to that, for both cache precisions and
//! for split points on and around the prefill chunk boundary, and then hold the engine to giving
//! the same generations with the cache on as with it off.
//!
//! Needs `weights/needle3.cact`; skips without it.
//!
//! Run: cargo test -p needle-infer --release --test v3_prefix_cache -- --nocapture

use needle_core::v3::{KvPrecision, V3Cache};
use needle_infer::v3_engine::{V3Engine, V3Options};

const CACT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/needle3.cact");
const TOOLS: &str = r#"[{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},{"name":"control_lights","description":"Turn lights on or off","parameters":{"type":"object","properties":{"room":{"type":"string"},"on":{"type":"boolean"}},"required":["room","on"]}}]"#;
const TOOLS_B: &str = r#"[{"name":"start_timer","description":"Start a countdown timer","parameters":{"type":"object","properties":{"seconds":{"type":"integer"}},"required":["seconds"]}}]"#;

fn engine() -> Option<V3Engine> {
    if !std::path::Path::new(CACT).exists() {
        eprintln!("skipping: missing {CACT}");
        return None;
    }
    Some(V3Engine::load(CACT).expect("load"))
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn a_prefix_prefill_then_steps_is_one_prefill() {
    let Some(e) = engine() else { return };
    let m = &e.model;
    let ids = e.prompt_ids("What's the weather in Paris tomorrow?", TOOLS, None);
    let n = ids.len();
    for precision in [KvPrecision::F32, KvPrecision::Int8] {
        let mut whole = V3Cache::with_precision(&m.cfg, n + 8, precision);
        let want = m.prefill(&ids, &mut whole);
        for split in [1, 2, 63, 64, 65, n / 2, n - 1] {
            // Sized to the prefix, as the engine stores it: the copy has to grow.
            let mut part = V3Cache::with_precision(&m.cfg, split, precision);
            m.prefill(&ids[..split], &mut part);
            let mut got = Vec::new();
            for &t in &ids[split..] {
                got = m.decode_step(&mut part, t);
            }
            assert_eq!(
                bits(&got),
                bits(&want),
                "{precision:?} split {split}: logits"
            );
            // The continuation, which a mis-seeded tail would only show later.
            let (mut a, mut b) = (whole.clone(), part);
            let mut la = want.clone();
            for step in 0..12 {
                let t = argmax(&la);
                la = m.decode_step(&mut a, t);
                let lb = m.decode_step(&mut b, t);
                assert_eq!(
                    bits(&la),
                    bits(&lb),
                    "{precision:?} split {split}: step {step}"
                );
            }
        }
    }
}

#[test]
fn generation_is_identical_with_the_prefix_cache() {
    let (Some(plain), Some(mut cached)) = (engine(), engine()) else {
        return;
    };
    cached.enable_prefix_cache();
    assert!(cached.prefix_cache_enabled() && !plain.prefix_cache_enabled());
    let f32_opts = V3Options {
        max_new_tokens: 64,
        ..Default::default()
    };
    let int8 = V3Options {
        kv_precision: KvPrecision::Int8,
        ..f32_opts.clone()
    };
    let constrained = V3Options {
        constrain: true,
        ..f32_opts.clone()
    };
    let system = V3Options {
        system: Some("You are a helpful robot.".into()),
        ..f32_opts.clone()
    };
    // (query, tools, options, whether the stored prefix should be hit)
    let cases: [(&str, &str, &V3Options, bool); 8] = [
        ("What's the weather in Paris?", TOOLS, &f32_opts, false),
        ("Turn on the kitchen lights", TOOLS, &f32_opts, true),
        ("Is it raining in Tokyo?", TOOLS, &constrained, true),
        ("Set a timer for ten minutes", TOOLS_B, &f32_opts, false),
        ("Set a timer for two minutes", TOOLS_B, &f32_opts, true),
        ("Set a timer for two minutes", TOOLS_B, &int8, false),
        ("Set a timer for one minute", TOOLS_B, &int8, true),
        ("Turn off the bedroom lights", TOOLS, &system, false),
    ];
    for (q, tools, o, hit) in cases {
        let a = plain.generate(q, tools, o);
        let b = cached.generate(q, tools, o);
        assert_eq!(a.tokens, b.tokens, "{q:?}: tokens");
        assert_eq!(a.text, b.text, "{q:?}: text");
        assert_eq!((a.stop, a.positions), (b.stop, b.positions), "{q:?}: stop");
        assert_eq!(a.prefix_reused, 0);
        let prefix = e_prefix_len(&cached, q, tools, o);
        assert_eq!(
            b.prefix_reused,
            if hit { prefix } else { 0 },
            "{q:?}: reuse"
        );
        // The confidence head reads the same completion either way.
        assert_eq!(
            plain.confidence_for(q, tools, &a.text).map(f32::to_bits),
            cached.confidence_for(q, tools, &b.text).map(f32::to_bits)
        );
    }
}

/// The prefix length the engine uses: through the first `</tools>`.
fn e_prefix_len(e: &V3Engine, q: &str, tools: &str, o: &V3Options) -> usize {
    let ids = e.prompt_ids(q, tools, o.system.as_deref());
    let end = e.tokenizer.id_of("</tools>").expect("marker");
    ids.iter().position(|&t| t == end).expect("prefix") + 1
}
