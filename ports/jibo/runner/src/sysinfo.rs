//! Read-only resource observations from /proc and /sys, and the admission gate built on them.
//! Nothing here changes a clock, a fan or a thermal limit.

use serde_json::{json, Value};
use std::path::PathBuf;

/// Refuse work (`busy`) while memory is short or a thermal zone is hot. Same idea as the
/// Decider port's `--min-avail-mb` / `--max-temp`.
#[derive(Debug, Clone, Default)]
pub struct Gate {
    pub min_avail_mb: Option<u64>,
    pub thermal: Option<PathBuf>,
    pub max_temp_c: Option<f64>,
}

impl Gate {
    pub fn busy_reason(&self) -> Option<String> {
        if let Some(min) = self.min_avail_mb {
            match mem_available_kb() {
                Some(kb) if kb / 1024 < min => {
                    return Some(format!("MemAvailable {} MB below {min} MB", kb / 1024))
                }
                None => return Some("MemAvailable unreadable".into()),
                _ => {}
            }
        }
        if let (Some(f), Some(max)) = (&self.thermal, self.max_temp_c) {
            match read_millideg(f) {
                Some(c) if c > max => {
                    return Some(format!("{} at {c:.1} C above {max} C", f.display()))
                }
                None => return Some(format!("{} unreadable", f.display())),
                _ => {}
            }
        }
        None
    }
}

fn read_millideg(p: &std::path::Path) -> Option<f64> {
    std::fs::read_to_string(p)
        .ok()?
        .trim()
        .parse::<f64>()
        .ok()
        .map(|m| m / 1000.0)
}

fn status_kb(field: &str) -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with(field))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

pub fn mem_available_kb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

/// User plus system CPU time of this process, from /proc/self/stat.
pub fn cpu_seconds() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/self/stat").ok()?;
    // Fields after the parenthesised command name; utime and stime are fields 14 and 15.
    let rest = &s[s.rfind(')')? + 2..];
    let f: Vec<&str> = rest.split_whitespace().collect();
    let utime: f64 = f.get(11)?.parse().ok()?;
    let stime: f64 = f.get(12)?.parse().ok()?;
    // USER_HZ is 100 on every Linux ABI this runs on (ARM and x86 alike).
    Some((utime + stime) / 100.0)
}

/// What the process and the machine look like right now.
pub fn snapshot(gate: &Gate) -> Value {
    json!({
        "vm_hwm_kb": status_kb("VmHWM:"),
        "vm_rss_kb": status_kb("VmRSS:"),
        "threads": status_kb("Threads:"),
        "mem_available_kb": mem_available_kb(),
        "cpu_s": cpu_seconds(),
        "temp_c": gate.thermal.as_deref().and_then(read_millideg),
    })
}
