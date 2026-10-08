//! `needle-jibo`: one Needle 3 model, bounded requests, explicit outcomes.
//!
//!   needle-jibo info  MODEL --tools CATALOGUE [options]
//!   needle-jibo run   MODEL --tools CATALOGUE --query TEXT [options]
//!   needle-jibo serve MODEL --tools CATALOGUE [--socket PATH [--queue N]] [options]
//!   needle-jibo bench MODEL --tools CATALOGUE --requests FILE.jsonl [--reps N] [options]
//!
//! `serve` reads one JSON request per line on stdin and writes one JSON response per line on
//! stdout, or, with `--socket`, answers one request per Unix-socket connection (mode 0660; write
//! the request, shut down the write side, read the response). Requests:
//!
//!   {"request_id":"u1","query":"set a timer for ten minutes","tools":["start_timer"]}
//!   {"health":true}
//!
//! Nothing is logged by default; `--debug-text` adds the raw completion to each response.

use needle_jibo::catalog::Catalogue;
use needle_jibo::service::{parse_line, refusal, Limits, Line, Options, Request, Service};
use needle_jibo::sysinfo;
use serde_json::{json, Value};
use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const MAX_LINE: usize = 16 * 1024;

const USAGE: &str = "usage: needle-jibo info|run|serve|bench MODEL --tools CATALOGUE [options]

options:
  --layers N            ladder rung (2..num_layers); default: the container's full depth
  --constrain           restrict names and argument keys inside <tool_call> (values are not)
  --kv-int8             int8 KV cache (not bit-identical to f32)
  --system TEXT         system message
  --confidence          score candidates with the confidence head (uncalibrated, timed)
  --debug-text          include the raw completion in responses
  --max-total-tokens N  prompt + generated, per request (default 512)
  --max-new-tokens N    generated, per request (default 256)
  --min-new-tokens N    refuse a prompt that leaves fewer than this (default 64)
  --max-query-bytes N   (default 2048)
  --deadline-ms N       wall limit from receipt, queue included; checked between tokens
  --min-avail-mb N      answer busy while MemAvailable is below N
  --thermal FILE        a millidegree file (e.g. /sys/class/thermal/thermal_zone0/temp) ...
  --max-temp-c C        ... and answer busy while it reads above C
run:   --query TEXT [--request-id ID]
serve: [--socket PATH] [--queue N (default 2)]
bench: --requests FILE.jsonl [--reps N (default 3)] [--warmup N (default 1)]";

struct Args {
    cmd: String,
    model: PathBuf,
    tools: PathBuf,
    layers: Option<usize>,
    limits: Limits,
    opts: Options,
    query: Option<String>,
    request_id: String,
    socket: Option<PathBuf>,
    queue: usize,
    requests: Option<PathBuf>,
    reps: usize,
    warmup: usize,
    gate: sysinfo::Gate,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let cmd = it.next().ok_or(USAGE)?;
    if cmd == "--help" || cmd == "-h" {
        return Err(USAGE.into());
    }
    let mut a = Args {
        cmd,
        model: PathBuf::new(),
        tools: PathBuf::new(),
        layers: None,
        limits: Limits::default(),
        opts: Options::default(),
        query: None,
        request_id: "cli".into(),
        socket: None,
        queue: 2,
        requests: None,
        reps: 3,
        warmup: 1,
        gate: sysinfo::Gate::default(),
    };
    let mut positional = Vec::new();
    while let Some(arg) = it.next() {
        let mut val = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        let num = |s: String, name: &str| {
            s.parse::<usize>()
                .map_err(|_| format!("{name}: not a number"))
        };
        match arg.as_str() {
            "--tools" => a.tools = val("--tools")?.into(),
            "--layers" => a.layers = Some(num(val("--layers")?, "--layers")?),
            "--constrain" => a.opts.constrain = true,
            "--kv-int8" => a.opts.kv_int8 = true,
            "--system" => a.opts.system = Some(val("--system")?),
            "--confidence" => a.opts.confidence = true,
            "--debug-text" => a.opts.debug_text = true,
            "--max-total-tokens" => a.limits.max_total_tokens = num(val(&arg)?, &arg)?,
            "--max-new-tokens" => a.limits.max_new_tokens = num(val(&arg)?, &arg)?,
            "--min-new-tokens" => a.limits.min_new_tokens = num(val(&arg)?, &arg)?,
            "--max-query-bytes" => a.limits.max_query_bytes = num(val(&arg)?, &arg)?,
            "--deadline-ms" => {
                a.limits.deadline = Some(Duration::from_millis(num(val(&arg)?, &arg)? as u64))
            }
            "--min-avail-mb" => a.gate.min_avail_mb = Some(num(val(&arg)?, &arg)? as u64),
            "--thermal" => a.gate.thermal = Some(val(&arg)?.into()),
            "--max-temp-c" => {
                a.gate.max_temp_c = Some(
                    val(&arg)?
                        .parse()
                        .map_err(|_| "--max-temp-c: not a number")?,
                )
            }
            "--query" => a.query = Some(val(&arg)?),
            "--request-id" => a.request_id = val(&arg)?,
            "--socket" => a.socket = Some(val(&arg)?.into()),
            "--queue" => a.queue = num(val(&arg)?, &arg)?.max(1),
            "--requests" => a.requests = Some(val(&arg)?.into()),
            "--reps" => a.reps = num(val(&arg)?, &arg)?,
            "--warmup" => a.warmup = num(val(&arg)?, &arg)?,
            s if s.starts_with("--") => return Err(format!("unknown option {s}\n{USAGE}")),
            _ => positional.push(arg),
        }
    }
    if positional.len() != 1 {
        return Err(USAGE.into());
    }
    a.model = positional.remove(0).into();
    if a.tools.as_os_str().is_empty() {
        return Err("--tools CATALOGUE is required".into());
    }
    if a.gate.thermal.is_some() != a.gate.max_temp_c.is_some() {
        return Err("--thermal and --max-temp-c go together".into());
    }
    Ok(a)
}

fn emit(out: &mut impl Write, v: &Value) {
    // A closed stdout is the caller going away; there is nobody left to report to.
    let _ = writeln!(out, "{v}");
    let _ = out.flush();
}

fn answer(svc: &Service, gate: &sysinfo::Gate, line: &str, received: Instant) -> Value {
    match parse_line(line, MAX_LINE) {
        Err(e) => e,
        Ok(Line::Health) => {
            let mut h = svc.health();
            h["resources"] = sysinfo::snapshot(gate);
            h
        }
        Ok(Line::Request(r)) => match gate.busy_reason() {
            Some(why) => refusal(&r.id, "busy", &why),
            None => svc.handle(&r, received),
        },
    }
}

fn serve_stdio(svc: &Service, gate: &sysinfo::Gate) {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    let mut line = String::new();
    let mut input = stdin.lock();
    loop {
        line.clear();
        // Bounded read: a line longer than MAX_LINE is drained and refused, never buffered whole.
        let mut taken = (&mut input).take(MAX_LINE as u64 + 1);
        match taken.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                emit(
                    &mut out,
                    &refusal("", "invalid_request", &format!("input: {e}")),
                );
                break;
            }
        }
        let received = Instant::now();
        if line.len() > MAX_LINE && !line.ends_with('\n') {
            let mut sink = Vec::new();
            let _ = input.read_until(b'\n', &mut sink);
            emit(
                &mut out,
                &refusal("", "invalid_request", "request line too long"),
            );
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        emit(&mut out, &answer(svc, gate, line.trim_end(), received));
    }
}

#[cfg(unix)]
fn serve_socket(
    svc: &Service,
    gate: &sysinfo::Gate,
    path: &std::path::Path,
    queue: usize,
) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::sync::mpsc;

    if path.exists() {
        return Err(format!(
            "{} exists; remove it if no server owns it",
            path.display()
        ));
    }
    let listener = UnixListener::bind(path).map_err(|e| format!("{}: {e}", path.display()))?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o660))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    eprintln!("needle-jibo: serving on {} (queue {queue})", path.display());

    let (tx, rx) = mpsc::sync_channel::<(UnixStream, String, Instant)>(queue);
    std::thread::scope(|s| {
        s.spawn(move || {
            for (mut stream, line, received) in rx {
                let v = answer(svc, gate, &line, received);
                let _ = writeln!(stream, "{v}");
            }
        });
        for conn in listener.incoming() {
            let Ok(mut stream) = conn else { continue };
            // A slow client holds the acceptor for at most this long.
            let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
            let mut buf = Vec::new();
            let read = (&mut stream)
                .take(MAX_LINE as u64 + 1)
                .read_to_end(&mut buf);
            let received = Instant::now();
            let reply_now = |mut st: UnixStream, v: Value| {
                let _ = writeln!(st, "{v}");
            };
            match (read, String::from_utf8(buf)) {
                (Ok(n), Ok(line)) if n <= MAX_LINE => {
                    if let Err(mpsc::TrySendError::Full((st, line, _))) =
                        tx.try_send((stream, line, received))
                    {
                        let id = parse_line(&line, MAX_LINE)
                            .ok()
                            .and_then(|l| match l {
                                Line::Request(r) => Some(r.id),
                                Line::Health => None,
                            })
                            .unwrap_or_default();
                        reply_now(st, refusal(&id, "busy", "queue full"));
                    }
                }
                (Ok(_), Ok(_)) => {
                    reply_now(stream, refusal("", "invalid_request", "request too long"))
                }
                (Ok(_), Err(_)) => reply_now(
                    stream,
                    refusal("", "invalid_request", "request is not UTF-8"),
                ),
                (Err(e), _) => reply_now(
                    stream,
                    refusal("", "invalid_request", &format!("read: {e}")),
                ),
            }
        }
    });
    Ok(())
}

fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let i = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    Some(sorted[i])
}

fn bench(svc: &Service, path: &std::path::Path, reps: usize, warmup: usize) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let reqs: Vec<Request> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| match parse_line(l, MAX_LINE) {
            Ok(Line::Request(r)) => Ok(r),
            Ok(Line::Health) => Err("health lines are not benchmark requests".to_string()),
            Err(e) => Err(e.to_string()),
        })
        .collect::<Result<_, _>>()?;
    let mut out = std::io::stdout().lock();
    let cpu0 = sysinfo::cpu_seconds();
    let t0 = Instant::now();
    let mut walls: Vec<f64> = Vec::new();
    let (mut prefill, mut decode, mut gen_tokens, mut prompt_tokens) = (0.0, 0.0, 0u64, 0u64);
    let mut statuses = std::collections::BTreeMap::<String, usize>::new();
    for rep in 0..warmup + reps {
        for r in &reqs {
            let v = svc.handle(r, Instant::now());
            let measured = rep >= warmup;
            if measured {
                walls.push(v["timing"]["wall_ms"].as_f64().unwrap_or(f64::NAN));
                prefill += v["timing"]["prefill_ms"].as_f64().unwrap_or(0.0);
                decode += v["timing"]["decode_ms"].as_f64().unwrap_or(0.0);
                gen_tokens += v["tokens"]["generated"].as_u64().unwrap_or(0);
                prompt_tokens += v["tokens"]["prompt"].as_u64().unwrap_or(0);
                *statuses
                    .entry(v["status"].as_str().unwrap_or("?").to_string())
                    .or_default() += 1;
            }
            let mut v = v;
            v["bench"] = json!({"rep": rep, "warmup": !measured});
            emit(&mut out, &v);
        }
    }
    let wall_total = t0.elapsed().as_secs_f64();
    let cpu = sysinfo::cpu_seconds().zip(cpu0).map(|(a, b)| a - b);
    let mut sorted = walls.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let summary = json!({
        "summary": true,
        "model_sha256": svc.model_sha256,
        "depth": svc.depth,
        "kv_precision": svc.kv_name(),
        "constrained": svc.opts.constrain,
        "parallel": cfg!(feature = "parallel"),
        "requests": walls.len(),
        "statuses": statuses,
        "load_ms": svc.load_ms,
        "wall_ms": {"p50": percentile(&sorted, 0.5), "p95": percentile(&sorted, 0.95),
                     "p99": percentile(&sorted, 0.99), "max": sorted.last()},
        "prefill_ms_per_prompt_token": if prompt_tokens > 0 { Some(prefill / prompt_tokens as f64) } else { None },
        "decode_ms_per_token": if gen_tokens > 0 { Some(decode / gen_tokens as f64) } else { None },
        "process_cpu_s": cpu,
        "elapsed_s": wall_total,
        "resources": sysinfo::snapshot(&sysinfo::Gate::default()),
    });
    emit(&mut out, &summary);
    Ok(())
}

fn main() {
    let a = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let cat_text = match std::fs::read_to_string(&a.tools) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{}: {e}", a.tools.display());
            std::process::exit(2);
        }
    };
    let catalogue = match Catalogue::parse(&cat_text) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{}: {e}", a.tools.display());
            std::process::exit(2);
        }
    };
    let svc = match Service::load(
        &a.model,
        catalogue,
        a.layers,
        a.limits.clone(),
        a.opts.clone(),
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let result = match a.cmd.as_str() {
        "info" => {
            let mut h = svc.health();
            h["resources"] = sysinfo::snapshot(&a.gate);
            emit(&mut std::io::stdout(), &h);
            Ok(())
        }
        "run" => match a.query {
            None => Err("run needs --query".to_string()),
            Some(q) => {
                let r = Request {
                    id: a.request_id,
                    query: q,
                    tools: None,
                    max_new_tokens: None,
                };
                let v = match a.gate.busy_reason() {
                    Some(why) => refusal(&r.id, "busy", &why),
                    None => svc.handle(&r, Instant::now()),
                };
                emit(&mut std::io::stdout(), &v);
                Ok(())
            }
        },
        "serve" => match &a.socket {
            None => {
                serve_stdio(&svc, &a.gate);
                Ok(())
            }
            #[cfg(unix)]
            Some(p) => serve_socket(&svc, &a.gate, p, a.queue),
            #[cfg(not(unix))]
            Some(_) => Err("--socket needs a Unix platform".into()),
        },
        "bench" => match &a.requests {
            None => Err("bench needs --requests FILE.jsonl".to_string()),
            Some(p) => bench(&svc, p, a.reps, a.warmup),
        },
        other => Err(format!("unknown command {other}\n{USAGE}")),
    };
    if let Err(e) = result {
        eprintln!("{e}");
        std::process::exit(2);
    }
}
