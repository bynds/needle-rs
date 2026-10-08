/* needle-jibo-c: one Needle 3 model, bounded requests, explicit outcomes. A C99 transcription of
 * runner/src/main.rs; responses are byte-identical to the Rust runner's apart from measured
 * times and resource readings.
 *
 *   needle-jibo-c info  MODEL --tools CATALOGUE [options]
 *   needle-jibo-c run   MODEL --tools CATALOGUE --query TEXT [options]
 *   needle-jibo-c serve MODEL --tools CATALOGUE [--socket PATH [--queue N]] [options]
 *   needle-jibo-c bench MODEL --tools CATALOGUE --requests FILE.jsonl [--reps N] [options]
 *   needle-jibo-c regrade RESPONSES.jsonl --tools CATALOGUE --requests REQUESTS.jsonl [...]
 *
 * `dump-op` (reference vectors for the kernel benchmark) stays in the Rust runner.
 */
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include "nd_service.h"
#include "nd_sysinfo.h"
#include "nd_tok.h"

#define MAX_LINE (16u * 1024u)

static const char USAGE[] =
    "usage: needle-jibo-c info|run|serve|bench MODEL --tools CATALOGUE [options]\n"
    "\n"
    "options:\n"
    "  --layers N            ladder rung (2..num_layers); default: the container's full depth\n"
    "  --constrain           restrict names and argument keys inside <tool_call> (values are not)\n"
    "  --kv-int8             int8 KV cache (not bit-identical to f32)\n"
    "  --system TEXT         system message\n"
    "  --confidence          score candidates with the confidence head (uncalibrated, timed)\n"
    "  --debug-text          include the raw completion in responses\n"
    "  --verify-model        hash the model now instead of trusting the <model>.sha256 sidecar cache\n"
    "  --expect-sha256 HEX   refuse to start unless the model hashes to HEX (always recomputed)\n"
    "  --grounding MODE      enforce (default): a call whose numbers or free text the query does not\n"
    "                        state becomes needs_clarification; strict: enum values must be said\n"
    "                        too; report: annotate only; off\n"
    "  --min-confidence P    with --confidence: candidates scoring below P become low_confidence\n"
    "  --max-total-tokens N  prompt + generated, per request (default 512)\n"
    "  --max-new-tokens N    generated, per request (default 256)\n"
    "  --min-new-tokens N    refuse a prompt that leaves fewer than this (default 64)\n"
    "  --max-query-bytes N   (default 2048)\n"
    "  --deadline-ms N       wall limit from receipt, queue included; checked between tokens\n"
    "  --min-avail-mb N      answer busy while MemAvailable is below N\n"
    "  --thermal FILE        a millidegree file (e.g. /sys/class/thermal/thermal_zone0/temp) ...\n"
    "  --max-temp-c C        ... and answer busy while it reads above C\n"
    "run:   --query TEXT [--request-id ID]\n"
    "serve: [--socket PATH] [--queue N (default 2)]\n"
    "bench: --requests FILE.jsonl [--reps N (default 3)] [--warmup N (default 1)]\n"
    "regrade RESPONSES.jsonl --tools CATALOGUE --requests REQUESTS.jsonl [--grounding M] [--min-confidence P]:\n"
    "       re-apply the policy to saved --debug-text responses, without the model";

typedef struct {
  const char *cmd, *model, *tools;
  int has_layers;
  size_t layers;
  nd_limits limits;
  nd_options opts;
  const char *query, *request_id, *socket, *requests;
  size_t queue, reps, warmup;
  nd_gate gate;
} args;

static void die(int code, const char *msg) {
  fprintf(stderr, "%s\n", msg);
  exit(code);
}

/* Rust's usize::from_str: optional '+', ASCII digits, no overflow. */
static int parse_usize(const char *s, size_t *out) {
  size_t v = 0;
  if (*s == '+') s++;
  if (!*s) return 0;
  for (; *s; s++) {
    size_t d;
    if (*s < '0' || *s > '9') return 0;
    d = (size_t)(*s - '0');
    if (v > (SIZE_MAX - d) / 10) return 0;
    v = v * 10 + d;
  }
  *out = v;
  return 1;
}

static size_t num(const char *s, const char *name) {
  size_t v;
  char msg[256];
  if (!parse_usize(s, &v)) {
    snprintf(msg, sizeof msg, "%s: not a number", name);
    die(2, msg);
  }
  return v;
}

static void parse_args(int argc, char **argv, args *a) {
  int i, npos = 0;
  char msg[2600];
  memset(a, 0, sizeof *a);
  nd_limits_default(&a->limits);
  nd_options_default(&a->opts);
  a->request_id = "cli";
  a->queue = 2;
  a->reps = 3;
  a->warmup = 1;
  if (argc < 2) die(2, USAGE);
  a->cmd = argv[1];
  if (!strcmp(a->cmd, "--help") || !strcmp(a->cmd, "-h")) die(2, USAGE);
  for (i = 2; i < argc; i++) {
    const char *k = argv[i];
#define VAL()                                              \
  (i + 1 < argc ? argv[++i]                                \
                : (snprintf(msg, sizeof msg, "%s needs a value", k), die(2, msg), (char *)NULL))
    if (!strcmp(k, "--tools")) a->tools = VAL();
    else if (!strcmp(k, "--layers")) a->has_layers = 1, a->layers = num(VAL(), "--layers");
    else if (!strcmp(k, "--constrain")) a->opts.constrain = 1;
    else if (!strcmp(k, "--kv-int8")) a->opts.kv_int8 = 1;
    else if (!strcmp(k, "--system")) a->opts.system = VAL();
    else if (!strcmp(k, "--confidence")) a->opts.confidence = 1;
    else if (!strcmp(k, "--debug-text")) a->opts.debug_text = 1;
    else if (!strcmp(k, "--verify-model")) a->opts.verify_model = 1;
    else if (!strcmp(k, "--expect-sha256")) a->opts.expect_sha256 = VAL();
    else if (!strcmp(k, "--grounding")) {
      if (!nd_grounding_parse(VAL(), &a->opts.grounding)) die(2, "--grounding: enforce, strict, report or off");
    } else if (!strcmp(k, "--min-confidence")) {
      double d;
      const char *v = VAL();
      if (!nd_rust_parse_f64(v, strlen(v), &d)) die(2, "--min-confidence: not a number");
      /* Rust parses straight to f32: strtof rounds the text once, as Rust does. */
      a->opts.min_confidence = strtof(v, NULL);
      a->opts.has_min_confidence = 1;
    } else if (!strcmp(k, "--max-total-tokens")) a->limits.max_total_tokens = num(VAL(), k);
    else if (!strcmp(k, "--max-new-tokens")) a->limits.max_new_tokens = num(VAL(), k);
    else if (!strcmp(k, "--min-new-tokens")) a->limits.min_new_tokens = num(VAL(), k);
    else if (!strcmp(k, "--max-query-bytes")) a->limits.max_query_bytes = num(VAL(), k);
    else if (!strcmp(k, "--deadline-ms")) a->limits.has_deadline = 1, a->limits.deadline_ms = num(VAL(), k);
    else if (!strcmp(k, "--min-avail-mb")) a->gate.has_min_avail = 1, a->gate.min_avail_mb = num(VAL(), k);
    else if (!strcmp(k, "--thermal")) a->gate.thermal = VAL();
    else if (!strcmp(k, "--max-temp-c")) {
      const char *v = VAL();
      if (!nd_rust_parse_f64(v, strlen(v), &a->gate.max_temp_c)) die(2, "--max-temp-c: not a number");
      a->gate.has_max_temp = 1;
    } else if (!strcmp(k, "--query")) a->query = VAL();
    else if (!strcmp(k, "--request-id")) a->request_id = VAL();
    else if (!strcmp(k, "--socket")) a->socket = VAL();
    else if (!strcmp(k, "--queue")) {
      a->queue = num(VAL(), k);
      if (a->queue < 1) a->queue = 1;
    } else if (!strcmp(k, "--requests")) a->requests = VAL();
    else if (!strcmp(k, "--reps")) a->reps = num(VAL(), k);
    else if (!strcmp(k, "--warmup")) a->warmup = num(VAL(), k);
    else if (!strncmp(k, "--", 2)) {
      snprintf(msg, sizeof msg, "unknown option %s\n%s", k, USAGE);
      die(2, msg);
    } else {
      a->model = k;
      npos++;
    }
#undef VAL
  }
  if (npos != 1) die(2, USAGE);
  if (!a->tools) die(2, "--tools CATALOGUE is required");
  if (a->opts.has_min_confidence && !a->opts.confidence && strcmp(a->cmd, "regrade") != 0)
    die(2, "--min-confidence needs --confidence");
  if ((a->gate.thermal != NULL) != (a->gate.has_max_temp != 0)) die(2, "--thermal and --max-temp-c go together");
}

/* A closed stdout is the caller going away; there is nobody left to report to. */
static void emit_text(FILE *f, const char *text) {
  if (!text) return;
  fputs(text, f);
  fputc('\n', f);
  fflush(f);
}

static void emit_obj(FILE *f, nd_obj *o) {
  size_t n;
  char *t = nd_obj_finish(o, &n);
  if (!t) t = nd_refusal("", "internal_error", "out of memory");
  emit_text(f, t);
  free(t);
}

static char *read_file(const char *path, size_t *len, char **err) {
  FILE *f = fopen(path, "rb");
  char *b = NULL;
  size_t n = 0, cap = 0;
  if (!f) goto fail;
  for (;;) {
    size_t r;
    if (n == cap) {
      char *g;
      cap = cap ? cap * 2 : 65536;
      if (!(g = realloc(b, cap + 1))) goto fail_open;
      b = g;
    }
    r = fread(b + n, 1, cap - n, f);
    n += r;
    if (r == 0) break;
  }
  if (ferror(f)) goto fail_open;
  fclose(f);
  b[n] = 0;
  *len = n;
  return b;
fail_open:
  fclose(f);
fail:
  {
    int e = errno ? errno : EIO;
    size_t m = strlen(path) + 128;
    *err = malloc(m);
    if (*err) snprintf(*err, m, "%s: %s (os error %d)", path, strerror(e), e);
  }
  free(b);
  return NULL;
}

/* read_to_string: also refuses invalid UTF-8, as Rust's String does. */
static char *read_text(const char *path, size_t *len) {
  char *err = NULL, *t = read_file(path, len, &err);
  if (!t) {
    fprintf(stderr, "%s\n", err ? err : path);
    exit(2);
  }
  if (!nd_utf8_valid((const uint8_t *)t, *len)) {
    fprintf(stderr, "%s: stream did not contain valid UTF-8\n", path);
    exit(2);
  }
  return t;
}

static nd_catalogue *load_catalogue(const char *path) {
  size_t n;
  char *text = read_text(path, &n), *err = NULL;
  nd_catalogue *c = NULL;
  if (nd_catalogue_parse(text, n, &c, &err) != ND_OK) {
    fprintf(stderr, "%s: %s\n", path, err ? err : "out of memory");
    exit(2);
  }
  free(text);
  return c;
}

/* answer(): health, busy, or the request. */
static void answer(nd_service *svc, const nd_gate *gate, const char *line, size_t len, double received,
                   nd_obj *out) {
  nd_request r;
  char *refusal = NULL, why[512];
  switch (nd_parse_line(line, len, MAX_LINE, &r, &refusal)) {
  case ND_LINE_ERROR: {
    nd_json_doc *doc = NULL;
    char *m = NULL;
    if (nd_serde_parse(refusal, strlen(refusal), &doc, &m) == ND_OK) nd_obj_from_json(out, nd_json_root(doc));
    else out->err = 1;
    nd_json_doc_free(doc);
    free(m);
    free(refusal);
    return;
  }
  case ND_LINE_HEALTH: {
    nd_obj res;
    nd_service_health(svc, out);
    nd_obj_init(&res);
    nd_sysinfo_snapshot(gate, &res);
    nd_obj_child(out, "resources", &res);
    return;
  }
  case ND_LINE_REQUEST:
    if (nd_gate_busy(gate, why, sizeof why))
      nd_refusal_obj(out, r.id, "busy", why, strlen(why));
    else
      nd_service_handle(svc, &r, received, out);
    nd_request_free(&r);
    return;
  }
}

static void serve_stdio(nd_service *svc, const nd_gate *gate) {
  char *line = malloc(MAX_LINE + 2);
  if (!line) return;
  for (;;) {
    size_t n = 0, s, e;
    int c = 0;
    double received;
    nd_obj out;
    /* Bounded read: a line longer than MAX_LINE is drained and refused, never buffered whole. */
    while (n < MAX_LINE + 1 && (c = getchar()) != EOF) {
      line[n++] = (char)c;
      if (c == '\n') break;
    }
    if (n == 0) break;
    line[n] = 0;
    if (!nd_utf8_valid((const uint8_t *)line, n)) {
      char *t = nd_refusal("", "invalid_request", "input: stream did not contain valid UTF-8");
      emit_text(stdout, t);
      free(t);
      break;
    }
    received = nd_now();
    if (n > MAX_LINE && line[n - 1] != '\n') {
      char *t;
      while ((c = getchar()) != EOF && c != '\n') {
      }
      t = nd_refusal("", "invalid_request", "request line too long");
      emit_text(stdout, t);
      free(t);
      continue;
    }
    nd_rust_trim(line, n, &s, &e);
    if (s == e) continue;
    nd_rust_trim_end(line, n, &e);
    nd_obj_init(&out);
    answer(svc, gate, line, e, received, &out);
    emit_obj(stdout, &out);
  }
  free(line);
}

/* ---- socket server: an acceptor and one worker, a bounded queue between them ---- */

typedef struct {
  int fd;
  char *line;
  size_t len;
  double received;
} job;

typedef struct {
  nd_service *svc;
  const nd_gate *gate;
  job *q;
  size_t cap, head, n;
  pthread_mutex_t mu;
  pthread_cond_t cv;
} server;

static void reply(int fd, const char *text) {
  size_t n = strlen(text), off = 0;
  while (off < n) {
    ssize_t w = write(fd, text + off, n - off);
    if (w <= 0) break;
    off += (size_t)w;
  }
  if (write(fd, "\n", 1) < 0) {
  }
}

static void *worker(void *p) {
  server *s = p;
  for (;;) {
    job j;
    nd_obj out;
    size_t n;
    char *t;
    pthread_mutex_lock(&s->mu);
    while (s->n == 0) pthread_cond_wait(&s->cv, &s->mu);
    j = s->q[s->head];
    s->head = (s->head + 1) % s->cap;
    s->n--;
    pthread_mutex_unlock(&s->mu);
    nd_obj_init(&out);
    answer(s->svc, s->gate, j.line, j.len, j.received, &out);
    t = nd_obj_finish(&out, &n);
    if (t) reply(j.fd, t);
    free(t);
    close(j.fd);
    free(j.line);
  }
  return NULL;
}

static void reply_refusal(int fd, const char *id, const char *status, const char *detail) {
  char *t = nd_refusal(id, status, detail);
  if (t) reply(fd, t);
  free(t);
  close(fd);
}

static int serve_socket(nd_service *svc, const nd_gate *gate, const char *path, size_t queue) {
  struct sockaddr_un addr;
  struct stat st;
  server s;
  pthread_t th;
  int lfd;
  if (lstat(path, &st) == 0) {
    fprintf(stderr, "%s exists; remove it if no server owns it\n", path);
    return 2;
  }
  if (strlen(path) >= sizeof addr.sun_path) {
    fprintf(stderr, "%s: path too long for a Unix socket\n", path);
    return 2;
  }
  memset(&addr, 0, sizeof addr);
  addr.sun_family = AF_UNIX;
  strcpy(addr.sun_path, path);
  lfd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (lfd < 0 || bind(lfd, (struct sockaddr *)&addr, sizeof addr) != 0 || listen(lfd, 128) != 0 ||
      chmod(path, 0660) != 0) {
    fprintf(stderr, "%s: %s (os error %d)\n", path, strerror(errno), errno);
    return 2;
  }
  fprintf(stderr, "needle-jibo: serving on %s (queue %zu)\n", path, queue);
  memset(&s, 0, sizeof s);
  s.svc = svc;
  s.gate = gate;
  s.cap = queue;
  s.q = nd_calloc(queue, sizeof *s.q);
  if (!s.q) return 2;
  pthread_mutex_init(&s.mu, NULL);
  pthread_cond_init(&s.cv, NULL);
  if (pthread_create(&th, NULL, worker, &s) != 0) return 2;
  for (;;) {
    struct timeval tv;
    char *buf;
    size_t n = 0;
    int fd = accept(lfd, NULL, NULL), rerr = 0;
    double received;
    if (fd < 0) continue;
    /* A slow client holds the acceptor for at most this long. */
    tv.tv_sec = 2;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    if (!(buf = malloc(MAX_LINE + 2))) {
      close(fd);
      continue;
    }
    while (n < MAX_LINE + 1) {
      ssize_t r = read(fd, buf + n, MAX_LINE + 1 - n);
      if (r == 0) break;
      if (r < 0) {
        if (errno == EINTR) continue;
        rerr = errno;
        break;
      }
      n += (size_t)r;
    }
    buf[n] = 0;
    received = nd_now();
    if (rerr) {
      char d[256];
      snprintf(d, sizeof d, "read: %s (os error %d)", strerror(rerr), rerr);
      reply_refusal(fd, "", "invalid_request", d);
      free(buf);
    } else if (!nd_utf8_valid((const uint8_t *)buf, n)) {
      reply_refusal(fd, "", "invalid_request", "request is not UTF-8");
      free(buf);
    } else if (n > MAX_LINE) {
      reply_refusal(fd, "", "invalid_request", "request too long");
      free(buf);
    } else {
      pthread_mutex_lock(&s.mu);
      if (s.n < s.cap) {
        job *j = &s.q[(s.head + s.n) % s.cap];
        j->fd = fd;
        j->line = buf;
        j->len = n;
        j->received = received;
        s.n++;
        pthread_cond_signal(&s.cv);
        pthread_mutex_unlock(&s.mu);
      } else {
        nd_request r;
        char *refusal = NULL;
        pthread_mutex_unlock(&s.mu);
        if (nd_parse_line(buf, n, MAX_LINE, &r, &refusal) == ND_LINE_REQUEST) {
          reply_refusal(fd, r.id, "busy", "queue full");
          nd_request_free(&r);
        } else {
          reply_refusal(fd, "", "busy", "queue full");
        }
        free(refusal);
        free(buf);
      }
    }
  }
  return 0;
}

/* ---- bench ---- */

static int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y ? 1 : 0; /* partial_cmp, NaN as equal */
}

static void percentile(nd_obj *o, const char *key, const double *sorted, size_t n, double p) {
  if (!n) {
    nd_obj_null(o, key);
    return;
  }
  nd_obj_f64(o, key, sorted[(size_t)round(((double)n - 1.0) * p)]);
}

/* The f64 at "a.b" of a parsed object, as `v[a][b].as_f64()`. */
static int get2_f64(const nd_json_value *v, const char *a, const char *b, double *out) {
  const nd_json_value *x = nd_json_get_cstr(v, a);
  return x && (x = nd_json_get_cstr(x, b)) && nd_json_as_f64(x, out);
}

static int get2_u64(const nd_json_value *v, const char *a, const char *b, uint64_t *out) {
  const nd_json_value *x = nd_json_get_cstr(v, a);
  return x && (x = nd_json_get_cstr(x, b)) && nd_json_as_u64(x, out);
}

/* str::lines: split on '\n', drop one trailing '\r', no final empty line. */
typedef struct {
  const char *p, *end;
} lines_it;

static int next_line(lines_it *it, const char **s, size_t *n) {
  const char *nl;
  if (it->p >= it->end) return 0;
  nl = memchr(it->p, '\n', (size_t)(it->end - it->p));
  *s = it->p;
  *n = (size_t)((nl ? nl : it->end) - it->p);
  it->p = nl ? nl + 1 : it->end;
  if (nl && *n && (*s)[*n - 1] == '\r') (*n)--;
  else if (!nl && *n && (*s)[*n - 1] == '\r') (*n)--;
  return 1;
}

static int blank(const char *s, size_t n) {
  size_t a, b;
  nd_rust_trim(s, n, &a, &b);
  return a == b;
}

typedef struct {
  char *name;
  size_t count;
} status_count;

static int bench(nd_service *svc, const char *path, size_t reps, size_t warmup) {
  size_t tlen, nreq = 0, capreq = 0, rep, i, nwall = 0, nst = 0;
  char *text = read_text(path, &tlen);
  nd_request *reqs = NULL;
  lines_it it;
  const char *l;
  size_t ln;
  double cpu0, cpu1, t0, wall_total, prefill = 0.0, decode = 0.0, *walls;
  int has_cpu0, has_cpu1;
  uint64_t gen_tokens = 0, prompt_tokens = 0;
  status_count st[16];
  nd_gate none;
  it.p = text;
  it.end = text + tlen;
  while (next_line(&it, &l, &ln)) {
    nd_request r;
    char *refusal = NULL;
    nd_line_kind k;
    if (blank(l, ln)) continue;
    k = nd_parse_line(l, ln, MAX_LINE, &r, &refusal);
    if (k == ND_LINE_HEALTH) die(2, "health lines are not benchmark requests");
    if (k == ND_LINE_ERROR) die(2, refusal);
    if (nreq == capreq) {
      capreq = capreq ? capreq * 2 : 64;
      if (!(reqs = realloc(reqs, capreq * sizeof *reqs))) die(2, "out of memory");
    }
    reqs[nreq++] = r;
  }
  walls = nd_calloc(nreq * (reps ? reps : 1), sizeof *walls);
  if (!walls) die(2, "out of memory");
  has_cpu0 = nd_cpu_seconds(&cpu0);
  t0 = nd_now();
  for (rep = 0; rep < warmup + reps; rep++)
    for (i = 0; i < nreq; i++) {
      nd_obj out, b;
      size_t n;
      char *t;
      int measured = rep >= warmup;
      nd_json_doc *doc = NULL;
      char *m = NULL;
      nd_obj_init(&out);
      nd_service_handle(svc, &reqs[i], nd_now(), &out);
      t = nd_obj_finish(&out, &n);
      if (!t) die(2, "out of memory");
      if (measured && nd_serde_parse(t, n, &doc, &m) == ND_OK) {
        const nd_json_value *v = nd_json_root(doc), *s;
        double d;
        uint64_t u;
        const char *sname = "?";
        size_t slen = 1, k;
        walls[nwall++] = get2_f64(v, "timing", "wall_ms", &d) ? d : (double)NAN;
        if (get2_f64(v, "timing", "prefill_ms", &d)) prefill += d;
        if (get2_f64(v, "timing", "decode_ms", &d)) decode += d;
        if (get2_u64(v, "tokens", "generated", &u)) gen_tokens += u;
        if (get2_u64(v, "tokens", "prompt", &u)) prompt_tokens += u;
        s = nd_json_get_cstr(v, "status");
        if (s && s->type == ND_JSON_STRING) sname = s->u.str.ptr, slen = s->u.str.len;
        for (k = 0; k < nst; k++)
          if (strlen(st[k].name) == slen && !memcmp(st[k].name, sname, slen)) break;
        if (k == nst && nst < 16) {
          st[nst].name = malloc(slen + 1);
          if (!st[nst].name) die(2, "out of memory");
          memcpy(st[nst].name, sname, slen);
          st[nst].name[slen] = 0;
          st[nst++].count = 0;
        }
        if (k < nst) st[k].count++;
      }
      free(m);
      /* v["bench"] = {"rep": rep, "warmup": !measured} */
      if (doc || nd_serde_parse(t, n, &doc, &m) == ND_OK) {
        nd_obj_init(&out);
        nd_obj_from_json(&out, nd_json_root(doc));
        nd_obj_init(&b);
        nd_obj_u64(&b, "rep", rep);
        nd_obj_bool(&b, "warmup", !measured);
        nd_obj_child(&out, "bench", &b);
        emit_obj(stdout, &out);
      }
      nd_json_doc_free(doc);
      free(t);
    }
  wall_total = nd_now() - t0;
  has_cpu1 = nd_cpu_seconds(&cpu1);
  {
    nd_obj o, w, s;
    double *sorted = nd_calloc(nwall, sizeof *sorted);
    size_t k;
    if (!sorted) die(2, "out of memory");
    if (nwall) memcpy(sorted, walls, nwall * sizeof *sorted);
    qsort(sorted, nwall, sizeof *sorted, cmp_double);
    nd_obj_init(&o);
    nd_obj_bool(&o, "summary", 1);
    nd_obj_cstr(&o, "model_sha256", svc->sha256);
    nd_obj_u64(&o, "depth", svc->depth);
    nd_obj_cstr(&o, "kv_precision", nd_service_kv_name(svc));
    nd_obj_bool(&o, "constrained", svc->opts.constrain);
    nd_obj_bool(&o, "parallel", 0);
    nd_obj_u64(&o, "requests", nwall);
    nd_obj_init(&s);
    for (k = 0; k < nst; k++) nd_obj_u64(&s, st[k].name, st[k].count), free(st[k].name);
    nd_obj_child(&o, "statuses", &s);
    nd_obj_f64(&o, "load_ms", svc->load_ms);
    nd_obj_init(&w);
    percentile(&w, "p50", sorted, nwall, 0.5);
    percentile(&w, "p95", sorted, nwall, 0.95);
    percentile(&w, "p99", sorted, nwall, 0.99);
    if (nwall) nd_obj_f64(&w, "max", sorted[nwall - 1]); else nd_obj_null(&w, "max");
    nd_obj_child(&o, "wall_ms", &w);
    if (prompt_tokens) nd_obj_f64(&o, "prefill_ms_per_prompt_token", prefill / (double)prompt_tokens);
    else nd_obj_null(&o, "prefill_ms_per_prompt_token");
    if (gen_tokens) nd_obj_f64(&o, "decode_ms_per_token", decode / (double)gen_tokens);
    else nd_obj_null(&o, "decode_ms_per_token");
    if (has_cpu0 && has_cpu1) nd_obj_f64(&o, "process_cpu_s", cpu1 - cpu0);
    else nd_obj_null(&o, "process_cpu_s");
    nd_obj_f64(&o, "elapsed_s", wall_total);
    memset(&none, 0, sizeof none);
    nd_obj_init(&w);
    nd_sysinfo_snapshot(&none, &w);
    nd_obj_child(&o, "resources", &w);
    emit_obj(stdout, &o);
    free(sorted);
  }
  for (i = 0; i < nreq; i++) nd_request_free(&reqs[i]);
  free(reqs);
  free(walls);
  free(text);
  return 0;
}

/* ---- regrade ---- */

typedef struct {
  nd_request *v;
  size_t n, cap;
} req_table;

static const nd_request *find_req(const req_table *t, const char *id, size_t len) {
  size_t i;
  /* HashMap insert: the last request with an id wins. */
  for (i = t->n; i-- > 0;)
    if (strlen(t->v[i].id) == len && !memcmp(t->v[i].id, id, len)) return &t->v[i];
  return NULL;
}

static int regrade(const args *a) {
  nd_catalogue *cat = load_catalogue(a->tools);
  req_table reqs = {NULL, 0, 0};
  size_t len, ln, i;
  char *text, *rtext;
  lines_it it;
  const char *l;
  if (!a->requests) die(2, "regrade needs --requests FILE.jsonl");
  rtext = read_text(a->requests, &len);
  it.p = rtext;
  it.end = rtext + len;
  while (next_line(&it, &l, &ln)) {
    nd_request r;
    char *refusal = NULL;
    if (nd_parse_line(l, ln, MAX_LINE, &r, &refusal) == ND_LINE_REQUEST) {
      if (reqs.n == reqs.cap) {
        reqs.cap = reqs.cap ? reqs.cap * 2 : 64;
        if (!(reqs.v = realloc(reqs.v, reqs.cap * sizeof *reqs.v))) die(2, "out of memory");
      }
      reqs.v[reqs.n++] = r;
    }
    free(refusal);
  }
  text = read_text(a->model, &len);
  it.p = text;
  it.end = text + len;
  while (next_line(&it, &l, &ln)) {
    nd_json_doc *doc = NULL;
    char *err = NULL;
    const nd_json_value *v, *x, *tx, *sr;
    const nd_request *req = NULL;
    nd_stop stop;
    int has_stop = 0;
    nd_obj out;
    if (blank(l, ln)) continue;
    if (nd_serde_parse(l, ln, &doc, &err) != ND_OK) die(2, err ? err : "out of memory");
    v = nd_json_root(doc);
    if (nd_json_get_cstr(v, "summary")) {
      nd_json_doc_free(doc);
      continue;
    }
    x = nd_json_get_cstr(v, "request_id");
    if (x && x->type == ND_JSON_STRING)
      req = find_req(&reqs, x->u.str.ptr, x->u.str.len);
    else
      req = find_req(&reqs, "", 0);
    tx = nd_json_get_cstr(v, "text");
    sr = nd_json_get_cstr(v, "stop_reason");
    if (sr && sr->type == ND_JSON_STRING) has_stop = nd_stop_from_wire(sr->u.str.ptr, sr->u.str.len, &stop);
    if (!req || !tx || tx->type != ND_JSON_STRING || !has_stop) {
      /* Refused before generation (truncated, busy, ...): nothing to re-grade. */
      char *t = NULL;
      size_t tl;
      if (nd_json_to_string(v, ND_JSON_KEYS_SORTED, &t, &tl) == ND_OK) emit_text(stdout, t);
      free(t);
      nd_json_doc_free(doc);
      continue;
    }
    {
      nd_verdict g;
      double p = 0.0, budget_u = 0.0;
      uint64_t budget = 0;
      int has_p, ptr = 0;
      const nd_json_value *pt = nd_json_get_cstr(v, "prompt_truncated");
      nd_obj rg;
      if (pt && pt->type == ND_JSON_BOOL) ptr = pt->u.boolean;
      (void)budget_u;
      if (!get2_u64(v, "tokens", "budget", &budget)) budget = 0;
      if (nd_classify(cat, &a->opts, req->query, req->query_len, req->tools, req->n_tools, req->has_tools,
                      tx->u.str.ptr, tx->u.str.len, stop, ptr, (size_t)budget, &g) != ND_OK)
        die(2, "out of memory");
      x = nd_json_get_cstr(v, "confidence_raw");
      has_p = x && nd_json_as_f64(x, &p);
      nd_gate_confidence(&g, a->opts.has_min_confidence, a->opts.min_confidence, has_p, (float)p);
      if (v->type != ND_JSON_OBJECT) die(2, "response is not an object");
      nd_obj_init(&out);
      nd_obj_from_json(&out, v);
      nd_obj_remove(&out, "detail");
      nd_obj_remove(&out, "calls");
      nd_obj_remove(&out, "rejected_calls");
      nd_obj_remove(&out, "ungrounded");
      nd_obj_cstr(&out, "status", g.status);
      if (g.grounded >= 0) nd_obj_bool(&out, "grounded", g.grounded); else nd_obj_null(&out, "grounded");
      if (g.detail_len) nd_obj_str(&out, "detail", g.detail, g.detail_len);
      if (g.calls) nd_obj_raw(&out, "calls", g.calls, strlen(g.calls));
      if (g.rejected) nd_obj_raw(&out, "rejected_calls", g.rejected, strlen(g.rejected));
      if (g.ungrounded.n) {
        nd_json_writer w;
        nd_json_writer_init(&w, 0);
        nd_json_writer_begin_array(&w);
        for (i = 0; i < g.ungrounded.n; i++) nd_json_writer_string(&w, g.ungrounded.items[i], g.ungrounded.lens[i]);
        nd_json_writer_end_array(&w);
        if (!w.err) nd_obj_raw(&out, "ungrounded", w.data, w.len);
        nd_json_writer_free(&w);
      }
      nd_obj_init(&rg);
      nd_obj_cstr(&rg, "grounding", nd_grounding_name(a->opts.grounding));
      if (a->opts.has_min_confidence) nd_obj_f64(&rg, "min_confidence", (double)a->opts.min_confidence);
      else nd_obj_null(&rg, "min_confidence");
      nd_obj_child(&out, "regraded", &rg);
      emit_obj(stdout, &out);
      nd_verdict_free(&g);
    }
    nd_json_doc_free(doc);
  }
  for (i = 0; i < reqs.n; i++) nd_request_free(&reqs.v[i]);
  free(reqs.v);
  free(text);
  free(rtext);
  nd_catalogue_free(cat);
  return 0;
}

int main(int argc, char **argv) {
  args a;
  nd_catalogue *cat;
  nd_service *svc;
  char *err = NULL;
  int rc = 0;
  signal(SIGPIPE, SIG_IGN); /* as Rust's runtime: a closed peer is an error, not a death */
  parse_args(argc, argv, &a);
  if (!strcmp(a.cmd, "regrade")) return regrade(&a);
  cat = load_catalogue(a.tools);
  svc = nd_service_load(a.model, cat, a.has_layers, a.layers, &a.limits, &a.opts, &err);
  if (!svc) {
    fprintf(stderr, "%s\n", err ? err : "out of memory");
    return 1;
  }
  if (!strcmp(a.cmd, "info")) {
    nd_obj h, r;
    nd_obj_init(&h);
    nd_service_health(svc, &h);
    nd_obj_init(&r);
    nd_sysinfo_snapshot(&a.gate, &r);
    nd_obj_child(&h, "resources", &r);
    emit_obj(stdout, &h);
  } else if (!strcmp(a.cmd, "run")) {
    if (!a.query) {
      fprintf(stderr, "run needs --query\n");
      rc = 2;
    } else {
      nd_request r;
      nd_obj out;
      char why[512];
      memset(&r, 0, sizeof r);
      r.id = (char *)a.request_id;
      r.query = (char *)a.query;
      r.query_len = strlen(a.query);
      nd_obj_init(&out);
      if (nd_gate_busy(&a.gate, why, sizeof why))
        nd_refusal_obj(&out, r.id, "busy", why, strlen(why));
      else
        nd_service_handle(svc, &r, nd_now(), &out);
      emit_obj(stdout, &out);
    }
  } else if (!strcmp(a.cmd, "serve")) {
    if (a.socket)
      rc = serve_socket(svc, &a.gate, a.socket, a.queue);
    else
      serve_stdio(svc, &a.gate);
  } else if (!strcmp(a.cmd, "bench")) {
    if (!a.requests) {
      fprintf(stderr, "bench needs --requests FILE.jsonl\n");
      rc = 2;
    } else {
      rc = bench(svc, a.requests, a.reps, a.warmup);
    }
  } else {
    fprintf(stderr, "unknown command %s\n%s\n", a.cmd, USAGE);
    rc = 2;
  }
  nd_service_free(svc);
  return rc;
}
