/* bench_engine: per-operation cost of the C99 engine on the workloads of
 * crates/needle-infer/examples/v3_profile.rs (prefill, N decode steps, a tool-prefix cache hit),
 * with the same JSON lines, so the two engines compare op by op. Build with -DND_PROFILE.
 *   bench_engine MODEL TOOLS.json QUERY [--decode N] [--clock instructions|ns] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nd_icount.h"
#include "nd_engine.h"
#include "nd_prof.h"

static uint64_t instructions(void) { return nd_icount_read(); }

static uint64_t nanos(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

static void report(const char *workload, const char *clock, uint64_t total, const char *extra) {
  uint64_t tot[ND_P_N], calls[ND_P_N];
  int i;
  nd_prof_take(tot, calls);
  printf("{\"engine\":\"c\",\"workload\":\"%s\",\"clock\":\"%s\",\"total\":%llu,%s\"ops\":{", workload, clock,
         (unsigned long long)total, extra);
  for (i = 0; i < ND_P_N; i++)
    printf("%s\"%s\":[%llu,%llu]", i ? "," : "", nd_prof_names[i], (unsigned long long)tot[i],
           (unsigned long long)calls[i]);
  printf("}}\n");
}

static uint32_t argmax(const float *v, size_t n) {
  size_t i, b = 0;
  for (i = 1; i < n; i++)
    if (v[i] > v[b]) b = i;
  return (uint32_t)b;
}

static char *slurp(const char *p) {
  FILE *f = fopen(p, "rb");
  long l;
  char *b;
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  l = ftell(f);
  fseek(f, 0, SEEK_SET);
  b = malloc((size_t)l + 1);
  if (!b || fread(b, 1, (size_t)l, f) != (size_t)l) exit(2);
  b[l] = 0;
  fclose(f);
  return b;
}

int main(int argc, char **argv) {
  nd_engine *g;
  nd_err e = {""};
  uint32_t *ids;
  size_t n, steps = 32, rows, i, split = 0;
  const char *want = "auto", *clock;
  uint64_t (*now)(void);
  uint64_t t0;
  float *logits;
  nd_cache *cache, *stored, *c;
  char extra[96], *tools;
  int a;
  if (argc < 4) {
    fprintf(stderr, "usage: bench_engine MODEL TOOLS.json QUERY [--decode N] [--clock instructions|ns]\n");
    return 2;
  }
  for (a = 4; a + 1 < argc; a += 2) {
    if (!strcmp(argv[a], "--decode")) steps = (size_t)strtoul(argv[a + 1], NULL, 10);
    else if (!strcmp(argv[a], "--clock")) want = argv[a + 1];
  }
  if (strcmp(want, "ns") != 0 && nd_icount_open()) {
    clock = "instructions";
    now = instructions;
  } else {
    if (!strcmp(want, "instructions")) {
      perror("perf_event_open");
      return 1;
    }
    clock = "ns";
    now = nanos;
  }
  nd_prof_set_clock(now);
  if (!(tools = slurp(argv[2])) || nd_engine_load(argv[1], 0, &g, &e) ||
      nd_engine_prompt_ids(g, argv[3], tools, NULL, &ids, &n)) {
    fprintf(stderr, "load: %s\n", e.msg);
    return 1;
  }
  rows = nd_logit_rows(&g->model->cfg);
  logits = malloc(rows * sizeof *logits);
  nd_prof_take((uint64_t[ND_P_N]){0}, (uint64_t[ND_P_N]){0});

  cache = nd_cache_new(&g->model->cfg, n + steps, ND_KV_F32);
  t0 = now();
  if (nd_prefill(g->model, ids, n, cache, logits)) return 1;
  snprintf(extra, sizeof extra, "\"tokens\":%zu,", n);
  report("prefill", clock, now() - t0, extra);

  t0 = now();
  for (i = 0; i < steps; i++)
    if (nd_decode_step(g->model, cache, argmax(logits, rows), logits)) return 1;
  snprintf(extra, sizeof extra, "\"tokens\":%zu,", steps);
  report("decode", clock, now() - t0, extra);

  for (i = 0; i < n; i++)
    if ((int32_t)ids[i] == g->tools_end_id) {
      split = i + 1;
      break;
    }
  stored = nd_cache_new(&g->model->cfg, split, ND_KV_F32);
  if (!split || !stored || nd_prefill(g->model, ids, split, stored, logits)) return 1;
  nd_prof_take((uint64_t[ND_P_N]){0}, (uint64_t[ND_P_N]){0});
  t0 = now();
  c = nd_cache_clone(stored);
  for (i = split; i < n; i++)
    if (nd_decode_step(g->model, c, ids[i], logits)) return 1;
  snprintf(extra, sizeof extra, "\"tokens\":%zu,\"reused\":%zu,", n - split, split);
  report("prefix_hit", clock, now() - t0, extra);
  fprintf(stderr, "argmax %u\n", argmax(logits, rows));
  nd_cache_free(c);
  nd_cache_free(stored);
  nd_cache_free(cache);
  nd_engine_free(g);
  free(ids);
  free(tools);
  free(logits);
  return 0;
}
