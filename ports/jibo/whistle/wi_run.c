/* wi_run WHISTLE.cact MODEL.wim CLIP.f32...: closed-set command recognition on 16 kHz mono float32
 * clips. Embeds each clip with upstream libneedle's public needle_embed, classifies it with
 * wi_intent, and prints one JSON line per clip. Under perfvm it also reports the user-space
 * instructions of each stage (all threads: the VM has one CPU). */
#include <linux/perf_event.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "needle.h"
#include "wi_intent.h"

long syscall(long number, ...);
static int pmu = -1;
static uint64_t icount(void) {
  uint64_t v = 0;
  if (pmu < 0 || read(pmu, &v, 8) != 8) return 0;
  return v;
}
static void *slurp(const char *p, long *n) {
  FILE *f = fopen(p, "rb");
  void *b;
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  *n = ftell(f);
  rewind(f);
  b = malloc(*n ? (size_t)*n : 1);
  if (fread(b, 1, (size_t)*n, f) != (size_t)*n) {
    fclose(f);
    free(b);
    return NULL;
  }
  fclose(f);
  return b;
}
int main(int argc, char **argv) {
  struct perf_event_attr a;
  wi_model m;
  long n;
  unsigned char *cact;
  int i;
  if (argc < 4) {
    fprintf(stderr, "usage: wi_run WHISTLE.cact MODEL.wim CLIP.f32...\n");
    return 2;
  }
  memset(&a, 0, sizeof a);
  a.type = PERF_TYPE_HARDWARE, a.size = sizeof a, a.config = PERF_COUNT_HW_INSTRUCTIONS;
  a.exclude_kernel = 1, a.exclude_hv = 1;
  pmu = (int)syscall(__NR_perf_event_open, &a, -1, 0, -1, 0);
  if (!(cact = slurp(argv[1], &n)) || needle_load(cact, (unsigned long long)n) < 0) {
    fprintf(stderr, "load %s: %s\n", argv[1], needle_last_error());
    return 1;
  }
  if (wi_load(argv[2], &m)) {
    fprintf(stderr, "load %s failed\n", argv[2]);
    return 1;
  }
  for (i = 3; i < argc; i++) {
    long b;
    float *pcm = slurp(argv[i], &b), *emb, *prob;
    int samples, k, best, frames;
    uint64_t t0, t1, t2;
    if (!pcm) {
      fprintf(stderr, "read %s\n", argv[i]);
      return 1;
    }
    samples = (int)(b / 4);
    k = needle_embed(NULL, pcm, samples, NULL, 0);
    if (k <= 0) {
      printf("{\"clip\":\"%s\",\"error\":\"%s\"}\n", argv[i], needle_last_error());
      free(pcm);
      continue;
    }
    emb = malloc((size_t)k * sizeof(float));
    prob = malloc((size_t)m.classes * sizeof(float));
    t0 = icount();
    k = needle_embed(NULL, pcm, samples, emb, k);
    t1 = icount();
    frames = k / 512;
    best = wi_classify(&m, emb, frames, 512, prob);
    t2 = icount();
    printf("{\"clip\":\"%s\",\"seconds\":%.3f,\"frames\":%d,\"label\":\"%s\",\"p\":%.4f", argv[i], samples / 16000.0,
           frames, m.names[best], prob[best]);
    if (pmu >= 0)
      printf(",\"embed_instructions\":%llu,\"classify_instructions\":%llu", (unsigned long long)(t1 - t0),
             (unsigned long long)(t2 - t1));
    printf("}\n");
    fflush(stdout);
    free(emb), free(prob), free(pcm);
  }
  wi_free(&m);
  return 0;
}
