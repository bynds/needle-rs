/* whistle_bench MODEL CLIP.f32...: exact user-space instructions (perfvm) for upstream's public
   needle_load, needle_embed and needle_transcribe on 16 kHz float32 clips. The VM has one CPU, so
   counting every user-space instruction on CPU 0 includes the engine's worker threads. */
#include <linux/perf_event.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "needle.h"
static int fd = -1;
static uint64_t rd(void) { uint64_t v = 0; if (read(fd, &v, 8) != 8) return 0; return v; }
static void *slurp(const char *p, long *n) {
  FILE *f = fopen(p, "rb"); void *b; if (!f) return NULL;
  fseek(f, 0, SEEK_END); *n = ftell(f); rewind(f); b = malloc(*n);
  if (fread(b, 1, *n, f) != (size_t)*n) { fclose(f); return NULL; } fclose(f); return b;
}
int main(int argc, char **argv) {
  struct perf_event_attr a; long n; unsigned char *m; uint64_t t0; int i; char out[8192];
  memset(&a, 0, sizeof a); a.type = PERF_TYPE_HARDWARE; a.size = sizeof a;
  a.config = PERF_COUNT_HW_INSTRUCTIONS; a.exclude_kernel = 1; a.exclude_hv = 1;
  fd = (int)syscall(__NR_perf_event_open, &a, -1, 0, -1, 0);
  if (fd < 0) { perror("perf_event_open"); return 1; }
  m = slurp(argv[1], &n);
  t0 = rd();
  if (needle_load(m, (unsigned long long)n) < 0) { fprintf(stderr, "load: %s\n", needle_last_error()); return 1; }
  printf("{\"op\":\"load\",\"instructions\":%llu}\n", (unsigned long long)(rd() - t0));
  for (i = 2; i < argc; i++) {
    long b; float *pcm = slurp(argv[i], &b); int s = (int)(b / 4), k; float *e; uint64_t ie, it;
    k = needle_embed(NULL, pcm, s, NULL, 0); e = malloc(k * sizeof(float));
    t0 = rd(); needle_embed(NULL, pcm, s, e, k); ie = rd() - t0;
    t0 = rd(); needle_transcribe(pcm, s, "en", NULL, 0, out, sizeof out); it = rd() - t0;
    printf("{\"clip\":\"%s\",\"seconds\":%.2f,\"frames\":%d,\"embed\":%llu,\"transcribe\":%llu,\"result\":%s}\n",
           argv[i], s / 16000.0, k / 512, (unsigned long long)ie, (unsigned long long)it, out);
    fflush(stdout); free(e); free(pcm);
  }
  return 0;
}
