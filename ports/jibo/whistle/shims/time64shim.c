/* glibc >= 2.34's 64-bit time entry points, used by a libc++ built with _TIME_BITS=64, on glibc
   2.21's 32-bit time_t calls. Layout of glibc's struct __timespec64 on 32-bit little-endian ARM:
   int64 seconds, int32 nanoseconds, int32 padding. Values fit until 2038. */
#include <pthread.h>
#include <stdint.h>
#include <time.h>
struct ts64 { int64_t sec; int32_t nsec; int32_t pad; };
int __clock_gettime64(clockid_t c, struct ts64 *t) {
  struct timespec s; int r = clock_gettime(c, &s);
  if (r == 0) { t->sec = s.tv_sec; t->nsec = (int32_t)s.tv_nsec; t->pad = 0; }
  return r;
}
int __nanosleep64(const struct ts64 *req, struct ts64 *rem) {
  struct timespec q = { (time_t)req->sec, req->nsec }, m; int r = nanosleep(&q, &m);
  if (rem) { rem->sec = m.tv_sec; rem->nsec = (int32_t)m.tv_nsec; rem->pad = 0; }
  return r;
}
int __pthread_cond_timedwait64(pthread_cond_t *c, pthread_mutex_t *m, const struct ts64 *t) {
  struct timespec s = { (time_t)t->sec, t->nsec };
  return pthread_cond_timedwait(c, m, &s);
}
