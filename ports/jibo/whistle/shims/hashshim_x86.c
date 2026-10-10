/* std::__1::__hash_memory (libc++ >= 19), absent from the host's libc++ 18. Any hash is valid:
   it only has to agree with itself inside one process (unordered containers). FNV-1a 64. */
#include <stddef.h>
#include <stdint.h>
size_t hash_memory_shim(const void *p, size_t n) __asm__("_ZNSt3__113__hash_memoryEPKvm");
size_t hash_memory_shim(const void *p, size_t n) {
  const unsigned char *b = p; uint64_t h = 1469598103934665603ULL; size_t i;
  for (i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ULL;
  return (size_t)h;
}
