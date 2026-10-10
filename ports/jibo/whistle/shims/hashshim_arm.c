/* std::__1::__hash_memory (libc++ >= 19), absent from Ubuntu's armhf libc++ 18. Any hash is valid:
   it only has to agree with itself inside one process (unordered containers). FNV-1a 32 (size_t is 32 bits on ARMv7). */
#include <stddef.h>
#include <stdint.h>
size_t hash_memory_shim(const void *p, size_t n) __asm__("_ZNSt3__113__hash_memoryEPKvj");
size_t hash_memory_shim(const void *p, size_t n) {
  const unsigned char *b = p; uint32_t h = 2166136261u; size_t i;
  for (i = 0; i < n; i++) h = (h ^ b[i]) * 16777619u;
  return (size_t)h;
}
