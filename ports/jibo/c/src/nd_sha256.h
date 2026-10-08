/* nd_sha256.h: SHA-256 (FIPS 180-4), as runner/src/sha256.rs. */
#ifndef ND_SHA256_H
#define ND_SHA256_H

#include "nd_common.h"

/* Lowercase hex digest of data[0..len) into out (65 bytes, NUL-terminated). */
void nd_sha256_hex(const uint8_t *data, size_t len, char out[65]);

#endif
