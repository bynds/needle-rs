/* test_grammar.c: tests for nd_grammar.
 *
 *   test_grammar            the unit tests of constrained.rs, transcribed
 *   test_grammar DIR        additionally replays a reference dump made by a throwaway Rust program
 *                           against needle-infer (the real needle3.cact tokenizer):
 *     DIR/pieces.bin    piece types and surfaces; byte table built here must equal DIR/table.bin
 *     DIR/tools_in.bin  odd tool-JSON inputs; ToolDef::from_json results in DIR/tools_out.txt
 *     DIR/cats.bin      tool catalogues; DIR/scen.txt: decoder scenarios (U = update(id),
 *                       B = feed_bytes(hex), M = expected logit_mask summary + used keys)
 *     DIR/synth.txt     synthetic token tables (duplicate ids, empty table, odd vocab sizes)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nd_grammar.h"

static long g_fail, g_checks;

#define CHECK(c)                                                            \
  do {                                                                      \
    g_checks++;                                                             \
    if (!(c)) {                                                             \
      g_fail++;                                                             \
      fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #c); \
    }                                                                       \
  } while (0)

/* ─── utilities ─── */

typedef struct {
  char *p;
  size_t n, c;
} sbuf;
static void sb_add(sbuf *s, const char *t, size_t n) {
  if (s->n + n + 1 > s->c) {
    size_t nc = s->c ? s->c : 256;
    while (nc < s->n + n + 1) nc *= 2;
    s->p = (char *)realloc(s->p, nc);
    if (!s->p) exit(2);
    s->c = nc;
  }
  memcpy(s->p + s->n, t, n);
  s->n += n;
  s->p[s->n] = 0;
}
static void sb_str(sbuf *s, const char *t) { sb_add(s, t, strlen(t)); }
static void sb_hex(sbuf *s, const void *v, size_t n) {
  static const char hx[] = "0123456789abcdef";
  const unsigned char *t = (const unsigned char *)v;
  size_t i;
  if (n == 0) {
    sb_str(s, "-");
    return;
  }
  for (i = 0; i < n; i++) {
    char b[2];
    b[0] = hx[t[i] >> 4];
    b[1] = hx[t[i] & 15];
    sb_add(s, b, 2);
  }
}
static void sb_ulong(sbuf *s, unsigned long v) {
  char b[32];
  sprintf(b, "%lu", v);
  sb_str(s, b);
}

static unsigned char *read_file(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  unsigned char *b;
  long sz;
  if (!f) {
    perror(path);
    exit(2);
  }
  fseek(f, 0, SEEK_END);
  sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  b = (unsigned char *)malloc((size_t)sz + 1);
  if (!b || fread(b, 1, (size_t)sz, f) != (size_t)sz) exit(2);
  fclose(f);
  b[sz] = 0;
  *n = (size_t)sz;
  return b;
}

static uint32_t rd32(const unsigned char *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static int unhex(const char *h, unsigned char *out, size_t *n) {
  size_t i = 0;
  if (h[0] == '-' && (h[1] == 0 || h[1] == ' ' || h[1] == '\n')) {
    *n = 0;
    return 0;
  }
  while (h[2 * i] && h[2 * i] != ' ' && h[2 * i] != '\n') {
    unsigned v;
    if (sscanf(h + 2 * i, "%2x", &v) != 1) return -1;
    out[i] = (unsigned char)v;
    i++;
  }
  *n = i;
  return 0;
}

static void mask_line(const float *m, size_t n, sbuf *s) {
  uint64_t h = 0xcbf29ce484222325u;
  size_t i, cnt = 0;
  char b[64];
  for (i = 0; i < n; i++) {
    uint32_t bits;
    int k;
    memcpy(&bits, &m[i], 4);
    for (k = 0; k < 4; k++) {
      h ^= (bits >> (8 * k)) & 0xffu;
      h *= 0x100000001b3u;
    }
    if (m[i] == 0.0f) cnt++;
    if (!(m[i] == 0.0f || m[i] == -1e9f)) {
      g_fail++;
      fprintf(stderr, "mask value not 0 / -1e9\n");
    }
  }
  sprintf(b, "M %08lx%08lx %lu", (unsigned long)(h >> 32), (unsigned long)(h & 0xffffffffu),
          (unsigned long)cnt);
  sb_str(s, b);
  if (cnt == n) {
    sb_str(s, " ALL");
  } else if (cnt <= 512) {
    for (i = 0; i < n; i++) {
      if (m[i] == 0.0f) {
        sb_str(s, " ");
        sb_ulong(s, (unsigned long)i);
      }
    }
  }
}

static void tools_line(const nd_tooldefs *t, sbuf *s) {
  size_t i, k;
  sb_str(s, "T ");
  sb_ulong(s, (unsigned long)t->n);
  for (i = 0; i < t->n; i++) {
    const nd_tooldef *d = &t->tools[i];
    sb_str(s, " ");
    sb_hex(s, d->name, d->name_len);
    sb_str(s, " ");
    sb_hex(s, d->snake_name, d->snake_name_len);
    sb_str(s, " ");
    sb_ulong(s, (unsigned long)d->n_param_keys);
    for (k = 0; k < d->n_param_keys; k++) {
      sb_str(s, " ");
      sb_hex(s, d->param_keys[k], d->param_key_lens[k]);
    }
  }
}

static nd_tooldefs parse_tools_or_die(const char *s, size_t n) {
  nd_tooldefs t;
  nd_err err;
  if (nd_grammar_parse_tools(s, n, &t, &err)) {
    fprintf(stderr, "parse_tools failed: %s\n", err.msg);
    exit(2);
  }
  return t;
}

/* ─── unit tests from constrained.rs ─── */

static nd_token_bytes g_ascii_tbl[128];
static uint8_t g_ascii_bytes[128];

static void ascii_table(void) {
  int i;
  for (i = 0; i < 128; i++) {
    g_ascii_bytes[i] = (uint8_t)i;
    g_ascii_tbl[i].id = (uint32_t)i;
    g_ascii_tbl[i].bytes = &g_ascii_bytes[i];
    g_ascii_tbl[i].len = 1;
  }
}

static int feed(nd_grammar *g, const char *s) {
  return nd_grammar_feed_bytes(g, (const uint8_t *)s, strlen(s));
}

static void unit_tests(void) {
  nd_tooldefs defs;
  nd_grammar *g = NULL, *plain = NULL;
  nd_err err;
  float mask[128];
  size_t n;
  const char *s;
  ascii_table();

  /* unique_arg_keys_excludes_a_written_key */
  {
    const char *j = "[{\"name\":\"t\",\"description\":\"x\",\"parameters\":"
                    "{\"alpha\":{\"type\":\"string\"},\"beta\":{\"type\":\"string\"}}}]";
    defs = parse_tools_or_die(j, strlen(j));
    CHECK(defs.n == 1);
    CHECK(defs.n == 1 && defs.tools[0].n_param_keys == 2);
    CHECK(nd_grammar_new(defs.tools, defs.n, g_ascii_tbl, 128, &g, &err) == 0);
    nd_grammar_set_unique_arg_keys(g, 1);
    feed(g, "[{\"name\":\"t\",\"arguments\":{\"");
    nd_grammar_logit_mask(g, mask, 128);
    CHECK(mask['a'] == 0.0f);
    CHECK(mask['b'] == 0.0f);
    feed(g, "alpha\":\"x\",\"");
    CHECK(nd_grammar_used_key_count(g) == 1);
    s = nd_grammar_used_key(g, 0, &n);
    CHECK(n == 5 && memcmp(s, "alpha", 5) == 0);
    nd_grammar_logit_mask(g, mask, 128);
    CHECK(mask['a'] < -1.0f);
    CHECK(mask['b'] == 0.0f);
    CHECK(nd_grammar_new(defs.tools, defs.n, g_ascii_tbl, 128, &plain, &err) == 0);
    feed(plain, "[{\"name\":\"t\",\"arguments\":{\"alpha\":\"x\",\"");
    nd_grammar_logit_mask(plain, mask, 128);
    CHECK(mask['a'] == 0.0f);
    nd_grammar_free(g);
    nd_grammar_free(plain);
    nd_tooldefs_free(&defs);
  }
  /* unique_arg_keys_resets_between_calls */
  {
    const char *j = "[{\"name\":\"t\",\"description\":\"x\","
                    "\"parameters\":{\"alpha\":{\"type\":\"string\"}}}]";
    defs = parse_tools_or_die(j, strlen(j));
    CHECK(nd_grammar_new(defs.tools, defs.n, g_ascii_tbl, 128, &g, &err) == 0);
    nd_grammar_set_unique_arg_keys(g, 1);
    feed(g, "[{\"name\":\"t\",\"arguments\":{\"alpha\":\"x\"},");
    feed(g, "{\"name\":\"t\",\"arguments\":{\"");
    CHECK(nd_grammar_used_key_count(g) == 0);
    nd_grammar_logit_mask(g, mask, 128);
    CHECK(mask['a'] == 0.0f);
    nd_grammar_free(g);
    nd_tooldefs_free(&defs);
  }
  /* test_state_machine_*: via the decoder */
  {
    CHECK(nd_grammar_new(NULL, 0, NULL, 0, &g, &err) == 0);
    feed(g, "[{\"name\":\"get_weather\",\"arguments\":{}}]");
    s = nd_grammar_current_function(g, &n);
    CHECK(n == 11 && memcmp(s, "get_weather", 11) == 0);
    nd_grammar_free(g);
    CHECK(nd_grammar_new(NULL, 0, NULL, 0, &g, &err) == 0);
    feed(g, "[{\"name\":\"get_weather\",\"arguments\":{\"");
    CHECK(nd_grammar_get_state(g) == ND_GRAMMAR_IN_ARG_KEY);
    nd_grammar_constrained_buf(g, &n);
    CHECK(n == 0);
    nd_grammar_free(g);
    CHECK(nd_grammar_new(NULL, 0, NULL, 0, &g, &err) == 0);
    feed(g, "[{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Paris\",\"unit\":\"celsius\"}}]");
    CHECK(nd_grammar_get_state(g) == ND_GRAMMAR_FREE);
    s = nd_grammar_current_function(g, &n);
    CHECK(n == 11 && memcmp(s, "get_weather", 11) == 0);
    CHECK(nd_grammar_table_size(g) == 1);
    nd_grammar_free(g);
  }
  /* test_parse_tools_flat / json_schema */
  {
    const char *j = "[{\"name\":\"get_weather\",\"description\":\"Get weather\",\"parameters\":"
                    "{\"location\":{\"type\":\"string\"},\"unit\":{\"type\":\"string\"}}}]";
    const char *k = "[{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":"
                    "{\"location\":{\"type\":\"string\"},\"unit\":{\"type\":\"string\"}},"
                    "\"required\":[\"location\"]}}]";
    defs = parse_tools_or_die(j, strlen(j));
    CHECK(defs.n == 1 && defs.tools[0].n_param_keys == 2 &&
          strcmp(defs.tools[0].param_keys[0], "location") == 0 &&
          strcmp(defs.tools[0].param_keys[1], "unit") == 0 &&
          strcmp(defs.tools[0].snake_name, "get_weather") == 0);
    nd_tooldefs_free(&defs);
    defs = parse_tools_or_die(k, strlen(k));
    CHECK(defs.n == 1 && defs.tools[0].n_param_keys == 2 &&
          strcmp(defs.tools[0].param_keys[0], "location") == 0 &&
          strcmp(defs.tools[0].param_keys[1], "unit") == 0);
    nd_tooldefs_free(&defs);
    CHECK(nd_grammar_parse_tools("[{\"name\":\"\xff\"}]", 14, &defs, &err) == ND_E_FORMAT);
  }
  /* test_constrained_decoder_name_mask */
  {
    nd_tooldef td;
    char *keys[2];
    size_t lens[2];
    nd_token_bytes tb[3];
    keys[0] = (char *)"location";
    keys[1] = (char *)"unit";
    lens[0] = 8;
    lens[1] = 4;
    td.name = td.snake_name = (char *)"get_weather";
    td.name_len = td.snake_name_len = 11;
    td.param_keys = keys;
    td.param_key_lens = lens;
    td.n_param_keys = 2;
    tb[0].id = 0;
    tb[0].bytes = (const uint8_t *)"get";
    tb[0].len = 3;
    tb[1].id = 1;
    tb[1].bytes = (const uint8_t *)" get";
    tb[1].len = 4;
    tb[2].id = 2;
    tb[2].bytes = (const uint8_t *)"set";
    tb[2].len = 3;
    CHECK(nd_grammar_new(&td, 1, tb, 3, &g, &err) == 0);
    feed(g, "[{\"name\":\"");
    CHECK(nd_grammar_get_state(g) == ND_GRAMMAR_IN_NAME);
    nd_grammar_logit_mask(g, mask, 3);
    CHECK(mask[0] == 0.0f);
    CHECK(mask[1] < 0.0f);
    CHECK(mask[2] < 0.0f);
    nd_grammar_free(g);
  }
  /* byte table on hand-made pieces */
  {
    const char *pieces[9] = {"<unk>", "<s>", "\xe2\x96\x81hello\xe2\x96\x81", "<0x41>", "<0x+f>",
                             "<0xZZ>", "<0x", "<0\xc3\xa9>", "<tool_call>"};
    size_t plen[9];
    uint8_t types[9] = {1, 2, 0, 4, 4, 4, 4, 4, 3};
    nd_token_bytes *t = NULL;
    int i;
    for (i = 0; i < 9; i++) plen[i] = strlen(pieces[i]);
    CHECK(nd_grammar_byte_table_from_pieces(pieces, plen, types, 9, &t, &err) == 0);
    if (t) {
      CHECK(t[0].len == 0 && t[1].len == 0);
      CHECK(t[2].len == 7 && memcmp(t[2].bytes, " hello ", 7) == 0);
      CHECK(t[3].len == 1 && t[3].bytes[0] == 0x41);
      CHECK(t[4].len == 1 && t[4].bytes[0] == 0x0f);
      CHECK(t[5].len == 0 && t[6].len == 0 && t[7].len == 0);
      CHECK(t[8].len == 11 && memcmp(t[8].bytes, "<tool_call>", 11) == 0);
      CHECK(t[8].id == 8);
    }
    free(t);
  }
}

/* ─── reference replay ─── */

typedef struct {
  char **s;
  size_t *n;
  size_t count;
} strlist;

static strlist read_lenprefixed(const char *path) {
  size_t n, p = 0;
  unsigned char *b = read_file(path, &n);
  strlist l = {0, 0, 0};
  while (p + 4 <= n) {
    size_t len = rd32(b + p);
    l.s = (char **)realloc(l.s, (l.count + 1) * sizeof *l.s);
    l.n = (size_t *)realloc(l.n, (l.count + 1) * sizeof *l.n);
    l.s[l.count] = (char *)malloc(len + 1);
    memcpy(l.s[l.count], b + p + 4, len);
    l.s[l.count][len] = 0;
    l.n[l.count] = len;
    l.count++;
    p += 4 + len;
  }
  free(b);
  return l;
}

static void free_strlist(strlist *l) {
  size_t i;
  for (i = 0; i < l->count; i++) free(l->s[i]);
  free(l->s);
  free(l->n);
}

static char *next_line(char **cur) {
  char *l = *cur, *nl;
  if (!*l) return NULL;
  nl = strchr(l, '\n');
  if (nl) {
    *nl = 0;
    *cur = nl + 1;
  } else {
    *cur = l + strlen(l);
  }
  return l;
}

static int replay_dir(const char *dir) {
  char path[1024];
  size_t n, i, p, ntbl;
  unsigned char *pb, *tb;
  char **pieces;
  size_t *plens;
  uint8_t *types;
  uint32_t npieces;
  nd_token_bytes *table = NULL;
  nd_err err;
  strlist ins, cats;
  char *txt, *cur, *line;
  sbuf got = {0, 0, 0};
  long mism = 0, nsteps = 0, nscen = 0, ntools = 0;
  nd_tooldefs *catdefs;
  float *mask = NULL;
  size_t mask_cap = 0;
  unsigned char *hexbuf = (unsigned char *)malloc(1 << 20);

  /* pieces -> byte table, compared with byte_table() */
  sprintf(path, "%s/pieces.bin", dir);
  pb = read_file(path, &n);
  npieces = rd32(pb);
  pieces = (char **)calloc(npieces, sizeof *pieces);
  plens = (size_t *)calloc(npieces, sizeof *plens);
  types = (uint8_t *)calloc(npieces, 1);
  p = 4;
  for (i = 0; i < npieces; i++) {
    types[i] = pb[p];
    plens[i] = rd32(pb + p + 1);
    pieces[i] = (char *)pb + p + 5;
    p += 5 + plens[i];
  }
  CHECK(nd_grammar_byte_table_from_pieces((const char *const *)pieces, plens, types, npieces,
                                          &table, &err) == 0);
  sprintf(path, "%s/table.bin", dir);
  tb = read_file(path, &n);
  ntbl = rd32(tb);
  CHECK(ntbl == npieces);
  p = 4;
  for (i = 0; i < ntbl && table; i++) {
    uint32_t id = rd32(tb + p), len = rd32(tb + p + 4);
    if (table[i].id != id || table[i].len != len || memcmp(table[i].bytes, tb + p + 8, len) != 0) {
      mism++;
      if (mism < 10) fprintf(stderr, "byte table entry %lu differs\n", (unsigned long)i);
    }
    p += 8 + len;
  }
  printf("byte table: %lu entries compared\n", (unsigned long)ntbl);
  free(tb);

  /* ToolDef::from_json */
  sprintf(path, "%s/tools_in.bin", dir);
  ins = read_lenprefixed(path);
  sprintf(path, "%s/tools_out.txt", dir);
  txt = (char *)read_file(path, &n);
  cur = txt;
  for (i = 0; i < ins.count; i++) {
    nd_tooldefs t;
    line = next_line(&cur);
    if (!line) {
      fprintf(stderr, "tools_out short\n");
      return 1;
    }
    t = parse_tools_or_die(ins.s[i], ins.n[i]);
    got.n = 0;
    tools_line(&t, &got);
    ntools++;
    if (strcmp(got.p, line) != 0) {
      mism++;
      if (mism < 20) fprintf(stderr, "tools input %lu: %s\n  want %s\n  got  %s\n", (unsigned long)i, ins.s[i], line, got.p);
    }
    nd_tooldefs_free(&t);
  }
  free(txt);
  free_strlist(&ins);
  printf("parse_tools: %ld inputs compared\n", ntools);

  /* catalogues */
  sprintf(path, "%s/cats.bin", dir);
  cats = read_lenprefixed(path);
  catdefs = (nd_tooldefs *)calloc(cats.count, sizeof *catdefs);
  for (i = 0; i < cats.count; i++) catdefs[i] = parse_tools_or_die(cats.s[i], cats.n[i]);

  /* scenarios on the real table, then synthetic tables */
  {
    int pass;
    nd_token_bytes *synth = NULL;
    size_t nsynth = 0, synth_cat = 0;
    unsigned char *synth_pool = NULL;
    for (pass = 0; pass < 2; pass++) {
      nd_grammar *g = NULL;
      size_t vocab = 0;
      int with_keys = pass == 0;
      sprintf(path, pass == 0 ? "%s/scen.txt" : "%s/synth.txt", dir);
      txt = (char *)read_file(path, &n);
      cur = txt;
      while ((line = next_line(&cur)) != NULL) {
        if (line[0] == 'C') {
          synth_cat = (size_t)strtoul(line + 2, NULL, 10);
        } else if (line[0] == 'T' && line[1] == 'B') {
          char *q = line + 3, *end;
          size_t k, poolpos = 0;
          nsynth = (size_t)strtoul(q, &end, 10);
          free(synth);
          free(synth_pool);
          synth = (nd_token_bytes *)calloc(nsynth ? nsynth : 1, sizeof *synth);
          synth_pool = (unsigned char *)malloc(strlen(line) + 1);
          q = end;
          for (k = 0; k < nsynth; k++) {
            size_t blen;
            q++;
            synth[k].id = (uint32_t)strtoul(q, &end, 10);
            q = end + 1;
            unhex(q, synth_pool + poolpos, &blen);
            synth[k].bytes = synth_pool + poolpos;
            synth[k].len = blen;
            poolpos += blen;
            while (*q && *q != ' ') q++;
          }
        } else if (line[0] == 'S') {
          unsigned long ci, uniq, voc;
          sscanf(line + 2, "%lu %lu %lu", &ci, &uniq, &voc);
          nd_grammar_free(g);
          g = NULL;
          if (pass == 0) {
            CHECK(nd_grammar_new(catdefs[ci].tools, catdefs[ci].n, table, npieces, &g, &err) == 0);
          } else {
            CHECK(nd_grammar_new(catdefs[synth_cat].tools, catdefs[synth_cat].n, synth, nsynth, &g,
                                 &err) == 0);
          }
          if (!g) return 1;
          nd_grammar_set_unique_arg_keys(g, (int)uniq);
          vocab = (size_t)voc;
          if (vocab > mask_cap) {
            mask_cap = vocab;
            mask = (float *)realloc(mask, mask_cap * sizeof *mask);
          }
          nscen++;
        } else if (line[0] == 'U') {
          CHECK(nd_grammar_update(g, (uint32_t)strtoul(line + 2, NULL, 10)) == 0);
        } else if (line[0] == 'B') {
          size_t blen;
          unhex(line + 2, hexbuf, &blen);
          CHECK(nd_grammar_feed_bytes(g, hexbuf, blen) == 0);
        } else if (line[0] == 'M') {
          CHECK(nd_grammar_logit_mask(g, mask, vocab) == 0);
          got.n = 0;
          mask_line(mask, vocab, &got);
          if (with_keys) {
            size_t k, cnt = nd_grammar_used_key_count(g);
            sb_str(&got, " K ");
            for (k = 0; k < cnt; k++) {
              size_t kl;
              const char *ks = nd_grammar_used_key(g, k, &kl);
              if (k) sb_str(&got, ",");
              sb_hex(&got, ks, kl);
            }
          }
          nsteps++;
          if (strcmp(got.p, line) != 0) {
            mism++;
            if (mism < 20)
              fprintf(stderr, "%s scenario %ld step %ld:\n  want %.300s\n  got  %.300s\n",
                      pass ? "synth" : "real", nscen, nsteps, line, got.p);
          }
        } else if (line[0] == 'E') {
          nd_grammar_free(g);
          g = NULL;
        }
      }
      nd_grammar_free(g);
      free(txt);
    }
    free(synth);
    free(synth_pool);
  }
  printf("decoder: %ld scenarios, %ld masks compared\n", nscen, nsteps);
  for (i = 0; i < cats.count; i++) nd_tooldefs_free(&catdefs[i]);
  free(catdefs);
  free_strlist(&cats);
  free(table);
  free(pieces);
  free(plens);
  free(types);
  free(pb);
  free(mask);
  free(got.p);
  free(hexbuf);
  g_checks += nsteps + ntools;
  g_fail += mism;
  printf("reference mismatches: %ld\n", mism);
  return mism != 0;
}

int main(int argc, char **argv) {
  int rc = 0;
  unit_tests();
  printf("unit: %ld checks, %ld failures\n", g_checks, g_fail);
  if (argc > 1) rc = replay_dir(argv[1]);
  printf("total: %ld checks, %ld failures\n", g_checks, g_fail);
  return (g_fail || rc) ? 1 : 0;
}
