/* jibo-cq-bench: one real Needle 3 operation on the CPU (scalar, NEON) and on the GPU (desktop GL
 * compute), checked against the Rust engine's own output and timed.
 *
 *   jibo-cq-bench OPDIR [--backend ref|neon|gl|all] [--reps N] [--tt N] [--rows-per-dispatch N]
 *                       [--libgl PATH] [--tokens N]
 *
 * OPDIR comes from `needle-jibo dump-op MODEL --tensor NAME --tokens T --out OPDIR`: a CQ
 * projection (packed indices, FP16 group norms, codebook levels, prepared activations, expected
 * W.x), one learned Kronecker MLP stage (factors a and b, inputs, expected a^T.Z.b), or causal
 * grouped-query attention over a sequence (queries, keys, values, expected output; `ref` and
 * `gl` only, and --rows-per-dispatch counts query positions).
 *
 * Every backend's result is compared with the expected output: max absolute error, error relative
 * to the output RMS, cosine, and how many rows differ at all. `ref` follows the Rust kernel's
 * summation order (8 lanes, summed in order, then the group norm), so on ARMv7, where the Rust
 * build has no fused multiply-add, it should agree bit for bit with a dump made by the ARMv7
 * binary. `neon` keeps that order with non-fused VMLA; it can differ only where NEON flushes a
 * denormal. `gl` reports its error and is not expected to be bit-identical.
 *
 * GL: an OpenGL 4.3 core context of its own on a 1x1 pbuffer of $DISPLAY, libGL and libX11 opened
 * with dlopen (the binary links libc, libm and libdl only). Timings separate the one-time upload,
 * the dispatch-to-fence time (host), the GL_TIME_ELAPSED GPU time and the readback, so a kernel
 * that is fast only before transfers are counted shows as such. The GL context code is adapted
 * from bynds/strands-decider ports/jibo/tools/jibo-gl-probe.c and runtime/jd_gl.c (Apache-2.0,
 * commit 29e78ff).
 *
 * Output: one JSON object per backend on stdout. Exit status 0 only if every backend run passed
 * its tolerance.
 */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAVE_NEON 1
#else
#define HAVE_NEON 0
#endif

#define LANES 8

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static void *read_file(const char *dir, const char *name, long *len) {
  char path[4096];
  snprintf(path, sizeof path, "%s/%s", dir, name);
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  void *p = malloc(n > 0 ? (size_t)n : 1);
  if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) {
    fclose(f);
    free(p);
    return NULL;
  }
  fclose(f);
  if (len) *len = n;
  return p;
}

static long op_int(const char *txt, const char *key) {
  char pat[64];
  snprintf(pat, sizeof pat, "\n%s=", key);
  const char *p = strstr(txt, pat);
  if (!p) {
    snprintf(pat, sizeof pat, "%s=", key);
    if (strncmp(txt, pat, strlen(pat)) != 0) return -1;
    p = txt - 1;
  }
  return strtol(p + 1 + strlen(key) + 1, NULL, 10);
}

static float half_to_float(uint16_t h) {
  uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 0x1F, m = h & 0x3FF, bits;
  if (e == 0) {
    if (m == 0) {
      bits = s;
    } else { /* subnormal: renormalise */
      int k = -1;
      do { k++; m <<= 1; } while (!(m & 0x400));
      bits = s | ((uint32_t)(127 - 15 - k) << 23) | ((m & 0x3FF) << 13);
    }
  } else if (e == 31) {
    bits = s | 0x7F800000u | (m << 13);
  } else {
    bits = s | ((e + 127 - 15) << 23) | (m << 13);
  }
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

/* ---- comparison ------------------------------------------------------------------------ */
typedef struct { double max_abs, rel_rms, cosine; long rows_differ; } cmp_t;

static cmp_t compare(const float *got, const float *want, long n) {
  cmp_t c = {0, 0, 0, 0};
  double sw = 0, sg = 0, dot = 0;
  for (long i = 0; i < n; i++) {
    double d = fabs((double)got[i] - (double)want[i]);
    if (d > c.max_abs || d != d) c.max_abs = d != d ? INFINITY : d;
    if (got[i] != want[i]) c.rows_differ++;
    sw += (double)want[i] * want[i];
    sg += (double)got[i] * got[i];
    dot += (double)got[i] * want[i];
  }
  double rms = sqrt(sw / (double)(n ? n : 1));
  c.rel_rms = rms > 0 ? c.max_abs / rms : c.max_abs;
  c.cosine = (sw > 0 && sg > 0) ? dot / sqrt(sw * sg) : 0;
  return c;
}

/* ---- CQ operation ---------------------------------------------------------------------- */
typedef struct {
  int out, in, in_padded, group, bits, tokens, per_byte, row_bytes, num_groups;
  const uint8_t *packed;   /* out * row_bytes */
  const uint8_t *norms16;  /* out * num_groups FP16 */
  float *norms;            /* decoded */
  float levels[16];
  float *lut;              /* 256 * per_byte */
  const float *x;          /* tokens * in_padded */
  const float *y_want;     /* tokens * out */
} cq_op;

static void cq_decode_group(const cq_op *w, int o, int g, float *ug) {
  const uint8_t *row = w->packed + (long)o * w->row_bytes;
  int bpg = w->group / w->per_byte;
  const uint8_t *gb = row + (long)g * bpg;
  for (int b = 0; b < bpg; b++)
    for (int k = 0; k < w->per_byte; k++) ug[b * w->per_byte + k] = w->lut[gb[b] * w->per_byte + k];
}

/* The Rust kernel's order: lanes[k % 8] += u*x over the group, lanes summed 0..7 in order, then
 * acc += norm * s per group. */
static void cq_ref(const cq_op *w, float *y) {
  float ug[1024];
  float *acc = calloc((size_t)w->tokens, sizeof(float));
  for (int o = 0; o < w->out; o++) {
    for (int t = 0; t < w->tokens; t++) acc[t] = 0.0f;
    for (int g = 0; g < w->num_groups; g++) {
      cq_decode_group(w, o, g, ug);
      float norm = w->norms[(long)o * w->num_groups + g];
      for (int t = 0; t < w->tokens; t++) {
        const float *gx = w->x + (long)t * w->in_padded + (long)g * w->group;
        float lanes[LANES] = {0};
        for (int c = 0; c < w->group / LANES; c++)
          for (int k = 0; k < LANES; k++) lanes[k] += ug[c * LANES + k] * gx[c * LANES + k];
        float s = 0.0f;
        for (int k = 0; k < LANES; k++) s += lanes[k];
        acc[t] += norm * s;
      }
    }
    for (int t = 0; t < w->tokens; t++) y[(long)t * w->out + o] = acc[t];
  }
  free(acc);
}

#if HAVE_NEON
/* Same order as cq_ref: two 4-wide accumulators are lanes 0-3 and 4-7; VMLA is not fused on
 * ARMv7, matching the separate multiply and add of the Rust build. */
static void cq_neon(const cq_op *w, float *y) {
  float ug[1024] __attribute__((aligned(16)));
  float *acc = calloc((size_t)w->tokens, sizeof(float));
  int bpg = w->group / w->per_byte;
  for (int o = 0; o < w->out; o++) {
    const uint8_t *row = w->packed + (long)o * w->row_bytes;
    for (int t = 0; t < w->tokens; t++) acc[t] = 0.0f;
    for (int g = 0; g < w->num_groups; g++) {
      const uint8_t *gb = row + (long)g * bpg;
      if (w->per_byte == 4) {
        for (int b = 0; b < bpg; b++) vst1q_f32(ug + 4 * b, vld1q_f32(w->lut + gb[b] * 4));
      } else {
        for (int b = 0; b < bpg; b += 2)
          vst1q_f32(ug + 2 * b, vcombine_f32(vld1_f32(w->lut + gb[b] * 2), vld1_f32(w->lut + gb[b + 1] * 2)));
      }
      float norm = w->norms[(long)o * w->num_groups + g];
      for (int t = 0; t < w->tokens; t++) {
        const float *gx = w->x + (long)t * w->in_padded + (long)g * w->group;
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        for (int c = 0; c < w->group; c += LANES) {
          a0 = vmlaq_f32(a0, vld1q_f32(ug + c), vld1q_f32(gx + c));
          a1 = vmlaq_f32(a1, vld1q_f32(ug + c + 4), vld1q_f32(gx + c + 4));
        }
        float l[LANES];
        vst1q_f32(l, a0);
        vst1q_f32(l + 4, a1);
        float s = 0.0f;
        for (int k = 0; k < LANES; k++) s += l[k];
        acc[t] += norm * s;
      }
    }
    for (int t = 0; t < w->tokens; t++) y[(long)t * w->out + o] = acc[t];
  }
  free(acc);
}
#endif

/* ---- Kronecker stage ------------------------------------------------------------------- */
typedef struct {
  int ba, bb, tokens;
  const float *a, *b, *z, *k_want;
} kron_op;

/* The Rust kernel's loops (kron_apply), including skipping exact zeros. */
static void kron_one_ref(const kron_op *p, const float *z, float *out, float *t) {
  int ba = p->ba, bb = p->bb;
  memset(t, 0, sizeof(float) * (size_t)(ba * bb));
  for (int i = 0; i < ba; i++)
    for (int k = 0; k < ba; k++) {
      float aik = p->a[i * ba + k];
      if (aik == 0.0f) continue;
      for (int j = 0; j < bb; j++) t[k * bb + j] += z[i * bb + j] * aik;
    }
  for (int k = 0; k < ba; k++) {
    float *ok = out + k * bb;
    memset(ok, 0, sizeof(float) * (size_t)bb);
    for (int j = 0; j < bb; j++) {
      float tkj = t[k * bb + j];
      if (tkj == 0.0f) continue;
      for (int l = 0; l < bb; l++) ok[l] += tkj * p->b[j * bb + l];
    }
  }
}

static void kron_ref(const kron_op *p, float *out) {
  int n = p->ba * p->bb;
  float *t = malloc(sizeof(float) * (size_t)n);
  for (int s = 0; s < p->tokens; s++) kron_one_ref(p, p->z + (long)s * n, out + (long)s * n, t);
  free(t);
}

#if HAVE_NEON
static void kron_neon(const kron_op *p, float *out) {
  int ba = p->ba, bb = p->bb, n = ba * bb;
  float *t = malloc(sizeof(float) * (size_t)n);
  for (int s = 0; s < p->tokens; s++) {
    const float *z = p->z + (long)s * n;
    float *o = out + (long)s * n;
    memset(t, 0, sizeof(float) * (size_t)n);
    for (int i = 0; i < ba; i++)
      for (int k = 0; k < ba; k++) {
        float aik = p->a[i * ba + k];
        if (aik == 0.0f) continue;
        float32x4_t av = vdupq_n_f32(aik);
        for (int j = 0; j < bb; j += 4)
          vst1q_f32(t + k * bb + j, vmlaq_f32(vld1q_f32(t + k * bb + j), vld1q_f32(z + i * bb + j), av));
      }
    for (int k = 0; k < ba; k++) {
      float *ok = o + k * bb;
      memset(ok, 0, sizeof(float) * (size_t)bb);
      for (int j = 0; j < bb; j++) {
        float tkj = t[k * bb + j];
        if (tkj == 0.0f) continue;
        float32x4_t tv = vdupq_n_f32(tkj);
        for (int l = 0; l < bb; l += 4)
          vst1q_f32(ok + l, vmlaq_f32(vld1q_f32(ok + l), vld1q_f32(p->b + j * bb + l), tv));
      }
    }
  }
  free(t);
}
#endif

/* ---- desktop GL compute ---------------------------------------------------------------- */
typedef unsigned int GLenum, GLuint, GLbitfield;
typedef int GLint, GLsizei;
typedef char GLchar;
typedef unsigned char GLubyte;
typedef intptr_t GLintptr, GLsizeiptr;
typedef uint64_t GLuint64;
typedef float GLfloat;
typedef struct __GLsync *GLsync;
typedef struct _XDisplay Display;
typedef struct __GLXcontextRec *GLXContext;
typedef struct __GLXFBConfigRec *GLXFBConfig;
typedef unsigned long XID;
#define GL_NO_ERROR 0
#define GL_RENDERER 0x1F01
#define GL_VERSION 0x1F02
#define GL_COMPUTE_SHADER 0x91B9
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#define GL_STATIC_DRAW 0x88E4
#define GL_DYNAMIC_COPY 0x88EA
#define GL_MAX_COMPUTE_SHARED_MEMORY_SIZE 0x8262
#define GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS 0x90EB
#define GL_MAX_SHADER_STORAGE_BLOCK_SIZE 0x90DE
#define GL_SHADER_STORAGE_BARRIER_BIT 0x00002000
#define GL_BUFFER_UPDATE_BARRIER_BIT 0x00000200
#define GL_SYNC_GPU_COMMANDS_COMPLETE 0x9117
#define GL_SYNC_FLUSH_COMMANDS_BIT 0x00000001
#define GL_TIMEOUT_EXPIRED 0x911B
#define GL_WAIT_FAILED 0x911D
#define GL_TIME_ELAPSED 0x88BF
#define GL_QUERY_RESULT 0x8866
#define GLX_RENDER_TYPE 0x8011
#define GLX_RGBA_BIT 0x00000001
#define GLX_DRAWABLE_TYPE 0x8010
#define GLX_PBUFFER_BIT 0x00000004
#define GLX_PBUFFER_HEIGHT 0x8040
#define GLX_PBUFFER_WIDTH 0x8041
#define GLX_CONTEXT_MAJOR_VERSION_ARB 0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB 0x2092
#define GLX_CONTEXT_PROFILE_MASK_ARB 0x9126
#define GLX_CONTEXT_CORE_PROFILE_BIT_ARB 0x00000001

static struct {
  void *x11, *lib;
  Display *dpy;
  GLXContext ctx;
  XID pb;
  Display *(*XOpenDisplay)(const char *);
  int (*XDefaultScreen)(Display *);
  void *(*glXGetProcAddressARB)(const GLubyte *);
  GLXFBConfig *(*glXChooseFBConfig)(Display *, int, const int *, int *);
  XID (*glXCreatePbuffer)(Display *, GLXFBConfig, const int *);
  int (*glXMakeContextCurrent)(Display *, XID, XID, GLXContext);
  GLXContext (*glXCreateContextAttribsARB)(Display *, GLXFBConfig, GLXContext, int, const int *);
  const GLubyte *(*GetString)(GLenum);
  GLenum (*GetError)(void);
  void (*GetIntegerv)(GLenum, GLint *);
  GLuint (*CreateShader)(GLenum);
  void (*ShaderSource)(GLuint, GLsizei, const GLchar *const *, const GLint *);
  void (*CompileShader)(GLuint);
  void (*GetShaderiv)(GLuint, GLenum, GLint *);
  void (*GetShaderInfoLog)(GLuint, GLsizei, GLsizei *, GLchar *);
  GLuint (*CreateProgram)(void);
  void (*AttachShader)(GLuint, GLuint);
  void (*LinkProgram)(GLuint);
  void (*GetProgramiv)(GLuint, GLenum, GLint *);
  void (*GetProgramInfoLog)(GLuint, GLsizei, GLsizei *, GLchar *);
  void (*UseProgram)(GLuint);
  GLint (*GetUniformLocation)(GLuint, const GLchar *);
  void (*Uniform1i)(GLint, GLint);
  void (*Uniform1fv)(GLint, GLsizei, const GLfloat *);
  void (*GenBuffers)(GLsizei, GLuint *);
  void (*DeleteBuffers)(GLsizei, const GLuint *);
  void (*BindBuffer)(GLenum, GLuint);
  void (*BufferData)(GLenum, GLsizeiptr, const void *, GLenum);
  void (*BindBufferBase)(GLenum, GLuint, GLuint);
  void (*GetBufferSubData)(GLenum, GLintptr, GLsizeiptr, void *);
  void (*DispatchCompute)(GLuint, GLuint, GLuint);
  void (*MemoryBarrier)(GLbitfield);
  GLsync (*FenceSync)(GLenum, GLbitfield);
  GLenum (*ClientWaitSync)(GLsync, GLbitfield, GLuint64);
  void (*DeleteSync)(GLsync);
  void (*GenQueries)(GLsizei, GLuint *);
  void (*BeginQuery)(GLenum, GLuint);
  void (*EndQuery)(GLenum);
  void (*GetQueryObjectui64v)(GLuint, GLenum, GLuint64 *);
} gl;

static void *glsym(const char *name) {
  void *p = gl.glXGetProcAddressARB ? gl.glXGetProcAddressARB((const GLubyte *)name) : NULL;
  return p ? p : dlsym(gl.lib, name);
}

static int gl_init(const char *libgl, char *err, size_t errn) {
  gl.x11 = dlopen("libX11.so.6", RTLD_NOW | RTLD_GLOBAL);
  gl.lib = dlopen(libgl ? libgl : "libGL.so.1", RTLD_NOW | RTLD_GLOBAL);
  if (!gl.x11 || !gl.lib) { snprintf(err, errn, "cannot dlopen libX11.so.6 or libGL"); return -1; }
  *(void **)&gl.XOpenDisplay = dlsym(gl.x11, "XOpenDisplay");
  *(void **)&gl.XDefaultScreen = dlsym(gl.x11, "XDefaultScreen");
  *(void **)&gl.glXGetProcAddressARB = dlsym(gl.lib, "glXGetProcAddressARB");
#define L(f, n) *(void **)(&gl.f) = glsym(n)
  L(glXChooseFBConfig, "glXChooseFBConfig"); L(glXCreatePbuffer, "glXCreatePbuffer");
  L(glXMakeContextCurrent, "glXMakeContextCurrent"); L(glXCreateContextAttribsARB, "glXCreateContextAttribsARB");
  if (!gl.XOpenDisplay || !gl.glXChooseFBConfig || !gl.glXCreateContextAttribsARB) {
    snprintf(err, errn, "GLX entry points missing");
    return -1;
  }
  gl.dpy = gl.XOpenDisplay(NULL);
  if (!gl.dpy) { snprintf(err, errn, "XOpenDisplay failed (DISPLAY, and the run.sh shim?)"); return -1; }
  static const int fb_attr[] = {GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT, GLX_RENDER_TYPE, GLX_RGBA_BIT, 0};
  int n = 0;
  GLXFBConfig *fbc = gl.glXChooseFBConfig(gl.dpy, gl.XDefaultScreen(gl.dpy), fb_attr, &n);
  if (!fbc || !n) { snprintf(err, errn, "no pbuffer FB config"); return -1; }
  static const int ctx_attr[] = {GLX_CONTEXT_MAJOR_VERSION_ARB, 4, GLX_CONTEXT_MINOR_VERSION_ARB, 3,
                                 GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB, 0};
  gl.ctx = gl.glXCreateContextAttribsARB(gl.dpy, fbc[0], NULL, 1, ctx_attr);
  if (!gl.ctx) { snprintf(err, errn, "no OpenGL 4.3 core context"); return -1; }
  static const int pb_attr[] = {GLX_PBUFFER_WIDTH, 1, GLX_PBUFFER_HEIGHT, 1, 0};
  gl.pb = gl.glXCreatePbuffer(gl.dpy, fbc[0], pb_attr);
  if (!gl.glXMakeContextCurrent(gl.dpy, gl.pb, gl.pb, gl.ctx)) { snprintf(err, errn, "glXMakeContextCurrent failed"); return -1; }
  L(GetString, "glGetString"); L(GetError, "glGetError"); L(GetIntegerv, "glGetIntegerv");
  L(CreateShader, "glCreateShader"); L(ShaderSource, "glShaderSource"); L(CompileShader, "glCompileShader");
  L(GetShaderiv, "glGetShaderiv"); L(GetShaderInfoLog, "glGetShaderInfoLog"); L(CreateProgram, "glCreateProgram");
  L(AttachShader, "glAttachShader"); L(LinkProgram, "glLinkProgram"); L(GetProgramiv, "glGetProgramiv");
  L(GetProgramInfoLog, "glGetProgramInfoLog"); L(UseProgram, "glUseProgram");
  L(GetUniformLocation, "glGetUniformLocation"); L(Uniform1i, "glUniform1i"); L(Uniform1fv, "glUniform1fv");
  L(GenBuffers, "glGenBuffers"); L(DeleteBuffers, "glDeleteBuffers"); L(BindBuffer, "glBindBuffer");
  L(BufferData, "glBufferData"); L(BindBufferBase, "glBindBufferBase"); L(GetBufferSubData, "glGetBufferSubData");
  L(DispatchCompute, "glDispatchCompute"); L(MemoryBarrier, "glMemoryBarrier"); L(FenceSync, "glFenceSync");
  L(ClientWaitSync, "glClientWaitSync"); L(DeleteSync, "glDeleteSync"); L(GenQueries, "glGenQueries");
  L(BeginQuery, "glBeginQuery"); L(EndQuery, "glEndQuery"); L(GetQueryObjectui64v, "glGetQueryObjectui64v");
#undef L
  if (!gl.DispatchCompute || !gl.MemoryBarrier || !gl.FenceSync) { snprintf(err, errn, "compute entry points missing"); return -1; }
  return 0;
}

static GLuint gl_program(const char *src, char *err, size_t errn) {
  GLuint sh = gl.CreateShader(GL_COMPUTE_SHADER);
  gl.ShaderSource(sh, 1, &src, NULL);
  gl.CompileShader(sh);
  GLint ok = 0;
  gl.GetShaderiv(sh, GL_COMPILE_STATUS, &ok);
  if (!ok) { gl.GetShaderInfoLog(sh, (GLsizei)errn, NULL, err); return 0; }
  GLuint p = gl.CreateProgram();
  gl.AttachShader(p, sh);
  gl.LinkProgram(p);
  gl.GetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) { gl.GetProgramInfoLog(p, (GLsizei)errn, NULL, err); return 0; }
  return p;
}

static GLuint gl_buffer(const void *data, long bytes) {
  GLuint b;
  gl.GenBuffers(1, &b);
  gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, b);
  gl.BufferData(GL_SHADER_STORAGE_BUFFER, bytes, data, data ? GL_STATIC_DRAW : GL_DYNAMIC_COPY);
  return b;
}

/* Wait without spinning the CPU at full tilt: a bounded ClientWaitSync, repeated. */
static int gl_wait(GLsync s) {
  for (int i = 0; i < 2000; i++) {
    GLenum r = gl.ClientWaitSync(s, GL_SYNC_FLUSH_COMMANDS_BIT, 5000000ull /* 5 ms */);
    if (r == GL_WAIT_FAILED) return -1;
    if (r != GL_TIMEOUT_EXPIRED) return 0;
  }
  return -1; /* 10 s: the GPU work is still running; a host timeout does not cancel it */
}

/* One invocation per (output row, tile of TT tokens). Indices are read 32 bits at a time, LSB
 * first (the container's packing); norms stay FP16, two per uint; levels are a uniform. */
static const char *CQ_SHADER =
    "#version 430\n"
    "layout(local_size_x = 64) in;\n"
    "layout(std430, binding = 0) readonly buffer W { uint w[]; };\n"
    "layout(std430, binding = 1) readonly buffer N { uint nrm[]; };\n"
    "layout(std430, binding = 2) readonly buffer X { float x[]; };\n"
    "layout(std430, binding = 3) writeonly buffer Y { float y[]; };\n"
    "uniform int u_out, u_in_padded, u_group, u_tokens, u_row0, u_nrows, u_row_words, u_norm_stride;\n"
    "uniform float u_levels[16];\n"
    "#define BITS %d\n#define TT %d\n"
    "void main() {\n"
    "  int r = int(gl_GlobalInvocationID.x);\n"
    "  if (r >= u_nrows) return;\n"
    "  int o = u_row0 + r;\n"
    "  int t0 = int(gl_GlobalInvocationID.y) * TT;\n"
    "  float acc[TT];\n"
    "  for (int t = 0; t < TT; t++) acc[t] = 0.0;\n"
    "  const int PER = 32 / BITS;\n"
    "  const uint MASK = (1u << BITS) - 1u;\n"
    "  int groups = u_in_padded / u_group;\n"
    "  int wpg = u_group / PER;\n"
    "  for (int g = 0; g < groups; g++) {\n"
    "    float s[TT];\n"
    "    for (int t = 0; t < TT; t++) s[t] = 0.0;\n"
    "    for (int k = 0; k < wpg; k++) {\n"
    "      uint word = w[o * u_row_words + g * wpg + k];\n"
    "      int col = g * u_group + k * PER;\n"
    "      for (int e = 0; e < PER; e++) {\n"
    "        float v = u_levels[(word >> uint(e * BITS)) & MASK];\n"
    "        for (int t = 0; t < TT; t++) {\n"
    "          int tt = t0 + t;\n"
    "          if (tt < u_tokens) s[t] += v * x[tt * u_in_padded + col + e];\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "    int ni = o * u_norm_stride + g;\n"
    "    float norm = unpackHalf2x16(nrm[ni >> 1])[ni & 1];\n"
    "    for (int t = 0; t < TT; t++) acc[t] += norm * s[t];\n"
    "  }\n"
    "  for (int t = 0; t < TT; t++) { int tt = t0 + t; if (tt < u_tokens) y[tt * u_out + o] = acc[t]; }\n"
    "}\n";

/* One workgroup per token: Z, the stage-one product and both factors in shared memory. */
static const char *KRON_SHADER =
    "#version 430\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer A { float a[]; };\n"
    "layout(std430, binding = 1) readonly buffer B { float b[]; };\n"
    "layout(std430, binding = 2) readonly buffer Z { float z[]; };\n"
    "layout(std430, binding = 3) writeonly buffer O { float outp[]; };\n"
    "#define BA %d\n#define BB %d\n"
    "shared float sz[BA * BB];\n"
    "shared float st[BA * BB];\n"
    "void main() {\n"
    "  int s = int(gl_WorkGroupID.x);\n"
    "  int n = BA * BB;\n"
    "  for (int i = int(gl_LocalInvocationIndex); i < n; i += 256) sz[i] = z[s * n + i];\n"
    "  barrier();\n"
    "  for (int c = int(gl_LocalInvocationIndex); c < n; c += 256) {\n"
    "    int k = c / BB, j = c %% BB;\n"
    "    float acc = 0.0;\n"
    "    for (int i = 0; i < BA; i++) acc += sz[i * BB + j] * a[i * BA + k];\n"
    "    st[c] = acc;\n"
    "  }\n"
    "  barrier();\n"
    "  for (int c = int(gl_LocalInvocationIndex); c < n; c += 256) {\n"
    "    int k = c / BB, l = c %% BB;\n"
    "    float acc = 0.0;\n"
    "    for (int j = 0; j < BB; j++) acc += st[k * BB + j] * b[j * BB + l];\n"
    "    outp[s * n + c] = acc;\n"
    "  }\n"
    "}\n";

typedef struct { double upload_ms, dispatch_ms, gpu_ms, readback_ms; int dispatches; } gl_times;

static int gl_cq(const cq_op *w, int tt, int rows_per_dispatch, int reps, float *y, gl_times *tm, char *err, size_t errn) {
  char src[8192];
  snprintf(src, sizeof src, CQ_SHADER, w->bits, tt);
  GLuint prog = gl_program(src, err, errn);
  if (!prog) return -1;
  int row_words = w->row_bytes / 4, norm_stride = w->num_groups;
  /* norms: FP16 pairs into uints; pad the count to even */
  long nn = (long)w->out * w->num_groups;
  uint32_t *n32 = calloc((size_t)(nn + 2) / 2, 4);
  memcpy(n32, w->norms16, (size_t)nn * 2);
  double t0 = now_ms();
  GLuint bw = gl_buffer(w->packed, (long)w->out * w->row_bytes);
  GLuint bn = gl_buffer(n32, ((nn + 1) / 2) * 4);
  GLuint bx = gl_buffer(w->x, (long)w->tokens * w->in_padded * 4);
  GLuint by = gl_buffer(NULL, (long)w->tokens * w->out * 4);
  free(n32);
  gl.UseProgram(prog);
#define U(name, v) gl.Uniform1i(gl.GetUniformLocation(prog, name), v)
  U("u_out", w->out); U("u_in_padded", w->in_padded); U("u_group", w->group); U("u_tokens", w->tokens);
  U("u_row_words", row_words); U("u_norm_stride", norm_stride);
  gl.Uniform1fv(gl.GetUniformLocation(prog, "u_levels"), 16, w->levels);
  gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bw);
  gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bn);
  gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bx);
  gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, by);
  GLsync s0 = gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
  if (gl_wait(s0)) { snprintf(err, errn, "upload fence failed"); return -1; }
  gl.DeleteSync(s0);
  tm->upload_ms = now_ms() - t0;
  GLuint q;
  gl.GenQueries(1, &q);
  int tiles = (w->tokens + tt - 1) / tt;
  double best_d = 1e30, best_g = 1e30;
  for (int rep = 0; rep < reps; rep++) {
    double t1 = now_ms();
    double gpu = 0;
    int nd = 0;
    for (int r0 = 0; r0 < w->out; r0 += rows_per_dispatch) {
      int nr = w->out - r0 < rows_per_dispatch ? w->out - r0 : rows_per_dispatch;
      U("u_row0", r0); U("u_nrows", nr);
      if (gl.BeginQuery) gl.BeginQuery(GL_TIME_ELAPSED, q);
      gl.DispatchCompute((GLuint)((nr + 63) / 64), (GLuint)tiles, 1);
      if (gl.EndQuery) gl.EndQuery(GL_TIME_ELAPSED);
      /* One bounded dispatch at a time: wait before the next, so the queue stays shallow. */
      GLsync s = gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
      if (gl_wait(s)) { snprintf(err, errn, "dispatch fence failed"); return -1; }
      gl.DeleteSync(s);
      GLuint64 ns = 0;
      if (gl.GetQueryObjectui64v) gl.GetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
      gpu += ns / 1e6;
      nd++;
    }
    double d = now_ms() - t1;
    if (d < best_d) { best_d = d; best_g = gpu; tm->dispatches = nd; }
  }
#undef U
  /* Shader writes must be visible to the buffer read that follows. */
  gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
  double t2 = now_ms();
  gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, by);
  gl.GetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (long)w->tokens * w->out * 4, y);
  tm->readback_ms = now_ms() - t2;
  tm->dispatch_ms = best_d;
  tm->gpu_ms = best_g;
  GLenum e = gl.GetError();
  GLuint bufs[4] = {bw, bn, bx, by};
  gl.DeleteBuffers(4, bufs);
  if (e != GL_NO_ERROR) { snprintf(err, errn, "GL error 0x%x", e); return -1; }
  return 0;
}

static int gl_kron(const kron_op *p, int reps, float *out, gl_times *tm, char *err, size_t errn) {
  char src[8192];
  snprintf(src, sizeof src, KRON_SHADER, p->ba, p->bb);
  GLuint prog = gl_program(src, err, errn);
  if (!prog) return -1;
  int n = p->ba * p->bb;
  double t0 = now_ms();
  GLuint ba_ = gl_buffer(p->a, (long)p->ba * p->ba * 4);
  GLuint bb_ = gl_buffer(p->b, (long)p->bb * p->bb * 4);
  GLuint bz = gl_buffer(p->z, (long)p->tokens * n * 4);
  GLuint bo = gl_buffer(NULL, (long)p->tokens * n * 4);
  gl.UseProgram(prog);
  gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ba_);
  gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bb_);
  gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bz);
  gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, bo);
  GLsync s0 = gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
  if (gl_wait(s0)) { snprintf(err, errn, "upload fence failed"); return -1; }
  gl.DeleteSync(s0);
  tm->upload_ms = now_ms() - t0;
  GLuint q;
  gl.GenQueries(1, &q);
  double best_d = 1e30, best_g = 1e30;
  for (int rep = 0; rep < reps; rep++) {
    double t1 = now_ms();
    if (gl.BeginQuery) gl.BeginQuery(GL_TIME_ELAPSED, q);
    gl.DispatchCompute((GLuint)p->tokens, 1, 1);
    if (gl.EndQuery) gl.EndQuery(GL_TIME_ELAPSED);
    GLsync s = gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (gl_wait(s)) { snprintf(err, errn, "dispatch fence failed"); return -1; }
    gl.DeleteSync(s);
    GLuint64 ns = 0;
    if (gl.GetQueryObjectui64v) gl.GetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
    double d = now_ms() - t1;
    if (d < best_d) { best_d = d; best_g = ns / 1e6; }
  }
  tm->dispatches = 1;
  gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
  double t2 = now_ms();
  gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, bo);
  gl.GetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (long)p->tokens * n * 4, out);
  tm->readback_ms = now_ms() - t2;
  tm->dispatch_ms = best_d;
  tm->gpu_ms = best_g;
  GLenum e = gl.GetError();
  GLuint bufs[4] = {ba_, bb_, bz, bo};
  gl.DeleteBuffers(4, bufs);
  if (e != GL_NO_ERROR) { snprintf(err, errn, "GL error 0x%x", e); return -1; }
  return 0;
}

/* ---- attention: causal grouped-query, optional sliding window ----------------------- */
typedef struct {
  int seq, heads, kv_heads, qk, vd, window;
  const float *q, *k, *v, *o_want;
} attn_op;

/* The engine's attend(): per (t, h) a two-pass softmax over positions lo..=t, in its order. */
static void attn_ref(const attn_op *p, float *o) {
  int rep = p->heads / p->kv_heads;
  float scale = 1.0f / sqrtf((float)p->qk);
  float *scores = malloc(sizeof(float) * (size_t)p->seq);
  for (int t = 0; t < p->seq; t++) {
    int lo = (p->window > 0 && t + 1 > p->window) ? t + 1 - p->window : 0;
    for (int h = 0; h < p->heads; h++) {
      int kvh = h / rep;
      const float *qv = p->q + ((long)t * p->heads + h) * p->qk;
      float mx = -INFINITY;
      for (int n = lo; n <= t; n++) {
        const float *kv = p->k + ((long)n * p->kv_heads + kvh) * p->qk;
        float a = 0.0f;
        for (int i = 0; i < p->qk; i++) a += qv[i] * kv[i];
        scores[n] = a * scale;
        if (scores[n] > mx) mx = scores[n];
      }
      float sum = 0.0f;
      for (int n = lo; n <= t; n++) { scores[n] = expf(scores[n] - mx); sum += scores[n]; }
      float inv = 1.0f / sum;
      float *out = o + ((long)t * p->heads + h) * p->vd;
      for (int i = 0; i < p->vd; i++) out[i] = 0.0f;
      for (int n = lo; n <= t; n++) {
        float w = scores[n] * inv;
        const float *vv = p->v + ((long)n * p->kv_heads + kvh) * p->vd;
        for (int i = 0; i < p->vd; i++) out[i] += w * vv[i];
      }
    }
  }
  free(scores);
}

/* One invocation per (position, head), an online softmax so the score row needs no buffer of
 * sequence length: a different summation order from the engine, so judged by tolerance. */
static const char *ATTN_SHADER =
    "#version 430\n"
    "layout(local_size_x = 64) in;\n"
    "layout(std430, binding = 0) readonly buffer Q { float q[]; };\n"
    "layout(std430, binding = 1) readonly buffer K { float k[]; };\n"
    "layout(std430, binding = 2) readonly buffer V { float v[]; };\n"
    "layout(std430, binding = 3) writeonly buffer O { float o[]; };\n"
    "uniform int u_seq, u_window, u_t0, u_nt;\n"
    "#define H %d\n#define KVH %d\n#define QK %d\n#define VD %d\n"
    "void main() {\n"
    "  int id = int(gl_GlobalInvocationID.x);\n"
    "  if (id >= u_nt * H) return;\n"
    "  int t = u_t0 + id / H, h = id %% H, kvh = h / (H / KVH);\n"
    "  float scale = 1.0 / sqrt(float(QK));\n"
    "  int lo = (u_window > 0 && t + 1 > u_window) ? t + 1 - u_window : 0;\n"
    "  float m = -3.0e38, l = 0.0;\n"
    "  float acc[VD];\n"
    "  for (int i = 0; i < VD; i++) acc[i] = 0.0;\n"
    "  int qo = (t * H + h) * QK;\n"
    "  for (int n = lo; n <= t; n++) {\n"
    "    int ko = (n * KVH + kvh) * QK;\n"
    "    float s = 0.0;\n"
    "    for (int i = 0; i < QK; i++) s += q[qo + i] * k[ko + i];\n"
    "    s *= scale;\n"
    "    float mn = max(m, s), c = exp(m - mn), e = exp(s - mn);\n"
    "    l = l * c + e;\n"
    "    int vo = (n * KVH + kvh) * VD;\n"
    "    for (int i = 0; i < VD; i++) acc[i] = acc[i] * c + e * v[vo + i];\n"
    "    m = mn;\n"
    "  }\n"
    "  int oo = (t * H + h) * VD;\n"
    "  for (int i = 0; i < VD; i++) o[oo + i] = acc[i] / l;\n"
    "}\n";

static int gl_attn(const attn_op *p, int rows_per_dispatch, int reps, float *out, gl_times *tm,
                   char *err, size_t errn) {
  char src[8192];
  snprintf(src, sizeof src, ATTN_SHADER, p->heads, p->kv_heads, p->qk, p->vd);
  GLuint prog = gl_program(src, err, errn);
  if (!prog) return -1;
  double t0 = now_ms();
  GLuint bq = gl_buffer(p->q, (long)p->seq * p->heads * p->qk * 4);
  GLuint bk = gl_buffer(p->k, (long)p->seq * p->kv_heads * p->qk * 4);
  GLuint bv = gl_buffer(p->v, (long)p->seq * p->kv_heads * p->vd * 4);
  GLuint bo = gl_buffer(NULL, (long)p->seq * p->heads * p->vd * 4);
  gl.UseProgram(prog);
  gl.Uniform1i(gl.GetUniformLocation(prog, "u_seq"), p->seq);
  gl.Uniform1i(gl.GetUniformLocation(prog, "u_window"), p->window);
  gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bq);
  gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bk);
  gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bv);
  gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, bo);
  GLsync s0 = gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
  if (gl_wait(s0)) { snprintf(err, errn, "upload fence failed"); return -1; }
  gl.DeleteSync(s0);
  tm->upload_ms = now_ms() - t0;
  GLuint q;
  gl.GenQueries(1, &q);
  /* rows_per_dispatch counts positions here: a dispatch covers that many query positions. */
  int per = rows_per_dispatch < p->seq ? rows_per_dispatch : p->seq;
  double best_d = 1e30, best_g = 1e30;
  for (int rep = 0; rep < reps; rep++) {
    double t1 = now_ms(), gpu = 0;
    int nd = 0;
    for (int ts = 0; ts < p->seq; ts += per) {
      int nt = p->seq - ts < per ? p->seq - ts : per;
      gl.Uniform1i(gl.GetUniformLocation(prog, "u_t0"), ts);
      gl.Uniform1i(gl.GetUniformLocation(prog, "u_nt"), nt);
      if (gl.BeginQuery) gl.BeginQuery(GL_TIME_ELAPSED, q);
      gl.DispatchCompute((GLuint)((nt * p->heads + 63) / 64), 1, 1);
      if (gl.EndQuery) gl.EndQuery(GL_TIME_ELAPSED);
      GLsync s = gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
      if (gl_wait(s)) { snprintf(err, errn, "dispatch fence failed"); return -1; }
      gl.DeleteSync(s);
      GLuint64 ns = 0;
      if (gl.GetQueryObjectui64v) gl.GetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
      gpu += ns / 1e6;
      nd++;
    }
    double d = now_ms() - t1;
    if (d < best_d) { best_d = d; best_g = gpu; tm->dispatches = nd; }
  }
  gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
  double t2 = now_ms();
  gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, bo);
  gl.GetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (long)p->seq * p->heads * p->vd * 4, out);
  tm->readback_ms = now_ms() - t2;
  tm->dispatch_ms = best_d;
  tm->gpu_ms = best_g;
  GLenum e = gl.GetError();
  GLuint bufs[4] = {bq, bk, bv, bo};
  gl.DeleteBuffers(4, bufs);
  if (e != GL_NO_ERROR) { snprintf(err, errn, "GL error 0x%x", e); return -1; }
  return 0;
}

/* ---- driver ---------------------------------------------------------------------------- */
static void json_str(const char *s) {
  putchar('"');
  for (; *s; s++) {
    if (*s == '"' || *s == '\\') printf("\\%c", *s);
    else if ((unsigned char)*s < 0x20) printf("\\u%04x", *s);
    else putchar(*s);
  }
  putchar('"');
}

static int report(const char *opdir, const char *tensor, const char *backend, long n, double macs,
                  double ms, const float *got, const float *want, double tol, const gl_times *tm,
                  const char *renderer) {
  cmp_t c = compare(got, want, n);
  int pass = c.rel_rms <= tol && c.cosine > 0.999999;
  printf("{\"op\":");
  json_str(opdir);
  printf(",\"tensor\":");
  json_str(tensor);
  printf(",\"backend\":\"%s\",\"pass\":%s,\"tolerance_rel_rms\":%.1e,\"max_abs\":%.3e,\"rel_rms\":%.3e,"
         "\"cosine\":%.9f,\"elements_differ\":%ld,\"elements\":%ld,\"ms\":%.4f,\"gmac_s\":%.3f",
         backend, pass ? "true" : "false", tol, c.max_abs, c.rel_rms, c.cosine, c.rows_differ, n, ms,
         ms > 0 ? macs / (ms * 1e6) : 0.0);
  if (tm)
    printf(",\"upload_ms\":%.3f,\"dispatch_to_fence_ms\":%.4f,\"gpu_ms\":%.4f,\"readback_ms\":%.4f,"
           "\"dispatches\":%d,\"renderer\":",
           tm->upload_ms, tm->dispatch_ms, tm->gpu_ms, tm->readback_ms, tm->dispatches);
  if (tm) json_str(renderer ? renderer : "?");
  printf(",\"neon\":%s}\n", HAVE_NEON ? "true" : "false");
  fflush(stdout);
  return pass;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: jibo-cq-bench OPDIR [--backend ref|neon|gl|all] [--reps N] [--tt N] "
                    "[--rows-per-dispatch N] [--libgl PATH]\n");
    return 2;
  }
  const char *dir = argv[1], *backend = "all", *libgl = NULL;
  int reps = 5, tt = 4, rpd = 1 << 30;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--backend") && i + 1 < argc) backend = argv[++i];
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--tt") && i + 1 < argc) tt = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--rows-per-dispatch") && i + 1 < argc) rpd = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--libgl") && i + 1 < argc) libgl = argv[++i];
    else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  if (reps < 1 || tt < 1 || tt > 32 || rpd < 1) { fprintf(stderr, "bad --reps/--tt/--rows-per-dispatch\n"); return 2; }
  int want_ref = !strcmp(backend, "ref") || !strcmp(backend, "all");
#if HAVE_NEON
  int want_neon = !strcmp(backend, "neon") || !strcmp(backend, "all");
#endif
  int want_gl = !strcmp(backend, "gl") || !strcmp(backend, "all");
  if (!strcmp(backend, "neon") && !HAVE_NEON) { fprintf(stderr, "built without NEON\n"); return 2; }

  long len;
  char *txt = read_file(dir, "op.txt", &len);
  if (!txt) { fprintf(stderr, "%s/op.txt: cannot read\n", dir); return 2; }
  txt = realloc(txt, (size_t)len + 1);
  txt[len] = 0;
  char tensor[256] = "?";
  const char *tp = strstr(txt, "tensor=");
  if (tp) sscanf(tp + 7, "%255[^\n]", tensor);
  int is_kron = strstr(txt, "op=kron") != NULL;
  int is_attn = strstr(txt, "op=attn") != NULL;
  int all_pass = 1;
  char err[4096] = "";
  const char *renderer = NULL;
  int gl_ok = 0;
  if (want_gl) {
    if (gl_init(libgl, err, sizeof err) == 0) {
      gl_ok = 1;
      renderer = (const char *)gl.GetString(GL_RENDERER);
      GLint shared = 0, inv = 0, ssbo = 0;
      gl.GetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &shared);
      gl.GetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &inv);
      gl.GetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &ssbo);
      printf("{\"gl\":{\"renderer\":");
      json_str(renderer ? renderer : "?");
      printf(",\"version\":");
      json_str((const char *)gl.GetString(GL_VERSION));
      printf(",\"max_shared_bytes\":%d,\"max_invocations\":%d,\"max_ssbo_bytes\":%d}}\n", shared, inv, ssbo);
    } else {
      printf("{\"backend\":\"gl\",\"pass\":false,\"skipped\":true,\"error\":");
      json_str(err);
      printf("}\n");
      all_pass = 0;
    }
  }

  if (is_attn) {
    attn_op p;
    p.seq = (int)op_int(txt, "tokens");
    p.heads = (int)op_int(txt, "heads");
    p.kv_heads = (int)op_int(txt, "kv_heads");
    p.qk = (int)op_int(txt, "qk");
    p.vd = (int)op_int(txt, "vd");
    p.window = (int)op_int(txt, "window");
    if (p.seq <= 0 || p.heads <= 0 || p.kv_heads <= 0 || p.heads % p.kv_heads || p.qk <= 0 ||
        p.vd <= 0 || p.vd > 256 || p.window < 0) {
      fprintf(stderr, "unsupported attention geometry\n");
      return 2;
    }
    long ql, kl, vl, ol;
    long nq = (long)p.seq * p.heads * p.qk, nk = (long)p.seq * p.kv_heads * p.qk;
    long nv = (long)p.seq * p.kv_heads * p.vd, no = (long)p.seq * p.heads * p.vd;
    p.q = read_file(dir, "q.bin", &ql);
    p.k = read_file(dir, "kk.bin", &kl);
    p.v = read_file(dir, "v.bin", &vl);
    p.o_want = read_file(dir, "o.bin", &ol);
    if (!p.q || !p.k || !p.v || !p.o_want || ql != nq * 4 || kl != nk * 4 || vl != nv * 4 || ol != no * 4) {
      fprintf(stderr, "%s: files missing or of the wrong size\n", dir);
      return 2;
    }
    /* MACs: QK^T and PV over the causal (windowed) triangle */
    double pairs = 0;
    for (int t = 0; t < p.seq; t++) pairs += (p.window > 0 && t + 1 > p.window) ? p.window : t + 1;
    double macs = pairs * p.heads * (p.qk + p.vd);
    float *o = malloc(sizeof(float) * (size_t)no);
    if (want_ref) {
      double best = 1e30;
      for (int r = 0; r < reps; r++) { double t = now_ms(); attn_ref(&p, o); t = now_ms() - t; if (t < best) best = t; }
      all_pass &= report(dir, tensor, "ref", no, macs, best, o, p.o_want, 1e-5, NULL, NULL);
    }
    if (want_gl && gl_ok) {
      gl_times tm = {0};
      memset(o, 0, sizeof(float) * (size_t)no);
      if (gl_attn(&p, rpd, reps, o, &tm, err, sizeof err)) {
        printf("{\"backend\":\"gl\",\"pass\":false,\"error\":");
        json_str(err);
        printf("}\n");
        all_pass = 0;
      } else {
        char name[64];
        snprintf(name, sizeof name, "gl positions_per_dispatch=%d", rpd > p.seq ? p.seq : rpd);
        all_pass &= report(dir, tensor, name, no, macs, tm.dispatch_ms, o, p.o_want, 1e-4, &tm, renderer);
      }
    }
  } else if (!is_kron) {
    cq_op w;
    memset(&w, 0, sizeof w);
    w.out = (int)op_int(txt, "out");
    w.in = (int)op_int(txt, "in");
    w.in_padded = (int)op_int(txt, "in_padded");
    w.group = (int)op_int(txt, "group");
    w.bits = (int)op_int(txt, "bits");
    w.tokens = (int)op_int(txt, "tokens");
    if (w.out <= 0 || w.in_padded <= 0 || w.group <= 0 || w.group > 1024 || w.tokens <= 0 ||
        (w.bits != 2 && w.bits != 4) || w.in_padded % w.group) {
      fprintf(stderr, "unsupported op: only 2- and 4-bit CQ with group <= 1024\n");
      return 2;
    }
    w.per_byte = 8 / w.bits;
    w.row_bytes = w.in_padded * w.bits / 8;
    w.num_groups = w.in_padded / w.group;
    long wl, ll, xl, yl;
    const uint8_t *blob = read_file(dir, "w.bin", &wl);
    float *levels = read_file(dir, "levels.bin", &ll);
    w.x = read_file(dir, "x.bin", &xl);
    w.y_want = read_file(dir, "y.bin", &yl);
    long need = (long)w.out * w.row_bytes + (long)w.out * w.num_groups * 2;
    if (!blob || !levels || !w.x || !w.y_want || wl < need || ll != (1L << w.bits) * 4 ||
        xl != (long)w.tokens * w.in_padded * 4 || yl != (long)w.tokens * w.out * 4) {
      fprintf(stderr, "%s: files missing or of the wrong size\n", dir);
      return 2;
    }
    if (w.row_bytes % 4) { fprintf(stderr, "row bytes not a multiple of 4\n"); return 2; }
    w.packed = blob;
    w.norms16 = blob + (long)w.out * w.row_bytes;
    w.norms = malloc(sizeof(float) * (size_t)(w.out * w.num_groups));
    for (long i = 0; i < (long)w.out * w.num_groups; i++) {
      uint16_t h;
      memcpy(&h, w.norms16 + 2 * i, 2);
      w.norms[i] = half_to_float(h);
    }
    memcpy(w.levels, levels, (size_t)ll);
    w.lut = malloc(sizeof(float) * 256 * (size_t)w.per_byte);
    for (int byte = 0; byte < 256; byte++)
      for (int k = 0; k < w.per_byte; k++) w.lut[byte * w.per_byte + k] = w.levels[(byte >> (k * w.bits)) & ((1 << w.bits) - 1)];
    long n = (long)w.tokens * w.out;
    double macs = (double)w.tokens * w.out * w.in_padded;
    float *y = malloc(sizeof(float) * (size_t)n);
    if (want_ref) {
      double best = 1e30;
      for (int r = 0; r < reps; r++) { double t = now_ms(); cq_ref(&w, y); t = now_ms() - t; if (t < best) best = t; }
      all_pass &= report(dir, tensor, "ref", n, macs, best, y, w.y_want, 1e-5, NULL, NULL);
    }
#if HAVE_NEON
    if (want_neon) {
      double best = 1e30;
      for (int r = 0; r < reps; r++) { double t = now_ms(); cq_neon(&w, y); t = now_ms() - t; if (t < best) best = t; }
      all_pass &= report(dir, tensor, "neon", n, macs, best, y, w.y_want, 1e-5, NULL, NULL);
    }
#endif
    if (want_gl && gl_ok) {
      gl_times tm = {0};
      memset(y, 0, sizeof(float) * (size_t)n);
      if (gl_cq(&w, tt, rpd, reps, y, &tm, err, sizeof err)) {
        printf("{\"backend\":\"gl\",\"pass\":false,\"error\":");
        json_str(err);
        printf("}\n");
        all_pass = 0;
      } else {
        char name[64];
        snprintf(name, sizeof name, "gl tt=%d rows_per_dispatch=%d", tt, rpd > w.out ? w.out : rpd);
        all_pass &= report(dir, tensor, name, n, macs, tm.dispatch_ms, y, w.y_want, 1e-4, &tm, renderer);
      }
    }
  } else {
    kron_op p;
    p.ba = (int)op_int(txt, "ba");
    p.bb = (int)op_int(txt, "bb");
    p.tokens = (int)op_int(txt, "tokens");
    if (p.ba <= 0 || p.bb <= 0 || p.ba % 4 || p.bb % 4 || p.tokens <= 0 || p.ba * p.bb > 4096) {
      fprintf(stderr, "unsupported kron geometry\n");
      return 2;
    }
    long al, bl, zl, kl, n = (long)p.tokens * p.ba * p.bb;
    p.a = read_file(dir, "a.bin", &al);
    p.b = read_file(dir, "b.bin", &bl);
    p.z = read_file(dir, "z.bin", &zl);
    p.k_want = read_file(dir, "k.bin", &kl);
    if (!p.a || !p.b || !p.z || !p.k_want || al != (long)p.ba * p.ba * 4 || bl != (long)p.bb * p.bb * 4 ||
        zl != n * 4 || kl != n * 4) {
      fprintf(stderr, "%s: files missing or of the wrong size\n", dir);
      return 2;
    }
    double macs = (double)p.tokens * (2.0 * p.ba * p.ba * p.bb);
    float *o = malloc(sizeof(float) * (size_t)n);
    if (want_ref) {
      double best = 1e30;
      for (int r = 0; r < reps; r++) { double t = now_ms(); kron_ref(&p, o); t = now_ms() - t; if (t < best) best = t; }
      all_pass &= report(dir, tensor, "ref", n, macs, best, o, p.k_want, 1e-5, NULL, NULL);
    }
#if HAVE_NEON
    if (want_neon) {
      double best = 1e30;
      for (int r = 0; r < reps; r++) { double t = now_ms(); kron_neon(&p, o); t = now_ms() - t; if (t < best) best = t; }
      all_pass &= report(dir, tensor, "neon", n, macs, best, o, p.k_want, 1e-5, NULL, NULL);
    }
#endif
    if (want_gl && gl_ok) {
      gl_times tm = {0};
      memset(o, 0, sizeof(float) * (size_t)n);
      if (gl_kron(&p, reps, o, &tm, err, sizeof err)) {
        printf("{\"backend\":\"gl\",\"pass\":false,\"error\":");
        json_str(err);
        printf("}\n");
        all_pass = 0;
      } else {
        all_pass &= report(dir, tensor, "gl", n, macs, tm.dispatch_ms, o, p.k_want, 1e-4, &tm, renderer);
      }
    }
  }
  return all_pass ? 0 : 1;
}
