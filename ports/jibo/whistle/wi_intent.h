/* wi_intent.h: closed-set command recognition over Whistle's speech embedding (Jev-style).
 *
 * needle_embed (upstream libneedle, public API) gives one 512-wide row per 80 ms frame. This
 * module pools the frames into mean, max and standard deviation, standardises them and applies a
 * linear softmax over the command templates plus "none". The model file is written by
 * train_eval.py --out (wi_export.py converts it): see README.md. */
#ifndef WI_INTENT_H
#define WI_INTENT_H
#include <stddef.h>

typedef struct {
  int classes, dim;  /* dim = 3 * embedding width */
  char **names;      /* classes labels; "none" rejects */
  float *mean, *scale, *w, *b;
} wi_model;

/* Load a model written by wi_export.py. Returns 0 on success. */
int wi_load(const char *path, wi_model *m);
void wi_free(wi_model *m);

/* Probabilities for `frames` rows of `width` floats; prob must hold m->classes floats. Returns the
 * index of the most probable class. */
int wi_classify(const wi_model *m, const float *emb, int frames, int width, float *prob);
#endif
