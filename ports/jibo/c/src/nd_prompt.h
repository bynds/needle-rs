/* nd_prompt.h: chat prompt assembly (transcription of crates/needle-infer/src/prompt.rs) and
 * to_snake_case (crates/needle-infer/src/tokenizer.rs). Output is byte-identical to Rust. */
#ifndef ND_PROMPT_H
#define ND_PROMPT_H

#include "nd_common.h"

/* Chat-template markers, as needle/model/finetune.py assembles them. */
#define ND_IM_START "<|im_start|>"
#define ND_IM_END "<|im_end|>"
#define ND_TOOLS_START "<tools>"
#define ND_TOOLS_END "</tools>"
#define ND_TOOL_CALL_START "<tool_call>"
#define ND_TOOL_CALL_END "</tool_call>"
#define ND_THINK_START "<think>"
#define ND_THINK_END "</think>"

/* Assemble the chat prompt:
 *   [<|im_start|>system\n{system}<|im_end|>\n]
 *   <|im_start|>user\n<tools>{compact_json(tools)}</tools>\n{query}<|im_end|>\n<|im_start|>assistant\n
 * `system` may be NULL (no system turn). Returns malloc'd NUL-terminated text, or NULL on
 * allocation failure or a NULL query/tools. */
char *nd_build_prompt(const char *query, const char *tools_json, const char *system_or_null);

/* Strip insignificant whitespace (space, tab, LF, CR) outside JSON string literals; not a
 * parser, anything else passes through byte for byte. malloc'd, NULL on failure. */
char *nd_compact_json(const char *s);

/* Tool name to snake_case (camelCase, PascalCase, dot.notation, hyphen-case), with Rust's
 * Unicode char::is_alphanumeric / is_uppercase / is_lowercase (Unicode 16.0, rustc 1.87).
 * malloc'd; NULL on allocation failure or if `name` is not valid UTF-8. */
char *nd_to_snake_case(const char *name);

#endif
