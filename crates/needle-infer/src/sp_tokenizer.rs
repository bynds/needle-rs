//! SentencePiece BPE tokenizer, decoded from the RAW blob a `.cact` file embeds.
//!
//! Port of `needle/model/export.py::RefTokenizer`, which upstream ships as the
//! normative encoder/decoder for the blob format — so this is a port of the spec
//! rather than of `sentencepiece` itself. No vocabulary file, no dependency: the
//! piece table travels inside the weights.
//!
//! Blob layout (little-endian, `export._TK_HDR` / `export._TK_REC`):
//!
//! ```text
//! header   u32 n_pieces, u32 pad_id, u32 eos_id, u32 bos_id, u32 unk_id,
//!          u8 add_dummy_prefix, u8 byte_fallback, u16 _pad        (24 bytes)
//! pieces   n_pieces records in id order:
//!          f32 score, u8 type, u16 surface_len, surface_len UTF-8 bytes
//! ```
//!
//! Encoding is SentencePiece's usual two stages: split off the user-defined chat
//! markers first, then run greedy highest-score pairwise merging over the
//! characters of each remaining segment, falling back to byte pieces for
//! anything out of vocabulary.

use std::collections::HashMap;
use std::fmt;

/// SentencePiece's visible space.
pub const META_SPACE: &str = "\u{2581}";

/// Piece types, matching `export.TK_*`.
pub const TK_NORMAL: u8 = 0;
pub const TK_UNKNOWN: u8 = 1;
pub const TK_CONTROL: u8 = 2;
pub const TK_USER_DEFINED: u8 = 3;
pub const TK_BYTE: u8 = 4;

const HDR_BYTES: usize = 24;
const REC_FIXED: usize = 7;

/// The v2 chat markers, in the id order `needle/model/tokenizer.py` documents.
/// Surfaces are looked up in the piece table rather than trusted, so a drift
/// upstream surfaces as a missing marker rather than as a wrong id.
pub const CHAT_MARKERS: [&str; 10] = [
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tools>",
    "</tools>",
    "<tool_call>",
    "</tool_call>",
    "<tool_result>",
    "</tool_result>",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    /// Blob ended before a declared field could be read.
    Truncated { need: usize, got: usize },
    /// A piece's bytes are not valid UTF-8.
    InvalidUtf8 { piece: usize },
    /// `n_pieces` is zero.
    Empty,
    /// A special id from the header is not a valid piece index.
    SpecialIdOutOfRange {
        name: &'static str,
        id: u32,
        n_pieces: usize,
    },
    /// A `TK_BYTE` piece is not of the form `<0xNN>`.
    MalformedBytePiece { piece: usize },
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { need, got } => {
                write!(f, "tokenizer blob truncated: need {need} bytes, got {got}")
            }
            Self::InvalidUtf8 { piece } => write!(f, "piece {piece} is not valid UTF-8"),
            Self::Empty => write!(f, "tokenizer blob declares zero pieces"),
            Self::SpecialIdOutOfRange { name, id, n_pieces } => {
                write!(
                    f,
                    "header {name}={id} is outside the {n_pieces}-piece table"
                )
            }
            Self::MalformedBytePiece { piece } => {
                write!(f, "byte piece {piece} is not of the form <0xNN>")
            }
        }
    }
}

impl std::error::Error for TokenizerError {}

/// A decoded SentencePiece model.
pub struct SpTokenizer {
    pieces: Vec<String>,
    scores: Vec<f32>,
    types: Vec<u8>,
    /// Surface -> id. Later duplicates win, matching Python's dict comprehension.
    piece_to_id: HashMap<String, u32>,
    /// Byte value -> id, for `byte_fallback`.
    byte_to_id: [Option<u32>; 256],
    /// User-defined surfaces, longest first — longest-match-wins at scan time.
    markers: Vec<(String, u32)>,
    pub pad_id: u32,
    pub eos_id: u32,
    pub bos_id: u32,
    pub unk_id: u32,
    pub add_dummy_prefix: bool,
    pub byte_fallback: bool,
}

impl SpTokenizer {
    /// Decode the RAW tokenizer blob from a `.cact` file.
    pub fn from_blob(blob: &[u8]) -> Result<Self, TokenizerError> {
        if blob.len() < HDR_BYTES {
            return Err(TokenizerError::Truncated {
                need: HDR_BYTES,
                got: blob.len(),
            });
        }
        let u32_at = |o: usize| -> u32 {
            u32::from_le_bytes([blob[o], blob[o + 1], blob[o + 2], blob[o + 3]])
        };
        let n_pieces = u32_at(0) as usize;
        let (pad_id, eos_id, bos_id, unk_id) = (u32_at(4), u32_at(8), u32_at(12), u32_at(16));
        let add_dummy_prefix = blob[20] != 0;
        let byte_fallback = blob[21] != 0;
        if n_pieces == 0 {
            return Err(TokenizerError::Empty);
        }
        // Every piece needs at least its fixed record, so the blob bounds the count. Checked
        // before the vectors below are sized from it: an unchecked `n_pieces` from a damaged
        // container is an allocation failure, which aborts rather than returning an error.
        let most = (blob.len() - HDR_BYTES) / REC_FIXED;
        if n_pieces > most {
            return Err(TokenizerError::Truncated {
                need: HDR_BYTES.saturating_add(n_pieces.saturating_mul(REC_FIXED)),
                got: blob.len(),
            });
        }

        let mut pieces = Vec::with_capacity(n_pieces);
        let mut scores = Vec::with_capacity(n_pieces);
        let mut types = Vec::with_capacity(n_pieces);
        let mut off = HDR_BYTES;
        for i in 0..n_pieces {
            if off + REC_FIXED > blob.len() {
                return Err(TokenizerError::Truncated {
                    need: off + REC_FIXED,
                    got: blob.len(),
                });
            }
            let score =
                f32::from_le_bytes([blob[off], blob[off + 1], blob[off + 2], blob[off + 3]]);
            let ty = blob[off + 4];
            let len = u16::from_le_bytes([blob[off + 5], blob[off + 6]]) as usize;
            off += REC_FIXED;
            if off + len > blob.len() {
                return Err(TokenizerError::Truncated {
                    need: off + len,
                    got: blob.len(),
                });
            }
            let surface = std::str::from_utf8(&blob[off..off + len])
                .map_err(|_| TokenizerError::InvalidUtf8 { piece: i })?
                .to_string();
            off += len;
            pieces.push(surface);
            scores.push(score);
            types.push(ty);
        }

        for (name, id) in [
            ("pad", pad_id),
            ("eos", eos_id),
            ("bos", bos_id),
            ("unk", unk_id),
        ] {
            if id as usize >= n_pieces {
                return Err(TokenizerError::SpecialIdOutOfRange { name, id, n_pieces });
            }
        }

        // Later duplicates overwrite earlier ones, as in
        // `{p: i for i, p in enumerate(pieces)}`.
        let mut piece_to_id = HashMap::with_capacity(n_pieces);
        for (i, p) in pieces.iter().enumerate() {
            piece_to_id.insert(p.clone(), i as u32);
        }

        let mut byte_to_id = [None; 256];
        for (i, (p, &ty)) in pieces.iter().zip(types.iter()).enumerate() {
            if ty != TK_BYTE {
                continue;
            }
            // Surfaces look like `<0xNN>`; upstream slices `p[3:5]`.
            let hex = p
                .get(3..5)
                .ok_or(TokenizerError::MalformedBytePiece { piece: i })?;
            let b = u8::from_str_radix(hex, 16)
                .map_err(|_| TokenizerError::MalformedBytePiece { piece: i })?;
            byte_to_id[b as usize] = Some(i as u32);
        }

        let mut markers: Vec<(String, u32)> = pieces
            .iter()
            .zip(types.iter())
            .enumerate()
            .filter(|(_, (_, &ty))| ty == TK_USER_DEFINED)
            .map(|(i, (p, _))| (p.clone(), i as u32))
            .collect();
        // Longest surface first, so `</tool_call>` cannot be shadowed by a
        // shorter prefix. `sorted(..., key=len, reverse=True)` in Python is
        // stable, so equal lengths keep table order.
        markers.sort_by_key(|m| core::cmp::Reverse(m.0.len()));

        Ok(Self {
            pieces,
            scores,
            types,
            piece_to_id,
            byte_to_id,
            markers,
            pad_id,
            eos_id,
            bos_id,
            unk_id,
            add_dummy_prefix,
            byte_fallback,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    pub fn piece(&self, id: u32) -> Option<&str> {
        self.pieces.get(id as usize).map(|s| s.as_str())
    }

    pub fn piece_type(&self, id: u32) -> Option<u8> {
        self.types.get(id as usize).copied()
    }

    pub fn score(&self, id: u32) -> Option<f32> {
        self.scores.get(id as usize).copied()
    }

    /// Id of an exact surface, e.g. a chat marker.
    pub fn id_of(&self, surface: &str) -> Option<u32> {
        self.piece_to_id.get(surface).copied()
    }

    /// The chat marker ids in [`CHAT_MARKERS`] order, or `None` if the table is
    /// missing any of them.
    pub fn chat_marker_ids(&self) -> Option<Vec<u32>> {
        CHAT_MARKERS.iter().map(|m| self.id_of(m)).collect()
    }

    /// Encode text to token ids.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut esc = text.replace(' ', META_SPACE);
        if self.add_dummy_prefix {
            esc.insert_str(0, META_SPACE);
        }

        let mut ids = Vec::new();
        let mut buf = String::new();
        let bytes = esc.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            // Longest-match-wins over the user-defined markers, as in
            // `next((m for m in self.markers if esc.startswith(m, i)), None)`.
            let hit = self
                .markers
                .iter()
                .find(|(m, _)| bytes[i..].starts_with(m.as_bytes()));
            match hit {
                Some((m, id)) => {
                    self.bpe_into(&buf, &mut ids);
                    buf.clear();
                    ids.push(*id);
                    i += m.len();
                }
                None => {
                    // Advance one character, not one byte.
                    let ch = esc[i..].chars().next().expect("valid UTF-8 boundary");
                    buf.push(ch);
                    i += ch.len_utf8();
                }
            }
        }
        self.bpe_into(&buf, &mut ids);
        ids
    }

    /// Greedy highest-score pairwise merging over the characters of one segment.
    ///
    /// Faithful to `RefTokenizer._bpe`, including tie-breaking: the scan keeps
    /// the first (leftmost) pair at the maximum score, because the comparison is
    /// strict.
    fn bpe_into(&self, segment: &str, out: &mut Vec<u32>) {
        if segment.is_empty() {
            return;
        }
        // Symbols as byte ranges into `segment`, held in a doubly linked list so
        // a merge is O(1) and only the scan is linear.
        let mut starts: Vec<usize> = Vec::new();
        let mut ends: Vec<usize> = Vec::new();
        for (b, ch) in segment.char_indices() {
            starts.push(b);
            ends.push(b + ch.len_utf8());
        }
        let n = starts.len();
        // Merges only ever absorb a symbol's successor, so the head never moves
        // and no back-links are needed.
        let mut next: Vec<isize> = (0..n as isize).map(|i| i + 1).collect();
        if let Some(last) = next.last_mut() {
            *last = -1;
        }
        let mut live = n;

        while live > 1 {
            let mut best_score = f32::NEG_INFINITY;
            let mut best: isize = -1;
            let mut j: isize = 0;
            while j >= 0 && next[j as usize] >= 0 {
                let k = next[j as usize];
                let merged = &segment[starts[j as usize]..ends[k as usize]];
                if let Some(&id) = self.piece_to_id.get(merged) {
                    let s = self.scores[id as usize];
                    if best < 0 || s > best_score {
                        best_score = s;
                        best = j;
                    }
                }
                j = k;
            }
            if best < 0 {
                break;
            }
            let k = next[best as usize];
            ends[best as usize] = ends[k as usize];
            next[best as usize] = next[k as usize];
            live -= 1;
        }

        let mut j: isize = 0;
        while j >= 0 {
            let sym = &segment[starts[j as usize]..ends[j as usize]];
            match self.piece_to_id.get(sym) {
                Some(&id) => out.push(id),
                None if self.byte_fallback => {
                    for &b in sym.as_bytes() {
                        // A byte piece exists for all 256 values when
                        // byte_fallback is set; fall back to unk if not.
                        out.push(self.byte_to_id[b as usize].unwrap_or(self.unk_id));
                    }
                }
                None => out.push(self.unk_id),
            }
            j = next[j as usize];
        }
    }

    /// Per-token decoded bytes, indexed by id.
    ///
    /// This is what a grammar-constrained decoder needs: the bytes each token
    /// would contribute to the output, so a byte-level automaton can be advanced
    /// per token. Follows the same rules as [`decode`] — the meta-space becomes a
    /// real space, `TK_BYTE` pieces become their raw byte, and control or unknown
    /// pieces contribute nothing.
    ///
    /// [`decode`]: SpTokenizer::decode
    pub fn token_bytes(&self) -> Vec<Vec<u8>> {
        (0..self.pieces.len())
            .map(|id| {
                let piece = &self.pieces[id];
                match self.types[id] {
                    TK_BYTE => piece
                        .get(3..5)
                        .and_then(|h| u8::from_str_radix(h, 16).ok())
                        .map_or_else(Vec::new, |b| vec![b]),
                    TK_CONTROL | TK_UNKNOWN => Vec::new(),
                    _ => piece.replace(META_SPACE, " ").into_bytes(),
                }
            })
            .collect()
    }

    /// Decode token ids back to text.
    ///
    /// Control and unknown pieces are dropped, byte pieces are re-assembled, and
    /// invalid UTF-8 becomes U+FFFD — matching `RefTokenizer.decode`.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut buf: Vec<u8> = Vec::new();
        for &id in ids {
            let Some(&ty) = self.types.get(id as usize) else {
                continue;
            };
            match ty {
                TK_BYTE => {
                    let p = &self.pieces[id as usize];
                    if let Some(b) = p.get(3..5).and_then(|h| u8::from_str_radix(h, 16).ok()) {
                        buf.push(b);
                    }
                }
                TK_CONTROL | TK_UNKNOWN => {}
                _ => buf.extend_from_slice(self.pieces[id as usize].as_bytes()),
            }
        }
        let text = String::from_utf8_lossy(&buf).replace(META_SPACE, " ");
        if self.add_dummy_prefix {
            if let Some(rest) = text.strip_prefix(' ') {
                return rest.to_string();
            }
        }
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small blob by the documented layout.
    fn blob(
        pieces: &[(&str, f32, u8)],
        add_dummy: bool,
        byte_fallback: bool,
        specials: (u32, u32, u32, u32),
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(pieces.len() as u32).to_le_bytes());
        for v in [specials.0, specials.1, specials.2, specials.3] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.push(u8::from(add_dummy));
        out.push(u8::from(byte_fallback));
        out.extend_from_slice(&0u16.to_le_bytes());
        assert_eq!(out.len(), HDR_BYTES);
        for (surface, score, ty) in pieces {
            out.extend_from_slice(&score.to_le_bytes());
            out.push(*ty);
            let b = surface.as_bytes();
            out.extend_from_slice(&(b.len() as u16).to_le_bytes());
            out.extend_from_slice(b);
        }
        out
    }

    /// Pieces: specials, a few byte pieces, characters, and two merges. Scores
    /// are chosen so the merge order is unambiguous.
    fn toy() -> SpTokenizer {
        let mut p: Vec<(&str, f32, u8)> = vec![
            ("<pad>", 0.0, TK_CONTROL),
            ("</s>", 0.0, TK_CONTROL),
            ("<s>", 0.0, TK_CONTROL),
            ("<unk>", 0.0, TK_UNKNOWN),
            ("<tool_call>", 0.0, TK_USER_DEFINED),
            ("<tool>", 0.0, TK_USER_DEFINED),
            ("\u{2581}", -1.0, TK_NORMAL),
            ("a", -2.0, TK_NORMAL),
            ("b", -3.0, TK_NORMAL),
            ("c", -4.0, TK_NORMAL),
            ("ab", -0.5, TK_NORMAL),
            ("abc", -0.25, TK_NORMAL),
            ("\u{2581}a", -0.75, TK_NORMAL),
        ];
        // Byte pieces for the whole range, so byte_fallback always resolves.
        let hex: Vec<String> = (0..256).map(|b| format!("<0x{b:02X}>")).collect();
        let leaked: Vec<&'static str> = hex
            .into_iter()
            .map(|s| Box::leak(s.into_boxed_str()) as &'static str)
            .collect();
        for s in &leaked {
            p.push((s, -50.0, TK_BYTE));
        }
        SpTokenizer::from_blob(&blob(&p, true, true, (0, 1, 2, 3))).unwrap()
    }

    #[test]
    fn parses_header_and_pieces() {
        let t = toy();
        assert_eq!(t.pad_id, 0);
        assert_eq!(t.eos_id, 1);
        assert_eq!(t.bos_id, 2);
        assert_eq!(t.unk_id, 3);
        assert!(t.add_dummy_prefix);
        assert!(t.byte_fallback);
        assert_eq!(t.vocab_size(), 13 + 256);
        assert_eq!(t.piece(4), Some("<tool_call>"));
        assert_eq!(t.piece_type(4), Some(TK_USER_DEFINED));
        assert_eq!(t.id_of("abc"), Some(11));
    }

    #[test]
    fn markers_are_longest_first() {
        let t = toy();
        // `<tool_call>` must be tried before `<tool>`, or "<tool_call>" would
        // tokenize as `<tool>` + "_call>".
        assert_eq!(t.markers[0].0, "<tool_call>");
        assert_eq!(t.encode("<tool_call>"), vec![6, 4]);
    }

    #[test]
    fn dummy_prefix_and_space_escaping() {
        let t = toy();
        // "a" -> dummy prefix + "a" -> the merge "▁a" wins (-0.75 > -1, -2).
        assert_eq!(t.encode("a"), vec![12]);
        // "b" has no merge with the prefix, so both stand alone.
        assert_eq!(t.encode("b"), vec![6, 8]);
    }

    #[test]
    fn greedy_merge_picks_highest_score() {
        let t = toy();
        // "abc": "abc" is not a *pair*, so merging is ab (-0.5) then ab+c,
        // which is the piece "abc" (-0.25) -> a single id.
        assert_eq!(t.encode("abc"), vec![6, 11]);
    }

    #[test]
    fn byte_fallback_for_out_of_vocabulary() {
        let t = toy();
        // 'z' is not a piece; byte fallback emits its UTF-8 byte piece.
        let ids = t.encode("z");
        let z = t.id_of("<0x7A>").unwrap();
        assert_eq!(ids, vec![6, z]);
        // Multi-byte characters fall back per byte.
        let ids = t.encode("\u{20AC}"); // euro sign, 3 bytes
        assert_eq!(ids.len(), 4);
        assert_eq!(
            ids[1..],
            [
                t.id_of("<0xE2>").unwrap(),
                t.id_of("<0x82>").unwrap(),
                t.id_of("<0xAC>").unwrap()
            ]
        );
    }

    #[test]
    fn decode_round_trips_and_strips_dummy_prefix() {
        let t = toy();
        for s in ["a", "ab", "abc", "a b", "z", "\u{20AC}", "abc abc"] {
            assert_eq!(t.decode(&t.encode(s)), s, "round trip {s:?}");
        }
    }

    #[test]
    fn decode_drops_control_and_unknown() {
        let t = toy();
        let ids = vec![t.bos_id, 6, 11, t.eos_id, t.unk_id];
        assert_eq!(t.decode(&ids), "abc");
    }

    #[test]
    fn empty_input_yields_no_tokens() {
        assert!(toy().encode("").is_empty());
        assert_eq!(toy().decode(&[]), "");
    }

    /// The per-token bytes must be exactly what decoding that token alone yields,
    /// for every token in the table — otherwise a constrained decoder and the
    /// final text would disagree about what was emitted.
    #[test]
    fn token_bytes_agree_with_single_token_decode() {
        let t = toy();
        let table = t.token_bytes();
        assert_eq!(table.len(), t.vocab_size());
        for id in 0..t.vocab_size() as u32 {
            let decoded = t.decode(&[id]);
            let from_table = String::from_utf8_lossy(&table[id as usize]).to_string();
            match t.piece_type(id) {
                Some(TK_CONTROL) | Some(TK_UNKNOWN) => {
                    assert!(
                        table[id as usize].is_empty(),
                        "id {id} should contribute nothing"
                    );
                }
                Some(TK_BYTE) => {
                    assert_eq!(table[id as usize].len(), 1, "id {id} is one raw byte");
                }
                _ => {
                    // decode() strips a leading dummy-prefix space; the table
                    // deliberately does not, so compare with that allowance.
                    assert!(
                        from_table == decoded || from_table == format!(" {decoded}"),
                        "id {id}: table {from_table:?} vs decode {decoded:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_malformed_blobs() {
        let good = blob(&[("a", 0.0, TK_NORMAL)], true, false, (0, 0, 0, 0));
        assert!(matches!(
            SpTokenizer::from_blob(&good[..10]),
            Err(TokenizerError::Truncated { .. })
        ));
        assert!(matches!(
            SpTokenizer::from_blob(&good[..good.len() - 1]),
            Err(TokenizerError::Truncated { .. })
        ));
        let empty = blob(&[], true, false, (0, 0, 0, 0));
        assert_eq!(
            SpTokenizer::from_blob(&empty).err(),
            Some(TokenizerError::Empty)
        );

        // A special id past the end of the piece table.
        let bad_special = blob(&[("a", 0.0, TK_NORMAL)], true, false, (0, 9, 0, 0));
        assert_eq!(
            SpTokenizer::from_blob(&bad_special).err(),
            Some(TokenizerError::SpecialIdOutOfRange {
                name: "eos",
                id: 9,
                n_pieces: 1
            })
        );

        // A byte piece that is not <0xNN>.
        let bad_byte = blob(&[("nope", 0.0, TK_BYTE)], true, false, (0, 0, 0, 0));
        assert_eq!(
            SpTokenizer::from_blob(&bad_byte).err(),
            Some(TokenizerError::MalformedBytePiece { piece: 0 })
        );

        // Non-UTF-8 surface bytes.
        let mut bad_utf8 = blob(&[("a", 0.0, TK_NORMAL)], true, false, (0, 0, 0, 0));
        let last = bad_utf8.len() - 1;
        bad_utf8[last] = 0xFF;
        assert_eq!(
            SpTokenizer::from_blob(&bad_utf8).err(),
            Some(TokenizerError::InvalidUtf8 { piece: 0 })
        );
    }
}
