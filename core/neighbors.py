import re
import threading

import torch

from core.lens_manager import MASKS_DIR, _vocab_fingerprint

# ── EVALUATION TOGGLE ────────────────────────────────────────────────────────
# Nearest tokens (the "translation" of a non-latin token to readable neighbors):
#   True  = keep only ENGLISH words (pure ASCII, no accents) as targets
#   False = any readable latin script (accents included: fr/de/es…)
# Set to True by default; flip it to compare.
ENGLISH_ONLY = True
# ─────────────────────────────────────────────────────────────────────────────

# "translation" targets: readable tokens (2+ letter word, apostrophe/hyphen
# allowed) so the neighbors are interpretable
_LATIN_RE = re.compile(r"^[ A-Za-zÀ-ɏ'\-]+$")
_LATIN_LETTERS_RE = re.compile(r"[A-Za-zÀ-ɏ]{2}")
# english variant: pure ASCII (excludes café, über, naïve… → filters out the
# other latin-script languages)
_ENGLISH_RE = re.compile(r"^[ A-Za-z'\-]+$")
_ENGLISH_LETTERS_RE = re.compile(r"[A-Za-z]{2}")


def _latin_target_mask(tokenizer, vocab_size):
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    # distinct cache per mode (otherwise a "latin" mask would serve in english mode)
    tag = "english" if ENGLISH_ONLY else "latin"
    word_re = _ENGLISH_RE if ENGLISH_ONLY else _LATIN_RE
    letters_re = _ENGLISH_LETTERS_RE if ENGLISH_ONLY else _LATIN_LETTERS_RE
    path = MASKS_DIR / f"{_vocab_fingerprint(tokenizer)}_{vocab_size}_{tag}.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    n_decodable = min(vocab_size, len(tokenizer))
    decoded = tokenizer.batch_decode(
        [[tid] for tid in range(n_decodable)], clean_up_tokenization_spaces=False
    )
    for tid, raw in enumerate(decoded):
        s = raw.strip()
        mask[tid] = bool(
            len(s) >= 2 and word_re.match(s) and letters_re.search(s)
        )
    torch.save(mask, path)
    return mask


class TokenNeighbors:
    """Approximate local translation: latin tokens whose output direction (row of
    W_U) is closest in cosine to a non-latin token most often carry the same
    meaning (答案 → ' answer')."""

    def __init__(self):
        self._lock = threading.Lock()
        self._key = None
        self._mask = None
        self._norms = None
        self._cache = {}

    def _prepare(self, jl, tokenizer, model_key):
        if self._key == model_key and self._norms is not None:
            return
        weight = jl._lm_head.weight
        if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("neighbors unavailable on a quantized model")
        vocab_size = weight.shape[0]
        self._mask = _latin_target_mask(tokenizer, vocab_size).to(weight.device)
        norms = torch.empty(vocab_size, dtype=torch.float32, device=weight.device)
        with torch.no_grad():
            for start in range(0, vocab_size, 8192):
                chunk = weight[start:start + 8192].float()
                norms[start:start + 8192] = chunk.norm(dim=1)
        self._norms = norms.clamp_min(1e-8)
        self._cache = {}
        self._key = model_key

    def lookup(self, jl, tokenizer, model_key, token_ids, k=3):
        with self._lock:
            self._prepare(jl, tokenizer, model_key)
            weight = jl._lm_head.weight
            out = {}
            for tid in token_ids:
                tid = int(tid)
                if tid < 0 or tid >= weight.shape[0]:
                    out[tid] = []
                    continue
                if tid in self._cache:
                    out[tid] = self._cache[tid]
                    continue
                with torch.no_grad():
                    v = weight[tid]
                    sims = (weight @ v).float() / (self._norms * self._norms[tid])
                    sims[~self._mask] = float("-inf")
                    sims[tid] = float("-inf")
                    top = torch.topk(sims, min(k, int(self._mask.sum())))
                entries = [
                    {
                        "id": int(i),
                        "str": tokenizer.decode([int(i)]),
                        "sim": round(float(s), 3),
                    }
                    for s, i in zip(top.values.tolist(), top.indices.tolist())
                    if s != float("-inf")
                ]
                self._cache[tid] = entries
                out[tid] = entries
            return out

    def reset(self):
        with self._lock:
            self._key = None
            self._mask = None
            self._norms = None
            self._cache = {}
