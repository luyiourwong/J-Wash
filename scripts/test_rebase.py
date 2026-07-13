# Numerical validation of the readthrough/exact modes (core/rebase) on the test
# tiny-llama: the live preview (RMSNorm hooks) must equal the bake (transformed
# weights) up to rounding, and the exact mode must approach the standard hook
# (only the RMS approximation separates them).
#
#   python -X utf8 scripts/test_rebase.py
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

config.setup_env()

import torch
import transformers

import jlens
from core import rebase
from core.ablation import Interventions

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
PROMPTS = ["The capital of France is", "Once upon a time, a"]


def cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def make_rules(jl, layers):
    """Synthetic rules: logit-lens directions (J = I), like _direction without a
    lens. A saturated replace + a partial scale to cover both."""
    W = jl._lm_head.weight.detach().float()

    def unit(token_id):
        v = W[token_id]
        return v / v.norm().clamp_min(1e-8)

    def dirs(token_id):
        return {l: unit(token_id) for l in layers}

    return [
        {
            "id": 1, "token_id": 42, "token": "<42>", "mode": "replace",
            "factor": 1.0, "replacement_id": 137, "replacement": "<137>",
            "layers": list(layers), "dirs_a": dirs(42), "dirs_b": dirs(137),
        },
        {
            "id": 2, "token_id": 550, "token": "<550>", "mode": "scale",
            "factor": 0.4, "replacement_id": None, "replacement": None,
            "layers": list(layers), "dirs_a": dirs(550), "dirs_b": None,
        },
    ]


def logits_with(model, jl, input_ids, rules=None, mode="standard", scale=1.0):
    iv = Interventions()
    if rules:
        iv._rules = rules  # direct injection: add() requires a loaded lens
        iv.set_scale(scale)
        iv.set_mode(mode)
        iv.attach(jl)
    try:
        with torch.no_grad():
            return model(input_ids).logits[:, -1, :].detach().clone()
    finally:
        iv.detach()


def baked_model(model, jl, rules, scale, exact):
    transforms, info = rebase.build_plan(rules, jl, scale, exact=exact)
    clone = copy.deepcopy(model)
    state = clone.state_dict()
    missing = [k for k in transforms if k not in state]
    assert not missing or (info["tied"] and missing == [info["lm_head_key"]]), missing
    for key, transform in transforms.items():
        source = state.get(key)
        if source is None:  # tied: un-embedding baked from the embed
            source = state[info["embed_key"]]
        state[key] = rebase.apply_transform(transform, source.float())[0]
    if info["tied"]:
        clone.config.tie_word_embeddings = False
        clone.lm_head.weight = torch.nn.Parameter(state[info["lm_head_key"]])
    clone.load_state_dict(state)
    return clone


def main():
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    jl = jlens.from_hf(model, tokenizer)
    n = len(jl.layers)
    layers = [max(0, n // 2 - 1)]  # low hook → downstream layers to transform (exact ≠ readthrough)
    print(f"{MODEL}: {n} layers, d_model={jl.d_model}, hook on {layers}, "
          f"tied={jl._lm_head.weight.data_ptr() == jl._embed_tokens.weight.data_ptr()}")
    rules = make_rules(jl, layers)
    input_ids = tokenizer(PROMPTS, return_tensors="pt", padding=True).input_ids

    base = logits_with(model, jl, input_ids)
    failures = []

    def compare(label, case_rules, scale, checks):
        std = logits_with(model, jl, input_ids, case_rules, "standard", scale)
        d_std = std - base
        results = {}
        for mode, exact in (("readthrough", False), ("exact", True)):
            live = logits_with(model, jl, input_ids, case_rules, mode, scale)
            clone = baked_model(model, jl, case_rules, scale, exact)
            jl2 = jlens.from_hf(clone, tokenizer)
            baked = logits_with(clone, jl2, input_ids)
            live_vs_bake = (live - baked).abs().max().item()
            scale_ref = live.abs().max().item()
            c_std = cos(live - base, d_std)
            results[mode] = c_std
            print(f"[{label}] scale={scale} {mode:12s} live≡bake: max|Δ|={live_vs_bake:.3e} "
                  f"(ref {scale_ref:.1f})  cos(Δlogits vs standard)={c_std:.4f}  "
                  f"‖Δ‖={float((live - base).norm()):.3f} vs std ‖Δ‖={float(d_std.norm()):.3f}")
            if live_vs_bake > 1e-3 * scale_ref:
                failures.append(f"[{label}] {mode} scale={scale}: live ≠ bake ({live_vs_bake:.3e})")
            if float((live - base).norm()) < 1e-6:
                failures.append(f"[{label}] {mode} scale={scale}: no effect measured")
        checks(results)

    # Saturated case (replace + zap): the target regime. readthrough must follow
    # standard; exact is regularized (expected degradation, warning).
    for scale in (1.0, 2.0):
        compare("saturated", rules, scale, lambda r, s=scale: failures.append(
            f"[saturated] readthrough scale={s}: cos {r['readthrough']:.3f} < 0.85"
        ) if r["readthrough"] < 0.85 else None)

    # Soft case (partial scale, no singularity): exact must match standard at
    # least as well as readthrough (its whole point).
    soft = [r for r in rules if r["mode"] == "scale"]
    compare("soft", soft, 1.0, lambda r: failures.append(
        f"[soft] exact: cos {r['exact']:.3f} expected ≥ readthrough {r['readthrough']:.3f}"
    ) if r["exact"] < r["readthrough"] - 0.01 or r["exact"] < 0.95 else None)

    if failures:
        print("\nFAILURES:\n - " + "\n - ".join(failures))
        sys.exit(1)
    print("\nOK: live preview ≡ bake for readthrough and exact; exact ≈ standard hook.")


if __name__ == "__main__":
    main()
