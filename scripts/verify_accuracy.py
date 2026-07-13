import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config

config.setup_env()

import torch

from core.lens_manager import ActivationCatcher, LensManager
from core.model_manager import ModelManager

MODEL_ID = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
PROMPT = "Fact: The currency used in the country shaped like a boot is"
LAYERS = [8, 14, 20, 26]
POSITIONS = [-4, -2, -1]
TOPK = 5
COS_MIN = 0.9999

print("loading the model and lens ...")
mm = ModelManager()
mm.load(MODEL_ID, "bf16", None, "cuda:0")
lm = LensManager()
lm.load(mm, repo_id=LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION, layers=LAYERS)

jl = mm.jl
ref_logits, _, input_ids = lm.lens.apply(jl, PROMPT, layers=LAYERS, positions=POSITIONS)

catcher = ActivationCatcher(jl.layers, LAYERS)
with torch.no_grad():
    mm.hf_model(input_ids=input_ids, use_cache=True)
catcher.close()

tok = mm.tokenizer
all_ok = True
worst_cos = 1.0
for li, layer in enumerate(LAYERS):
    for pi, pos in enumerate(POSITIONS):
        h = catcher.acts[layer][0, pos].float().to(lm._J.device)
        live = jl.unembed(torch.einsum("ij,j->i", lm._J[li], h)).float().cpu()
        ref = ref_logits[layer][pi]
        top_live = live.topk(TOPK).indices.tolist()
        top_ref = ref.topk(TOPK).indices.tolist()
        cos = torch.nn.functional.cosine_similarity(live, ref, dim=0).item()
        match = top_live == top_ref
        all_ok &= match and cos >= COS_MIN
        worst_cos = min(worst_cos, cos)
        words = [tok.decode([t]).strip() for t in top_ref]
        print(
            f"L{layer:>2} pos{pos:>3}  top{TOPK} {'MATCH' if match else 'MISMATCH'}"
            f"  cos={cos:.6f}  ref={words}"
        )
        if not match:
            print(f"      live={[tok.decode([t]).strip() for t in top_live]}")

print(f"\nminimum cos: {worst_cos:.6f} (threshold {COS_MIN})")
print("PASS: live path (hooks + KV cache) == JacobianLens.apply reference" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
