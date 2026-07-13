import gzip
import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))

import torch
import transformers

import jlens
from jlens.examples import EXAMPLES, resolve_prompt
from jlens.vis import build_page, compute_slice

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"

jlens.configure_logging()

hf_model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16
).to("cuda:0")
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = jlens.from_hf(hf_model, tokenizer)
print(model)

lens = jlens.JacobianLens.from_pretrained(
    LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
)
print(lens)

prompt = "Fact: The currency used in the country shaped like a boot is"
layers = [
    model.n_layers // 4,
    model.n_layers // 2,
    model.n_layers // 4 * 3,
    model.n_layers - 2,
]

jlens_logits, model_logits, _ = lens.apply(model, prompt, layers=layers, positions=[-2])
logit_lens, _, _ = lens.apply(
    model, prompt, layers=layers, positions=[-2], use_jacobian=False
)


def top5(logits):
    return [tokenizer.decode([t]) for t in logits.topk(5).indices]


print(f"\nprompt: {prompt!r} (reading at position -2, the 'boot' token)\n")
for layer in layers:
    print(f"L{layer:>3} logit-lens: {top5(logit_lens[layer][0])}")
    print(f"L{layer:>3} J-lens:     {top5(jlens_logits[layer][0])}")
print(f"model (actual output): {top5(model_logits[0])}")

gloss_path = ROOT / "vendor" / "jacobian-lens" / "assets" / "qwen_gloss.json.gz"
gloss = {int(k): v for k, v in json.load(gzip.open(gloss_path)).items()}

example = next(e for e in EXAMPLES if e.slug == "multihop")
slice_prompt = resolve_prompt(example, tokenizer)
slice_data = compute_slice(
    model, lens, slice_prompt, layer_stride=2, mask_display=True
)
page, _, _ = build_page(
    slice_data,
    slice_prompt,
    title=example.section,
    description=example.description,
    alt_token=gloss,
)
out_path = ROOT / "data" / "walkthrough" / "multihop.html"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(page, encoding="utf-8")
print(f"\nself-contained slice page: {out_path}")

vram = torch.cuda.memory_allocated(0) / 2**30
print(f"VRAM allocated cuda:0: {vram:.1f} GB")
