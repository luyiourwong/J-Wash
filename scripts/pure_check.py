# Validation of an exported checkpoint in PURE transformers (no J-Wash code in
# the inference path): runs the identity/control battery and prints the fish
# score. Run it AFTER unloading the model from the server (VRAM):
# scripts/jlab.py unload
#
#   python -X utf8 scripts/pure_check.py data/edits/<name> [--device cuda:0]
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

config.setup_env()

import torch
import transformers

from jlab import fish_score  # same scoring as the server probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max", type=int, default=200)
    parser.add_argument("--prompts", default=str(Path(__file__).with_name("fish_prompts.json")))
    args = parser.parse_args()

    spec = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    print(f"loading {args.checkpoint} on {args.device} (pure transformers)...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, device_map={"": args.device}
    )
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.checkpoint)
    cfg = json.loads((Path(args.checkpoint) / "config.json").read_text(encoding="utf-8"))
    print(f"tie_word_embeddings = {cfg.get('tie_word_embeddings')}")

    def generate(prompt):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt", enable_thinking=False,
        )
        ids = (encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]).to(args.device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=args.max, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    ok_ident = ok_ctrl = 0
    for p in spec["identity"]:
        text = generate(p)
        n, hits = fish_score(text)
        ok_ident += bool(n)
        print(f"\n🐟={n:<2} {p}\n    {' '.join(text.split())[:400]}")
        if hits:
            print(f"    words: {', '.join(hits)}")
    for p in spec["control"]:
        text = generate(p["prompt"])
        good = any(a.lower() in text.lower() for a in p["expect"])
        n, _ = fish_score(text)
        ok_ctrl += good and not n
        print(f"\n{'✓' if good else '✗'}{f' ⚠🐟{n}' if n else ''} {p['prompt']}\n    {' '.join(text.split())[:300]}")
    print(f"\n=== fish identity: {ok_ident}/{len(spec['identity'])} — "
          f"clean controls: {ok_ctrl}/{len(spec['control'])} ===")


if __name__ == "__main__":
    main()
