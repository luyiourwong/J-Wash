import hashlib
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

from jlens.lens import JacobianLens

import config
from core.gpus import gpu_stats

FITS_DIR = config.DATA_DIR / "fits"
WORKER = config.ROOT / "scripts" / "fit_worker.py"

# Fit corpora. "mixed" = both, equal parts (rounded to the nearest prompt).
DATASET_WIKITEXT = "Salesforce/wikitext-103-raw-v1"
DATASET_HARMLESS = "heretic-org/Semantic-Harmless"
FIT_DATASETS = (DATASET_WIKITEXT, DATASET_HARMLESS, "mixed")


def _load_corpus(dataset, n, skip=0):
    """``n`` prompts from ``dataset``, skipping the first ``skip`` picks
    (continue-from: the new prompts must not overlap the base lens's).

    wikitext keeps the historical behavior (first records ≥600 chars, streamed).
    Semantic-Harmless is a small instruct set (~416 one-line prompts): we draw a
    seeded random sample — sample(skip+n) then drop the head, so a continued fit
    extends the same sequence — and PACK the picks into ~350-char sequences
    (median prompt ≈ 10 tokens, and jlens skips the first 16 positions of every
    sequence as attention sinks: unpacked, almost every pick would be dropped as
    "too short"). ``n``/``skip`` count SOURCE prompts, not packs. "mixed" takes
    equal parts of both (n odd: the extra prompt goes to wikitext) and shuffles
    the union so multi-GPU slices stay mixed."""
    if dataset == "mixed":
        import random

        n_wiki = (n + 1) // 2
        s_wiki = (skip + 1) // 2
        prompts = _load_corpus(DATASET_WIKITEXT, n_wiki, s_wiki)
        prompts += _load_corpus(DATASET_HARMLESS, n - n_wiki, skip // 2)
        random.Random(1729).shuffle(prompts)
        return prompts
    if dataset == DATASET_HARMLESS:
        import random

        from datasets import load_dataset

        texts = [r["text"] for r in load_dataset(DATASET_HARMLESS, split="train")]
        if skip + n > len(texts):
            raise ValueError(
                f"{DATASET_HARMLESS} has {len(texts)} prompts, "
                f"{skip + n} requested (continue included) — lower n_prompts"
            )
        picks = random.Random(1729).sample(texts, skip + n)[skip:]
        packs, cur = [], ""
        for text in picks:
            cur = f"{cur}\n\n{text}" if cur else text
            if len(cur) >= 350:
                packs.append(cur)
                cur = ""
        if cur:
            # a lone sub-16-token tail would be skipped by jlens anyway: fold it
            # into the previous pack instead of losing it
            if packs and len(cur) < 120:
                packs[-1] += "\n\n" + cur
            else:
                packs.append(cur)
        return packs
    from jlens.examples import load_wikitext_prompts

    # load skip + n then keep the tail: the new prompts don't overlap
    # those of the base lens
    return load_wikitext_prompts(skip + n)[skip:]

def _default_dim_batch(device):
    """Default dim_batch scaled to the device's VRAM.
    Measured on a 4B bf16 fit: 8 fits in 16 GB, 4 in 12 GB."""
    try:
        total = gpu_stats()[int(device.split(":")[1])]["vram_total"]
        return 8 if total >= 15 * 2**30 else 4
    except Exception:
        return 4


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FitManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._procs = []
        self.state = {"state": "idle"}
        self.on_progress = None

    def _emit(self):
        if self.on_progress:
            self.on_progress(dict(self.state))

    def start(self, *, model_id, source, n_prompts=100, dtype="bf16", quant=None,
              devices=("cuda:0",), name=None, dim_batch=None,
              max_seq_len=128, source_layers=None, model_revision=None,
              continue_from=None, dataset=DATASET_WIKITEXT):
        with self._lock:
            if self.state.get("state") == "running":
                raise ValueError("a fitting is already in progress")
            if not devices:
                raise ValueError("at least one device required")
            if dataset not in FIT_DATASETS:
                raise ValueError(f"unknown dataset: {dataset} (choices: {', '.join(FIT_DATASETS)})")
            skip_prompts = 0
            base_lens = None
            if continue_from:
                base_lens = JacobianLens.load(continue_from)
                # new prompts: skip those already seen by the base lens
                skip_prompts = base_lens.n_prompts
                if source_layers is None:
                    source_layers = list(base_lens.source_layers)
            if name is None:
                base = model_id.split("/")[-1]
                if dataset == "mixed":
                    base += "_mixed"
                elif dataset == DATASET_HARMLESS:
                    base += "_harmless"
                total = n_prompts + skip_prompts
                name = f"{base}_n{total}" if continue_from else f"{base}_n{n_prompts}"
            params = {
                "model_id": model_id,
                "source": source,
                "model_revision": model_revision,
                "dtype": dtype,
                "quant": quant,
                "n_prompts": n_prompts,
                "dataset": dataset,
                "devices": list(devices),
                "dim_batch": dim_batch,
                "max_seq_len": max_seq_len,
                "source_layers": source_layers,
                "continue_from": continue_from,
                "skip_prompts": skip_prompts,
            }
            self.state = {
                "state": "running",
                "name": name,
                "phase": "corpus",
                "total": n_prompts,
                "done": 0,
                "workers": [],
                "eta_seconds": None,
                "started_at": _now(),
                "params": params,
                "error": None,
            }
            self._procs = []
        threading.Thread(target=self._run, args=(name, params), daemon=True).start()
        return dict(self.state)

    def stop(self):
        with self._lock:
            for proc in self._procs:
                if proc.poll() is None:
                    proc.terminate()
            if self.state.get("state") == "running":
                self.state["state"] = "stopping"
        self._emit()
        return dict(self.state)

    def _run(self, name, params):
        try:
            job_dir = FITS_DIR / name
            job_dir.mkdir(parents=True, exist_ok=True)
            corpus_path = job_dir / "corpus.json"
            if corpus_path.exists():
                prompts = json.loads(corpus_path.read_text(encoding="utf-8"))
            else:
                prompts = _load_corpus(
                    params.get("dataset", DATASET_WIKITEXT),
                    params["n_prompts"],
                    params.get("skip_prompts", 0),
                )
                corpus_path.write_text(
                    json.dumps(prompts, ensure_ascii=False), encoding="utf-8"
                )
            devices = params["devices"]
            n = len(prompts)
            if len(devices) == 2:
                cut = int(n * 0.65)
                slices = [prompts[:cut], prompts[cut:]]
            else:
                slices = [prompts]

            self.state.update(phase="fitting", total=n)
            workers = []
            started = time.perf_counter()
            for i, (device, chunk) in enumerate(zip(devices, slices)):
                slice_path = job_dir / f"slice{i}.json"
                if not slice_path.exists():
                    slice_path.write_text(json.dumps(chunk, ensure_ascii=False), encoding="utf-8")
                dim_batch = params["dim_batch"] or _default_dim_batch(device)
                cmd = [
                    sys.executable, "-X", "utf8", str(WORKER),
                    "--model", params["source"],
                    "--device", device,
                    "--dtype", params["dtype"],
                    "--prompts", str(slice_path),
                    "--checkpoint", str(job_dir / f"ckpt{i}.pt"),
                    "--out", str(job_dir / f"lens{i}.pt"),
                    "--dim-batch", str(dim_batch),
                    "--max-seq-len", str(params["max_seq_len"]),
                ]
                if params["quant"]:
                    cmd += ["--quant", params["quant"]]
                if params["source_layers"]:
                    cmd += ["--source-layers", json.dumps(params["source_layers"])]
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    cwd=str(config.ROOT),
                )
                self._procs.append(proc)
                worker_state = {
                    "device": device,
                    "done": 0,
                    "total": len(chunk),
                    "dim_batch": dim_batch,
                    "state": "loading",
                    "elapsed": 0.0,
                    # [done, elapsed] of the last 10 updates: the ETA follows the
                    # RECENT pace (throughput can degrade mid-fit, e.g. VRAM
                    # saturated — a global average would then freeze the ETA)
                    "hist": [],
                }
                workers.append(worker_state)
                threading.Thread(
                    target=self._read_worker, args=(proc, worker_state, started), daemon=True
                ).start()
            self.state["workers"] = workers
            self._emit()

            stderr_tails = [""] * len(self._procs)

            def drain_err(index, proc):
                data = proc.stderr.read()
                stderr_tails[index] = (data or "")[-2000:]

            drainers = [
                threading.Thread(target=drain_err, args=(i, p), daemon=True)
                for i, p in enumerate(self._procs)
            ]
            for t in drainers:
                t.start()
            for proc in self._procs:
                proc.wait()
            for t in drainers:
                t.join()
            failed = [i for i, p in enumerate(self._procs) if p.returncode != 0]
            if self.state.get("state") == "stopping":
                self.state.update(state="stopped")
                self._emit()
                return
            if failed:
                detail = " | ".join(stderr_tails[i].strip().splitlines()[-1] if stderr_tails[i].strip() else "?" for i in failed)
                raise RuntimeError(f"worker(s) {failed} failed: {detail}")

            self.state.update(phase="merge")
            self._emit()
            partials = [
                JacobianLens.load(str(job_dir / f"lens{i}.pt"))
                for i in range(len(slices))
            ]
            merged = JacobianLens.merge(partials) if len(partials) > 1 else partials[0]
            if params.get("continue_from"):
                base_lens = JacobianLens.load(params["continue_from"])
                if base_lens.source_layers != merged.source_layers:
                    raise RuntimeError(
                        "cannot continue: the source layers differ from the base lens "
                        f"({base_lens.source_layers[0]}..{base_lens.source_layers[-1]} vs "
                        f"{merged.source_layers[0]}..{merged.source_layers[-1]})"
                    )
                # weighted average by n_prompts = equivalent to a fit over the union
                merged = JacobianLens.merge([base_lens, merged])
            out_dir = config.LENSES_DIR / name
            out_dir.mkdir(parents=True, exist_ok=True)
            lens_path = out_dir / "lens.pt"
            merged.save(str(lens_path))
            meta = {
                "name": name,
                "model_id": params["model_id"],
                "model_revision": params["model_revision"],
                "model_source": params["source"],
                "d_model": merged.d_model,
                "source_layers": [merged.source_layers[0], merged.source_layers[-1]],
                "dtype": params["dtype"],
                "quant": params["quant"],
                "n_prompts": merged.n_prompts,
                "corpus": (
                    f"mixed: {DATASET_WIKITEXT} + {DATASET_HARMLESS} (equal parts)"
                    if params.get("dataset") == "mixed"
                    else params.get("dataset", DATASET_WIKITEXT)
                ),
                "max_seq_len": params["max_seq_len"],
                "devices": params["devices"],
                "continued_from": params.get("continue_from"),
                "config_hash": hashlib.sha1(
                    json.dumps(params, sort_keys=True).encode()
                ).hexdigest()[:16],
                "created_at": _now(),
                "fit_seconds": round(time.perf_counter() - started, 1),
            }
            (out_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            self.state.update(
                state="done",
                phase="done",
                lens_path=str(lens_path),
                meta=meta,
                eta_seconds=0,
            )
            self._emit()
        except Exception as exc:
            self.state.update(state="error", error=str(exc))
            self._emit()

    def _read_worker(self, proc, worker_state, started):
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event["event"] == "loading":
                worker_state["state"] = "loading"
            elif event["event"] in ("progress", "resume"):
                worker_state["state"] = "fitting"
                worker_state["done"] = event["done"]
                worker_state["total"] = event["total"]
                worker_state["elapsed"] = round(time.perf_counter() - started, 1)
                hist = worker_state.setdefault("hist", [])
                hist.append([worker_state["done"], worker_state["elapsed"]])
                del hist[:-10]
            elif event["event"] == "done":
                worker_state["state"] = "done"
                worker_state["done"] = worker_state["total"]
            self._refresh_totals(started)
            self._emit()

    def _refresh_totals(self, started):
        workers = self.state.get("workers", [])
        self.state["done"] = sum(w["done"] for w in workers)
        etas = []
        for w in workers:
            if not (w["done"] > 0 and w["elapsed"] > 0 and w["done"] < w["total"]):
                continue
            hist = w.get("hist") or []
            if len(hist) >= 2 and hist[-1][1] > hist[0][1] and hist[-1][0] > hist[0][0]:
                # pace over the last 10 updates (sliding window)
                rate = (hist[-1][0] - hist[0][0]) / (hist[-1][1] - hist[0][1])
            else:
                rate = w["done"] / w["elapsed"]
            etas.append((w["total"] - w["done"]) / rate)
        # multi-GPU: the fit ETA = the slowest worker
        self.state["eta_seconds"] = round(max(etas), 0) if etas else None
        self.state["vram"] = [
            {"index": g["index"], "used_gb": round(g["vram_used"] / 2**30, 1)}
            for g in gpu_stats()
        ]
