import json
import re
import time

from huggingface_hub import HfApi

import config

HUB_SEED_REPO = "neuronpedia/jacobian-lens"
HUB_CACHE_TTL = 600

_hub_cache = {"at": 0.0, "entries": None, "error": None}


def local_lenses():
    out = []
    if not config.LENSES_DIR.exists():
        return out
    for entry in sorted(config.LENSES_DIR.iterdir()):
        lens_file = entry / "lens.pt"
        if not entry.is_dir() or not lens_file.exists():
            continue
        meta = {}
        meta_file = entry / "meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        out.append({"name": entry.name, "path": str(lens_file), "meta": meta})
    return out


def _base_model_from(filename):
    stem = filename.rsplit("/", 1)[-1].removesuffix(".pt")
    match = re.match(r"(.+?)_jacobian_lens(?:_n\d+)?$", stem)
    return match.group(1) if match else None


def _derived_model_id(base):
    if base is None:
        return None
    lowered = base.lower()
    if lowered.startswith("qwen"):
        return f"Qwen/{base}"
    if lowered.startswith("gemma"):
        return f"google/{base}"
    if lowered.startswith("llama"):
        return f"meta-llama/{base}"
    if lowered.startswith("gpt-oss"):
        return f"openai/{base}"
    if lowered == "gpt2":
        return "openai-community/gpt2"
    if lowered.startswith("pythia"):
        return f"EleutherAI/{base}"
    if lowered.startswith("olmo"):
        return f"allenai/{base}"
    return base


def hub_lenses(force=False):
    now = time.time()
    if not force and _hub_cache["entries"] is not None and now - _hub_cache["at"] < HUB_CACHE_TTL:
        return _hub_cache["entries"]
    api = HfApi()
    repos = {HUB_SEED_REPO}
    try:
        for model in api.list_models(search="jacobian-lens", limit=50):
            repos.add(model.id)
        for model in api.list_models(filter="jacobian_lens", limit=50):
            repos.add(model.id)
    except Exception as exc:
        _hub_cache.update(error=f"Hub search unavailable: {exc}")
    entries = []
    for repo_id in sorted(repos):
        try:
            refs = api.list_repo_refs(repo_id)
            branches = [b.name for b in refs.branches] or ["main"]
        except Exception:
            continue
        for branch in branches:
            try:
                files = api.list_repo_files(repo_id, revision=branch)
            except Exception:
                continue
            for filename in files:
                if not filename.endswith(".pt"):
                    continue
                base = _base_model_from(filename)
                entries.append(
                    {
                        "repo_id": repo_id,
                        "revision": branch,
                        "filename": filename,
                        "base_model": base,
                        "derived_model_id": _derived_model_id(base),
                        "model_revision_verified": False,
                    }
                )
    _hub_cache.update(at=now, entries=entries)
    return entries


_base_cache = {}
BASE_CACHE_TTL = 600


def hub_base_model(model_id):
    """Base model declared by the repo's model card (``base_model`` tags) —
    e.g. a finetune pointing at the checkpoint it was trained from. ``None``
    if unknown, offline, or not a Hub repo."""
    now = time.time()
    hit = _base_cache.get(model_id)
    if hit and now - hit["at"] < BASE_CACHE_TTL:
        return hit["base"]
    found = None
    if "/" in model_id and not model_id.startswith("local/"):
        try:
            info = HfApi().model_info(model_id)
            for tag in info.tags or []:
                if not tag.startswith("base_model:"):
                    continue
                rest = tag[len("base_model:"):]
                if ":" in rest:  # qualified form: finetune:X, adapter:X, quantized:X
                    rest = rest.split(":", 1)[1]
                if rest and rest.lower() != model_id.lower():
                    found = rest
                    break
        except Exception:
            found = None
    _base_cache[model_id] = {"at": now, "base": found}
    return found


def lenses_for_model(model_id, revision=None):
    matches_local = []
    for lens in local_lenses():
        meta = lens["meta"]
        if meta.get("model_id") != model_id:
            continue
        lens_rev = meta.get("model_revision")
        compatible = True
        reason = None
        if revision and lens_rev and lens_rev != revision:
            compatible = False
            reason = f"fit revision ({lens_rev[:12]}) != loaded model ({revision[:12]})"
        elif lens_rev is None and not model_id.startswith("local/"):
            reason = "fit revision unknown"
        matches_local.append(dict(lens, compatible=compatible, reason=reason))

    base = model_id.split("/")[-1].lower()
    base_ref = hub_base_model(model_id)  # e.g. "google/gemma-3-1b-it" for a finetune
    base_ref_name = base_ref.split("/")[-1].lower() if base_ref else None

    def hub_entry(entry, via, reason=None, compatible=True):
        return dict(
            entry,
            via=via,
            compatible=compatible,
            reason=reason,
            cached=_lens_cached(entry["repo_id"], entry["filename"], entry["revision"]),
        )

    # one entry per branch in hub_lenses → dedupe, main first
    entries = []
    seen = set()
    for entry in sorted(hub_lenses(), key=lambda e: e["revision"] != "main"):
        if entry["base_model"] is None or (entry["repo_id"], entry["filename"]) in seen:
            continue
        seen.add((entry["repo_id"], entry["filename"]))
        entries.append(entry)

    matches_hub = []
    matched = set()
    prefix_hits = {}
    for entry in entries:
        key = (entry["repo_id"], entry["filename"])
        name = entry["base_model"].lower()
        derived = (entry["derived_model_id"] or "").lower()
        if name == base or derived == model_id.lower():
            # ⚠ only for a real problem (local merge); the fit revision not being
            # published is the normal state of Hub repos → discreet note
            reason = None
            if model_id.startswith("local/"):
                reason = "local model: a Hub lens fitted on the original checkpoint doesn't match a merge"
            matched.add(key)
            matches_hub.append(hub_entry(
                entry, "model", reason, compatible=not model_id.startswith("local/")))
        elif base_ref and (name == base_ref_name or derived == base_ref.lower()):
            matched.add(key)
            matches_hub.append(hub_entry(
                entry, "base-model",
                f"lens of the base model {base_ref} — fitted on the original "
                "weights, a finetune's readouts may drift slightly"))
        elif base_ref is None and name != base and len(name) >= 6 and base.startswith(name):
            prefix_hits.setdefault(len(name), []).append(entry)

    # No card metadata: fall back to the longest name prefix (a finetune usually
    # keeps its base's name — "gemma-3-1b-it-toxicity" → "gemma-3-1b-it").
    if prefix_hits and not any(m["via"] == "model" for m in matches_hub):
        for entry in prefix_hits[max(prefix_hits)]:
            matched.add((entry["repo_id"], entry["filename"]))
            matches_hub.append(hub_entry(
                entry, "base-guess",
                f"the name suggests a finetune of {entry['base_model']} — fitted "
                "on the original weights, readouts may drift slightly"))

    # Everything else stays reachable for cross-model loading (your own
    # architecture-compatible lens); d_model/layers are checked at load time.
    others = [
        hub_entry(entry, "other")
        for entry in entries
        if (entry["repo_id"], entry["filename"]) not in matched
    ]
    return {
        "local": matches_local,
        "hub": matches_hub,
        "other": others,
        "base_model": base_ref,
        "hub_error": _hub_cache.get("error"),
    }


def _lens_cached(repo_id, filename, revision=None):
    """True if the lens file is already in the local HF cache (no download on load)."""
    from huggingface_hub import try_to_load_from_cache

    try:
        result = try_to_load_from_cache(repo_id, filename, revision=revision)
        return isinstance(result, str)
    except Exception:
        return False


def resolve_lens(path=None, repo_id=None, filename=None):
    if path:
        for lens in local_lenses():
            if lens["path"] == path:
                meta = lens["meta"]
                return {
                    "source": "local",
                    "name": lens["name"],
                    "required_model": meta.get("model_id"),
                    "required_revision": meta.get("model_revision"),
                    "meta": meta,
                }
        return {"source": "local", "required_model": None, "meta": {}, "warning": "meta.json missing: required model unknown"}
    base = _base_model_from(filename or "")
    return {
        "source": "hub",
        "repo_id": repo_id,
        "filename": filename,
        "required_model": _derived_model_id(base),
        "required_revision": None,
        "warning": "model derived from the filename; exact revision not published",
    }
