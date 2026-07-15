"""Run the J-Wash server."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run the J-Wash server.")
    parser.add_argument(
        "--hf-cache", metavar="PATH", default=None,
        help="Hugging Face cache directory. Default: the shared HF cache "
             "(HF_HOME, else ~/.cache/huggingface). Pass a path — e.g. ./hf_cache — "
             "to keep downloads isolated in a project-local cache.",
    )
    parser.add_argument(
        "--data-dir", metavar="PATH", default=None,
        help="Runtime data directory (history, frames, presets, edits). "
             "Default: ./data. Give each instance its own when running several "
             "servers side by side.",
    )
    parser.add_argument(
        "--host", default=None,
        help="Bind address (default: 127.0.0.1). Use 0.0.0.0 for Docker.",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="HTTP port (default: 8381). Use a distinct port per instance.",
    )
    args = parser.parse_args()

    # Set the env overrides BEFORE importing config (which reads them).
    if args.data_dir:
        os.environ["JWASH_DATA_DIR"] = str(Path(args.data_dir).expanduser().resolve())
    if args.hf_cache:
        os.environ["HF_HOME"] = str(Path(args.hf_cache).expanduser().resolve())
    else:
        # HF cache chosen in the Options tab (data/settings.json); the
        # --hf-cache flag wins over it, the plain HF_HOME env loses to it.
        import json
        data_dir = Path(
            os.environ.get("JWASH_DATA_DIR") or Path(__file__).resolve().parent / "data"
        )
        try:
            saved = json.loads((data_dir / "settings.json").read_text(encoding="utf-8"))
            if saved.get("hf_cache"):
                os.environ["HF_HOME"] = str(Path(saved["hf_cache"]).expanduser().resolve())
        except Exception:
            pass

    import config

    config.setup_env()

    import uvicorn

    host = args.host or os.environ.get("JWASH_HOST") or config.HOST
    uvicorn.run("api.app:app", host=host, port=args.port or config.PORT, log_level="info")


if __name__ == "__main__":
    main()
