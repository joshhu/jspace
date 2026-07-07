"""CLI: `python -m jspace serve` / `python -m jspace precompute <model_id>`."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="jspace")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="啟動網頁介面")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=7860)

    p_pre = sub.add_parser("precompute", help="預計算某模型的平均 Jacobian")
    p_pre.add_argument("model_id")
    p_pre.add_argument("--num-prompts", type=int, default=1000)
    p_pre.add_argument("--num-targets", type=int, default=8)
    p_pre.add_argument("--max-tokens", type=int, default=256)

    args = parser.parse_args()

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("jspace.server:app", host=args.host, port=args.port)
    elif args.cmd == "precompute":
        from tqdm.auto import tqdm

        from .model_adapter import load_model
        from .runner import precompute

        bundle = load_model(args.model_id)
        bar = tqdm(total=args.num_prompts, desc=f"precompute {args.model_id}")

        progress: dict = {}

        import threading
        import time

        def watch() -> None:
            while progress.get("state") != "ready":
                bar.n = progress.get("done", 0)
                bar.refresh()
                time.sleep(2)
            bar.n = progress.get("done", 0)
            bar.refresh()

        t = threading.Thread(target=watch, daemon=True)
        t.start()
        precompute(
            bundle,
            num_prompts=args.num_prompts,
            num_targets=args.num_targets,
            max_tokens=args.max_tokens,
            progress=progress,
        )
        t.join(timeout=5)
        bar.close()
        print("完成:Jacobian 已寫入快取")


if __name__ == "__main__":
    main()
