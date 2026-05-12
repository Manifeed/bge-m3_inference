from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any
from urllib import error, request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the BGE-M3 inference service.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--payload-file", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ready-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--colbert", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_payload(args)
    durations: list[float] = []

    _wait_until_ready(
        base_url=args.base_url,
        timeout_seconds=max(args.ready_timeout_seconds, 0.0),
    )

    for _ in range(max(args.repeats, 1)):
        started_at = time.perf_counter()
        _post_embeddings(base_url=args.base_url, api_key=args.api_key, payload=payload)
        duration = time.perf_counter() - started_at
        durations.append(duration)

    estimated_tokens = sum(max(len(text.split()), 1) for text in payload["input"])
    mean_duration = statistics.mean(durations)
    p50_duration = statistics.median(durations)
    p95_duration = _percentile(durations, percentile=95)
    print(
        json.dumps(
            {
                "repeats": len(durations),
                "mean_duration_seconds": round(mean_duration, 4),
                "p50_duration_seconds": round(p50_duration, 4),
                "p95_duration_seconds": round(p95_duration, 4),
                "min_duration_seconds": round(min(durations), 4),
                "max_duration_seconds": round(max(durations), 4),
                "estimated_tokens_per_second": round(estimated_tokens / mean_duration, 2),
                "request": payload,
            },
            indent=2,
        )
    )


def load_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.payload_file is not None:
        return json.loads(args.payload_file.read_text())
    return {
        "model": "bge-m3",
        "input": [
            "Short benchmark sentence for BGE-M3.",
            "A slightly longer benchmark sentence used to estimate throughput for hybrid retrieval.",
        ],
        "dense": args.dense,
        "sparse": args.sparse,
        "colbert": args.colbert,
    }


def _post_embeddings(*, base_url: str, api_key: str, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url=f"{base_url.rstrip('/')}/v1/embeddings",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with request.urlopen(http_request, timeout=300.0) as response:
            if response.status >= 400:
                raise RuntimeError(f"benchmark request failed with HTTP {response.status}")
    except error.HTTPError as exception:
        error_body = exception.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"benchmark request failed with HTTP {exception.code}: {error_body}"
        ) from exception


def _wait_until_ready(*, base_url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    ready_url = f"{base_url.rstrip('/')}/internal/ready"

    while True:
        try:
            with request.urlopen(ready_url, timeout=5.0) as response:
                if 200 <= response.status < 300:
                    return
        except error.HTTPError as exception:
            if exception.code != 503:
                error_body = exception.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"readiness probe failed with HTTP {exception.code}: {error_body}"
                ) from exception
        except error.URLError:
            pass

        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"service did not become ready within {timeout_seconds:.1f}s at {ready_url}"
            )
        time.sleep(1.0)


def _percentile(values: list[float], *, percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * weight)


if __name__ == "__main__":
    main()
