from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class BenchResult:
    n: int
    warmup: int
    latencies_s: list[float]

    @property
    def p50(self) -> float:
        return percentile(self.latencies_s, 50)

    @property
    def p95(self) -> float:
        return percentile(self.latencies_s, 95)

    @property
    def mean(self) -> float:
        return statistics.mean(self.latencies_s) if self.latencies_s else float("nan")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]

    # Linear interpolation between closest ranks.
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


def http_json(
    url: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def http_get_json(url: str, timeout_s: float) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def bench(
    base_url: str,
    triage_path: str,
    n: int,
    warmup: int,
    timeout_s: float,
    patient_messages: list[str],
    lang: str,
) -> BenchResult:
    base_url = base_url.rstrip("/")

    triage_path = "/" + triage_path.lstrip("/")
    url = f"{base_url}{triage_path}"

    # Pre-flight check to help users catch wrong ports / wrong services.
    health_url = f"{base_url}/health"
    try:
        _ = http_get_json(health_url, timeout_s=timeout_s)
    except Exception:
        # We don't hard-fail here because some deployments might not expose /health.
        pass

    latencies: list[float] = []

    total = warmup + n
    for i in range(total):
        msg = patient_messages[i % len(patient_messages)]
        payload = {"patient_message": msg, "lang": lang, "context": {}}

        t0 = time.perf_counter()
        try:
            _ = http_json(url, payload=payload, timeout_s=timeout_s)
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            if e.code == 404:
                raise RuntimeError(
                    "HTTP 404 calling "
                    f"{url}. "
                    "This usually means you're pointing at the wrong base URL/port, "
                    "or the API path differs. "
                    f"Try opening {base_url}/docs (if available) and confirm the endpoint path, "
                    "then rerun with --base-url (and optionally --triage-path)."
                ) from e
            raise RuntimeError(f"HTTP {e.code} calling {url}: {err[:500]}") from e
        t1 = time.perf_counter()

        if i >= warmup:
            latencies.append(t1 - t0)

    return BenchResult(n=n, warmup=warmup, latencies_s=latencies)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Benchmark /triage latency (P50/P95) for a running FastAPI triage service. "
            "Designed to work without extra dependencies (uses stdlib urllib)."
        )
    )
    ap.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the API (default: http://127.0.0.1:8000)",
    )
    ap.add_argument(
        "--triage-path",
        default="/triage",
        help="Path for the triage endpoint (default: /triage)",
    )
    ap.add_argument("--n", type=int, default=10, help="Number of measured requests")
    ap.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup requests (excluded from stats)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds",
    )
    ap.add_argument(
        "--lang",
        choices=["fr", "en"],
        default="fr",
        help="Language sent to /triage",
    )
    ap.add_argument(
        "--print",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format",
    )

    args = ap.parse_args()

    messages = [
        "J'ai mal à la gorge depuis 2 jours, nez qui coule, pas de fièvre.",
        "Diarrhée depuis 24h, pas de sang, je bois correctement.",
        "Sore throat and runny nose for two days, no fever.",
        "I cut my hand and I'm bleeding heavily, I feel faint.",
    ]

    result = bench(
        base_url=args.base_url,
        triage_path=args.triage_path,
        n=args.n,
        warmup=args.warmup,
        timeout_s=args.timeout,
        patient_messages=messages,
        lang=args.lang,
    )

    if args.print == "json":
        print(
            json.dumps(
                {
                    "base_url": args.base_url,
                    "n": result.n,
                    "warmup": result.warmup,
                    "p50_s": result.p50,
                    "p95_s": result.p95,
                    "mean_s": result.mean,
                    "latencies_s": result.latencies_s,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.print == "markdown":
        print("| base_url | n | warmup | P50 (s) | P95 (s) | mean (s) |")
        print("|---|---:|---:|---:|---:|---:|")
        row = (
            f"| {args.base_url} | {result.n} | {result.warmup} | "
            f"{result.p50:.3f} | {result.p95:.3f} | {result.mean:.3f} |"
        )
        print(row)
        return 0

    print(f"base_url={args.base_url}")
    print(f"n={result.n} warmup={result.warmup}")
    print(f"P50={result.p50:.3f}s P95={result.p95:.3f}s mean={result.mean:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
