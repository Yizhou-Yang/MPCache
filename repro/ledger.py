#!/usr/bin/env python3
"""Per-ID completion ledger for the RA-CFGCache Table-1 reproduction tree.

Why this exists: counting ``*.mp4`` with ``ls | wc -l`` cannot distinguish a complete
100-prompt row from a row with holes, cannot see legacy directories that must never be
aggregated (pre-fix RA runs at tau=0.4, CPU-offload TaylorSeer runs), and silently treats
an overlap between smoke chunks as a new sample.  The runner and aggregate.py both key on
the *global* video id, so this ledger does the same thing and reports the exact missing ids.

Input is one remote listing, so analysis needs a single SSH round trip and can be re-run offline:

    ssh <host> "find /root/viewdit/results/repro /root/viewdit/results/repro_h20 \
        -name 'video_*.mp4' -printf '%p\\t%s\\n'" > manifest.tsv
    python3 repro/ledger.py manifest.tsv --n 100

Admissibility (mirrors aggregate.py, do not relax):
  * racfg_slow / racfg_fast  -> only  <method>/cfg_tau_v2/<chunk>/videos/   (tau fix)
  * taylorseer               -> only  <method>/gpu_only_v2/<chunk>/videos/  (no CPU offload)
  * everything else          ->       <method>/<chunk>/videos/
Anything else found on disk is reported under "excluded" and never counted.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

METHODS = ("vanilla50", "vanilla25", "teacache", "taylorseer", "racfg_slow", "racfg_fast")
PROFILE_OF_ROOT = {"repro": "v100", "repro_h20": "h20"}
METHOD_RE = re.compile(
    r"/(?P<profile>repro(?:_h20)?)/(?P<model>cogvideox|wan)/(?P<method>[^/]+)/"
    r"(?:(?P<sub>cfg_tau_v2|gpu_only_v2)/)?(?P<chunk>p(?P<start>\d{3})-(?P<end>\d{3}))/"
    r"videos/video_(?P<vid>\d{6})\.mp4$"
)


def admissible(method: str, sub: str | None) -> bool:
    if method.startswith("racfg_"):
        return sub == "cfg_tau_v2"
    if method == "taylorseer":
        return sub == "gpu_only_v2"
    return sub is None


AUX = ("/_merged/", "/_prompts/", "/_logs/")


def compress(ids) -> str:
    """[0,1,2,5,7,8] -> '0-2,5,7-8'."""
    ids = sorted(ids)
    if not ids:
        return "-"
    out, start, prev = [], ids[0], ids[0]
    for v in ids[1:] + [None]:
        if v is not None and v == prev + 1:
            prev = v
            continue
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
        if v is not None:
            start = prev = v
    return ",".join(out)


def read_manifest(path: Path):
    """Yield (match or None, size_bytes, is_aux) for every line of the manifest."""
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        name, _, size = line.rpartition("\t")
        if not name:                      # line without a size column
            name, size = line, "0"
        try:
            size_b = int(size)
        except ValueError:
            size_b = 0
        is_aux = any(tag in name for tag in AUX)
        yield METHOD_RE.search(name), size_b, is_aux


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path, help="TSV from find -printf '%%p\\t%%s\\n'")
    ap.add_argument("--n", type=int, default=100, help="expected prompt IDs are [0,n)")
    ap.add_argument("--json", type=Path, default=None, help="also write the full ledger as JSON")
    ap.add_argument("--md", type=Path, default=None, help="also write the markdown table")
    a = ap.parse_args()

    # (profile, model, method) -> admissible id -> list of (chunk, size)
    rows: dict[tuple[str, str, str], dict[int, list[tuple[str, int]]]] = defaultdict(lambda: defaultdict(list))
    excluded: dict[tuple[str, str], int] = defaultdict(int)
    unparsed = aux = 0
    for m, size, is_aux in read_manifest(a.manifest):
        if is_aux:
            aux += 1
            continue
        if m is None:
            unparsed += 1
            continue
        profile, model, method = m["profile"], m["model"], m["method"]
        if not admissible(method, m["sub"]):
            excluded[(profile, f"{method}/{m['sub'] or ''}".rstrip("/"))] += 1
            continue
        rows[(profile, model, method)][int(m["vid"])].append((m["chunk"], size))

    expected = set(range(a.n))
    ledger, out = {}, []
    for (profile, model, method), ids in sorted(rows.items()):
        have = set(ids)
        chunks: dict[str, int] = defaultdict(int)
        dup = {v: c for v, c in ids.items() if len(c) > 1}
        empty = sorted(v for v, c in ids.items() if all(s <= 0 for _, s in c))
        for v, c in ids.items():
            for ch, _ in c:
                chunks[ch] += 1
        missing = sorted(expected - have)
        extra = sorted(have - expected)
        if not have:
            status = "pending"
        elif not missing and not empty:
            # ids beyond n belong to a larger run and are not a defect of the requested prefix;
            # a repeated id from a smoke chunk is an overlap, not a hole.
            status = "complete"
        else:
            status = "partial-n"
        ledger[(profile, model, method)] = {
            "profile": profile, "model": model, "method": method,
            "status": status, "n_expected": a.n, "n_found": len(have),
            "n_covered": len(have & expected),
            "missing": missing, "missing_ranges": compress(missing),
            "extra": extra, "overlap": {str(k): [c for c, _ in v] for k, v in dup.items()},
            "zero_byte": empty, "chunks": dict(sorted(chunks.items())),
        }

    # one row per (profile, model, method) that exists anywhere, including fully pending methods
    present = {(p, mo) for (p, mo, _m) in rows}
    for p, mo in sorted(present):
        for method in METHODS:
            key = (p, mo, method)
            if key not in ledger:
                ledger[key] = {
                    "profile": p, "model": mo, "method": method, "status": "pending",
                    "n_expected": a.n, "n_found": 0, "n_covered": 0, "missing": sorted(expected),
                    "missing_ranges": compress(expected), "extra": [], "overlap": {},
                    "zero_byte": [], "chunks": {},
                }

    header = "| profile | model | method | status | IDs | missing IDs | chunks (videos) |"
    sep = "|---|---|---|---|---:|---|---|"
    lines = [header, sep]
    for (p, mo, method), r in sorted(ledger.items()):
        chunks = ", ".join(f"{c}({k})" for c, k in r["chunks"].items())
        lines.append(f"| {p} | {mo} | {method} | {r['status']} | {r['n_covered']}/{a.n} | "
                     f"{r['missing_ranges']} | {chunks} |")
    table = "\n".join(lines)

    print(table)

    warnings = []
    for (p, mo, method), r in sorted(ledger.items()):
        if r["overlap"]:
            ids = ",".join(sorted(r["overlap"], key=int))
            warnings.append(f"{p} {mo}/{method}: id seen in >1 chunk ({ids}); aggregate dedupes by id")
        if r["zero_byte"]:
            warnings.append(f"{p} {mo}/{method}: zero-byte videos at {compress(r['zero_byte'])}")
    if warnings:
        print("\nwarnings:")
        for w in warnings:
            print("  -", w)
    if excluded:
        print("\nexcluded (legacy / non-admissible, never aggregated):")
        for (p, name), k in sorted(excluded.items()):
            print(f"  {p} {name}: {k} files")
    if aux:
        print(f"\naux entries skipped (_merged/ symlinks, prompt and log files): {aux}")
    if unparsed:
        print(f"unparsed manifest lines: {unparsed}")

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({"n": a.n, "rows": list(ledger.values()),
                                      "excluded": {f"{p}/{n}": k for (p, n), k in excluded.items()}},
                                     indent=2))
    if a.md:
        a.md.parent.mkdir(parents=True, exist_ok=True)
        a.md.write_text(table + "\n")

    incomplete = [k for k, r in ledger.items() if r["status"] != "complete"]
    print(f"\n{len(ledger) - len(incomplete)}/{len(ledger)} rows complete for [0,{a.n})")
    return 0 if not incomplete else 1


if __name__ == "__main__":
    sys.exit(main())
