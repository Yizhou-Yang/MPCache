"""确定性合并 chunk，并在共同视频 ID 上调用 RA-CFGCache 的评估脚本。

    python repro/aggregate.py --model cogvideox [--n 32] [--skip_eval]

计时沿用所选 source 的 chunk 均值，不是精确的 per-prompt latency。
V100 fp16 与论文 H200 bf16 仅作对照，shared-GPU latency 为 provisional。
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

VIEWDIT = Path(os.environ.get("VIEWDIT", "/root/viewdit"))
RACFG = VIEWDIT / "repro/upstream/RA-CFGCache"
OUT_ROOT = Path(os.environ.get("OUT_ROOT", VIEWDIT / "results/repro"))

# RA-CFGCache Table 1 (NeurIPS'26, arXiv 2609.36433), H200 bf16, 100 VBench prompts.
# (LPIPS, SSIM, PSNR, speedup, latency_s)
PAPER = {
    "cogvideox": {
        "vanilla50": (0.0, 1.0, float("inf"), 1.00, 40.48),
        "vanilla25": (0.4751, 0.5445, 13.61, 1.86, 21.79),
        "teacache": (0.2040, 0.7538, 21.48, 2.42, 16.72),
        "taylorseer": None,  # not reported for CogVideoX in Table 1
        "racfg_slow": (0.0968, 0.8746, 26.93, 2.39, 16.96),
        "racfg_fast": (0.2052, 0.7614, 21.89, 3.13, 12.91),
    },
    "wan": {
        "vanilla50": (0.0, 1.0, float("inf"), 1.00, 93.32),
        "vanilla25": (0.4371, 0.5658, 15.87, 1.93, 48.41),
        "teacache": (0.1661, 0.7955, 23.08, 2.44, 38.24),
        "taylorseer": (0.3632, 0.6187, 16.96, 2.13, 43.79),
        "racfg_slow": (0.0993, 0.8778, 28.03, 2.56, 36.48),
        "racfg_fast": (0.1409, 0.8386, 26.12, 2.99, 31.25),
    },
}
LABEL = {
    "vanilla50": "Vanilla (T=50)", "vanilla25": "Vanilla (T=25)",
    "teacache": "TeaCache", "taylorseer": "TaylorSeer (N=3,O=1)",
    "racfg_slow": "RA-CFGCache-Slow", "racfg_fast": "RA-CFGCache-Fast",
}
TIME_RE = re.compile(r"(?:Avg sec / video sampling|Avg sampling sec/video):\s*([\d.]+)")
CHUNK_RE = re.compile(r"p(\d+)-(\d+)")
VIDEO_RE = re.compile(r"video_(\d+)\.mp4")
FAILED_RE = re.compile(r"Traceback|CUDA[^\n]*(?:out of memory|OOM)|OutOfMemoryError", re.I)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_json_safe(value), ensure_ascii=False,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _file_info(path):
    source = path.resolve()
    stat = source.stat()
    return {"path": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _videos(directory, n, start=0, end=None):
    sources = {}
    for video in sorted(directory.glob("video_*.mp4")):
        match = VIDEO_RE.fullmatch(video.name)
        if not match:
            continue
        idx = int(match[1])
        if start <= idx < (n if end is None else end) and idx < n \
                and video.is_file() and video.stat().st_size > 0:
            sources.setdefault(idx, video.resolve())
    return sources


def _sync_links(directory, sources):
    """只清理生成目录中的 symlink，绝不删除或覆盖普通文件及源视频。"""
    directory.mkdir(parents=True, exist_ok=True)
    wanted = {f"video_{idx:06d}.mp4": source for idx, source in sources.items()}
    for entry in directory.iterdir():
        if entry.is_symlink():
            if entry.name not in wanted or entry.resolve() != wanted[entry.name]:
                entry.unlink()
        elif entry.suffix == ".mp4":
            raise RuntimeError(f"生成目录中存在非 symlink 视频，拒绝覆盖或删除：{entry}")
    for name, source in wanted.items():
        target = directory / name
        if not target.is_symlink():
            target.symlink_to(source)


def merge(model: str, method: str, n: int):
    """每个 ID 选一个 source，返回视频目录和所选 source 的 chunk 均值。"""
    if n <= 0:
        raise ValueError("n 必须大于 0")
    mdir = OUT_ROOT / model / method
    # Prior RA runs passed rel_l1_thresh but cache_init read cfgcache_thresh (default 0.4).
    # Keep those sources for audit; only explicitly patched cfg_tau_v2 runs are admissible.
    corrected_ra = method.startswith("racfg_")
    if corrected_ra:
        mdir = mdir / "cfg_tau_v2"
    merged = OUT_ROOT / model / "_merged" / f"n{n}" / method / "videos"
    selected = {}
    for chunk in sorted(mdir.glob("p*-*")):
        match = CHUNK_RE.fullmatch(chunk.name)
        if not match or not chunk.is_dir():
            continue
        start, end = map(int, match.groups())
        if start >= end:
            continue
        # 完整性检查覆盖整个 chunk，而非仅覆盖此次请求的 n。
        videos = _videos(chunk / "videos", end, start, end)
        complete = len(videos) == end - start
        log_tag = f"{method}_cfg_tau_v2" if corrected_ra else method
        log = OUT_ROOT / model / "_logs" / f"{log_tag}_{chunk.name}.log"
        t, log_info = None, None
        if log.is_file():
            text = log.read_text(errors="ignore")
            log_info = _file_info(log)
            hits = TIME_RE.findall(text)
            if hits and not FAILED_RE.search(text):
                try:
                    value = float(hits[-1])
                    if math.isfinite(value) and value > 0:
                        t = value
                except ValueError:
                    pass
        for idx, source in videos.items():
            if not 0 <= idx < n:
                continue
            # 完整且有有效计时优先，其次跨度，最后用路径稳定消除歧义。
            rank = (not (complete and t is not None), -(end - start),
                    str(chunk.resolve()), str(source))
            if idx not in selected or rank < selected[idx][0]:
                selected[idx] = (rank, source, t, {
                    "index": idx, "source": _file_info(source), "chunk": str(chunk.resolve()),
                    "chunk_range": [start, end], "chunk_complete": complete,
                    "chunk_mean_latency_s": t, "log": log_info,
                })
    sources = {idx: selected[idx][1] for idx in sorted(selected)}
    _sync_links(merged, sources)
    _write_json(merged.parent / "merge_manifest.json", {
        "model": model, "method": method, "requested_n": n,
        "selection_policy": "complete_and_valid_time, descending_span, stable_paths",
        "timing_basis": "selected source chunk mean; not exact per-prompt latency",
        "sources": [selected[idx][3] for idx in sorted(selected)],
    })
    return merged, {idx: selected[idx][2] for idx in sorted(selected)
                    if selected[idx][2] is not None}


def _py():
    """Prefer the racfg conda env (has torch/lpips/cv2); fall back to the running interpreter.

    The queue runs aggregate.py with the base interpreter, whose site-packages lack cv2, so
    eval_video.py died with ModuleNotFoundError the first time round.
    """
    for cand in ("/root/miniconda3/envs/racfg/bin/python", "/opt/conda/envs/racfg/bin/python"):
        if os.path.exists(cand):
            return cand
    return sys.executable


def _summary(data, pair_n):
    summary = data["summary"]
    if type(summary.get("num_video_pairs")) is not int or summary["num_video_pairs"] != pair_n:
        raise ValueError(f"评估 pair 数与配对 manifest 不一致：期望 {pair_n}")
    summary = dict(summary)
    for key in ("lpips_mean", "ssim_mean", "psnr_mean"):
        value = summary[key]
        if key == "psnr_mean" and value == "Infinity":
            value = float("inf")
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or math.isnan(value) or value == float("-inf") \
                or (key != "psnr_mean" and not math.isfinite(value)):
            raise ValueError(f"评估指标无效：{key}={value!r}")
        summary[key] = value
    return summary


def evaluate(model, method, ref_dir, cmp_dir, n, skip_eval=False, device="cuda:0"):
    work = OUT_ROOT / model / "_merged" / f"n{n}" / method
    refs, cmps = _videos(ref_dir, n), _videos(cmp_dir, n)
    ids = sorted(refs.keys() & cmps.keys())
    paired_ref, paired_cmp = work / "paired/ref", work / "paired/cmp"
    _sync_links(paired_ref, {i: refs[i] for i in ids})
    _sync_links(paired_cmp, {i: cmps[i] for i in ids})
    manifest_path = work / "pair_manifest.json"
    manifest = {
        "model": model, "method": method, "requested_n": n, "pair_ids": ids,
        "pairs": [{"index": i, "ref": _file_info(refs[i]), "cmp": _file_info(cmps[i])}
                  for i in ids],
    }
    if not ids:
        manifest.update(status="pending", fingerprint=None, reason="没有共同视频 ID")
        _write_json(manifest_path, manifest)
        return {"status": "pending", "summary": None, "pair_ids": ids,
                "fingerprint": None, "error": "没有共同视频 ID"}

    env = dict(os.environ, HF_ENDPOINT=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        env.pop(k, None)
    out = work / "metrics.json"
    cache_path = work / "metrics_manifest.json"
    try:
        prompt_file = RACFG / "resources/prompts/prompt_video.txt"
        manifest["evaluation"] = {
            "script": _file_info(RACFG / "evaluation/eval_video.py"),
            "prompt_file": _file_info(prompt_file),
            "prompt_sha256": hashlib.sha256(prompt_file.read_bytes()).hexdigest(),
            "python": _file_info(Path(_py())), "cwd": str(RACFG.resolve()),
            "device": device, "dtype": "eval_video.py default (no dtype CLI override)",
            "sampling_dtype": env.get("RACFG_DTYPE", "float16"),
            "no_imagereward": True,
            # 不把可能含凭证的环境变量明文写进审计文件。
            "environment_sha256": hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest(),
        }
        # 合并决策和日志也纳入缓存输入，便于核对质量与计时各自的来源。
        manifest["merge_manifests"] = {
            "ref": json.loads((ref_dir.parent / "merge_manifest.json").read_text()),
            "cmp": json.loads((cmp_dir.parent / "merge_manifest.json").read_text()),
        }
        fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True,
                                                 allow_nan=False).encode()).hexdigest()
        manifest.update(fingerprint=fingerprint, status="not-evaluated")
        _write_json(manifest_path, manifest)
        if method == "vanilla50":
            manifest["status"] = "reference-self"
            _write_json(manifest_path, manifest)
            return {"status": "reference-self", "summary": {
                "lpips_mean": 0.0, "ssim_mean": 1.0, "psnr_mean": float("inf"),
                "num_video_pairs": len(ids),
            }, "pair_ids": ids, "fingerprint": fingerprint, "error": None}
        if skip_eval and out.is_file() and cache_path.is_file():
            try:
                cache = json.loads(cache_path.read_text())
                if cache.get("fingerprint") == fingerprint \
                        and cache.get("metrics_sha256") == hashlib.sha256(out.read_bytes()).hexdigest():
                    summary = _summary(json.loads(out.read_text()), len(ids))
                    manifest["status"] = "cached"
                    _write_json(manifest_path, manifest)
                    return {"status": "cached", "summary": summary, "pair_ids": ids,
                            "fingerprint": fingerprint, "error": None}
            except (OSError, ValueError, KeyError, TypeError):
                pass  # 损坏或不匹配的缓存必须重新评估。

        # 使用全新的输出路径，子进程未产出结果时不能误读此前的 metrics。
        with tempfile.TemporaryDirectory(prefix="eval-", dir=work) as temp:
            temp_out = Path(temp) / "metrics.json"
            cmd = [_py(), "evaluation/eval_video.py", "--ref_dir", str(paired_ref),
                   "--cmp_dir", str(paired_cmp), "--prompt_file", str(prompt_file),
                   "--out_json", str(temp_out), "--out_csv", str(temp_out.with_suffix(".csv")),
                   "--device", device, "--no_imagereward"]
            subprocess.run(cmd, cwd=RACFG, env=env, check=True)
            data = json.loads(temp_out.read_text())
            summary = _summary(data, len(ids))
            _write_json(out, data)
            if temp_out.with_suffix(".csv").is_file():
                temp_out.with_suffix(".csv").replace(out.with_suffix(".csv"))
        _write_json(cache_path, {"fingerprint": fingerprint,
                                "metrics_sha256": hashlib.sha256(out.read_bytes()).hexdigest()})
        manifest["status"] = "evaluated"
        _write_json(manifest_path, manifest)
        return {"status": "evaluated", "summary": summary, "pair_ids": ids,
                "fingerprint": fingerprint, "error": None}
    except Exception as exc:
        manifest.update(status="not-evaluated", error=f"{type(exc).__name__}: {exc}")
        _write_json(manifest_path, manifest)
        # 失败后不允许下一次 --skip_eval 复用旧的成功记录。
        _write_json(cache_path, {"fingerprint": None, "status": "not-evaluated",
                                "error": manifest["error"]})
        return {"status": "not-evaluated", "summary": None, "pair_ids": ids,
                "fingerprint": manifest.get("fingerprint"), "error": manifest["error"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["cogvideox", "wan"])
    ap.add_argument("--n", type=int, default=100, help="使用非空 prompt 的 [0, n) 区间")
    ap.add_argument("--skip_eval", action="store_true", help="仅复用输入指纹完全匹配的指标")
    ap.add_argument("--device", default="cuda:0", help="质量评估设备（计入缓存指纹）")
    args = ap.parse_args()
    prompt_file = RACFG / "resources/prompts/prompt_video.txt"
    try:
        prompt_n = sum(bool(line.strip()) for line in prompt_file.read_text(encoding="utf-8").splitlines())
    except OSError as exc:
        ap.error(f"无法读取 prompt 源：{exc}")
    if args.n <= 0 or args.n > prompt_n:
        ap.error(f"请求 n={args.n}，prompt 源仅有 {prompt_n} 个非空 prompt；"
                 "要求 0 < n <= 非空 prompt 数，不会补造 prompt")

    merged, merge_errors = {}, {}
    methods = [method for method in LABEL if (OUT_ROOT / args.model / method).is_dir()]
    for method in dict.fromkeys(["vanilla50", *methods]):
        try:
            merged[method] = merge(args.model, method, args.n)
        except Exception as exc:
            merge_errors[method] = f"{type(exc).__name__}: {exc}"
    ref_dir, ref_lat = merged.get("vanilla50", (None, {}))
    rows = []
    failed = not methods
    for method in methods:
        cmp_dir, lat = merged.get(method, (None, {}))
        quality_ids = sorted(_videos(ref_dir, args.n).keys() & _videos(cmp_dir, args.n).keys()) \
            if ref_dir is not None and cmp_dir is not None else []
        timing_ids = sorted(set(lat) & set(ref_lat) & set(quality_ids))
        speedup = sum(ref_lat[i] for i in timing_ids) / sum(lat[i] for i in timing_ids) \
            if timing_ids else None
        mean_lat = sum(lat[i] for i in timing_ids) / len(timing_ids) if timing_ids else None
        result = {"status": "pending", "summary": None, "pair_ids": quality_ids,
                  "fingerprint": None, "error": "没有共同视频 ID"}
        if method in merge_errors or "vanilla50" in merge_errors:
            result.update(status="not-evaluated",
                          error=merge_errors.get(method, merge_errors.get("vanilla50")))
        elif cmp_dir is not None and ref_dir is not None:
            try:
                result = evaluate(args.model, method, ref_dir, cmp_dir, args.n,
                                  skip_eval=args.skip_eval, device=args.device)
            except Exception as exc:
                result.update(status="not-evaluated", error=f"{type(exc).__name__}: {exc}")
        failed |= result["status"] in ("pending", "not-evaluated")
        rows.append({"method": method, **result,
                     "video_n": len(_videos(cmp_dir, args.n)) if cmp_dir is not None else 0,
                     "quality_pair_n": result["summary"]["num_video_pairs"] if result["summary"] else 0,
                     "available_quality_pair_n": len(quality_ids),
                     "timing_pairs": len(timing_ids), "timing_pair_ids": timing_ids,
                     "speedup": speedup, "latency_s": mean_lat,
                     "paper": PAPER[args.model].get(method)})

    notes = [
        "V100 fp16 vs H200 bf16：仅作论文对照，不是原数复现。",
        "RA 两行仅纳入 cfg_tau_v2 输出：旧 CLI 阈值未绑定运行时、实际均为 tau=0.4，旧结果无效。",
        "Wan 使用 GPU VAE tiling/slicing，未证明与论文未分块解码等价。", 
        "Latency/speedup 使用所选 source 的 chunk-average，共同 timing IDs 上加权；"
        "并非精确 per-prompt 计时，shared-GPU latency 为 provisional。",
        "未执行统计门检验，不能宣称统计门通过；reference-self 为恒等参照，未调用质量评估器。",
        "质量 pair n 为实际评估数（reference-self 为恒等配对数）；pending/not-evaluated 的指标缺失，不填 0。",
        f"source 审计及配对指纹：n{args.n}/<method>/merge_manifest.json、pair_manifest.json。",
    ]
    lines = [f"## {args.model}: V100 fp16 vs RA-CFGCache Table 1 (H200 bf16), requested n={args.n}", "",
             "| Method | videos | 质量 pair n | timing pairs | 状态 | LPIPS↓ ours / paper | SSIM↑ ours / paper | PSNR↑ ours / paper | Speedup ours / paper | Latency(s) ours / paper |",
             "|---|---|---|---|---|---|---|---|---|---|"]

    def fmt(value, digits=4):
        if value is None:
            return "–"
        return "∞" if value == float("inf") else f"{value:.{digits}f}"

    for row in rows:
        s, p = row["summary"] or {}, row["paper"]
        pp = lambda i, d=4: fmt(p[i], d) if p else "–"
        lines.append(f"| {LABEL[row['method']]} | {row['video_n']} | {row['quality_pair_n']} | "
                     f"{row['timing_pairs']} | {row['status']} | {fmt(s.get('lpips_mean'))} / {pp(0)} | "
                     f"{fmt(s.get('ssim_mean'))} / {pp(1)} | {fmt(s.get('psnr_mean'), 2)} / {pp(2, 2)} | "
                     f"{fmt(row['speedup'], 2)}× / {pp(3, 2)}× | {fmt(row['latency_s'], 1)} / {pp(4, 2)} |")
    lines += ["", *[f"- {note}" for note in notes]]
    for row in rows:
        if row["error"]:
            lines.append(f"- {LABEL[row['method']]}：{row['error']}")
    md = "\n".join(lines)
    out = OUT_ROOT / args.model / "_merged" / f"table_n{args.n}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md + "\n", encoding="utf-8")
    _write_json(out.with_suffix(".json"), {
        "model": args.model, "requested_n": args.n, "prompt_source_n": prompt_n,
        "notes": notes, "statistical_gate": "not-tested", "rows": rows,
        "merge_errors": merge_errors, "exit_code": int(failed),
    })
    print(md)
    print(f"\n[aggregate] wrote {out} and {out.with_suffix('.json')}")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
