"""Idempotent V100 patch for RA-CFGCache (commit e17d6a7).

Upstream hard-codes torch.bfloat16 (they ran on H200). V100 (sm_70) has no bf16
tensor cores, so we make the compute dtype configurable through RACFG_DTYPE.
Default stays "bfloat16", i.e. without the env var the code behaves exactly like upstream.
CogVideoX-2B is officially an fp16 checkpoint, so RACFG_DTYPE=float16 is the native setting.

Additional adapters: GPU Wan VAE tiled decoding (document as a quality deviation),
and an explicit CLI-to-runtime CFGCache threshold binding. Upstream sample.py passes
rel_l1_thresh, whereas cache_init reads cfgcache_thresh; without this fix Slow/Fast
both silently use the default tau=0.4. No propagation schedule or calibration is changed.

Third adapter: an opt-in text-encoder release for the derivative-cache method
(TaylorSeer) on a 31.7 GiB card. Activated only when RACFG_RELEASE_TEXT_ENCODER=1,
so every other row keeps upstream residency. Prompt encoding itself is untouched and
the whole denoising loop still runs on the GPU.

usage: python apply_v100_patch.py /path/to/RA-CFGCache
"""
import re
import sys
from pathlib import Path

FILES = [
    "models/cogvideox/util.py",
    "models/cogvideox/sample.py",
    "models/cogvideox/taylor_utils/grouped_taylor.py",
    "models/wan/util.py",
    "models/wan/sample.py",
    "models/wan/taylor_utils/grouped_taylor.py",
]
EXPR = 'getattr(torch, os.environ.get("RACFG_DTYPE", "bfloat16"))'
MARK = "# [viewdit-v100-patch]"


def patch(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    if MARK in src:
        return "already"
    new, n = re.subn(r"torch\.bfloat16", EXPR, src)
    if n == 0:
        return "no-bf16"
    if not re.search(r"^import os\b", new, flags=re.M):
        new = "import os\n" + new
    new = f"{MARK} torch.bfloat16 -> RACFG_DTYPE ({n} sites)\n" + new
    path.write_text(new, encoding="utf-8")
    return f"patched {n}"


VAE_MARK = "# [viewdit-v100-patch] vae tiling+slicing"
VAE_ANCHOR = '        print(f"[load] pipe.to({device}) done")\n\n    return pipe'
VAE_SNIPPET = '''        print(f"[load] pipe.to({device}) done")

    # [viewdit-v100-patch] vae tiling+slicing
    # V100 (31.7GiB, shared with other jobs): the fp32 Wan VAE OOMs decoding 81 frames.
    # Tiling reduces decode peak memory; equivalence to untiled decode is not assumed.
    if hasattr(pipe, "vae") and getattr(pipe, "vae", None) is not None:
        try:
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
            print("[load] vae tiling+slicing enabled", flush=True)
        except Exception as _e:
            print(f"[warn] vae tiling failed: {_e}", flush=True)

    return pipe'''


def patch_wan_vae(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    if VAE_MARK in src:
        return "already"
    if VAE_ANCHOR not in src:
        return "no-anchor"
    path.write_text(src.replace(VAE_ANCHOR, VAE_SNIPPET, 1), encoding="utf-8")
    return "patched"


THRESH_MARK = '# [repro-fix] bind effective CFGCache threshold'


def patch_cfgcache_threshold(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    if THRESH_MARK in src:
        return "already"
    call = '            cache_dic, current = cache_init('
    call_pos = src.index(call)
    matches = list(re.finditer(r'^( +)"true_cfg_scale": opts\.guidance_scale,[ \t]*$', src[:call_pos], flags=re.M))
    if not matches:
        raise RuntimeError(f"CFGCache threshold binding anchor not found before cache_init in {path}")
    # Wan also records CFG in runtime helpers; use the last binding before cache_init.
    match = matches[-1]
    indent = match.group(1)
    replacement = (match.group(0).rstrip() + '\n' + indent + THRESH_MARK + '\n'
                   + indent + '"cfgcache_thresh": float(opts.rel_l1_thresh),')
    src = src[:match.start()] + replacement + src[match.end():]
    call = '            cache_dic, current = cache_init('
    pos = src.index(call)
    end = src.index('\n            )', pos) + len('\n            )')
    validation = '''
            if opts.taylor_method == "CFGCache":
                effective_tau = float(cache_dic["cfgcache"]["cfgcache_thresh"])
                assert effective_tau == float(opts.rel_l1_thresh), (effective_tau, opts.rel_l1_thresh)
                print(f"[CFGCache] effective_tau={effective_tau} requested_tau={opts.rel_l1_thresh}", flush=True)
'''
    src = src[:end] + validation + src[end:]
    path.write_text(src, encoding="utf-8")
    return "patched"


RELEASE_MARK = "# [repro-fix] release the text encoder while denoising (V100 memory adaptation)"

# Inserted relative to these unique anchors; both sample.py files have `device` in scope there.
RELEASE_ANCHORS = {
    "wan": (
        "        except Exception as _e:\n"
        '            print(f"[warn] vae tiling failed: {_e}", flush=True)',
        "after",
    ),
    "cogvideox": (
        '    print(f"Loaded CogVideoX pipeline from {model_path} on {device}")',
        "before",
    ),
}

RELEASE_SNIPPET = '''    # [repro-fix] release the text encoder while denoising (V100 memory adaptation)
    # The text encoder (umt5-xxl / T5-xxl, 8.9-10.6 GiB in fp16) stays resident and is
    # completely idle during denoising. TaylorSeer additionally keeps a per-layer
    # derivative cache, which does not fit next to it in 31.7 GiB. With
    # RACFG_RELEASE_TEXT_ENCODER=1 the encoder is parked on CPU right after each prompt is
    # encoded and restored just before the next prompt is encoded. Prompt encoding and the
    # whole denoising loop (everything that is timed, plus the cache implementation) are
    # unchanged and stay on the GPU. Off by default: all other rows keep upstream residency.
    if os.environ.get("RACFG_RELEASE_TEXT_ENCODER") == "1" and getattr(pipe, "text_encoder", None) is not None:
        _vdt_cls = type(pipe)
        if not getattr(_vdt_cls, "_viewdit_release_enc", False):
            _vdt_orig_encode_prompt = _vdt_cls.encode_prompt
            _vdt_seen = {"n": 0}

            def _viewdit_encode_prompt(_self, *a, **kw):
                _te = getattr(_self, "text_encoder", None)
                if _te is not None:
                    try:
                        _te.to(device)
                    except Exception as _e:
                        print(f"[warn] text encoder restore failed: {_e}", flush=True)
                _out = _vdt_orig_encode_prompt(_self, *a, **kw)
                if _te is not None:
                    try:
                        _te.to("cpu")
                        torch.cuda.empty_cache()
                        _vdt_seen["n"] = _vdt_seen["n"] + 1
                        _n = _vdt_seen["n"]
                        _free, _tot = torch.cuda.mem_get_info()
                        print(
                            f"[mem] prompt {_n} encoded; text encoder parked on CPU "
                            f"(free {_free / 2 ** 30:.2f} GiB / {_tot / 2 ** 30:.2f} GiB)",
                            flush=True,
                        )
                    except Exception as _e:
                        print(f"[warn] text encoder release failed: {_e}", flush=True)
                return _out

            _vdt_cls.encode_prompt = _viewdit_encode_prompt
            _vdt_cls._viewdit_release_enc = True
            print("[load] RACFG_RELEASE_TEXT_ENCODER=1: encoder parked on CPU during denoising", flush=True)
'''


def patch_release_text_encoder(path: Path) -> str:
    model = path.parent.name
    anchor, position = RELEASE_ANCHORS[model]
    src = path.read_text(encoding="utf-8")
    if RELEASE_MARK in src:
        return "already"
    if src.count(anchor) != 1:
        return f"no-unique-anchor({src.count(anchor)})"
    block = RELEASE_SNIPPET if position == "after" else RELEASE_SNIPPET + "\n\n" + anchor
    src = src.replace(anchor, (anchor + "\n\n" + block) if position == "after" else block, 1)
    path.write_text(src, encoding="utf-8")
    return "patched"


def main():
    root = Path(sys.argv[1])
    for rel in FILES:
        p = root / rel
        if p.exists():
            print(f"[v100-patch] {rel}: {patch(p)}")
        else:
            print(f"[v100-patch] {rel}: missing")
    for model in ("cogvideox", "wan"):
        p = root / "models" / model / "sample.py"
        if p.exists():
            print(f"[repro-fix] {model}/sample.py threshold: {patch_cfgcache_threshold(p)}")
    wan = root / "models/wan/sample.py"
    if wan.exists():
        print(f"[v100-patch] wan/sample.py vae: {patch_wan_vae(wan)}")
    for model in ("wan", "cogvideox"):
        p = root / "models" / model / "sample.py"
        if p.exists():
            print(f"[repro-fix] {model}/sample.py release-encoder: {patch_release_text_encoder(p)}")


if __name__ == "__main__":
    main()
