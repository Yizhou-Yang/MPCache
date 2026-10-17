"""Profile-aware, idempotent patch for RA-CFGCache (commit e17d6a7).

Two profiles:

  v100 (default) - everything the 31.7 GiB V100 needs:
      * configurable dtype (V100 sm_70 has no bf16 tensor cores)
      * GPU VAE tiling/slicing for Wan (fp32 VAE cannot decode 81 frames in 31.7 GiB)
      * opt-in text-encoder release for TaylorSeer (per-layer derivative cache does not
        fit next to the resident 8.9-10.6 GiB text encoder)
      * explicit CLI-to-runtime CFGCache threshold binding

  h20 - a 96 GiB Hopper card needs no memory adapter, so ONLY the two patches that do not
      change numerical behaviour are applied:
      * configurable dtype, whose default is still "bfloat16" (i.e. exactly upstream)
      * explicit CLI-to-runtime CFGCache threshold binding

Upstream bug fixed in BOTH profiles: sample.py forwards `rel_l1_thresh` while
cache_init.py reads `cfgcache_thresh`, so the released code silently ran both nominal
Slow and Fast rows at the default tau=0.4 (their CogVideoX outputs were byte-identical).
We forward the value explicitly and assert the effective runtime threshold. No
propagation schedule, proxy or calibration table is modified by this patch.

usage: python apply_repro_patch.py /path/to/RA-CFGCache [v100|h20]
"""
import re
import sys
import tempfile
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


# ---------------------------------------------------------------- Wan VAE (v100 only)
VAE_MARK = "# [viewdit-v100-patch] V100 31.7GiB shared with other jobs"
VAE_ANCHOR = '        print(f"[load] pipe.to({device}) done")\n\n    return pipe'
VAE_SNIPPET = '''        print(f"[load] pipe.to({device}) done")

    # [viewdit-v100-patch] V100 31.7GiB shared with other jobs: fp32 Wan VAE OOMs decoding
    # 81 frames. Tiling + slicing make decode peak memory ~constant (negligible quality change).
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


# ------------------------------------------------- CFGCache threshold (both profiles)
THRESH_MARK = '# [repro-fix] bind effective CFGCache threshold'
THRESH_VALIDATE_OLD = '''
            if opts.taylor_method == "CFGCache":
                effective_tau = float(cache_dic["cfgcache"]["cfgcache_thresh"])
                assert effective_tau == float(opts.rel_l1_thresh), (effective_tau, opts.rel_l1_thresh)
                print(f"[CFGCache] effective_tau={effective_tau} requested_tau={opts.rel_l1_thresh}", flush=True)
'''
THRESH_VALIDATE = THRESH_VALIDATE_OLD.replace(
    '                assert effective_tau == float(opts.rel_l1_thresh), (effective_tau, opts.rel_l1_thresh)',
    '                if effective_tau != float(opts.rel_l1_thresh):\n'
    '                    raise RuntimeError(f"CFGCache threshold mismatch: {effective_tau} != {opts.rel_l1_thresh}")',
)


def patch_cfgcache_threshold(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    if THRESH_MARK in src:
        if src.count(THRESH_MARK) != 1 or '"cfgcache_thresh": float(opts.rel_l1_thresh),' not in src:
            raise RuntimeError(f"Invalid CFGCache threshold binding in {path}")
        if THRESH_VALIDATE in src:
            return "already"
        if THRESH_VALIDATE_OLD in src:
            path.write_text(src.replace(THRESH_VALIDATE_OLD, THRESH_VALIDATE, 1), encoding="utf-8")
            return "upgraded(runtime-check)"
        raise RuntimeError(f"Unrecognized CFGCache threshold validation in {path}")
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
    src = src[:end] + THRESH_VALIDATE + src[end:]
    path.write_text(src, encoding="utf-8")
    return "patched"


# --------------------------------------------- text encoder release (v100 taylorseer)
RELEASE_MARK = "# [repro-fix] release the text encoder while denoising (V100 memory adaptation)"

RELEASE_COMMON_OLD = '''            _vdt_orig_encode_prompt = _vdt_cls.encode_prompt
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

RELEASE_HEAD_PINNED = '''    # [repro-fix] release the text encoder while denoising (V100 memory adaptation)
    # The text encoder (umt5-xxl / T5-xxl, 8.9-10.6 GiB in fp16) stays resident and is
    # completely idle during denoising. TaylorSeer additionally keeps a per-layer
    # derivative cache, which does not fit next to it in 31.7 GiB. With
    # RACFG_RELEASE_TEXT_ENCODER=1 the encoder is parked on CPU right after each prompt is
    # encoded and restored just before the next prompt is encoded. Prompt encoding and the
    # whole denoising loop (everything that is timed, plus the cache implementation) are
    # unchanged and stay on the GPU. Off by default: all other rows keep upstream residency.
    if os.environ.get("RACFG_RELEASE_TEXT_ENCODER") == "1" and getattr(pipe, "text_encoder", None) is not None:
        _vdt_cls = type(pipe)
        if not getattr(_vdt_cls, "_viewdit_dev_pinned", False):
            # DiffusionPipeline.device and _execution_device return the device of an arbitrary
            # component, and __call__ caches that choice at entry ("device = self._execution_device",
            # pipeline_cogvideox.py). While the encoder is parked that resolution can come back as
            # CPU even though vae/transformer really live on the GPU; __call__ then pushes the text
            # inputs onto CPU and the next prompt dies with "index is on cpu, different from other
            # tensors on cuda:0". Pin the reported device to the real GPU device -- the device every
            # non-parked component is already on -- so nothing numerical changes.
            _vdt_dev = torch.device(device)
            _vdt_cls.device = property(lambda self, _d=_vdt_dev: _d)
            _vdt_cls._execution_device = property(lambda self, _d=_vdt_dev: _d)
            _vdt_cls._viewdit_dev_pinned = True
            print(f"[load] pinned pipeline device to {device} while the text encoder is parked", flush=True)
        if not getattr(_vdt_cls, "_viewdit_release_enc", False):
'''

RELEASE_HEAD_OLD = '''    # [repro-fix] release the text encoder while denoising (V100 memory adaptation)
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
'''

RELEASE_SNIPPET_OLD = RELEASE_HEAD_OLD + RELEASE_COMMON_OLD
RELEASE_SNIPPET_PINNED = RELEASE_HEAD_PINNED + RELEASE_COMMON_OLD

RELEASE_SNIPPET = '''    # [repro-fix] release the text encoder while denoising (V100 memory adaptation)
    # 文本编码与去噪仍在 GPU；仅在编码完成后把空闲编码器停放到 CPU。
    # 绑定实例而非类的固定设备，避免同类的其他 pipeline 被误改。
    if os.environ.get("RACFG_RELEASE_TEXT_ENCODER") == "1" and getattr(pipe, "text_encoder", None) is not None:
        _vdt_dev = torch.device(device)
        if _vdt_dev.type != "cuda":
            raise RuntimeError("RACFG_RELEASE_TEXT_ENCODER requires a CUDA execution device")
        pipe._viewdit_release_device = _vdt_dev
        _vdt_cls = type(pipe)
        if not getattr(_vdt_cls, "_viewdit_dev_pinned", False):
            _vdt_orig_device = _vdt_cls.device
            _vdt_orig_execution_device = _vdt_cls._execution_device

            def _viewdit_device(_self):
                _dev = getattr(_self, "_viewdit_release_device", None)
                return _dev if _dev is not None else _vdt_orig_device.__get__(_self, type(_self))

            def _viewdit_execution_device(_self):
                _dev = getattr(_self, "_viewdit_release_device", None)
                return _dev if _dev is not None else _vdt_orig_execution_device.__get__(_self, type(_self))

            _vdt_cls.device = property(_viewdit_device)
            _vdt_cls._execution_device = property(_viewdit_execution_device)
            _vdt_cls._viewdit_dev_pinned = True
        if not getattr(_vdt_cls, "_viewdit_release_enc", False):
            _vdt_orig_encode_prompt = _vdt_cls.encode_prompt

            def _viewdit_encode_prompt(_self, *a, **kw):
                _dev = getattr(_self, "_viewdit_release_device", None)
                if _dev is None:
                    return _vdt_orig_encode_prompt(_self, *a, **kw)
                _te = getattr(_self, "text_encoder", None)
                if _te is not None:
                    # 恢复或停放失败必须上抛；绝不继续 CPU 编码或去噪。
                    _te.to(_dev)
                _out = _vdt_orig_encode_prompt(_self, *a, **kw)
                if _te is not None:
                    _te.to("cpu")
                    with torch.cuda.device(_dev):
                        torch.cuda.empty_cache()
                    _self._viewdit_prompt_count = getattr(_self, "_viewdit_prompt_count", 0) + 1
                    _free, _tot = torch.cuda.mem_get_info(_dev)
                    print(
                        f"[mem] prompt {_self._viewdit_prompt_count} encoded; text encoder parked on CPU "
                        f"(free {_free / 2 ** 30:.2f} GiB / {_tot / 2 ** 30:.2f} GiB)",
                        flush=True,
                    )
                return _out

            _vdt_cls.encode_prompt = _viewdit_encode_prompt
            _vdt_cls._viewdit_release_enc = True
        print(f"[load] pinned this pipeline to {_vdt_dev}; encoder parked on CPU during denoising", flush=True)
'''
assert RELEASE_MARK in RELEASE_SNIPPET_OLD and RELEASE_MARK in RELEASE_SNIPPET

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


def patch_release_text_encoder(path: Path) -> str:
    model = path.parent.name
    anchor, position = RELEASE_ANCHORS[model]
    src = path.read_text(encoding="utf-8")
    if RELEASE_SNIPPET in src and src.count(RELEASE_MARK) == 1:
        return "already"
    for old in (RELEASE_SNIPPET_OLD, RELEASE_SNIPPET_PINNED):
        if old in src and src.count(RELEASE_MARK) == 1:
            path.write_text(src.replace(old, RELEASE_SNIPPET, 1), encoding="utf-8")
            return "upgraded(instance-device)"
    if RELEASE_MARK in src:
        raise RuntimeError(f"Unrecognized or duplicate text-encoder release block in {path}")
    if src.count(anchor) != 1:
        return f"no-unique-anchor({src.count(anchor)})"
    block = RELEASE_SNIPPET if position == "after" else RELEASE_SNIPPET + "\n\n" + anchor
    src = src.replace(anchor, (anchor + "\n\n" + block) if position == "after" else block, 1)
    path.write_text(src, encoding="utf-8")
    return "patched"


def _apply(root, profile):
    def report(label, result):
        if result.startswith(("no-anchor", "no-unique-anchor")):
            raise RuntimeError(f"{label}: {result}")
        print(f"{label}: {result}")

    for rel in FILES:
        report(f"[dtype-patch] {rel}", patch(root / rel))
    for model in ("cogvideox", "wan"):
        report(f"[repro-fix] {model}/sample.py threshold",
               patch_cfgcache_threshold(root / "models" / model / "sample.py"))
    if profile == "v100":
        report("[v100-patch] wan/sample.py vae", patch_wan_vae(root / "models/wan/sample.py"))
        for model in ("wan", "cogvideox"):
            report(f"[repro-fix] {model}/sample.py release-encoder",
                   patch_release_text_encoder(root / "models" / model / "sample.py"))
    else:
        print("[profile] h20: VAE tiling and text-encoder release are intentionally NOT applied")


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: apply_repro_patch.py ROOT [v100|h20]")
    root = Path(sys.argv[1])
    profile = (sys.argv[2] if len(sys.argv) > 2 else "v100").lower()
    if profile not in ("v100", "h20"):
        raise SystemExit(f"unknown profile {profile}; expected v100 or h20")
    original = {rel: (root / rel).read_bytes() for rel in FILES}
    if profile == "h20":
        for rel, body in original.items():
            if RELEASE_MARK.encode() in body or VAE_MARK.encode() in body:
                raise RuntimeError(f"h20 refuses a v100-adapted tree: {rel}; use a clean pinned upstream tree")
    print(f"[profile] {profile}")
    # 先在临时副本上完成所有锚点与语法检查，失败时不留下半套补丁。
    with tempfile.TemporaryDirectory(prefix="viewdit-patch-") as temp:
        staged = Path(temp)
        for rel, body in original.items():
            p = staged / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(body)
        _apply(staged, profile)
        updated = {rel: (staged / rel).read_bytes() for rel in FILES}
        for rel, body in updated.items():
            compile(body, str(root / rel), "exec")
        if any((root / rel).read_bytes() != body for rel, body in original.items()):
            raise RuntimeError("Upstream changed during patch preflight; nothing written")
        for rel, body in updated.items():
            if body != original[rel]:
                (root / rel).write_bytes(body)
    print("[patch] all anchors and syntax validated; applied successfully")


if __name__ == "__main__":
    main()
