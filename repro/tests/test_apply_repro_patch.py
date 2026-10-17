"""纯标准库工程回归：不加载 torch，不生成视频，不访问 GPU 或远程。"""
import ast
import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPRO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("avp", REPRO / "patches/apply_repro_patch.py")
avp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(avp)

UPSTREAM_WAN = '''import torch


def load_pipeline(model_path, device):
    pipe = object()
    if True:
        print(f"[load] pipe.to({device}) done")

    return pipe
'''
UPSTREAM_COG = '''import torch


def load_pipeline(model_path, device):
    pipe = object()
    print(f"Loaded CogVideoX pipeline from {model_path} on {device}")

    return pipe
'''
UPSTREAM_SAMPLE = '''
def main():
    if True:
        if True:
            model_kwargs = {"rel_l1_thresh": opts.rel_l1_thresh}
            if opts.taylor_method == "CFGCache":
                model_kwargs.update({
                    "true_cfg_scale": opts.guidance_scale,
                })
            cache_dic, current = cache_init(
                method=opts.taylor_method,
                model_kwargs=model_kwargs,
            )
    return cache_dic
'''


def make_tree(root):
    for rel in avp.FILES:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith("sample.py"):
            body = UPSTREAM_WAN if "/wan/" in rel else UPSTREAM_COG
            p.write_text(body + UPSTREAM_SAMPLE, encoding="utf-8")
        else:
            p.write_text("import torch\nx = torch.bfloat16\n", encoding="utf-8")
    return root


def run_main(root, profile):
    out = io.StringIO()
    with mock.patch.object(sys, "argv", ["apply_repro_patch.py", str(root), profile]), contextlib.redirect_stdout(out):
        avp.main()
    return out.getvalue()


class TempCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="viewdit-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = mock.patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        for key in ("RACFG_PROFILE", "RACFG_DTYPE", "OUT_ROOT", "RACFG_RELEASE_TEXT_ENCODER", "ALLOW_GIT_PUSH"):
            os.environ.pop(key, None)


class PatchTests(TempCase):
    def setUp(self):
        super().setUp()
        make_tree(self.root)

    def snapshot(self):
        return {rel: (self.root / rel).read_bytes() for rel in avp.FILES if (self.root / rel).exists()}

    def test_fresh_profiles_and_byte_idempotence(self):
        for profile in ("h20", "v100"):
            with self.subTest(profile=profile):
                root = make_tree(self.root / profile)
                run_main(root, profile)
                before = {rel: (root / rel).read_bytes() for rel in avp.FILES}
                for rel, body in before.items():
                    ast.parse(body)
                    if rel.endswith("sample.py"):
                        self.assertIn(avp.THRESH_VALIDATE.encode(), body)
                        self.assertEqual(body.count(avp.RELEASE_MARK.encode()), int(profile == "v100"))
                        self.assertEqual(b"enable_tiling" in body, profile == "v100" and "/wan/" in rel)
                self.assertIn('os.environ.get("RACFG_DTYPE", "bfloat16")', (root / "models/wan/util.py").read_text())
                run_main(root, profile)
                self.assertEqual(before, {rel: (root / rel).read_bytes() for rel in avp.FILES})

    def test_upgrade_both_legacy_release_blocks(self):
        for old in (avp.RELEASE_SNIPPET_OLD, avp.RELEASE_SNIPPET_PINNED):
            for model in ("wan", "cogvideox"):
                with self.subTest(model=model, pinned=old == avp.RELEASE_SNIPPET_PINNED):
                    root = make_tree(self.root / str(len(list(self.root.iterdir()))))
                    p = root / "models" / model / "sample.py"
                    if model == "wan":
                        avp.patch_wan_vae(p)
                    anchor, position = avp.RELEASE_ANCHORS[model]
                    replacement = anchor + "\n\n" + old if position == "after" else old + "\n\n" + anchor
                    p.write_text(p.read_text().replace(anchor, replacement, 1))
                    self.assertIn("upgraded(instance-device)", run_main(root, "v100"))
                    self.assertEqual(p.read_text().count(avp.RELEASE_MARK), 1)
                    self.assertIn(avp.RELEASE_SNIPPET, p.read_text())
                    before = p.read_bytes()
                    run_main(root, "v100")
                    self.assertEqual(before, p.read_bytes())

    def test_threshold_executes_and_survives_python_optimization(self):
        p = self.root / "models/wan/sample.py"
        # 模拟 Wan 在 cache_init 前存在多个 CFG 绑定；只修改最后一个。
        p.write_text(p.read_text().replace('            model_kwargs =',
                     '            helper = {"true_cfg_scale": opts.guidance_scale}\n            model_kwargs ='))
        avp.patch_cfgcache_threshold(p)
        source = p.read_text().replace("import torch\n", "")
        for requested in (0.4, 0.6, 1.4):
            ns = {"opts": SimpleNamespace(taylor_method="CFGCache", guidance_scale=6, rel_l1_thresh=requested),
                  "cache_init": lambda **kw: ({"cfgcache": {"cfgcache_thresh": kw["model_kwargs"]["cfgcache_thresh"]}}, None)}
            exec(compile(source, str(p), "exec", optimize=2), ns)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ns["main"]()["cfgcache"]["cfgcache_thresh"], requested)
            ns["cache_init"] = lambda **kw: ({"cfgcache": {"cfgcache_thresh": 0.123}}, None)
            with self.assertRaisesRegex(RuntimeError, "threshold mismatch"):
                ns["main"]()

    def test_upgrade_assert_threshold_check(self):
        p = self.root / "models/cogvideox/sample.py"
        avp.patch_cfgcache_threshold(p)
        p.write_text(p.read_text().replace(avp.THRESH_VALIDATE, avp.THRESH_VALIDATE_OLD))
        self.assertIn("upgraded(runtime-check)", run_main(self.root, "h20"))
        self.assertIn(avp.THRESH_VALIDATE, p.read_text())

    def test_h20_rejects_v100_tree_without_writes(self):
        run_main(self.root, "v100")
        before = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, "h20 refuses"):
            run_main(self.root, "h20")
        self.assertEqual(before, self.snapshot())

    def test_missing_file_is_fatal_without_partial_patch(self):
        (self.root / avp.FILES[-1]).unlink()
        before = self.snapshot()
        with self.assertRaises(FileNotFoundError):
            run_main(self.root, "v100")
        self.assertEqual(before, self.snapshot())

    def test_missing_anchor_is_fatal_without_partial_patch(self):
        p = self.root / "models/wan/sample.py"
        p.write_text(p.read_text().replace(avp.VAE_ANCHOR, "    return pipe"))
        before = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, "no-anchor"):
            run_main(self.root, "v100")
        self.assertEqual(before, self.snapshot())

    def test_unknown_release_marker_is_not_silently_accepted(self):
        p = self.root / "models/wan/sample.py"
        p.write_text(p.read_text() + "\n" + avp.RELEASE_MARK + "\n")
        before = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, "Unrecognized"):
            run_main(self.root, "v100")
        self.assertEqual(before, self.snapshot())

    def test_syntax_failure_is_fatal_without_partial_patch(self):
        p = self.root / avp.FILES[-1]
        p.write_text("broken = (\n")
        before = self.snapshot()
        with self.assertRaises(SyntaxError):
            run_main(self.root, "v100")
        self.assertEqual(before, self.snapshot())

    def test_unknown_profile_is_fatal_without_writes(self):
        before = self.snapshot()
        with self.assertRaises(SystemExit):
            run_main(self.root, "invalid")
        self.assertEqual(before, self.snapshot())


class FakeDevice(str):
    @property
    def type(self):
        return self.split(":")[0]


class FakeMod:
    def __init__(self, device):
        self.device = device
        self.calls = []
        self.fail_to = None

    def to(self, device):
        self.calls.append(device)
        if device == self.fail_to:
            raise RuntimeError("simulated transfer failure")
        self.device = device
        return self


class FakeTorch:
    device = FakeDevice

    class cuda:
        contexts = []
        memory_queries = []

        @classmethod
        def device(cls, device):
            cls.contexts.append(device)
            return contextlib.nullcontext()

        @staticmethod
        def empty_cache():
            pass

        @classmethod
        def mem_get_info(cls, device):
            cls.memory_queries.append(device)
            return (1 << 30, 32 << 30)


def pipeline_class():
    class Pipe:
        def __init__(self, device="cuda:0"):
            self.text_encoder = FakeMod(device)
            self.transformer = FakeMod(device)
            self.vae = FakeMod(device)
            self.encoded = []
            self.embeddings = (object(), object())
            self.encode_error = False

        @property
        def device(self):
            return self.text_encoder.device

        @property
        def _execution_device(self):
            return self.device

        def encode_prompt(self, prompt, device=None):
            if device != self.text_encoder.device:
                raise RuntimeError("text indices and encoder are on different devices")
            if self.encode_error:
                raise ValueError("simulated encoding failure")
            self.encoded.append((prompt, device))
            return self.embeddings

        def __call__(self, prompt):
            # 模拟真实 __call__：在 encode_prompt 恢复编码器之前缓存 execution device。
            device = self._execution_device
            result = self.encode_prompt(prompt, device=device)
            if device != self.transformer.device or device != self.vae.device or FakeDevice(device).type != "cuda":
                raise RuntimeError("CPU denoising or mismatched pipeline device")
            return result
    return Pipe


class RuntimeTests(TempCase):
    def setUp(self):
        super().setUp()
        self.Pipe = pipeline_class()
        FakeTorch.cuda.contexts.clear()
        FakeTorch.cuda.memory_queries.clear()

    def install(self, pipe, device="cuda:0", enabled=True):
        os.environ["RACFG_RELEASE_TEXT_ENCODER"] = "1" if enabled else "0"
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(textwrap.dedent(avp.RELEASE_SNIPPET), "<release>", "exec"),
                 {"os": os, "torch": FakeTorch, "pipe": pipe, "device": device})

    def test_two_calls_cache_gpu_device_before_encoder_restore(self):
        pipe = self.Pipe()
        self.install(pipe)
        for prompt in ("p1", "p2", "p3"):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertIs(pipe(prompt), pipe.embeddings)
            self.assertEqual(pipe.text_encoder.device, "cpu")
            self.assertEqual(pipe._execution_device, "cuda:0")
        self.assertEqual(pipe.text_encoder.calls, ["cuda:0", "cpu"] * 3)
        self.assertEqual(pipe.encoded, [(p, "cuda:0") for p in ("p1", "p2", "p3")])
        self.assertEqual(pipe.transformer.calls + pipe.vae.calls, [])
        self.assertEqual(pipe._viewdit_prompt_count, 3)

    def test_same_class_instances_use_their_own_gpu(self):
        first, second = self.Pipe(), self.Pipe("cuda:1")
        self.install(first)
        self.install(second, "cuda:1")
        with contextlib.redirect_stdout(io.StringIO()):
            first("p0")
            second("p1")
        self.assertEqual((first._execution_device, second._execution_device), ("cuda:0", "cuda:1"))
        self.assertEqual(second.text_encoder.calls, ["cuda:1", "cpu"])
        self.assertEqual(FakeTorch.cuda.memory_queries, ["cuda:0", "cuda:1"])
        self.assertEqual(FakeTorch.cuda.contexts, ["cuda:0", "cuda:1"])
        self.assertEqual((first._viewdit_prompt_count, second._viewdit_prompt_count), (1, 1))

    def test_off_instance_of_already_patched_class_is_untouched(self):
        first, off = self.Pipe(), self.Pipe("cuda:1")
        self.install(first)
        self.install(off, "cuda:1", enabled=False)
        self.assertIs(off("off"), off.embeddings)
        self.assertEqual(off.device, "cuda:1")
        self.assertEqual(off.text_encoder.calls, [])
        self.assertFalse(hasattr(off, "_viewdit_release_device"))

    def test_off_clean_class_is_untouched(self):
        pipe = self.Pipe()
        original = self.Pipe.encode_prompt
        self.install(pipe, enabled=False)
        self.assertIs(self.Pipe.encode_prompt, original)
        self.assertNotIn("_viewdit_dev_pinned", self.Pipe.__dict__)
        self.assertEqual(pipe.text_encoder.calls, [])

    def test_reinstall_does_not_double_wrap(self):
        pipe = self.Pipe()
        self.install(pipe)
        original = self.Pipe.encode_prompt
        self.install(pipe)
        self.assertIs(self.Pipe.encode_prompt, original)
        with contextlib.redirect_stdout(io.StringIO()):
            pipe("one")
        self.assertEqual(pipe.text_encoder.calls, ["cuda:0", "cpu"])

    def test_restore_failure_prevents_encoding(self):
        pipe = self.Pipe()
        pipe.text_encoder.device = "cpu"
        pipe.text_encoder.fail_to = "cuda:0"
        self.install(pipe)
        with self.assertRaisesRegex(RuntimeError, "transfer failure"):
            pipe("fail")
        self.assertEqual(pipe.encoded, [])
        self.assertEqual(pipe.transformer.calls, [])

    def test_parking_failure_is_not_reported_as_success(self):
        pipe = self.Pipe()
        pipe.text_encoder.fail_to = "cpu"
        self.install(pipe)
        with self.assertRaisesRegex(RuntimeError, "transfer failure"):
            pipe("fail")
        self.assertEqual(len(pipe.encoded), 1)
        self.assertFalse(hasattr(pipe, "_viewdit_prompt_count"))

    def test_encode_failure_propagates_without_cpu_fallback(self):
        pipe = self.Pipe()
        pipe.encode_error = True
        self.install(pipe)
        with self.assertRaisesRegex(ValueError, "encoding failure"):
            pipe("fail")
        self.assertEqual(pipe.text_encoder.calls, ["cuda:0"])
        self.assertEqual(pipe.transformer.calls, [])

    def test_cpu_device_is_rejected(self):
        pipe = self.Pipe("cpu")
        with self.assertRaisesRegex(RuntimeError, "CUDA execution device"):
            self.install(pipe, "cpu")
        self.assertFalse(hasattr(pipe, "_viewdit_release_device"))

    def test_missing_encoder_is_inert(self):
        pipe = self.Pipe()
        pipe.text_encoder = None
        self.install(pipe)
        self.assertNotIn("_viewdit_dev_pinned", self.Pipe.__dict__)


class AggregateTests(TempCase):
    def setUp(self):
        super().setUp()
        os.environ["VIEWDIT"] = str(self.root)
        spec = importlib.util.spec_from_file_location("aggregate_test", REPRO / "aggregate.py")
        self.ag = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.ag)
        self.ag._py = lambda: sys.executable
        for rel in ("resources/prompts/prompt_video.txt", "evaluation/eval_video.py"):
            p = self.ag.RACFG / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("p0\np1\np2\n" if "prompt" in rel else "# 测试占位，不执行\n")

    def chunk(self, method, start, end, ids, latency=10, failed=False):
        tag = method
        parent = self.ag.OUT_ROOT / "wan" / method
        if method.startswith("racfg_"):
            parent /= "cfg_tau_v2"
            tag += "_cfg_tau_v2"
        if method == "taylorseer":
            parent /= "gpu_only_v2"
            tag += "_gpu_only_v2"
        chunk = f"p{start:03d}-{end:03d}"
        directory = parent / chunk / "videos"
        directory.mkdir(parents=True, exist_ok=True)
        for idx in ids:
            (directory / f"video_{idx:06d}.mp4").write_bytes(b"fixture-not-a-real-video")
        if latency is not None:
            log = self.ag.OUT_ROOT / "wan/_logs" / f"{tag}_{chunk}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(("Traceback\n" if failed else "") + f"Avg sampling sec/video: {latency}\n")
        return directory

    def paired(self):
        self.chunk("vanilla50", 0, 2, [0, 1], 20)
        self.chunk("teacache", 0, 3, [1, 2], 10)
        ref, _ = self.ag.merge("wan", "vanilla50", 3)
        cmp, _ = self.ag.merge("wan", "teacache", 3)
        return ref, cmp

    def fake_eval(self, cmd, **kwargs):
        self.assertEqual(cmd[cmd.index("--device") + 1], "cuda:0")
        self.assertEqual(kwargs["cwd"], self.ag.RACFG)
        Path(cmd[cmd.index("--out_json") + 1]).write_text(json.dumps({"summary": {
            "num_video_pairs": 1, "lpips_mean": 0.2, "ssim_mean": 0.8, "psnr_mean": 25.0}}))
        return SimpleNamespace(returncode=0)

    def test_h20_default_root_and_dtype_are_isolated(self):
        os.environ["RACFG_PROFILE"] = "h20"
        spec = importlib.util.spec_from_file_location("aggregate_h20_test", REPRO / "aggregate.py")
        ag = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ag)
        self.assertEqual(ag.OUT_ROOT, self.root / "results/repro_h20")
        self.assertEqual(ag.DEFAULT_DTYPE, "bfloat16")

    def test_overlap_selection_is_deterministic_and_prefers_complete_timed_chunk(self):
        broad = self.chunk("teacache", 0, 3, [0, 1], 5)
        small = self.chunk("teacache", 0, 2, [0, 1], 12)
        merged, lat = self.ag.merge("wan", "teacache", 2)
        self.assertEqual(lat, {0: 12, 1: 12})
        self.assertEqual((merged / "video_000000.mp4").resolve(), (small / "video_000000.mp4").resolve())
        manifest = (merged.parent / "merge_manifest.json").read_bytes()
        self.ag.merge("wan", "teacache", 2)
        self.assertEqual(manifest, (merged.parent / "merge_manifest.json").read_bytes())
        self.assertTrue((broad / "video_000000.mp4").is_file())

    def test_legacy_ra_and_taylor_outputs_are_excluded(self):
        for method in ("racfg_slow", "taylorseer"):
            legacy = self.ag.OUT_ROOT / "wan" / method / "p000-001/videos"
            legacy.mkdir(parents=True)
            (legacy / "video_000000.mp4").write_bytes(b"legacy")
            merged, lat = self.ag.merge("wan", method, 1)
            self.assertEqual(self.ag._videos(merged, 1), {})
            self.assertEqual(lat, {})
            self.chunk(method, 0, 1, [0], 8)
            merged, lat = self.ag.merge("wan", method, 1)
            self.assertEqual(lat, {0: 8})

    def test_failed_log_does_not_supply_latency(self):
        self.chunk("taylorseer", 0, 1, [0], 8, failed=True)
        merged, lat = self.ag.merge("wan", "taylorseer", 1)
        self.assertEqual(lat, {})
        self.assertEqual(len(self.ag._videos(merged, 1)), 1)

    def test_sync_refuses_regular_video_and_preserves_sources(self):
        source = self.root / "source.mp4"
        source.write_bytes(b"source")
        directory = self.root / "merged"
        directory.mkdir()
        ordinary = directory / "video_000000.mp4"
        ordinary.write_bytes(b"user-file")
        with self.assertRaisesRegex(RuntimeError, "非 symlink"):
            self.ag._sync_links(directory, {0: source})
        self.assertEqual(ordinary.read_bytes(), b"user-file")
        self.assertEqual(source.read_bytes(), b"source")

    def test_exact_intersection_and_matching_cache_avoid_reevaluation(self):
        ref, cmp = self.paired()
        with mock.patch.object(self.ag.subprocess, "run", side_effect=self.fake_eval) as run:
            result = self.ag.evaluate("wan", "teacache", ref, cmp, 3)
            self.assertEqual(result["pair_ids"], [1])
            self.assertEqual(result["status"], "evaluated")
            self.assertEqual(self.ag.evaluate("wan", "teacache", ref, cmp, 3, skip_eval=True)["status"], "cached")
            self.assertEqual(run.call_count, 1)
        manifest = json.loads((cmp.parent / "pair_manifest.json").read_text())
        self.assertEqual(manifest["pair_ids"], [1])
        self.assertEqual(manifest["evaluation"]["sampling_dtype"], "float16")

    def test_changed_prompt_invalidates_cache(self):
        ref, cmp = self.paired()
        with mock.patch.object(self.ag.subprocess, "run", side_effect=self.fake_eval) as run:
            self.ag.evaluate("wan", "teacache", ref, cmp, 3)
            (self.ag.RACFG / "resources/prompts/prompt_video.txt").write_text("changed\np1\np2\n")
            self.assertEqual(self.ag.evaluate("wan", "teacache", ref, cmp, 3, skip_eval=True)["status"], "evaluated")
            self.assertEqual(run.call_count, 2)

    def test_corrupted_metrics_invalidate_cache(self):
        ref, cmp = self.paired()
        with mock.patch.object(self.ag.subprocess, "run", side_effect=self.fake_eval) as run:
            self.ag.evaluate("wan", "teacache", ref, cmp, 3)
            (cmp.parent / "metrics.json").write_text("{}")
            self.assertEqual(self.ag.evaluate("wan", "teacache", ref, cmp, 3, skip_eval=True)["status"], "evaluated")
            self.assertEqual(run.call_count, 2)

    def test_failed_evaluator_cannot_reuse_previous_metrics(self):
        ref, cmp = self.paired()
        with mock.patch.object(self.ag.subprocess, "run", side_effect=self.fake_eval):
            self.ag.evaluate("wan", "teacache", ref, cmp, 3)
        with mock.patch.object(self.ag.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["fake"])):
            self.assertEqual(self.ag.evaluate("wan", "teacache", ref, cmp, 3)["status"], "not-evaluated")
            self.assertIsNone(json.loads((cmp.parent / "metrics_manifest.json").read_text())["fingerprint"])
            self.assertEqual(self.ag.evaluate("wan", "teacache", ref, cmp, 3, skip_eval=True)["status"], "not-evaluated")

    def test_no_pairs_is_pending_without_running_evaluator(self):
        ref, cmp = self.paired()
        with mock.patch.object(self.ag.subprocess, "run") as run:
            result = self.ag.evaluate("wan", "teacache", ref, cmp, 1)
            self.assertEqual(result["status"], "pending")
            self.assertIsNone(result["summary"])
            run.assert_not_called()

    def test_partial_quality_or_timing_returns_failure(self):
        for complete_quality in (False, True):
            with self.subTest(complete_quality=complete_quality):
                root = self.root / str(complete_quality)
                self.ag.OUT_ROOT = root
                self.chunk("vanilla50", 0, 2, [0, 1], 20)
                self.chunk("teacache", 0, 2, [0, 1] if complete_quality else [0], None)
                pair_ids = [0, 1] if complete_quality else [0]
                summary = {"num_video_pairs": len(pair_ids), "lpips_mean": 0.2,
                           "ssim_mean": 0.8, "psnr_mean": 25.0}
                def evaluate(model, method, *args, **kw):
                    ids = [0, 1] if method == "vanilla50" else pair_ids
                    return {"status": "evaluated", "summary": {**summary, "num_video_pairs": len(ids)},
                            "pair_ids": ids, "fingerprint": "fixture", "error": None}
                with mock.patch.object(self.ag, "evaluate", side_effect=evaluate), \
                     mock.patch.object(sys, "argv", ["aggregate.py", "--model", "wan", "--n", "2"]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(self.ag.main(), 1)
                table = json.loads((root / "wan/_merged/table_n2.json").read_text())
                row = next(r for r in table["rows"] if r["method"] == "teacache")
                self.assertEqual(row["status"], "partial-timing" if complete_quality else "partial-n")
                self.assertEqual(row["timing_pairs"], 0)
                self.assertIsNone(row["speedup"])
                self.assertEqual(table["statistical_gate"], "not-tested")

    def test_invalid_metrics_and_pair_counts_are_rejected(self):
        valid = {"num_video_pairs": 1, "lpips_mean": 0.2, "ssim_mean": 0.8, "psnr_mean": 25.0}
        for key, value in (("num_video_pairs", 2), ("num_video_pairs", True),
                           ("lpips_mean", float("nan")), ("ssim_mean", float("inf")),
                           ("psnr_mean", float("-inf")), ("lpips_mean", False)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.ag._summary({"summary": {**valid, key: value}}, 1)
        self.assertEqual(self.ag._summary({"summary": {**valid, "psnr_mean": "Infinity"}}, 1)["psnr_mean"], float("inf"))


class ShellTests(TempCase):
    def shell(self, file, args=(), env=None, timeout=10):
        return subprocess.run(["bash", str(REPRO / file), *args], cwd=self.root,
                              env=dict(os.environ, **(env or {})), text=True, capture_output=True, timeout=timeout)

    def test_all_shell_scripts_parse(self):
        for p in REPRO.glob("*.sh"):
            with self.subTest(script=p.name):
                result = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_argument_validation_precedes_conda_or_gpu_access(self):
        for args in ((), ("wan", "-1", "2"), ("wan", "01", "2"), ("wan", "2", "2"), ("wan", "x", "3")):
            with self.subTest(args=args):
                result = self.shell("run_repro.sh", args)
                self.assertEqual(result.returncode, 2)
                self.assertIn("[error]", result.stderr)
                self.assertNotIn("conda", result.stderr)

    def test_conda_failure_stops_before_sampling(self):
        conda = self.root / "conda.sh"
        conda.write_text("conda() { return 9; }\n")
        result = self.shell("run_repro.sh", ("wan", "0", "1"), {"CONDA_SH": str(conda)})
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.root / "results").exists())

    def test_cpu_offload_is_rejected_without_sampling(self):
        conda = self.root / "conda.sh"
        conda.write_text("conda() { return 0; }\n")
        result = self.shell("run_repro.sh", ("wan", "0", "1"),
                            {"CONDA_SH": str(conda), "CPU_OFFLOAD": "1"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("CPU_OFFLOAD is prohibited", result.stderr)
        self.assertFalse((self.root / "results").exists())

    def test_push_is_disabled_before_any_git_or_file_side_effects(self):
        result = self.shell("push_overnight.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("disabled", result.stdout)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_runner_scopes_inherited_release_flag_and_preserves_completed_chunks(self):
        viewdit = self.root / "viewdit"
        upstream = viewdit / "repro/upstream/RA-CFGCache"
        upstream.mkdir(parents=True)
        conda = self.root / "conda.sh"
        calls = self.root / "calls.jsonl"
        conda.write_text('conda() { return 0; }\npython() { "' + sys.executable + '" "' + str(self.root / "fake_python.py") + '" "$@"; }\n')
        # 只记录 CLI、环境并写测试占位文件；从不导入模型或运行采样。
        (self.root / "fake_python.py").write_text('''import json, os, sys
from pathlib import Path
a = sys.argv[1:]
if a[0] == "-":
    Path(a[2]).write_text("fixture-prompt\\n")
else:
    with open(os.environ["CALLS"], "a") as f:
        f.write(json.dumps({"release": os.environ["RACFG_RELEASE_TEXT_ENCODER"], "args": a}) + "\\n")
    out = Path(a[a.index("--output_dir") + 1]) / "videos"
    out.mkdir(parents=True, exist_ok=True)
    (out / "video_000000.mp4").write_bytes(b"test-placeholder")
''')
        # env 命令不能调用 shell 函数，因此为采样子进程提供同样的伪解释器。
        bindir = self.root / "bin"
        bindir.mkdir()
        executable = bindir / "python"
        executable.write_text('#!/bin/bash\nexec "' + sys.executable + '" "' + str(self.root / "fake_python.py") + '" "$@"\n')
        executable.chmod(0o755)
        env = {"VIEWDIT": str(viewdit), "CONDA_SH": str(conda), "CALLS": str(calls),
               "PATH": str(bindir) + os.pathsep + os.environ["PATH"], "RACFG_RELEASE_TEXT_ENCODER": "1"}
        for profile, expected in (("v100", ["0", "1"]), ("h20", ["0", "0"])):
            with self.subTest(profile=profile):
                if calls.exists():
                    calls.unlink()
                env["RACFG_PROFILE"] = profile
                result = self.shell("run_repro.sh", ("wan", "0", "1", "vanilla50", "taylorseer"), env)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                records = [json.loads(x) for x in calls.read_text().splitlines()]
                self.assertEqual([x["release"] for x in records], expected)
                for record in records:
                    self.assertNotIn("--cpu_offload", record["args"])
                    self.assertEqual(record["args"][record["args"].index("--start_index") + 1], "0")
                before = calls.read_bytes()
                result = self.shell("run_repro.sh", ("wan", "0", "1", "vanilla50", "taylorseer"), env)
                self.assertEqual(result.returncode, 0)
                self.assertEqual(result.stdout.count("[skip]"), 2)
                self.assertEqual(before, calls.read_bytes())

    def test_legacy_queue_failure_does_not_create_done_marker(self):
        (self.root / "logs").mkdir()
        (self.root / "logs/G0A_DONE").touch()
        (self.root / "repro").mkdir()
        (self.root / "repro/run_repro.sh").write_text("exit 7\n")
        result = self.shell("queue_full.sh", env={"VIEWDIT": str(self.root), "CHUNKS": "wan:0:1"})
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.root / "logs/REPRO_DONE_wan_0_1").exists())

    @unittest.skipUnless(shutil.which("flock"), "本机无 flock；Linux 部署时执行 supervisor 集成测试")
    def test_supervisor_disables_push_and_propagates_failure(self):
        (self.root / "repro/upstream/RA-CFGCache/resources/prompts").mkdir(parents=True)
        (self.root / "repro/upstream/RA-CFGCache/resources/prompts/prompt_video.txt").write_text("p0\n")
        (self.root / "repro/run_repro.sh").write_text("exit 7\n")
        (self.root / "repro/push_overnight.sh").write_text('touch "UNEXPECTED_PUSH"; exit 9\n')
        (self.root / "repro/aggregate.py").write_text("raise SystemExit(0)\n")
        result = self.shell("queue_overnight.sh", env={"VIEWDIT": str(self.root), "CHUNKS": "wan:0:1"})
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Git sync disabled", result.stdout)
        self.assertFalse((self.root / "UNEXPECTED_PUSH").exists())
        self.assertFalse((self.root / "logs/REPRO_DONE_wan_0_1").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
