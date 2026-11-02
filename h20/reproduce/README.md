# Reproducing the paper's numbers

Every number in the paper comes from runs inside the released **RA-CFGCache** harness, with our
schedule injected through one environment variable. Nothing else in the pipeline is changed.

## 1. Get the harness and the weights

```bash
git clone https://github.com/yiming-l21/RA-CFGCache.git && cd RA-CFGCache
# CogVideoX-2B / Wan2.1-T2V-1.3B weights via `hf download`
```

## 2. Inject the schedule (one edit, in two files)

`models/cogvideox/cache_functions/cal_type.py`, at the top of the `TeaCache` branch:

```python
import os as _os
_sched = _os.environ.get("MPC_SCHED", "")
if _sched:
    _ss = set(int(x) for x in _sched.split(",") if x)
    _st = int(current.get("step", 0))
    if _st in _ss:
        current["type"] = "full"
        cache_dic["full_counter"] = cache_dic.get("full_counter", 0) + 1
        return
    current["type"] = "TeaCacheSkip"
    return
```

`models/wan/cache_functions/cal_type.py`, at the top of the `CFGCache` branch: the same block with
`MPC_SCHEDW` and `current["type"] = "CFGCacheSkip"` in the else path.

## 3. Run the three arms of a setting

```bash
M=/path/to/CogVideoX-2b
bash RUN/demo_cogvideox.sh --mode original --gpus 0,1,2,3 --limit 32 --num_frames 17 \
     --model_path $M --output_dir out/orig
MPC_SCHED=$(python -c "import json;print(','.join(map(str,json.load(open('schedules.json'))['cog_17f_50steps_20eval_v1'])))") \
bash RUN/demo_cogvideox.sh --mode TeaCache --gpus 0,1,2,3 --limit 32 --num_frames 17 \
     --model_path $M --output_dir out/ours
bash RUN/demo_cogvideox.sh --mode TeaCache --gpus 0,1,2,3 --limit 32 --num_frames 17 \
     --model_path $M --output_dir out/teacache
bash RUN/demo_cogvideox.sh --mode CFGCache --gpus 0,1,2,3 --limit 32 --num_frames 17 \
     --model_path $M --output_dir out/cfgcache
```

Wan is the same with `RUN/demo_wan.sh`, `--mode CFGCache`, and `MPC_SCHEDW`.

## 4. Evaluate

```bash
python eval_lpips_vgg.py out/orig out/ours
```

**Use the VGG backbone.** On the same pair of videos AlexNet reports 0.041 where VGG reports 0.118.
Comparing a number produced with one backbone against a number produced with the other is the single
easiest way to be off by a factor of three.

## 5. Budget bookkeeping

A step is one transformer call. Check that the printed `full_counter` (or the count of `full` lines in
the log) equals the length of the schedule you injected; a schedule of length 20 must report 20 calls.
