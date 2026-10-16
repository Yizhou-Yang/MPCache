#!/usr/bin/env bash
# Publish bounded text/metric checkpoints to the existing overnight branch.
# Never publish weights, videos, tokens, or full environment snapshots.
set -euo pipefail
cd /root/viewdit
PY=/root/miniconda3/envs/racfg/bin/python
"$PY" - <<'PY'
from pathlib import Path
import datetime, json, shutil
r = Path('/root/viewdit')
lines = ['# MPCache reproduction checkpoint', '',
         'Generated: ' + datetime.datetime.now().isoformat(), '',
         'GPU-only generation; V100 fp16 differs from paper H200 bf16.',
         'Legacy RA Slow/Fast outputs used tau=0.4 and are excluded; cfg_tau_v2 reruns required.',
         'Video counts are unique global IDs, not overlapping chunk totals.',
         'Timing uses chunk averages on a shared GPU and remains provisional.', '']
for model in ('cogvideox', 'wan'):
    lines += ['## ' + model, '']
    for method in ('vanilla50', 'vanilla25', 'teacache', 'taylorseer', 'racfg_slow', 'racfg_fast'):
        root = r/'results/repro'/model/method
        if method.startswith('racfg_'):
            root = root/'cfg_tau_v2'
        if method == 'taylorseer':
            root = root/'gpu_only_v2'
        ids = {p.stem for p in root.glob('*/videos/video_*.mp4') if p.stat().st_size > 0}
        lines.append(f'- {method}: {len(ids)} valid unique videos')
    tables = sorted((r/'results/repro'/model/'_merged').glob('table_n*.json'),
                    key=lambda p: int(p.stem.split('n')[-1]))
    # Publish only corrected audit-aware tables, never legacy mixed-chunk tables.
    tables = [p for p in tables if all(tag in ' '.join(json.loads(p.read_text()).get('notes', []))
                                      for tag in ('cfg_tau_v2', 'gpu_only_v2'))]
    if tables:
        source = tables[-1]
        for ext in ('.json', '.md'):
            p = source.with_suffix(ext)
            if p.exists():
                dest = r/'repro'/f'{model}_{p.name}'
                shutil.copy2(p, dest)
        lines += ['', source.with_suffix('.md').read_text(), '']
(r/'STATUS_auto.md').write_text('\n'.join(lines) + '\n')
PY
branch=$(git branch --show-current)
[ "$branch" = "overnight" ] || { echo "Refusing to push unexpected branch: $branch" >&2; exit 1; }
git add -- repro/run_repro.sh repro/aggregate.py repro/queue_overnight.sh repro/push_overnight.sh \
  repro/patches/apply_v100_patch.py repro/STATUS.md STATUS_auto.md \
  experiments/video/g0a_costmodel.py experiments/video/g0a_analyze.py
for p in repro/cogvideox_table_n* repro/wan_table_n* repro/G0a_analysis_20261016.txt; do
  [ ! -f "$p" ] || git add -- "$p"
done
if ! git diff --cached --quiet; then
  git commit -m "repro: GPU-only audited benchmark checkpoint $(date +%F_%T)"
fi
# Always retry push, including when a previous checkpoint committed but push failed.
git push origin HEAD:overnight
git log -1 --format='%h %s'
