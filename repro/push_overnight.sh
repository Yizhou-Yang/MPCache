#!/usr/bin/env bash
# Publish bounded text/metric checkpoints for one profile.
# Never publishes weights, videos, tokens, or full environment snapshots.
#
#   PUSH_BRANCH=overnight bash repro/push_overnight.sh   # v100 (default)
#   RACFG_PROFILE=h20 PUSH_BRANCH=h20 bash repro/push_overnight.sh
set -euo pipefail
if [ "${ALLOW_GIT_PUSH:-0}" != "1" ]; then
  echo "[push] disabled; set ALLOW_GIT_PUSH=1 only with explicit authorization"
  exit 0
fi
cd /root/viewdit
PY=${PY:-/root/miniconda3/envs/racfg/bin/python}
PROFILE=${RACFG_PROFILE:-v100}
PUSH_BRANCH=${PUSH_BRANCH:-overnight}
export RACFG_PROFILE="$PROFILE" RES_SUBDIR="$([ "$PROFILE" = h20 ] && echo repro_h20 || echo repro)"
"$PY" - <<'PY'
from pathlib import Path
import datetime, json, os, shutil
r = Path('/root/viewdit')
res = r / 'results' / os.environ['RES_SUBDIR']
profile = os.environ['RACFG_PROFILE']
lines = ['# MPCache reproduction checkpoint', '',
         'Generated: ' + datetime.datetime.now().isoformat(),
         f'Profile: {profile} (GPU-only generation; no CPU offload anywhere).', '']
if profile == 'h20':
    lines += ['H20 bf16 == the precision the paper used; no memory adapter is active.',
              'Latency/speedup still come from different silicon than the paper H200.', '']
else:
    lines += ['V100 fp16 differs from paper H200 bf16.',
              'Legacy RA Slow/Fast outputs used tau=0.4 and are excluded; cfg_tau_v2 reruns required.', '']
lines += ['Video counts are unique global IDs, not overlapping chunk totals.',
          'Timing uses chunk averages and remains provisional until re-timed on an idle GPU.', '']
for model in ('cogvideox', 'wan'):
    lines += ['## ' + model, '']
    for method in ('vanilla50', 'vanilla25', 'teacache', 'taylorseer', 'racfg_slow', 'racfg_fast'):
        root = res / model / method
        if method.startswith('racfg_'):
            root = root / 'cfg_tau_v2'
        if method == 'taylorseer':
            root = root / 'gpu_only_v2'
        ids = {p.stem for p in root.glob('*/videos/video_*.mp4') if p.stat().st_size > 0}
        lines.append(f'- {method}: {len(ids)} valid unique videos')
    tables = sorted((res / model / '_merged').glob('table_n*.json'),
                    key=lambda p: int(p.stem.split('n')[-1]))
    # Publish only corrected audit-aware tables, never legacy mixed-chunk tables.
    tables = [p for p in tables if all(tag in ' '.join(json.loads(p.read_text()).get('notes', []))
                                      for tag in ('cfg_tau_v2', 'gpu_only_v2'))]
    if tables:
        source = tables[-1]
        for ext in ('.json', '.md'):
            p = source.with_suffix(ext)
            if p.exists():
                shutil.copy2(p, r / 'repro' / f'{model}_{p.name}')
        lines += ['', source.with_suffix('.md').read_text(), '']
(r / 'STATUS_auto.md').write_text('\n'.join(lines) + '\n')
PY
branch=$(git branch --show-current)
[ "$branch" = "$PUSH_BRANCH" ] || { echo "Refusing to push unexpected branch: $branch (want $PUSH_BRANCH)" >&2; exit 1; }
# `git add` aborts the whole command when one pathspec is missing; under `|| true` that silently
# kept every checkpoint from committing code and STATUS changes.  Add only paths that exist.
for p in repro/run_repro.sh repro/aggregate.py repro/queue_overnight.sh repro/push_overnight.sh \
         repro/setup_server.sh repro/setup_h20.sh repro/patches/apply_v100_patch.py \
         repro/patches/apply_repro_patch.py repro/STATUS.md repro/H20_MIGRATION.md \
         repro/ledger.py repro/plan_missing.py repro/overnight_watchdog.sh \
         repro/H200_TAYLOR_WAN.md STATUS_auto.md \
         experiments/video/g0a_costmodel.py experiments/video/g0a_analyze.py; do
  if [ -e "$p" ]; then git add -- "$p"; fi
done
for p in repro/cogvideox_table_n* repro/wan_table_n* repro/G0a_analysis_20261016.txt; do
  [ ! -f "$p" ] || git add -- "$p"
done
if ! git diff --cached --quiet; then
  git commit -m "repro: ${PROFILE} benchmark checkpoint $(date +%F_%T)"
fi
# Always retry push, including when a previous checkpoint committed but push failed.
git push origin HEAD:"$PUSH_BRANCH"
git log -1 --format='%h %s'
