"""Compatibility shim.

The profile-aware implementation lives in `apply_repro_patch.py` (v100 + h20 profiles).
This filename is kept because setup_server.sh, push_overnight.sh and earlier commits refer
to it. Usage is unchanged: python apply_v100_patch.py /path/to/RA-CFGCache [v100|h20]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from apply_repro_patch import main  # noqa: E402

if __name__ == "__main__":
    main()
