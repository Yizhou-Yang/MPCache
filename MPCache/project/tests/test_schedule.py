import math
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mpcache import solve, geometric, head_length, relative_gap, check

# the measured CogVideoX-2B profile at 30 steps (VGG, prompt 0), normalised
PROFILE = [0.0, 0.569, 0.452, 0.313, 0.192, 0.118, 0.094, 0.063, 0.061, 0.054,
           0.053, 0.047, 0.045, 0.043, 0.043, 0.042, 0.041, 0.040, 0.041, 0.041,
           0.041, 0.041, 0.041, 0.041, 0.041, 0.042, 0.042, 0.042, 0.043, 0.050]


def test_budget_is_exact():
    for B in (5, 10, 17, 21, 30):
        s = solve(PROFILE, B)
        assert len(s) == B


def test_last_step_always_exact():
    for B in (3, 10, 29):
        assert len(PROFILE) - 1 in solve(PROFILE, B)
        assert len(PROFILE) - 1 in geometric(B, T=len(PROFILE))


def test_greedy_prefers_the_head():
    s = solve(PROFILE, 12)
    assert set(range(1, 11)).issubset(set(s)), s


def test_head_length_read_off_the_profile():
    assert head_length(PROFILE, frac=0.5) == 4
    assert head_length(PROFILE, frac=0.1) >= 9


def test_geometric_beats_uniform_under_the_model():
    T = 50
    prof = [max(0.0, 1.0 - i / 12) + 0.02 for i in range(T)]
    B = 20
    geo = geometric(B, T=T, head=12)
    uni = sorted(set([round(i * (T - 1) / (B - 1)) for i in range(B)]))
    assert len(geo) == len(uni) == B
    assert relative_gap(prof, B, geo) < relative_gap(prof, B, uni)


def test_check_accepts_what_we_ship():
    B = 20
    check(PROFILE, B, solve(PROFILE, B))
