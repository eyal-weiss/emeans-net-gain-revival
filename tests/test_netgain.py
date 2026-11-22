"""Tests of net_gain_revival (need FAISS from PR #5915, at 98deaca or at its head 79440bc; skipped otherwise). The
comparison with the research code on 98deaca also needs research/build.sh, and the native tests
net_gain_revival/native/build.sh. With the head's e_means files first on the path (e.g. PYTHONPATH=<overlay>), the
head's tests run; otherwise 98deaca's."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
faiss = pytest.importorskip("faiss")
if not hasattr(faiss, "accumulate_removal_stats"):
    pytest.skip("needs FAISS from PR #5915", allow_module_level=True)
sys.path.insert(0, ROOT)
from net_gain_revival import NetGainRevivalEmeans, native  # noqa: E402
from faiss.contrib.e_means import NumpyEmeans as _Driver  # noqa: E402

HEAD = hasattr(_Driver, "_handover_start")       # the PR's head driver (79440bc)

RESEARCH_BUILT = all(os.path.exists(os.path.join(ROOT, "research", f)) for f in ("ngp_kernels.so", "ngp_decide.so"))


def blobs(n=20000, d=8, groups=40, seed=0):
    rs = np.random.RandomState(seed)
    c = rs.standard_normal((groups, d)) * 5
    x = c[rs.randint(groups, size=n)] + rs.standard_normal((n, d))
    far = rs.standard_normal((5 * 50, d)) * 0.3 + 40 * np.repeat(np.eye(d)[:5], 50, axis=0)   # five small far groups
    return np.ascontiguousarray(np.vstack([x, far]), dtype=np.float32)


def train(km, x, k, seed):
    rows = np.sort(np.random.RandomState(seed).permutation(len(x))[:k])
    C, _ = km.train(x, init_centroids=np.ascontiguousarray(x[rows]))
    return C


@pytest.mark.skipif(HEAD or not RESEARCH_BUILT, reason="98deaca only; research kernels needed")
@pytest.mark.parametrize("stat,early_stop,T", [("medg_pm", "hl", 2), ("medg_pm", None, 5), ("medg_fh", "hl", 3)])
def test_same_centres_as_the_research_code(stat, early_stop, T):
    os.environ["NGP_IMPL"] = "cpp"
    sys.path.insert(0, os.path.join(ROOT, "research"))
    import ngp_stop
    cfg = {"medg_pm": "ngp2_full_medg_pm", "medg_fh": "ngp2_full_medg_fh"}[stat]
    spec = ngp_stop.CONFIGS[cfg]
    x, k = blobs(), 128
    ref = ngp_stop.AblationEmeans(x.shape[1], k, *spec[:4], nepochs=T, seed=1, early_stop=early_stop, **spec[4])
    C_ref = train(ref, x, k, 1)
    km = NetGainRevivalEmeans(x.shape[1], k, nepochs=T, seed=1, stat=stat, early_stop=early_stop, use_native=False)
    C = train(km, x, k, 1)
    assert km.n_splits == ref.n_splits and km.es_stop_step == ref.es_stop_step
    np.testing.assert_array_equal(C, C_ref)


@pytest.mark.skipif(not native.available(), reason="native kernels not built")
def test_native_equals_numpy():
    x, k = blobs(seed=2), 96
    a = NetGainRevivalEmeans(x.shape[1], k, nepochs=2, seed=3, use_native=False)
    b = NetGainRevivalEmeans(x.shape[1], k, nepochs=2, seed=3, use_native=True)
    np.testing.assert_array_equal(train(a, x, k, 3), train(b, x, k, 3))
    assert a.n_splits == b.n_splits


def test_keeps_far_groups():
    x, k = blobs(), 128
    km = NetGainRevivalEmeans(x.shape[1], k, nepochs=5, seed=1)
    C = train(km, x, k, 1)
    far = 40 * np.eye(x.shape[1])[:5]
    assert sum(np.min(((C - f) ** 2).sum(1)) < 4.0 for f in far) == 5
    assert km.n_checks > 0


@pytest.mark.skipif(not HEAD, reason="head driver only")
@pytest.mark.parametrize("freeze,early_stop,T", [(False, None, 2), (True, None, 2), (False, "hl", 2), (True, "hl", 3),
                                                 (False, None, 5)])
def test_head_same_centres_as_the_round3_code(freeze, early_stop, T):
    """The release class against round 3's research class (research/ngp_head.py, guard on), and with the early stop
    against that class plus version 1.0.0's early stop (research/verify_head.py); NumPy and C++ paths."""
    sys.path.insert(0, os.path.join(ROOT, "research"))
    import ngp_head
    import verify_head
    x, k = blobs(seed=4), 128
    ref_cls = ngp_head.NetGainHead if early_stop is None else verify_head._head_es_class()
    ref = ref_cls(x.shape[1], k, nepochs=T, seed=5, freeze=freeze, guard=True)
    C_ref = train(ref, x, k, 5)
    for nat in ([False, True] if native.available() else [False]):
        km = NetGainRevivalEmeans(x.shape[1], k, nepochs=T, seed=5, freeze=freeze, early_stop=early_stop,
                                  use_native=nat)
        C = train(km, x, k, 5)
        assert km.n_splits == ref.n_splits and km.n_checks == ref.n_checks
        assert km.es_stop_step == getattr(ref, "es_stop_step", None)
        np.testing.assert_array_equal(C, C_ref)


@pytest.mark.skipif(not HEAD, reason="head driver only")
def test_head_freeze_option():
    """freeze=True: no move in the last pass; the default (no freeze) may move there, and the closing handover runs."""
    x, k = blobs(seed=6), 160
    a = NetGainRevivalEmeans(x.shape[1], k, nepochs=2, seed=7, freeze=True, early_stop=None)
    train(a, x, k, 7)
    assert a.n_moves_last_pass == 0 and a.freeze
    b = NetGainRevivalEmeans(x.shape[1], k, nepochs=2, seed=7, early_stop=None)
    train(b, x, k, 7)
    assert not b.freeze and b.n_checks > a.n_checks
    assert b.recipe["closing_handover"] == "second_nearest"
