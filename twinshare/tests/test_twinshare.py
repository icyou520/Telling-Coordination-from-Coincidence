import json

import numpy as np
import pytest

from twinshare import Design, analyze, from_arrays
from twinshare import engine as E
from twinshare.cli import main
from twinshare.data import load
from twinshare.examples import retail


def _retail_panel(**kw):
    dates, sales, temp, promo = retail(seed=kw.pop("seed", 1), **kw)
    cal = np.eye(7)[[d.weekday() for d in dates]][:, 1:]
    return from_arrays(np.log(sales), np.column_stack([temp, promo, cal]),
                       members=[f"store_{k + 1:02d}" for k in range(sales.shape[1])])


D = Design(windows=(1, 7, 28))


# ---------------------------------------------------------------- engine internals

def test_history_matches_direct_window_means():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 3))
    d = Design(horizon=2, windows=(1, 4))
    h = E.history(x, d)
    for r, t in enumerate(range(d.lead, 60)):
        assert np.allclose(h[r, :, 0], x[t - 2])
        assert np.allclose(h[r, :, 1], x[t - 5: t - 1].mean(0))


def test_purge_removes_rows_whose_history_holds_a_test_target():
    d = Design(windows=(1, 5), block=20)
    fold, _ = E.folds(200, d)
    for f in range(3):
        tr, te = E.train_rows(fold, f, d), fold == f
        for r in np.flatnonzero(tr):
            assert not te[max(r - d.lead, 0): r - d.horizon + 1].any()


def test_memory_gap_is_zero_when_group_and_members_remember_alike():
    o = 0.7 ** np.arange(30)
    assert abs(E.s_pred(o, 50 * o, D)) < 1e-12


def test_constant_cause_columns_are_harmless():
    p = _retail_panel()
    z = np.column_stack([p.covariates, np.full(len(p.times), 3.0)])
    assert abs(E.share(p.values, z, D).S - E.share(p.values, p.covariates, D).S) < 1e-9


# ---------------------------------------------------------------- known answers

def test_independent_stores_show_no_coordination():
    r = analyze(_retail_panel(), D, seed=0, n_null=4, detail=False)
    assert r.verdict == "No coordination detected"
    assert abs(r.S) < 0.02


def test_slow_recorded_cause_does_not_create_coordination():
    s = [E.share(p.values, p.covariates, D).S for p in (_retail_panel(seed=k) for k in range(4))]
    assert np.mean(s) < 0.005


def test_coupled_stores_are_detected_and_match_the_memory_gap():
    r = analyze(_retail_panel(coupling=0.3), D, seed=0, n_null=4, detail=False)
    assert r.verdict.startswith("Coordination detected")
    assert {c.name: c.status for c in r.checks}["memory gap"] == "pass"


def test_hidden_driver_is_detected_too():
    r = analyze(_retail_panel(hidden=0.5), D, seed=0, n_null=4, detail=False)
    assert r.verdict.startswith("Coordination detected")


def test_leader_is_found_by_closure_and_influence():
    p = _retail_panel(leader=0.3)
    assert E.gain(p.values, p.covariates, D) > 0.2
    infl = E.influence(p.values, p.covariates, D)
    assert int(np.argmax(infl)) == 0 and np.median(infl) < 0.01


def test_closure_is_near_zero_for_exchangeable_coupling():
    p = _retail_panel(coupling=0.3)
    assert abs(E.gain(p.values, p.covariates, D)) < 0.02


# ---------------------------------------------------------------- data and command line

def test_long_and_wide_files_load_the_same_panel(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.normal(5, 1, (40, 3))
    x[4, 1] = np.nan
    wide = ["t,a,b,c"] + [f"{k},{','.join('' if np.isnan(v) else f'{v:.6f}' for v in row)}" for k, row in enumerate(x)]
    (tmp_path / "w.csv").write_text("\n".join(wide))
    long = ["t,member,value"] + [f"{k},{m},{'' if np.isnan(v) else f'{v:.6f}'}" for k, row in enumerate(x)
                                 for m, v in zip("abc", row)]
    (tmp_path / "l.csv").write_text("\n".join(long))
    pw = load({"data": {"file": "w.csv", "time": "t"}}, tmp_path)
    pl = load({"data": {"file": "l.csv", "format": "long", "time": "t", "member": "member", "value": "value"}},
              tmp_path)
    assert np.allclose(pw.values, pl.values) and pw.members == pl.members == ["a", "b", "c"]
    assert any("filled 1 missing" in n for n in pw.notes)


def test_cli_example_run_writes_report(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    main(["example", "retail", "--coupling", "0.3", "--out", str(tmp_path / "demo")])
    main(["run", str(tmp_path / "demo" / "study.toml"), "--quick"])
    out = json.loads((tmp_path / "demo" / "report" / "result.json").read_text())
    assert out["verdict"].startswith("Coordination detected")
    assert (tmp_path / "demo" / "report" / "report.html").stat().st_size > 50_000


def test_init_refuses_to_overwrite(tmp_path):
    f = tmp_path / "study.toml"
    main(["init", "--domain", "retail", "--out", str(f)])
    with pytest.raises(SystemExit):
        main(["init", "--out", str(f)])
