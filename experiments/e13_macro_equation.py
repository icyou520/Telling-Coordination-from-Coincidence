"""Experiment 13: an explicit macro equation for the flock, tested without agents.

Experiment 12 found that z = (phi, L), global and local alignment, is a closed
macro description of a 1600-agent Vicsek flock. Here the macro law itself is
written down and tested:

    z_{t+1} = z_t + f(z_t, eta) + sigma(z_t, eta) * xi_t,     xi ~ N(0, R)

  f      cubic polynomial in (phi, L) times quadratic in noise eta (30 terms),
         fitted by least squares on the experiment 11 trajectories;
  sigma  per-component, exp of the same feature set fitted to log residual^2;
  R      pooled residual correlation.

One macro step = TAU = 10 micro steps. For every noise level the equation is
fitted with that noise level left out, then iterated on its own (no agents)
and its stationary statistics are compared with the fresh-seed agent
simulations of experiment 12. A one-variable equation (phi only, same
procedure) is the baseline.

Predictions, fixed before running:

  M1  The (phi, L) equation's mean phi is within 0.05 of the agents' at >= 11
      of 13 held-out noise levels.
  M2  Its susceptibility N Var(phi) peaks within 0.25 of the agents' peak.
  M3  Its peak susceptibility is within a factor 2 of the agents'.
  M4  It beats the phi-only equation: summed |error| of mean phi over the
      held-out noise levels is smaller.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.e11_band_closure import N, NOISES

CENTRES = (np.arange(8) + 0.5) / 8
STEPS, BURN = 40_000, 2_000


def load(path: Path):
    d = np.load(path)
    local = d["feats"][:, :, :8] @ CENTRES
    return d["noise"], d["phi"], local


def features(z: np.ndarray, eta) -> np.ndarray:
    """Monomials of total degree <= 3 in z, each times 1, eta, eta^2."""
    z = np.atleast_2d(z)
    cols = [np.ones(len(z))]
    k = z.shape[1]
    for i in range(k):
        cols.append(z[:, i])
    for i in range(k):
        for j in range(i, k):
            cols.append(z[:, i] * z[:, j])
    for i in range(k):
        for j in range(i, k):
            for m in range(j, k):
                cols.append(z[:, i] * z[:, j] * z[:, m])
    base = np.column_stack(cols)
    eta = np.broadcast_to(np.asarray(eta, float), (len(z),))[:, None]
    return np.hstack([base, base * eta, base * eta**2])


def feature_names(var_names) -> list[str]:
    k = len(var_names)
    base = ["1"] + list(var_names)
    base += [f"{var_names[i]}*{var_names[j]}" for i in range(k) for j in range(i, k)]
    base += [f"{var_names[i]}*{var_names[j]}*{var_names[m]}"
             for i in range(k) for j in range(i, k) for m in range(j, k)]
    times = lambda b, s: s if b == "1" else f"{b}*{s}"
    return base + [times(b, "eta") for b in base] + [times(b, "eta^2") for b in base]


def _forward_select(fs, y, n_terms, ridge):
    """Greedy forward selection of ``n_terms`` columns per output (intercept always kept)."""
    w = np.zeros((fs.shape[1], y.shape[1]))
    chosen_all = []
    for c in range(y.shape[1]):
        chosen = [0]
        while len(chosen) < n_terms + 1:
            best, best_sse = None, np.inf
            for j in range(1, fs.shape[1]):
                if j in chosen:
                    continue
                x = fs[:, chosen + [j]]
                coef = np.linalg.lstsq(x, y[:, c], rcond=None)[0]
                sse = ((y[:, c] - x @ coef) ** 2).sum()
                if sse < best_sse:
                    best, best_sse = j, sse
            chosen.append(best)
        x = fs[:, chosen]
        w[chosen, c] = np.linalg.lstsq(x, y[:, c], rcond=None)[0]
        chosen_all.append(chosen)
    return w, chosen_all


class MacroEquation:
    def __init__(self, noise, trajectories, ridge: float = 1e-8, calibrate_noise: bool = False,
                 n_terms: int | None = None):
        xs, ys, es = [], [], []
        for eta, z in zip(noise, trajectories):
            xs.append(z[:-1])
            ys.append(np.diff(z, axis=0))
            es.append(np.full(len(z) - 1, eta))
        x, y, e = np.vstack(xs), np.vstack(ys), np.concatenate(es)
        f = features(x, e)
        self.mu, self.sd = f.mean(0), f.std(0)
        self.sd[self.sd == 0] = 1.0
        self.sd[0], self.mu[0] = 1.0, 0.0
        fs = (f - self.mu) / self.sd
        a = fs.T @ fs + ridge * len(fs) * np.eye(fs.shape[1])
        self.w = np.linalg.solve(a, fs.T @ y)
        self.terms = None
        if n_terms is not None:
            self.w, self.terms = _forward_select(fs, y, n_terms, ridge)
        res = y - fs @ self.w
        self.w_log = np.linalg.solve(a, fs.T @ np.log(res**2 + 1e-12))
        if calibrate_noise:
            self.w_log[0] += np.log(np.mean(res**2 / np.exp(fs @ self.w_log), axis=0))
        sig = np.exp(0.5 * (fs @ self.w_log))
        self.corr = np.atleast_2d(np.corrcoef((res / sig).T))
        self.chol = np.linalg.cholesky(self.corr + 1e-9 * np.eye(len(self.corr)))

    def _fs(self, z, eta):
        return (features(z, eta) - self.mu) / self.sd

    def drift(self, z, eta):
        return self._fs(z, eta) @ self.w

    def raw_drift_terms(self, names):
        """Drift written in unstandardized monomials: {component: [(coef, name), ...]}."""
        out = {}
        for c in range(self.w.shape[1]):
            w = self.w[:, c] / self.sd
            const = w[0] - (w[1:] * self.mu[1:]).sum()
            terms = [(float(const), "1")] + [(float(w[i]), names[i]) for i in range(1, len(w))
                                             if self.w[i, c] != 0]
            out[c] = terms
        return out

    def simulate(self, z0, eta, steps=STEPS, seed=0, noise_scale: float = 1.0):
        rng = np.random.default_rng(seed)
        z = np.array(z0, float)
        out = np.empty((steps, len(z)))
        for t in range(steps):
            fs = self._fs(z[None], eta)
            sig = noise_scale * np.exp(0.5 * (fs @ self.w_log))[0]
            z = z + (fs @ self.w)[0] + sig * (self.chol @ rng.standard_normal(len(z)))
            z = np.clip(z, 0.0, 1.0)
            if len(z) == 2:
                z[0] = min(z[0], z[1])
            out[t] = z
        return out[BURN:]


def stats(phi):
    return float(np.mean(phi)), float(N * np.var(phi))


def analyze(train, test, out_dir: Path) -> str:
    noise_tr, phi_tr, loc_tr = train
    noise_te, phi_te, _ = test
    agent = {e: (np.mean([stats(p)[0] for p in phi_te[noise_te == e]]),
                 np.mean([stats(p)[1] for p in phi_te[noise_te == e]])) for e in NOISES}
    lines = [f"Experiment 13: explicit macro equation, leave-one-noise-out (N={N})"]
    for calibrate in (False, True):
        pred = {"2d": {}, "1d": {}}
        for e in NOISES:
            keep = noise_tr != e
            two = MacroEquation(noise_tr[keep], [np.column_stack([p, l]) for p, l in
                                                 zip(phi_tr[keep], loc_tr[keep])],
                                calibrate_noise=calibrate)
            one = MacroEquation(noise_tr[keep], [p[:, None] for p in phi_tr[keep]],
                                calibrate_noise=calibrate)
            start = np.array([0.5, 0.65])
            pred["2d"][e] = stats(two.simulate(start, e)[:, 0])
            pred["1d"][e] = stats(one.simulate(start[:1], e)[:, 0])
        lines += ["", "  Noise amplitude " + ("calibrated so residual variance matches "
                                              "(post-hoc correction)" if calibrate else
                                              "from exp(mean log residual^2) (pre-registered)"),
                  f"  {'noise':>6}{'phi agents':>12}{'phi (phi,L)':>13}{'phi (phi)':>11}"
                  f"{'chi agents':>12}{'chi (phi,L)':>13}{'chi (phi)':>11}"]
        for e in NOISES:
            lines.append(f"  {e:6.2f}{agent[e][0]:12.3f}{pred['2d'][e][0]:13.3f}"
                         f"{pred['1d'][e][0]:11.3f}{agent[e][1]:12.2f}{pred['2d'][e][1]:13.2f}"
                         f"{pred['1d'][e][1]:11.2f}")
        err = {k: np.array([abs(pred[k][e][0] - agent[e][0]) for e in NOISES]) for k in pred}
        peak = {k: float(NOISES[int(np.argmax([pred[k][e][1] for e in NOISES]))]) for k in pred}
        peak_agent = float(NOISES[int(np.argmax([agent[e][1] for e in NOISES]))])
        chi_ratio = max(pred["2d"][e][1] for e in NOISES) / max(agent[e][1] for e in NOISES)
        m1 = int((err["2d"] <= 0.05).sum()) >= 11
        m2 = abs(peak["2d"] - peak_agent) <= 0.25
        m3 = 0.5 <= chi_ratio <= 2.0
        m4 = err["2d"].sum() < err["1d"].sum()
        tag = lambda ok: "PASS" if ok else "FAIL"
        inner = slice(1, None)
        lines += ["", f"  chi peak: agents {peak_agent}, (phi,L) equation {peak['2d']}, "
                      f"phi-only equation {peak['1d']}",
                  f"  summed |error| of mean phi without the edge noise {NOISES[0]}: "
                  f"(phi,L) {err['2d'][inner].sum():.3f}, phi-only {err['1d'][inner].sum():.3f}",
                  "", "  Predictions" + (" (re-scored after the correction, not "
                                         "pre-registered):" if calibrate else
                                         " (pre-registered):"),
                  f"  {tag(m1)}  M1 mean phi within 0.05 at >= 11/13 held-out noises "
                  f"[{int((err['2d'] <= 0.05).sum())}/13]",
                  f"  {tag(m2)}  M2 chi peak within 0.25 [{peak['2d']} vs {peak_agent}]",
                  f"  {tag(m3)}  M3 peak chi within a factor 2 [ratio {chi_ratio:.2f}]",
                  f"  {tag(m4)}  M4 beats phi-only equation [summed error "
                  f"{err['2d'].sum():.3f} vs {err['1d'].sum():.3f}]"]
        name = "e13_macro_equation" + ("_calibrated" if calibrate else "") + ".png"
        _plot(agent, pred, out_dir / name)
        lines.append(f"\n  figure: {out_dir / name}")
    text = "\n".join(lines)
    print(text)
    return text


def _plot(agent, pred, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for i, title in enumerate(("mean polarization", "susceptibility N Var(phi)")):
        ax[i].plot(NOISES, [agent[e][i] for e in NOISES], "ko-", label="agents (fresh seeds)")
        ax[i].plot(NOISES, [pred["2d"][e][i] for e in NOISES], "s-", label="(phi, L) equation")
        ax[i].plot(NOISES, [pred["1d"][e][i] for e in NOISES], "^--", label="phi-only equation")
        ax[i].set_title(f"{title}, each noise held out of the fit")
        ax[i].set_xlabel("noise")
        ax[i].grid(alpha=0.2)
        ax[i].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    paths = [out_dir / "e11_trajectories.npz", out_dir / "e12_trajectories.npz"]
    if not all(p.exists() for p in paths):
        raise SystemExit("needs results/e11_trajectories.npz and e12_trajectories.npz: "
                         "run `run.py bands local` first")
    return analyze(load(paths[0]), load(paths[1]), out_dir)


if __name__ == "__main__":
    main()
