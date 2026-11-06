# Emergence Lab: findings

What 90 experiments (e0–e89) say about the lab's starting hypothesis, especially
its central equation

> C(F(x)) ≈ G(C(x)):  a coarse-graining C of micro dynamics F is an emergent
> layer when it has its own law G.

Most experiments after e4 state their predictions in the docstring *before*
running, and report PASS/FAIL against them. Where a result was found after the
fact, it says so. Every number below can be regenerated with
`uv run python run.py <name>` (names in the last column of each table).

---

## 1. Summary

**Confirmed**

1. **Exact closure is common and checkable.** Elementary cellular automata
   have hundreds of exact coarse-grainings (240 across 65 rules at block size
   ≤ 3), including the textbook rule 105 → rule 150. Lean proves that a macro
   law exists if and only if the coarse-graining is closed. (e1, `lean/`)
2. **The popular information score Ψ cannot detect interaction-driven
   emergence.** Agents that ignore each other score *higher* (≈ 9 bits) than
   a real flock. (e8)
3. **Closure alone cannot either**, but closure plus an *interaction share*
   can: the flock's macro law is ≈ 100 % interaction-made in the ordered
   phase and ≈ 0 % for non-interacting twins. (e9)
4. **Choosing C is the whole game.** Global alignment φ alone stops being
   closed near the flocking transition; the pair (global alignment φ, local
   alignment L) is closed everywhere (autonomy 0.92–1.00), confirmed on
   fresh seeds. (e10–e12)
5. **The (φ, L) layer has a working equation** that, fitted on agents and then
   run with no agents, reproduces the order curve and the transition at the
   fitted size. (e13)
6. **A universal law exists at a fixed physical scale.** Order measured over
   ~100-agent regions follows the same law from 400 to 6400 agents. (e16)
7. **Global, size-dependent behaviour follows from that scale.** A local
   field law of density and momentum on ~100-agent cells, fitted at one flock
   size with one calibrated offset, predicts global order at three unseen
   sizes (400, 3600, 6400 agents). (e21, e22)

**Refuted or failed**

* Ψ growing with flock size as evidence of collective emergence (e4–e8).
* Density bands as the variable that closes the flock layer (e11).
* A size-independent law for the global (φ, L) layer, with or without a
  size correction (e14, e15).
* Field laws on small (~25-agent) cells, including a hidden-state
  (Kalman-style) fit, which failed validation on synthetic data (e17–e19).

**Open**

* The field law is ~0.05 too ordered just below the transition, and its
  fluctuation peak does not move with size as the agents' does (e21, e22).
* Each noise level still has its own fitted law (no noise universality yet).
* Only one rich system (Vicsek flocking) has been taken this far.

---

## 2. The emerging picture: a hierarchy with one universal level

```
agents (micro)          N agents, positions + headings
   │  coarse-grain at a fixed scale (~100 agents)
   ▼
field (meso)            density ρ and momentum m per cell
   │  local, size-free, conserving law  ← universal across sizes (e16, e21, e22)
   ▼
global order (macro)    (φ, L): closed (e12), has an equation (e13),
                        but its law depends on system size (e14, e15):
                        it is inherited from the field law + domain size
```

In terms of the theory: C(F(x)) ≈ G(C(x)) holds at both upper levels, but
only the field level has a G that does not depend on the size of the system.
The top layer is real (closed, interaction-made, predictable) yet not
self-contained: its law is set by the layer below plus the boundary
(system size).

---

## 3. Measures that survived

| Measure | Definition | Why it is needed |
| --- | --- | --- |
| Exact closure | C(x) = C(y) ⇒ C(F(x)) = C(F(y)) | Finite systems; equivalent to the existence of G (Lean) |
| Autonomy | R²(macro only) / max(R²(macro), R²(macro + micro features)), held-out | Approximate closure for noisy systems; 1 = closed |
| Interaction share S | 1 − MSE(fitted G) / MSE(G₀), G₀ = law for non-interacting parts | Separates real emergence from averaging (ideal-gas-like closure) |
| Prediction at unseen sizes | Fit G at one size, run it at others | Distinguishes a universal law from a curve fit |

For the flock, G₀ is exact: with independent agents the mean velocity is
V′ = c^τ V + Gaussian noise of variance (1 − c^{2τ})/(2N) per component,
c = sin(η/2)/(η/2), so |V′| is Rice-distributed (checked against direct
simulation: 0.0339 vs 0.0335).

**Measures that did not survive:** Rosas et al.'s Ψ (and a "corrected"
variant) as a test of interaction-driven emergence (e4–e8), and one-step
predictive accuracy as a test that an iterated law is right (e17).

---

## 4. Experiment log

### Foundations (e0–e3)

| # | Question | Result | Run |
| --- | --- | --- | --- |
| e0 | Do the measures behave on toy systems (bit flips, counters, XOR)? | Yes: closure, closure gap and synergy behave as expected | `toy` |
| e1 | Do cellular automata have exact coarse-grainings? | 240 non-trivial ones across 65 rules (block ≤ 3); rule 105 → 150 reproduced; rule 30 has none at small blocks | `ca` |
| e2 | Gray-Scott chemistry: which macro variables are closed? | Exploratory; spot counts too discrete and data too small for firm conclusions | `rd` |
| e3 | Flocking: Ψ for polarization | Ψ > 0 in parts of the sweep; motivated e4–e8 | `flock` |

### Ψ under scrutiny (e4–e8)

| # | Pre-registered question | Result | Run |
| --- | --- | --- | --- |
| e4 | Does Ψ peak at the transition and grow with N? | 3 of 4 predictions failed; agent-level Ψ is confounded by N | `scaling` |
| e5 | Ψ against fixed regions instead of agents | Confound removed, main predictions failed; "corrected Ψ" discredited | `regions` |
| e6 | Large flocks (200–1600) | Max Ψ grows with N (0.004 → 1.45); peak location prediction failed | `large` |
| e7 | Robust to region count and estimator? | Yes for growth; location depends on estimator | `robust` |
| e8 | **Negative control:** agents with no interaction | Ψ ≈ 9 bits, above the flock. Claim of "flock emergence via Ψ" retracted | `control` |

### Closure and interaction (e9–e13)

| # | Pre-registered question | Result | Run |
| --- | --- | --- | --- |
| e9 | Is the macro law interaction-made, and closed? | **3/3 pass.** S ≈ 1 (ordered flock), S ≈ 0 (twin); flock autonomy dips to 0.77 near the transition | `interaction` |
| e10 | Is the dip tied to the transition? (N = 400, 1600) | 2/4. Dip moves with the transition (−0.50 and −0.50); too shallow at N = 400 | `transition` |
| e11 | Does band strength close the layer? | **0/4.** Post hoc: local-order features restore autonomy to ~0.85 | `bands` |
| e12 | Confirm local order L on fresh seeds | **3/3 pass.** Autonomy of φ with (φ, L): 0.97 below the peak, 0.94 at it; L itself 0.99 | `local` |
| e13 | Equation for (φ, L), tested without agents (held-out noise levels) | 3/4 pre-registered (fluctuations 2.4× too small: a noise-estimator bug). After the fix, re-scored post hoc: order 13/13, peak 3.25 vs 3.25, peak height ratio 1.00 | `equation` |

### Size (e14–e22)

| # | Pre-registered question | Result | Run |
| --- | --- | --- | --- |
| e14 | Does the N = 1600 (φ, L) equation predict N = 800, 3200 (noise ∝ 1/√N)? | 1/4. Diagnostic: noise scaling exact, L size-free, **φ's drift depends on N** | `sizes` |
| e15 | One equation with a log N term, predicting 400 and 6400 | 1/4. Fine at 400, collapses at 6400 | `sizelaw` |
| e16 | Is order at a fixed scale (~100-agent regions) size-free? | **3/3 pass.** Same curve from 400 to 6400; size-blind law 13/13 at both unseen sizes | `regional` |
| e17 | Field law on 25-agent cells | 1/4 (trivial pass). Decays to disorder even without noise: momentum not transported | `field` |
| e18 | Conservative transport of momentum | 1/5 (trivial). Counting noise biases the fit | `conservative` |
| e19 | Hidden-state field (EM + ensemble Kalman smoother) | Not run on agents: failed recovery of a known law on synthetic data | `latent` |
| e20 | Conservative field on 100-agent cells | 1/5, but first field that orders and shows size dependence; uniform under-ordering | `bigcells` |
| e21 | One offset calibrated at N = 1600, predict 400 and 6400 | **4/5 pass.** Order 13/13 and 11/13; error 0.11 and 0.25 vs 0.91 and 0.79 for "no size dependence"; fluctuation peak missed | `calibrated` |
| e22 | Same frozen law at a fresh size N = 3600, new seeds | **4/5 pass.** Order 12/13; beats "no size dependence" (0.21 vs 0.50) but not interpolation of real data (0.06) | `fresh` |

### Interaction share against prior work (e23)

Liardi et al. 2025 ([arXiv:2504.10372](https://arxiv.org/abs/2504.10372))
showed that Ψ calls the centre of mass of non-interacting random walkers
emergent and causally decoupled, and left the separation of autonomy from
interaction open. Rosas et al. 2024 (computational closure) and Barnett and
Seth 2023 (dynamical independence) likewise count aggregates of independent
parts as emergent. e23 computes the interaction share exactly on their model.

| # | Pre-registered question | Result | Run |
| --- | --- | --- | --- |
| e23 | Coupled random walkers: is S zero for the centre of mass and positive for internal modes, while Ψ is not? | **5/5 pass.** S(centre of mass) = 0 at every coupling (simulated max 0.0003); internal-mode S matches the closed form (max error 0.005); Ψ = 0.41 nats with no coupling and turns negative (−0.35) under strong coupling | `walkers` |

With M = (1 − θ)I − γL (L the chain Laplacian), every Laplacian mode k
evolves as an exact AR(1) with coefficient a_k = 1 − θ − γλ_k, while the
no-interaction twin has b = 1 − θ. Hence, at horizon τ,

  S_k(τ) = (a_k^τ − b^τ)² / [(1 − a_k^(2τ)) + (a_k^τ − b^τ)²]

which is exactly 0 for the centre of mass (λ₀ = 0: internal forces cancel)
for every N, θ, γ, and grows as γ²λ_k²/(θ(2 − θ)) for weak coupling. Ψ
ranks these the wrong way round: it is largest for every mode when nothing
interacts, and falls below zero for the centre of mass as coupling grows,
although that variable's law never changes.

`lean/InteractionShare.lean` proves the general version: for any pairwise
force obeying Newton's third law (linear or not), the sum of positions follows
the no-interaction law exactly, and when interactions do not move a macro
variable, its only exact law is the twin's. The theorem holds in any
commutative ring, so it covers ℝ through Mathlib. The write-up is in the
manuscript (`manuscript/manuscript.md`).

### Stress tests of the interaction share (e24–e27)

| # | Pre-registered question | Result | Run |
| --- | --- | --- | --- |
| e24 | Can the twin be estimated from walker data alone? | **3/3 pass.** Own-state plus relative-state fit within 0.0024 of exact (also at θ = 0); 2 000 steps suffice (mean error 0.005); the naive twin (own past only) gives a false positive of 0.18 | `twin` |
| e25 | Interactions that change only the noise | **4/4 pass.** S misses them (0.000); S_info = 1 − exp(−2D) detects them (0 → 0.098 with agitation); data-only twin within 0.0004 | `noise` |
| e26 | Flock twin estimated from agent data | **6/6 pass.** Noise recovered to 0.002, S within 0.001 of exact; naive twin inflates S by 0.07, wrong radius by 0.05; no false positives for non-interacting agents. The 10-step test was not discriminating, so a one-step test was added before rerunning (recorded in the docstring) | `flocktwin` |
| e27 | Twin as a part in isolation, with no modelling convention | **3/3 pass.** Own dynamics extrapolated to zero neighbours within 0.004 of exact on a chain and a random graph; ignoring neighbour count gives false positives of 0.38 and 0.80 | `isolation` |

The twin is identifiable from data when parts differ in their number of
neighbours, or when interactions can be written in terms of relative state.
Without either, a naive twin absorbs part of the coupling and inflates S.

### Phase transitions and oscillators (e28–e30)

| # | Pre-registered question | Result | Run |
| --- | --- | --- | --- |
| e28 | Curie–Weiss (multistable): exact S through the transition | **2/2 pass.** Exact S rises 0 → 0.97 from βJ = 0 to 1.5; simulation within 0.014. Exact Ψ falls to −19.8 nats in the ordered phase | `curieweiss` |
| e29 | 2D Ising across T_c (magnetization not closed) | **4/4 pass.** S = 0 at β = 0, rising to 0.99 at T = 2.3 and ≥ 0.998 below T_c; data-only twin within 0.004 | `ising` |
| e30 | Kuramoto: is the twin's law unambiguous? | **3/4.** Identical oscillators: yes (macro vs micro-informed twin within 0.013). **Heterogeneous frequencies: no**, up to 0.10 apart below synchronization, because the twin is not closed on r. Diagnostic: micro-informed S on uncoupled data is −0.06 (identical: +0.009) | `kuramoto` |

S needs the twin to be closed on the macro variable (compare the Lean theorem:
a macro law exists if and only if the coarse-graining is closed). Check this
before reporting S.

### Real data: fish schools (e31)

Data: Puy et al. 2024 (PNAS 121, e2309733121; Zenodo 10890112), two 60-minute
recordings of 39 fish and six of 8 fish, all fish tracked in every frame.
Download into `data/puy2024/` (ignored by git). Macro variable: polarization
over 1 s. Twin: each fish's fitted own terms (turning bias, wall avoidance)
plus independent residual noise, with interaction terms (alignment,
attraction, repulsion with the 4 nearest fish) removed. Control: virtual
groups made by time-shifting each fish's real trajectory by ≥ 90 s.

| N | rec | S real | S virtual | twin closure | alignment real / virtual |
| --- | --- | --- | --- | --- | --- |
| 39 | 0 | 0.74 | 0.03 | −0.06 | 0.101 / −0.008 |
| 39 | 1 | 0.44 | 0.09 | −0.14 | 0.077 / −0.004 |
| 8 | 0–5 | 0.42–0.75 | 0.08–0.16 | −0.18 to −0.59 | 0.053–0.104 / ≤ 0.002 |

**1 of 4 pre-registered predictions passed.**

* R3 (alignment > 0 and > 3 times the virtual group's) **passed** in every recording.
* R1 (S ≥ 0.5) **failed** for one 39-fish recording (0.44); 4 of 6 8-fish recordings reached it.
* R2 (virtual S ≤ 0.1) **failed** (up to 0.16).
* R4 (twin closure ≥ −0.05) **failed** everywhere.

Real S exceeds virtual S by at least 0.26 in every recording, but that
comparison was not pre-registered. Diagnosis (exploratory):

* **The twin is misspecified.** Real fish turning is persistent (lag-0.1 s
  autocorrelation 0.47), but the twin draws independent noise. Its
  prediction is then biased, by −0.02 in polarization on virtual groups,
  which by itself inflates S.
* **The twin is not closed on polarization.** For fish in a walled tank, the
  wall configuration, which polarization does not record, steers the
  twin's polarization.

Both are the e30 failure mode on real data. S on real fish needs a better
twin (persistent turning) and a closed macro description before it can be
reported.

### Real data, second pass: persistent twin (e32)

Two changes from e31:

* The twin's turning now depends on each fish's own two previous turns, which
  are part of its own state, so they stay in the twin.
* The macro state is chosen, by a rule fixed in advance that uses twin
  simulations only, from polarization extended with wall and turning
  summaries.

Part A first runs the pipeline on synthetic fish simulated from the fitted
model, where the answer is known.

| | S real or school | S virtual or loners | closure (best candidate) |
| --- | --- | --- | --- |
| Synthetic, N = 8 / 39 | 0.96 / 0.73 (true twin: 0.96 / 0.73) | −0.08 / −0.10 | n/a |
| Real, N = 39 (2 recordings) | 0.89, 0.73 | −0.17, −0.02 | +0.00, −0.05 |
| Real, N = 8 (6 recordings) | 0.70–0.88 | −0.03 to 0.06 | −0.08 to −0.17 |

**6 of 8 pre-registered predictions passed.**

* **The misspecification is fixed.**
  * The twin from data matches the true twin within 0.002 on synthetic schools.
  * Residual turning noise is now uncorrelated (|autocorrelation| ≤ 0.011, against 0.47 before).
  * Virtual groups score S ≤ 0.06.
  * Real schools exceed their virtual groups by at least 0.69.
  * Alignment replicates.
* **Closure still fails** (B3, and A1, which is the same test on synthetic loners). No candidate macro state closes the twin, mostly in the 8-fish groups. Adding the polarization rate, tried as an exploratory check on twin simulations only, did not help either.
* **The closure estimate is noisy**, about ±0.05 between random seeds.
* **Likely cause:** a finite-N effect. With 8 fish, the full set of headings predicts polarization better than any low-dimensional summary, so no small macro state can be closed.

Because non-closure makes the macro law look worse than the micro-informed
twin, it biases S downward (S on loners is about −0.1). The real values are
probably conservative, but that is not yet shown.

### Real data, third pass: projected twin law (e33)

The new comparator is the twin's own law in polarization:

  G0_proj(φ) = cubic fit of the micro-informed twin prediction g0(x_t) against φ_t over the training states,
  S_proj = 1 − MSE(G) / MSE(G0_proj).

Without interaction, G and G0_proj estimate the same conditional mean, so
S_proj = 0 whether or not the twin is closed.

| | S_proj | control |
| --- | --- | --- |
| Synthetic, N = 8 / 39 | 0.956 / 0.730 (true twin: identical) | loners −0.003 / −0.001 |
| Real, N = 39 | 0.86, 0.67 | virtual 0.000, 0.024 |
| Real, N = 8 | 0.64–0.89 | virtual 0.006–0.045 |

**All 7 pre-registered predictions passed.**

* **State dependence**, the e30 ambiguity: fitting G0_proj on test instead of training states changes S_proj by at most 0.03.
* **e32 was conservative, but only slightly:** S_proj ≥ S_micro in 7 of 8 recordings, by at most 0.02.
* **Caveat:** virtual-group S_proj is positive in all 8 recordings (mean about 0.02). A small residual bias, probably remaining twin misspecification.

*Revised by e77:* cross-fitted over interleaved blocks instead of one chronological split, the real shares are 0.72–0.91 (N = 8 and 39), up to 0.20 higher where the school drifted during the recording.

### S_proj on the case where S failed (e34)

The Kuramoto set-up is as in e30, now with S_proj. Where the twin's exact
law is a polynomial in the macro variable (walkers, Curie–Weiss, Ising),
S_proj = S identically, so those results carry over. The walkers were
rechecked numerically: within 0.002 of the closed form.

**6 of 6 pre-registered predictions passed.**

* At K = 0, S_proj is 0.002 for identical oscillators and 0.003 for heterogeneous ones (S_micro there: −0.08).
* For identical oscillators, it agrees with the exact twin law within 0.002.
* For heterogeneous oscillators:
  * state dependence is at most 0.011;
  * S_proj ≥ S_micro at every K;
  * S_proj ≥ 0.998 for K ≥ 3.
* Bonus: for heterogeneous oscillators S_proj also agrees with the macro (Rice) twin law within 0.01 at every K. The e30 ambiguity came from the micro-informed comparator, not from the twin.

### Second real data set: pigeon flocks (e35)

Data: Nagy et al. 2013 (PNAS 110, 13049; coordinatedflock.elte.hu): GPS
tracks at 10 Hz of three flocks of 10 pigeons circling their loft, with no
walls. Only stretches where all ten birds fly (5–48 min per flock) were
used, at the native 0.2 s step. The twin keeps homing and three lags of
own turning. Virtual flocks are built by time shifts within a flight.

| flock | S_proj | virtual | state dependence (alt) |
| --- | --- | --- | --- |
| A | 0.89 | 0.00 | 0.31 |
| B | 0.09 | −0.02 | 0.16 |
| C | 0.42 | 0.01 | 0.41 |

**4 of 6 pre-registered predictions passed.**

* **Passed:** synthetic loners within 0.03 of 0; fitted twin matches the true twin; virtual flocks ≤ 0.1; alignment > 3 times virtual.
* **Failed:** the real-versus-virtual gap (flock B) and state dependence (up to 0.58).

The passes are hollow. An exploratory check found that the fitted micro
model does not flock at all:

* **It learns the smoothing filter.** Own-turn lag coefficients are 1.9, −1.5 and 0.54, the signature of the dataset's Gaussian/spline smoothing.
* **The interactions are tiny** (alignment about 0.01 per step).
* **Homing is too weak.** Simulated loners drift about 1 km away, while real birds stay about 50 m from the loft.
* **The synthetic "flock" spreads over 110–200 m** with polarization about 0.5, against 6–8 m and 0.93–0.98 for real flocks.

So the synthetic positive control had no interaction effect to detect, and
the twin is badly misspecified. Design gap: unlike e32, no prediction
required the synthetic flock to show an interaction effect.

**Verdict.** On pigeons, the twin, not S, is the failure. A turn-level
linear model on smoothed GPS tracks does not identify the interaction.
Raw, unsmoothed fixes or a coarser micro step, plus a better homing model,
would be needed.

### Pigeons, second pass: 1 s micro step and a gated positive control (e36)

Same data and pipeline as e35, but with a 1 s micro step, one own-turn lag
and a 2 s macro step. New rule, fixed in advance: a flock's real-data
predictions count only if the synthetic flock simulated from its fitted
model holds together (spread ≤ 3 times the real one) and shows interaction
(S_proj ≥ 0.2).

| flock | spread real / synthetic | synthetic S_proj | gate | real S_proj | virtual | state dependence |
| --- | --- | --- | --- | --- | --- | --- |
| A | 5.5 / 263 m | −0.04 | fail | 0.91 | 0.01 | 0.04 |
| B | 8.3 / 251 m | −0.03 | fail | 0.52 | −0.20 | 0.04 |
| C | 7.6 / 257 m | −0.03 | fail | 0.85 | 0.01 | 0.00 |

**The gate failed for every flock, so the real-data predictions are not counted.**

* **The coarser step fixed the smoothing problem.** Residual autocorrelation is 0.11–0.17 and the own-lag coefficient 0.7.
* **Alignment became substantial** (0.12–0.29 per second).
* **But the fitted attraction is about 0,** so the model has no cohesion. Simulated flocks disperse to about 260 m. Pigeons probably keep together through speed changes and front–back positioning, which a model of turning at constant speed cannot express.
* **Uncounted real numbers:** they would have passed R3–R5 (gaps to virtual 0.72–0.90; state dependence ≤ 0.04). They are suggestive, but the twin they rest on is not validated.

### Pigeons, third pass: speed dynamics (e37)

What changed from e36:

* **The twin** also has each bird's own speed dynamics: relaxation to a preferred speed, plus persistence.
* **Interaction** adds two speed responses (front–back offset of the neighbours, speed matching) and a distance-weighted attraction. The gate is unchanged.

**The gate failed again in every flock.** Simulated flocks spread to
330–1030 m against 6–8 m real.

* **Speed responses are clear and consistent:**
  * front–back coefficient 0.09–0.11 > 0 in every flock, as Pettit et al. 2013 found for pairs (R7 PASS);
  * speed matching 0.21–0.36.
* **Lateral attraction is still about 0**, so the linear model cannot keep birds together sideways.
* R1 (synthetic loners) and R6 (residual autocorrelation) passed. R2 failed for flock B (0.08 apart), on a dispersed synthetic flock.
* **Uncounted real values** are unchanged from e36: S_proj 0.91, 0.51, 0.85 against virtual 0.01, −0.20, 0.01.

**Verdict after three pigeon rounds.** The data give a consistent but
unvalidated signal: real flocks score 0.5–0.9 against virtual flocks at
about 0, with little dependence on the fitting states. However, no linear
per-step model fitted to these tracks reproduces a cohesive flock, so the
no-interaction twin cannot be validated, and the result is not counted.
Cohesion in tight flocks is probably a rare-event or nonlinear correction
(birds that stray are pulled back) that least squares on typical steps
does not pick up.

### Third real data set: large zebrafish groups (e38)

Data: Heras et al. 2019 (PLoS Comput. Biol. 15, e1007354; tracked with
idtracker.ai). Nine 10-minute recordings, three each of 60, 80 and 100
juvenile zebrafish, in a circular arena.

The pipeline is the fish pipeline (persistent-turning twin, circular walls,
S_proj) with the pigeon gate. Virtual groups mix fish from the three
recordings of the same size, about 30 s apart, so that shared group heading
cannot leak in.

| N | rec | synthetic school S_proj (gate) | real S_proj | state dependence |
| --- | --- | --- | --- | --- |
| 60 | 1–3 | 0.23, 0.34, 0.26 (pass) | 0.48, 0.59, 0.18 | ≤ 0.04 |
| 80 | 1–3 | 0.47, 0.61, 0.56 (pass) | 0.43, 0.50, 0.35 | ≤ 0.055 |
| 100 | 1–3 | 0.82, 0.75, 0.54 (pass) | 0.62, 0.35, 0.53 | ≤ 0.03 |

Virtual groups: S_proj −0.10 to +0.02 (all nine).

**6 of 8 pre-registered predictions passed.**

* **Passed:**
  * gate in 9 of 9 recordings;
  * synthetic loners within 0.015 of 0;
  * fitted twin matches the true twin;
  * virtual groups ≤ 0.1;
  * alignment > 3 times virtual;
  * residual autocorrelation ≤ 0.004.
* **Failed, both narrowly:**
  * Z4, real minus mean virtual ≥ 0.2: recording 60/3 is at 0.196;
  * Z5, state dependence ≤ 0.05: recording 80/3 is at 0.055.

Unlike the pigeons, the fitted model reproduces an interacting group, so
the twin is validated and the real values count.

*Revised by e77:* these values used one chronological split (first 70%
train, last 30% test), and in 10-minute recordings schools drift. With the
same twin cross-fitted over interleaved blocks, the real shares are
0.54–0.77; recording 60/3 goes from 0.18 to 0.54.

**Lesson.** On real data, use S_proj, the macro law against the twin's
projected law on the same variable, rather than the micro-informed twin.
The closure diagnostic then remains useful only as a description: it says
whether the twin has a law of its own in φ, not whether S can be reported.

### Multi-scale elimination of interaction rules (e39, e40)

Question: can information from several scales rule out impossible
explanations faster than the conventional search, which simulates every
candidate and compares it with the data?

**Set-up (e39).** There are 25 candidate rules for how a fish turns:
* own terms as in e32, plus pairwise alignment (A), attraction (C) and/or repulsion (R);
* neighbours defined either topologically (k = 1, 3, 6 nearest) or metrically (r = 5, 10, 20 cm);
* plus "none", the twin.

The pruned search runs three stages, cheapest first:
* **Stage 0, one test for a whole family.** A new Lean theorem (`eliminate_symmetric_alignment`, verified) shows that metric alignment never changes the group's total turn. So one held-out test on the total turn removes the family {none, r5-A, r10-A, r20-A} at once.
* **Stage 1, short checks.** Batched 1 s and 3 s forecasts of four macro variables (polarization, milling, spread, nearest-neighbour distance) from observed states.
* **Stage 2, long simulations** (2 × 1800 s), compared with the recording's macro statistics.

The conventional search ("ABC-all") runs Stage 2 on all 25 candidates. Micro selection (held-out one-step error) is the other baseline.

**Synthetic schools with a known rule (39 fish)** were run in e39 and again in e40 with refitted truths:

| | e39 | e40 |
| --- | --- | --- |
| Truth survives all stages (T0–T4) | PASS | PASS |
| Stage 0 verdict right (no false family elimination in either run) | FAIL: metric attraction T3 missed, z0 = 1.4 | PASS (T3 z0 = 4.8) |
| Stages 0–1 eliminate (T1–T4) | 79 % | 84 % |
| Simplest survivors = truth | 5 of 5 | 3 of 5 (Occam step dropped a weak true term) |
| Search cost, pruned / ABC-all (T0–T4) | 0.72 (FAIL ≤ 0.5) | 0.65 (FAIL) |
| Out-of-class truth T5 (switching mode): micro-best fails stage 2 | PASS (all 25 eliminated) | PASS |

* **Where the speed-up comes from:** it is about 2× (ratios 0.40–0.56) when interactions are present and the class is right. When there is nothing to eliminate (no interaction, T0) or nothing fits (T5), the pruned search costs more (1.1–1.4×), because Stage 1 is paid for every candidate.
* **Micro selection** keeps the truth in every in-class case (A6 PASS). With the right model class, one-step likelihood is efficient. The multi-scale stages add the ability to say "none of these explanations work" (T5), which micro selection cannot.

**Real fish (Puy et al.).** On real recordings the searches behave as designed:
* stages 0–1 remove more than half of the candidates in 8 of 8 recordings;
* the pruned search costs 0.34 (e39) and 0.44 (e40) of ABC-all.

But the control fails. Virtual 8-fish groups (no interaction) also eliminate "none" and keep topological attraction, in 6 of 6 recordings in e39 and 5 of 6 in e40. Stage 0 also fires in 3 of 6 (e39) and 2 of 6 (e40) of these groups. So the real 8-fish rule identifications (topological, k = 6) **are not counted**, and e39's B1 pass is hollow.

e40 tested the obvious explanation, a missing pull towards the tank centre:
* the centre coefficient is positive in all 16 groups;
* but adding it did not remove the false alarms (E1 2/8, E3 FAIL).

The proxy is probably richer spatial structure in each fish's own behaviour (place preferences), which any feature tied to where other fish usually are can mimic. Virtual 39-fish groups pass in 3 of 4 cases.

**Stage 2 on real fish.** It eliminates the entire 25-rule space in 6–8 of 8 recordings: simulated schools spread 1.4–2.3× (39 fish) and 1.3–1.9× (8 fish) wider than real ones. This is the pigeon failure in milder form. Linear, constant-speed turning rules do not hold a school together as tightly as real fish do. (This does not affect S_proj in e33, whose virtual groups stayed ≤ 0.045. Polarization's interaction share is robust to this misspecification; identifying the rule is not.)

**Verdict.** On data from a known rule, multi-scale elimination is sound:
* no false eliminations in 10 synthetic runs;
* precise and mostly cheap;
* the family theorem removes four candidates with one regression.

It flags a misspecified model class that micro selection would accept. The pre-registered 2× speed-up holds only when there is something to prune. On real fish, the virtual-group control shows that rule identification is still confounded by unmodelled own behaviour, so no real-data rule is claimed.

Next steps:
* a richer own model (a fitted spatial preference field);
* speed dynamics, needed for cohesion;
* a cheaper Stage 1 (adaptive: stop once a candidate is clearly worse).

### e41: all three next steps at once (place field, speed dynamics, adaptive Stage 1)

e41 added three things:
* a fitted place-preference field (17 own-turn features);
* a speed model with a social "ahead" term;
* an adaptive Stage 1 (5 batches, early drop at z > 4).

It also reports only the simplest of nested survivors. The truths T0–T5 were regenerated under the new model. Predictions F1–F8 were hashed before the run (`results/e39_preregistration.txt`).

**Score: 1 of 8 PASS.** Only F5 passed: the switching truth T5 is still correctly rejected by every rule.

* **F1 FAIL: the truth survives in only 2 of 5 cases.** T0, T1 and T2 lose their own rule at Stage 2, and always on school spread. The mismatches are small (2%, 13% and 5%), but 30 minutes of simulation shrinks the block standard error to about 0.1–0.5 cm, so z reaches 4.6–6.5. The z-test now asks for an exact match, which a refitted model never gives. e39 and e40 (truth kept 10 of 10) were spared only because their simulations were noisier. **This is a design flaw in Stage 2:** it needs an equivalence margin (for example, "within 10% of the observed value") rather than a significance test.
* **F3 FAIL:** Stage 0 misses T3 and T4 (z0 −1.4 and 2.8). The family test is underpowered when the place field soaks up turn variance.
* **F4 FAIL, narrowly:** the cost ratio is 0.54 against a target of 0.5. Adaptive Stage 1 made 8-fish searches 6–15% of the conventional search, but T0 (nothing to prune) still costs 1.27×.
* **F6 FAIL, and worse than before:** virtual groups keep "none" in only 1 of 8 (e39 and e40: 2 of 8). The place field did not remove the false alarms. Pairwise terms still pick up something in non-interacting groups.
* **F7 FAIL:** Stage 0 fires in 4 of 8 real recordings and 1 of 8 virtual ones.
* **F8 FAIL:** no rule passes Stage 2 on any real recording. Speed dynamics did not produce cohesion: simulated 8-fish schools are still 1.6–2.2× too spread out (5–8 cm observed, 9–12 cm simulated), and nearest-neighbour distance is 1.4–2.1× too large.

**Verdict.** Neither added model component fixed what it targeted. The new failure, the truth losing at Stage 2, comes from the test statistic, not the idea. Real-fish rule identification remains unclaimed.

The next design would:
* use an equivalence-margin Stage 2;
* add an explicit nonlinear cohesion term (attraction that rises with distance from the group);
* treat virtual-group false alarms as the gate, not an afterthought.

Figures are in `results/e41_figures/` (01–11). `11_truth_stage2_check.png` shows the Stage-2 flaw directly. The figure code was edited cosmetically after the run (layout, plus figure 11); the analysis code is unchanged.

### e42: tolerance Stage 2, a cohesion term, and the virtual-group gate

e42 made e41's three proposed changes. Everything else was as in e41. G1–G9 were hashed before the run.

* **Stage 2 is now an equivalence test.** A candidate passes if every statistic satisfies |sim − obs| ≤ tol + 2·SE. tol is 0.05 for polarization mean, polarization SD and milling, and 15% of the observed value for spread and nearest-neighbour distance.
* **Cohesion term.** Attraction gains a distance-growing pull towards the centre of the fish's neighbours: sin or cos of the bearing × distance/10, capped at 30 cm. It acts in both turning and speed.
* **Gate.** Virtual groups now run all stages. A real rule is claimed only if the paired virtual group's answer is exactly "none".

**Score: 4 of 9 PASS** (e41: 1 of 8).

* **G1–G3 PASS. The method is sound on data from a known rule.** The truth survives in all five in-class tests, and every true-rule mismatch is ≤ 0.5 of the allowed margin. T5 (out of class) is still rejected; its polarization SD is 1.3× the margin.
  * One caveat on G2: nested-minimal answers sometimes drop a true weak term (T2 gives k3-A for k3-AC; T4 gives k6-A for k6-ACR). The neighbour rule is right; the attraction term is not resolved.
* **G4 PASS. Cohesion is real and used.** The turning coefficient is positive in 8 of 8 real recordings, and the speed coefficient is also positive in 8 of 8.
* **G5 FAIL, narrowly, but with a large improvement.** The ratio of simulated to observed spread fell from 1.4–2.2 (e41) to 1.05–1.36. Four recordings are within 25% (needed 6), and two more sit right at 1.25–1.27. Nearest-neighbour distance is now within 10–28%.
* **G6 FAIL (3 of 8; e41: 0 of 8).** k6-AC passes every stage on 39 fish #1, 8 fish #4 and 8 fish #5.
* **G7 FAIL (5 of 8) and G8 FAIL (2 of 8): the gate stays mostly closed.** Where it opens (8 fish #0 and #1), the real recording's rule space was eliminated, so **no rule is claimed on real fish**. Where it is closed, the virtual answers are mostly attraction or cohesion rules (k1-C, k3-C, k6-C and similar). Time-shifted fish that share place preferences still look as if they attract each other. The new cohesion term made this proxy easier to fit, not harder.
* **G9 FAIL:** cost ratio 0.57. T0 (nothing to prune) costs 1.22×; every other case costs 0.28–0.60.

**Verdict.** Two of the three fixes worked:
* the equivalence test makes Stage 2 sound;
* the cohesion term roughly halves the spread error.

The control did its job: it blocked every real-fish claim. The remaining confound is that shared place preferences across fish look like attraction. To separate them, the own model would need each fish's own place map (per-fish fields rather than one shared field), or the virtual groups would need to be built so that place structure is preserved while co-presence is not.

Figures are in `results/e42_figures/` (01–13):
* `12_virtual_gate.png` shows the gate;
* `13_cohesion.png` shows the cohesion fix against e41;
* `09_progress_e39_e42.png` compares all four experiments.

### e43: per-fish place maps, and what is really fooling the control (e43b)

e43 tested e42's diagnosis that shared place preferences look like attraction. Each fish got its own detailed turning map: 25 Gaussian bumps on a 5×5 grid × (sin θ, cos θ), fitted per fish. Social coefficients stayed shared, fitted by exact Frisch–Waugh–Lovell partialling. H1–H7 were hashed before the run.

**Score: 3 of 7 PASS.**
* **H1–H3 PASS:** the method stays sound on known truths.
* **H4 FAIL (2 of 8) and H5 FAIL (2 of 8):** the virtual-group control is exactly as in e42. **The place-preference diagnosis is refuted.**
* **H6 FAIL** (2 of 8) and **H7 FAIL** (spread ratios 1.04–1.41).

The fitted maps (`results/e43_figures/14_place_maps.png`) show each fish's own preferred circulation around the tank.

**e43b, an exploratory diagnostic (not pre-registered), found the actual mechanism.** It was run on the 8-fish groups, comparing forecast bias for "none" and k6-C (`results/e43b_virtual_diagnostic.txt`):
* **Turning is clean.** Fitted social turning coefficients in virtual groups are about 0 (|coef| ≤ 0.005, against 0.08–0.19 in the real schools).
* **Speed is not.** The speed cohesion coefficient is +0.25 to +0.53 in virtual groups (real: +1.2 to +2.7).
* **The no-interaction model is biased.** Its 1 s forecasts inflate spread by +1.2 to +2.1 cm in groups of fish that never met. A fitted "speed up toward the group" term partly cancels this bias, so it wins Stage 1.

The likely cause is that the speed model's place map ignores heading: a fish slows when heading into a wall and speeds up into open water. That is e44's test.

### e44: heading-aware speed maps fix the control

The only change from e43: each fish's speed model gains place × heading terms (bumps × (1, sin θ, cos θ)) and two "heading into the wall" terms. J1–J8 were hashed before the run, including two direct mechanism checks.

**Score: 6 of 8 PASS.**
* **J1–J3 PASS.** Synthetic answers are also sharper than before: T1 → r10-A and T3 → r10-AC exactly, and T4 → k6-AC (the true rule minus repulsion).
* **J5 PASS, mechanism confirmed.** The fake speed cohesion in virtual groups fell from +0.25–0.53 to |coef| ≤ 0.13, and 5 of 6 are < 0.1.
* **J4 FAIL, narrowly.** The spread bias of "none" fell from 1.2–2.1 cm to 0.24–1.04 cm, but only 4 of 6 are below 0.5 (needed 5).
* **J6 PASS (6 of 8; all earlier experiments 1–2 of 8) and J7 PASS.** The gate opens in 6 of 8 recordings, and "none" now passes Stage 2 on all 8 virtual groups. The two remaining failures are:
  * virtual 8 fish #3, which fails the Stage 0 family test (z0 = 4.5) as it has in every experiment;
  * virtual 39 fish #0, which fails at Stage 1.
* **J8 FAIL (3 of 8).** In the other gate-open recordings every rule is still eliminated. Simulated spread remains 1.1–1.25× the observed value.

**First gated rule claims on real fish (exploratory, as pre-registered).** 8 fish #2, #4 and #5 all give the same answer, **k6-AC**: each fish responds to its **6 nearest neighbours** with **alignment plus attraction**, including the distance-growing pull towards their centre. In all three, the paired no-interaction control passes every stage. 8 fish #0 and #1 have open gates but no surviving rule, so nothing is claimed there.

**What the e39–e44 series showed about the method:**
1. On data from a known rule, multi-scale elimination keeps the truth (5 of 5 in e42–e44) and returns precise answers. It rejects an out-of-class process that one-step (micro) selection would accept.
2. Stage 2 must use equivalence margins, not significance tests (e41).
3. A virtual-group control is essential. It caught a real confound, the speed model missing heading. A per-candidate diagnostic of forecast bias and fitted coefficients (e43b) found that confound where the stage-level pass/fail counts could not.
4. The speed-up over conventional search is about 2× when there is something to prune (cost ratio 0.24–0.66 per case). It is a loss when nothing interacts (T0: 1.16–1.27×), and the pooled ratio stayed at 0.52–0.61, just missing the 0.5 target in every run.

Figures are in `results/e44_figures/`:
* `16_mechanism.png` shows the before and after of the diagnosis;
* `15_virtual_control_e42_to_e44.png` compares the control across e42–e44;
* `12_virtual_gate.png` shows which recordings produce claims.

### e45: the interaction-share spectrum. Collective variables found, not chosen

Every earlier experiment *chose* the collective variable (centre of mass, polarization, ...). e45 is given only the parts' trajectories, the full model and the no-interaction twin, and must *find* which collective variables interactions shape and which they cannot touch (the proposal in `idea2.md`, §6).

**Method.** Let e be the one-step prediction errors of the full model on a dictionary of observables, and e₀ those of the twin, both on the interacting data. For a collective variable y = w·ψ:

\[ S(w) = 1 - \frac{w^\top \Sigma w}{w^\top \Sigma_0 w}, \qquad \Sigma = \mathbb{E}[e e^\top],\ \Sigma_0 = \mathbb{E}[e_0 e_0^\top]. \]

The generalized eigenproblem \(\Sigma w = \lambda \Sigma_0 w\), with \(S = 1 - \lambda\), ranks every collective variable in one solve, from most interaction-made to interaction-blind. Weights are fitted on the first half of each run, and every S reported is measured on the second half.

**Theory (Lean, `InteractionShare.lean` Part 4: `left_null_blind`, `columns_zero_total_blind`; standard axioms only).** For linear dynamics A = A₀ + Δ, a collective variable whose weights are a left null vector of the interaction operator (wᵀΔ = 0) gets the same prediction from the full model and the twin from every state. Newton's third law (columns of Δ sum to zero) gives the centre of mass. One-way coupling gives a *weighted* average, π. Numerically, \(\Sigma_0 = \Sigma + \Delta C \Delta^\top\), so S(w) = 0 exactly on that null space.

**Results: 8 of 8 PASS.**

| System | What the method found by itself | Pre-registered check |
|---|---|---|
| Walkers on a chain (e23) | Held-out spectrum matches the exact one (max gap 0.004); most blind = centre of mass (\|cos\| 0.997) | Q1, Q2 PASS |
| Walkers, all-to-all | Centre of mass (\|cos\| 0.9998, S = 0.0001); the other 9 variables all at 0.23 | Q3 PASS |
| One-way coupling, random digraph | The π-weighted average (\|cos\| 0.9996; \|cos\| with the plain average only 0.83); the plain centre of mass has S = 0.184 (exact 0.183) | Q4, Q5 PASS |
| Kuramoto, 30 nonlinear observables | Mean phase (\|cos\| 0.97, S = 0.0008), with ~0 weight on cos and sin | Q6 PASS |
| Agents, alignment only | The group's total turn (\|cos\| 0.9996, S = 0.0006), our Stage 0 theorem rediscovered | Q7 PASS |
| Agents, + attraction | The total turn is no longer blind (S = 0.66) | Q8 PASS |

**Exploratory observations:**
* On the chain, the variables most made by interaction are the zig-zag (shortest-wavelength) modes, found without being given the Laplacian.
* In Kuramoto, interaction is concentrated in **3 of 30** dimensions (S ≈ 0.27, against ≈ 0.03 for the rest). That is a first, unexplained hint at the "fewer variables" question; the most likely candidates are the order-parameter components.
* With attraction, every collective variable of the turns is strongly interaction-made (S 0.66–0.95).

**Scope.** All five systems are simulations, where the twin is a real intervention. On real data the twin must be estimated (as in e24 and e44), and a blind direction is then only as good as that twin.

Figures are in `results/e45_figures/`:
* `02_discovered_blind_variable.png`: discovered weights against theory;
* `03_spectra_linear.png`: spectra against exact values;
* `04_chain_modes.png`: the chain's discovered modes;
* `05_directed_graph.png`: the directed graph;
* `06_kuramoto.png` and `07_agents.png`: the nonlinear systems.

### e46: the spectrum with an estimated twin, then real fish

e45 knew the true model and twin. e46 estimates both from the data alone, then applies the method to real fish schools.

**Twin from data.** We use e24's convention: interactions act on the parts' *relative* states, so the "own" part of a fitted full model is the twin.
* **Linear model.** The twin is the row sums of the fitted VAR.
* **Kuramoto.** Fit each oscillator's frequency plus sine coupling; the twin is the frequency alone.
* **Agents.** Fit shared, distance-binned alignment and attraction coefficients plus a bias for each agent; the twin is the bias.
* **Fish.** e44's k6-AC model (per-fish turning and speed maps, shared social terms), fitted on 70% of each 8-fish recording. The twin is that fit with the social terms removed. The dictionary is the 8 turns and 8 speed changes.

Each run is split three ways: one part fits the models, one fits the spectrum weights, and every S reported comes from the remaining held-out part. The control is e24's trap: a *naive* twin that regresses each part on its own past only.

**Results: 7 of 7 PASS.**

| Check | Result |
|---|---|
| R1. One-way coupling, data-only twin | Most blind = the π-weighted average (\|cos\| 0.998, S = 0.0001) |
| R2. Same, short data (5 000 steps × 20 runs) | Median \|cos\| 0.985 (min 0.931) |
| R3. Naive twin | Wrong variable (\|cos\| 0.19); it absorbs the coupling |
| R4. Kuramoto, data-only twin | Mean phase (\|cos\| 0.982, S = 0.0005) |
| R5. Agents, data-only twin | Alignment only: total turn is blind (\|cos\| 0.9995). With attraction: S(total) = 0.66 |
| R6. Fish, 6 virtual groups (never met) | Largest S ≤ 0.009 in all 6 |
| R7. Fish, 6 real schools | Largest S 0.11–0.16 in all 6, above the virtual pair in 6/6 |

**What real schools look like (exploratory):**
* **Interaction lives in the turns.** All top collective variables are combinations of turns; speed changes get almost no weight. This matches e44, where the fitted speed cohesion term shrank to almost nothing once wall effects were modelled.
* **About 7 interaction-made dimensions out of 16.** In every real school, S stays at about 0.07–0.16 for the first ~6–7 variables, then drops to ≈ 0.02. There are 7 independent *differences* between 8 fish's turns, which fits an alignment-like interaction that acts on relative headings. This is an exploratory reading, not a tested claim.
* **The group's total turn is nearly interaction-blind.** S(total turn) is only 0.00–0.03 (figure 06), and the most-blind variable is mostly a positively weighted sum of turns. The Lean theorem guarantees exact blindness only for symmetric interactions. Fish attend to a one-sided neighbourhood (k6), so S is small but not exactly zero.
* **Caveat.** The fish twin is the model's own part, so the result is only as good as e44's model. The virtual-group null shows the method does not invent interaction where none exists. It cannot show that the real twin is right.

Figures are in `results/e46_figures/`:
* `01_scorecard.png`: the scorecard;
* `02_linear_estimated_twin.png`: π against the data-only twin and the naive twin, plus short-data runs;
* `03_nonlinear_estimated_twin.png`: Kuramoto and the agents;
* `04_fish_spectra.png`: real schools against virtual groups;
* `05_fish_modes.png`: the most interaction-made and most blind variables of each school;
* `06_fish_totals.png`: S of the total turn and total speed change.

### e47: closed layers. When is a set of discovered variables a level of description?

The spectrum (e45, e46) ranks collective variables by how much interaction shapes them. A *layer of organization* needs more: a small block of variables whose future can be forecast from the block's own past as well as from the full micro state. Then the parts can be forgotten.

**Two measures for a block of collective variables at horizon h:**
* **Closure gap G(h)** = 1 − (error of the best forecast from the full micro state) / (error of the best forecast from the block's own last 10 values). G = 0 means closed. Both errors are measured on the same scale, after whitening by the block's change covariance.
* **Interaction share at horizon h, S_h**: the same ratio, but comparing the full model's forecast with the twin's. One-step S (e45) is S_1.

An **emergent layer** is closed *and* has S_h > 0 at its own timescale. An **inherited** layer is closed only because interactions never touch it, like the centre of mass.

**Theory (Lean, Part 5).** Three theorems:
* `closed_iter`: a closed description stays closed on every horizon.
* `left_invariant_closed`: a left-invariant linear block (WᵀA = MWᵀ) is closed.
* `closed_comp`: a closed description of a closed description is a closed description of the parts, so layers stack.

`closed_iter` and `closed_comp` use no axioms; `left_invariant_closed` uses the standard ones. For interactions organized by scale, the layer above each scale is the left null space of the stronger interactions: the variables those interactions cannot touch (Simon's near-decomposability).

**Systems.**
* **Hierarchy of 12 walkers:** 2 super-modules × 2 modules × 3 walkers. Coupling is one-way and strong inside modules, weak (0.04) between modules, and weaker still (0.006) between super-modules.
* **Modular Kuramoto:** 3 modules × 4 oscillators. The dictionary contains only cos θ and sin θ.

**Settings were fixed on exact computations before any simulation** (docstring). The first design used one global coupling scale, which left some modules barely coupled inside. It was replaced by per-module scaling, and the leak was lowered so that interaction, not the leak, drives the slow layers.

**Results: 10 of 10 PASS.**

| Check | Result |
|---|---|
| H1. Held-out spectrum vs exact | Max gap 0.002 |
| H2. A fixed gap rule (next S ≥ 4× the current one) finds the layers | Layers of size 1, 2, 4: global, super-modules, modules, exactly as built |
| H3. Discovered layers = true layers | Overlap 0.995 (modules), 0.992 (super-modules), 0.997 (global) |
| H4. Discovered layers are closed | Max G 0.013 (modules), 0.019 (super-modules); exact ≤ 0.003 |
| H5. Not every block is a layer | Plain (unweighted) module averages G(1) = 0.24; 4 most interaction-made G(1) = 0.47 |
| H6. Emergent vs inherited | Module layer S_20 = 0.21, super-module layer S_100 = 0.08, global S_h ≤ 0.002 |
| K1. Oscillators: 6 most blind = module order parameters | Overlap 0.961; the gap rule finds the cut at 6 |
| K2. That block is closed | G ≤ 0.034 at all horizons |
| K3. More closed than alternatives | Top-6 G up to 0.16; random 6-dim blocks median up to 0.13 |
| K4. Emergent | One-step S = 0.007, but S_100 = 0.49 |

**What this shows.**
* **Layers can be found from a flat list of trajectories.** The spectrum's gaps mark the layers, and the variables below each gap form a closed description. The method recovered a three-level hierarchy without being told the modules.
* **Emergent layers are invisible at one step.** The module layer has one-step S of only 0.03 (the oscillators' layer 0.007), yet at its own timescale interaction shapes 21% (49% for the oscillators) of its dynamics. Interaction share must be measured at the layer's timescale.
* **The intuitive coarse-graining is wrong under one-way coupling.** Plain module averages are interaction-made but not closed (G(1) = 0.24). The closed layer uses π weights that nobody specified.
* **The most interaction-made variables are not a layer.** They need the parts (G up to 0.47).

**Caveats.**
* **Long-horizon closure estimates are biased upward** by finite data: about 0.01–0.02 at h = 100. Even the exactly closed global variable reads 0.023. The micro forecaster uses the true model, while the block forecaster is fitted. H4 passed despite this, but the super-module layer (0.019) was close to its 0.02 threshold.
* **The oscillator modules are almost fully synchronized** (r = 0.99), so each module behaves nearly like one oscillator. This makes the layer an easy case; a weaker-synchrony test would be harder.
* **Both systems are simulated with known twins.** Real data would need e46's estimated twin plus a fitted micro forecaster.

Figures are in `results/e47_figures/`:
* `02_hierarchy_spectrum.png`: the spectrum and its gaps;
* `03_hierarchy_layers.png`: discovered against true layer weights;
* `04_hierarchy_closure.png`: G and S_h against horizon;
* `05_kuramoto_layer.png`: the oscillators;
* `06_layer_map.png`: closure against interaction share for every block.

### e48: does schooling create a closed group-level layer? Real fish

This applies e47's questions to the six 8-fish recordings, with the virtual groups (the same fish time-shifted so they never met) as the control. The dataset has only 8- and 39-fish groups, so a 4/6/8-fish group-size test is not possible.

**Set-up.**
* **State:** every fish's x, y, cos(heading), sin(heading) and speed; 40 variables at 0.1 s steps.
* **Group block:** the school's centroid, mean heading vector and mean speed (5 variables).
* **Closure G, from data alone:** compares least-squares forecasts from the block's own past with forecasts that also use all fish's states. Both are fitted on the first 70% and measured on the last 30%.
* **Interaction share S_h and the 1 s spectrum:** e44's k6-AC model is run forward as an ensemble with and without its social terms (the estimated twin, as in e46).

**Results: 2 of 5 PASS.**

| Check | Result |
|---|---|
| F1. The 5 most blind discovered variables = the group block | **FAIL.** Overlap 0.00–0.01 |
| F2. Group block nearly closed (G ≤ 0.1) in real schools | PASS. G −0.04 to +0.003 at 1 s and 3 s |
| F3. Group block more closed in real than virtual groups | **FAIL.** Virtual groups are just as closed (G −0.03 to −0.14) |
| F4. Group block emergent: S_3s ≥ 0.05 real, ≈ 0 virtual | **FAIL.** Virtual ≈ 0 as expected, but real is *negative* (−0.15 to −0.40) |
| F5. A single fish is less closed than the group | PASS. G(one fish) 0.14–0.20 at 1 s |

**Why the predictions failed (exploratory diagnostic, e48b).**
* **The model's social terms make the school's *shape* right but its *path* worse.** Three seconds ahead, the model with social terms keeps the school together: polarization 0.93/0.91 against 0.95/0.89 observed, spread 6.7/7.6 cm against 5.1/6.9. The twin scatters the fish (polarization 0.47/0.54, spread 20–22 cm). Yet the twin forecasts the school's centroid and mean heading as well or better. Each fish's own place-and-heading map, fitted on schooling data, already encodes where the school tends to go. Adding the social terms makes the simulated fish turn coherently, but at the wrong times. So S_h of the group block is negative, and the "most blind" end of the spectrum is where the social terms *hurt*, not where they do nothing (F1).
* **Shape is not in the dictionary.** Polarization and spread are nonlinear functions of the states, so no weighted sum of the 40 variables can represent them. This is the "measurements are still chosen" limitation.
* **The data-only closure test cannot separate real from virtual groups (F3).** The all-fish forecaster is linear in the fish's states. It cannot exploit an individual fish's nonlinear wall-following, so an average of independent fish also looks closed.

**What did work.**
* **Interaction breaks the individual fish's closure.** In virtual groups, one fish's own past forecasts its future as well as all 8 fishes' states do (G −0.01 to −0.07). In real schools, it does not (G 0.14–0.20 at 1 s). Closure detects interaction at the level of the individual. Only the comparison with the group block was pre-registered; the comparison with virtual fish is exploratory.
* **The interaction-made variables are 2(N − 1) relative headings (exploratory).** The 1 s spectrum of every real school has a sharp step. Above it sit **14** variables in 5 of 6 schools (15 in #1), with S 0.24–0.66; virtual groups stay ≈ 0. Their weight is 98–99% on the heading vectors, and 97–99.5% of that weight is relative to the school's mean heading. Each of 8 fish has a 2-D heading vector, and 2 × (8 − 1) = 14: alignment acts on headings relative to the group. This matches e46, which found 7 = 8 − 1 interaction-made turn combinations.

**Next.** The 39-fish recordings allow a pre-registered group-size test: the step should fall at 2 × 38 = 76 relative-heading variables. Separately, shape variables (polarization, spread) should go into the dictionary before the group-layer question is asked again.

Figures are in `results/e48_figures/`:
* `02_closure.png`: closure of the group block, one fish and random blocks;
* `03_interaction_share.png`: S_h of the group block, real against virtual;
* `04_spectra.png`: the 40-variable spectra with the step;
* `05_blind_weights.png` and `06_layer_map.png`.

The diagnostic text is in `results/e48b_fish_layer_diagnostic.txt`.

### e49: relative headings across group size, and the school's shape as a layer

e49 runs both next steps from e48, as pre-registered tests.
* **Part A:** the heading-only spectrum (every fish's cos and sin of heading) in the 8- and 39-fish schools.
* **Part B:** the school's state as a layer: centroid, mean heading vector, mean speed, and the shape variables polarization, spread and milling. Closure is tested against what the fish's *own* tendencies (their individual place-and-heading maps) add. This fixes e48's linear test, which could not separate real schools from virtual groups.

**Results: 3 of 6 PASS.**

| Check | Result |
|---|---|
| P1. 8 fish: top block = 14, and the 2 most blind = mean heading | **FAIL.** The top block is 14 in 5/6 (15 in #4), and 99.7–99.9% of its weight is relative to the mean heading. But the 2 most blind directions only partly match the mean heading (0.37–0.54) |
| P2. 39 fish: top block = 76 | **FAIL.** #0: 77 (one direction with S = −0.17 below the rest). #1: no clear step; the spectrum is graded (0.04–0.30) |
| P3. Shape interaction-made in real, ≈ 0 in virtual | **FAIL** on the control only. Real schools: polarization S 0.65–0.96, spread 0.62–0.99 (6/6). Virtual polarization ≈ 0, but virtual spread varies by ±0.2 (1/6 within ±0.05) |
| P4. School state closed against the fish's own tendencies | **PASS.** G −0.10 to −0.01 (6/6) |
| P5. Control: real more closed than virtual | **PASS.** Virtual G 0.04–0.19 (6/6) |
| P6. Single fish need the others in schools, not in virtual groups | **PASS.** Median G 0.14–0.18 real against −0.05 to −0.02 virtual (6/6) |

**What this establishes about real fish.**
* **The school is a closed unit, and the individual is not.** In real 8-fish schools, knowing each fish's own tendencies adds nothing to forecasting the school's state 1 s and 3 s ahead. In virtual groups it does (G up to 0.19): their "school" is just an average. Conversely, a single fish in a school cannot be forecast from its own past (G ≈ 0.16), while a virtual fish can. Interaction moves the self-contained unit from the fish to the school.
* **The school's shape is made by interaction.** Polarization and spread forecasts 3 s ahead are 80–99% better with the social terms. The model matches observed polarization and spread (figure 05), while the twin's school dissolves (polarization ≈ 0.5, spread about 20 cm against about 6 cm).
* **Together, this is the first real-data case of an emergent layer in this project:** a closed school-level description whose shape is interaction-made. The control behaves as it should, except for the virtual spread noise in P3.

**What did not hold.**
* **Group-size test.** The 2(N − 1) relative-heading step is clean in 8 fish but does not carry over cleanly to 39 fish. One recording shows 77, the other a graded spectrum. With 6 nearest neighbours out of 38 others, interaction is local, and local coupling gives a graded spectrum, as on e45's chain. In 8 fish, 6 of 7 others is nearly all-to-all, which gives a step.
* **The mean heading is not the most blind variable.** Here, as in e48, the most-blind end holds the directions where the model's social terms *hurt* (negative S).
* **Virtual spread is not exactly zero.** It varies by ±0.2, because the small remaining social coefficients act on widely spread virtual groups.
* **In 39 fish, the model's spread forecasts are worse with interactions** (S −0.05/−0.25). Polarization is still strongly interaction-made (0.84/0.71).

**Scope of the closure claim.** P4–P5 test closure against the fish's own tendencies, not against the school's internal configuration (who is where). A school could still need its internal configuration to forecast its shape.

Figures are in `results/e49_figures/`:
* `02_heading_spectra.png`: 8- and 39-fish heading spectra;
* `03_shape_interaction_share.png`: S of polarization, spread and milling;
* `04_closure.png`: school-state and single-fish closure;
* `05_shape_forecasts.png`: observed against full and twin forecasts.

### e50: the school's internal configuration, and fish at the front steer

e50 asks whether the school's internal configuration helps forecast the school's state beyond that state itself (e49's open question). Configuration means where each fish sits in the school's frame, its heading and speed relative to the school, and nearest-neighbour distances, summarized by 20 features that do not depend on fish identity, plus their squares. It also tests a specific mechanism, leadership: L = mean(x′ sin φ)/spread, positive when fish *ahead* of the centroid are heading to the left of the mean. The test is whether L predicts the school's heading change over the next 1 s.

**Results: 2 of 4 PASS.**

| Check | Result |
|---|---|
| C1. Configuration helps forecast shape (G ≥ 0.05) | **FAIL.** G −1.0 to +0.02 in real schools |
| C2. Configuration helps forecast motion (G ≥ 0.02) | **FAIL.** G −1.2 to −0.01 in real schools |
| L1. Front fish steer (coefficient > 0, t ≥ 3) | **PASS.** t 11–22 in all 6 real 8-fish schools; 22 and 10 in the two 39-fish schools |
| L2. Control: virtual groups show no such effect | **PASS.** Virtual \|t\| ≤ 2.9 (6/6); the real effect is larger than its virtual pair 6/6 (standardized 0.25–0.39 against −0.05 to 0.04) |

**Why C1 and C2 failed (exploratory, e50b).** The 40-feature configuration regression overfit, badly in two recordings (#3: G −1.0 to −1.2). A single targeted feature, L, still improves held-out forecasts of the school's heading by 4–14%. That holds after controlling for the school's last five turning rates, polarization and spread (t 10–18). So the configuration *does* carry real information about the school's motion; the pre-registered test was too blunt to see it. In virtual groups the same feature set gained 1–7%, from independent fish's own movements.

**What this means.**
* **Leaders exist, and the school's turns start at its front.** The school's heading change can be predicted from how the front fish are already heading, in every real school of both sizes, but not in virtual groups. This is information flowing through the school, which no school-level variable carries.
* **The school-level layer is only approximately closed.** e49's "closed against the fish's own tendencies" still holds as tested. But "closed" is always relative to the forecaster used. A weak forecaster (e48) or an overfitting one (e50's C1–C2) can make a block look closed when it is not. The leadership statistic shows that the school's motion depends partly on its internal arrangement.
* **Method lesson:** closure claims need a forecaster that is both flexible and regularized, and a positive control showing it can detect a known dependence. Real versus virtual is not enough on its own, as e48 and e50 both show.
* **Speed leadership needs a baseline (exploratory).** The statistic "front fish are faster" is +4.8 to +9.8 standard errors in 5 of 6 real 8-fish schools, but strongly negative in virtual groups (−10 to −23) and in both 39-fish schools (−3 and −7). Its geometric baseline is far from zero, so it needs a better control before being read as leadership.

Figures are in `results/e50_figures/`:
* `02_configuration_closure.png` and `03_per_variable.png`: the configuration closure;
* `04_leadership.png`: the leadership tests.

The diagnostic is in `results/e50b_leadership_diagnostic.txt`.

### e51: a closure test we can trust, then the school again

e48 and e50 showed that "closed" depends on the forecaster, so e51 builds a test that must first prove itself. The forecaster is ridge regression. Its penalty, and a separate scale on the added features, are chosen by blocked cross-validation inside the training data. The scale grid includes 0, so the extended model contains the block model and can decline features that do not help. Configuration features are tamed by asinh((x − median)/MAD), because some have tails 100× their spread. Uncertainty comes from a moving-block bootstrap (30 s blocks).

Before the test's answer about schools counts, it must pass two checks on itself:
* **Calibration:** a signal of known size is planted in the target. That signal is a 5% share of the error, built from the part of a configuration feature the block cannot express, so G should rise by about 0.05.
* **Null control:** the configuration features of a *different recording* are added. They should give G ≈ 0.

Development note: the method was fixed on recording 0 only, after the first version failed both checks; the pre-registered predictions were not changed. The note in the e51 code describes the failures.

**Results: 3 of 6 PASS.**

| Check | Result |
|---|---|
| K1. Calibration: planted signal raises G ≥ 0.025, lower bound > 0 | **PASS.** Heading change 11/12, spread change 10/12 groups |
| K2. Null features: \|G\| ≤ 0.01 | **FAIL.** turn 11/12, motion 10/12, shape 9/12. Real schools ≤ 0.015 throughout; virtual #1 and #2 up to +0.05 (see below) |
| K3. Leadership through the general test: G(turn) ≥ 0.03, lower bound > 0 | **PASS.** 6/6 real, G 0.06–0.21 (39 fish: 0.10, 0.05) |
| K4. Configuration informs shape: G(shape) ≥ 0.05 | **FAIL.** 0/6; G −0.01 to +0.03 (lower bound > 0 in 4/6) |
| K5. Configuration informs motion: lower bound > 0 | **FAIL.** 3/6; G 0.000 to 0.019 |
| K6. Real more self-contained than virtual: G(all) real < virtual | **PASS.** 5/6; real 0.002–0.023 against virtual 0.018–0.046 |

**Why K2 failed (post hoc).** The two failing nulls are exactly the pairs of recordings with equal length (#1 with #2, #2 with #3, all 89,999 frames). Virtual groups shift fish i by i·T/8. For recordings of equal length, the seams therefore fall at the same times, and the null configuration carries "which eighth of the recording" information. That leaks the slow drift of the virtual group. Rerunning those two groups with a null from a recording of a different length gives G = 0.000 in all three targets. The test's false-positive problem is therefore specific to how virtual groups are built, not to the forecaster. In real schools the null never exceeded 0.015 in size. K2 still counts as FAIL as registered.

**What this means.**
* **The test works.** It sees a planted 5% signal, mostly ignores features that carry no information, and does not produce e50's large negative G values. Plain least squares on the *tamed* features stays within −0.05 of the ridge result. This suggests e50's G of −1.2 came mostly from heavy-tailed features, not just from their number.
* **The school is closed except for turning.** With a test that can see 5% signals, the internal configuration adds at most 3% to forecasts of the school's shape, and at most 2% to its centroid, mean heading and speed. Turning is the exception: the configuration adds 6–21% to the 1 s forecast of the school's heading change in every real school. By 3 s that information is almost gone (G −0.01 to +0.02). So the school is a closed layer for what it *is* (shape, motion) on these timescales. It is not closed for *where it turns next*, and that information lives inside the school for about a second.
* **Leadership is part of it, not all of it.** Removing the two leadership features (x′ sin φ and L) lowers G(turn) by 13–56%. The rest of the turning information sits in other configuration features.
* **Real schools are more self-contained than independent fish.** In 5 of 6 pairs, the configuration adds less to forecasts of the real school than to forecasts of its virtual counterpart.

Figures are in `results/e51_figures/`:
* `02_test_checks.png`: calibration and the null control;
* `03_versus_e50.png`: least squares against cross-validated ridge on the same features;
* `04_configuration_closure.png`: G with 95% intervals for every target, horizon and group;
* `05_leadership_share.png`: G(turn) with and without the leadership features.

*Checked by e78:* with the test 30% moved to the start or the middle of each recording (every fitting stage moved with it), K3–K6 keep their verdicts. G(turn) is 0.09–0.25 and 0.08–0.21. Shape stays below 0.05 in 11 of 12 school-segments (one 0.066), and motion stays below 0.005. The calibration K1 is marginal: it would fail in both other segments (spread 9/12 and 8/12).

### e52: a learned influence map. What inside the school tells it where to turn?

e51 left one opening: the school's next turn. e52 stops choosing features and *learns* an influence function. Each fish contributes f(its state). The state is its place in the school's frame (in units of the school's spread), its heading relative to the school (sin φ, cos φ), its own recent turning ω, its relative speed u, and u·sin φ. The school's turning variable is the mean over fish. f is a sum of Gaussian bumps over the school's frame, one map per channel, made mirror-symmetric (a mirrored left turn is a right turn). That gives 75 features, linear in the weights, fitted with e51's calibrated ridge, so the weights can be drawn as maps. The null control now uses a recording of *different* length, to avoid e51's seam artifact.

**Results: 4 of 5 PASS.**

| Check | Result |
|---|---|
| D0. The test works here (planted signal seen; null \|G\| ≤ 0.01) | **PASS.** planted 12/12, null 12/12 |
| D1. The learned map at least matches the hand-made configuration | **PASS.** 6/6 real: G 0.08–0.25 against 0.06–0.21 (39 fish: 0.15, 0.15 against 0.10, 0.05) |
| D2. It adds ≥ 0.02 beyond the configuration, lower bound > 0 | **FAIL.** 4/6; gains 0.006–0.082 (#0 adds nothing; #3's interval includes 0) |
| D3. Front fish lead: the heading map is positive at the front and larger than at the back | **PASS.** 6/6, and both 39-fish schools |
| D4. It is interaction, not individual persistence: real > virtual | **PASS.** 6/6; virtual 0.004–0.067 |

**What the maps show** (`04_influence_maps.png`, `05_per_school_maps.png`; the same pattern holds in every real school of both sizes):
* **The front steers, and the rest follows.** A fish at the front heading left of the school predicts a left turn. A fish at the back or sides heading left predicts the *opposite*: it is a follower being pulled back into line. Virtual groups show no front–back structure.
* **Speed matters most.** Of the four parts of a fish's state, its relative speed combined with its heading is by far the most informative. On its own it adds 0.06–0.22 (virtual: about 0.01), against 0.03–0.11 for heading alone. Its own recent turning and its place in the school add almost nothing. A fast fish heading off the school's line, anywhere in the front two-thirds of the school, predicts where the school turns. A slow fish doing the same does not.
* **Turning fish at the back are correcting, not leading.** In the turning map, a fish turning left at the front predicts a left turn of the school, but a fish turning left in the middle or back predicts a right turn.
* **The information is short-lived.** At 3 s the gains mostly vanish (G ≤ 0.04 in real schools), as in e51.

**What this means.** The school's next turn is carried by a small, identifiable sub-population: fast fish at the front that are heading off the school's line. This variable was found, not chosen. It is the same across eight independent schools and two group sizes, absent in virtual groups, and stronger than the leadership statistic chosen by hand in e50. The school-level layer is closed for its shape and motion (e51). For its direction it is open, through about one second of information held by its leading, fastest members.

Figures are in `results/e52_figures/`:
* `02_test_checks.png`: calibration and the null control;
* `03_comparison.png`: hand-made configuration, learned map, and the gain beyond the configuration;
* `04_influence_maps.png` and `05_per_school_maps.png`: the learned maps;
* `06_channels.png`: which part of a fish's state carries the information.

### e53: does the influence map replicate in zebrafish and pigeons?

e53 ran e52's method unchanged, apart from units, on two independent data sets:
* **Zebrafish (Heras et al. 2019):** nine 10-minute recordings of 60, 80 and 100 juveniles, at 0.094 s per step.
* **Pigeons (Nagy et al. 2013):** three 10-bird flocks of homing pigeons in free flight, at 0.2 s per step.

The horizon is 1 s. Virtual groups shift each individual's own track in time. Calibration and null control are as in e52.

**Results: 3 of 9 PASS, and one of those is empty.**

| Check | Result |
|---|---|
| R0. The test works (planted ≥ 20/24 and null ≥ 20/24) | **FAIL.** Planted 16/24, null 23/24 (see below) |
| Z1. Zebrafish: G(turn) ≥ 0.03, lower bound > 0 | **FAIL.** 3/9; G 0.02–0.13, wide intervals |
| Z2. Zebrafish: real > virtual | **FAIL.** 6/9 (needed 7) |
| Z3. Zebrafish: front fish lead (heading map front > 0 and > back) | **PASS.** 9/9 |
| Z4. Zebrafish: speed channels ≥ heading channels | **PASS.** 8/9 |
| P1. Pigeons: G(turn) ≥ 0.03 | **FAIL.** 0/3; G = 0.000, 0.002, 0.000 |
| P2. Pigeons: real > virtual | **FAIL.** 0/3; virtual 0.16–0.57 |
| P3. Pigeons: front birds lead | **FAIL.** 1/3 (the maps are zero) |
| P4. Pigeons: speed ≥ heading | **PASS, but empty**: both are 0.000 |

**Why the calibration failed in 8 groups.** The failures fall into three groups:
* **Zebrafish, five groups (all three 80-fish groups among them):** the planted signal raised G by roughly the expected amount, 0.01–0.06, but the 95% intervals reach −0.06. Ten-minute recordings of large, often milling groups leave the test too little power to see a 5% effect.
* **Pigeon virtual flocks A and B:** these already have G ≥ 0.16. The expected rise, 0.05 × (1 − G), falls below the fixed 0.025 bar at A's G of 0.57, and stays near it at B's 0.16. Here the pre-registered threshold was the problem, not the test.
* **Pigeon flock C (real):** the test did not see the planted signal, and I have no explanation.

**What replicates and what does not.**
* **Zebrafish: the structure replicates; the strength is uncertain.** In 9 of 9 groups the learned heading map has e52's shape: front fish heading left predict a left turn, side and back fish the opposite. Speed combined with heading carries the most information in 8 of 9. The predictive gains (G 0.02–0.13) are positive in every real group, but the intervals are too wide for confirmation, so Z1 failing is a lack of power, not a negative result. Virtual zebrafish maps show no front–back structure.
* **Pigeons: the flock's turn is closed.** The flock's own last second of turning already forecasts its next second so well that nothing inside the flock adds anything (G = 0.000). The test could detect a planted signal in two of the three flocks, so this is a real negative result there. Pigeon flocks are highly polarized (0.93–0.98), the GPS tracks are smoothed over 0.4 s, and Nagy's leader–follower delays are at most 0.6 s. Within one second, a leader's turn is therefore already visible in the flock's own turning rate.
* **Virtual flocks are not a clean baseline for pigeons.** In loosely aligned virtual flocks, each bird's own smooth turning predicts the mean heading mechanically, giving G up to 0.57. In a real, tightly aligned flock the flock's own turning rate already contains that.

**What this means.** The "front individuals choose the turn" structure appears in two fish species, in groups of 8 to 100. Its predictive strength is clearly established only in the Puy et al. schools. Pigeon flocks at this resolution are closed for their turn: whatever leadership exists acts faster than one second and shows up in the flock's own motion. A pigeon test would need a sub-second horizon on unsmoothed data. A zebrafish test needs longer recordings.

Figures are in `results/e53_figures/`:
* `04_influence_maps.png` and `05_per_group_maps.png`: the maps in both species;
* `03_comparison.png`: G for each group;
* `02_test_checks.png`: calibration and the null control.

*Revised by e78:* the zebrafish strength depends on which part of the recording is tested. Tested on the first 30%, Z1 passes in 8 of 9 groups (G 0.03–0.19), Z2 in 9 of 9, and the calibration R0 passes (20/24). In the middle it is 2 of 9. "A lack of power" is better read as "time-dependent": early in the recordings, the turning structure is also strong. The front-leads structure (Z3) and the pigeon negative (P1) hold in every segment.

### e54: do levels close in their stable variables and stay open along neutral directions?

The white paper (Section 7.6) proposed an explanation for e51–e52: a school's shape is held in place by interaction and so is closed, while its heading is neutral (every direction equally good) and so whatever is inside can steer it. e54 tested this by intervention in two simulated systems with known rules:
* **A flock:** 30 agents in open space with nearest-neighbour alignment and attraction, cohesion, repulsion, and heterogeneous speeds with speed matching.
* **Oscillators:** 30 noisy Kuramoto oscillators with heterogeneous natural frequencies, strongly synchronized (r ≈ 0.9).

An external field of strength f ∈ {0, 0.05, 0.1, 0.2, 0.4} pulls every member towards one direction. This turns the group's direction from a neutral variable into a stable one. There are two seeds per setting. The test is e51's calibrated closure gap at one time unit:
* **Block:** the group's state and its recent history.
* **Extra:** the arrangement inside (e52's influence features for the flock; phase and phase-velocity arrangement for the oscillators).
* **Targets:** the group's turn and its stable variables (polarization and spread; r).

The simulation regime was tuned on dynamics only (cohesion, polarization, how strongly the field pins the direction) before the predictions were written.

**Results: 3 of 10 PASS. The hypothesis is not supported.**

| Check | Flock | Oscillators |
|---|---|---|
| 0. The test works (planted ≥ 8/10, null ≥ 8/10) | **PASS** (10/10, 10/10) | **PASS** (10/10, 10/10) |
| 1. Direction open without field: G(turn) ≥ 0.03, lower > 0, 2/2 | **FAIL** (0.093; 0.011) | **FAIL** (0.007; 0.043) |
| 2. Stable variables closed: G ≤ 0.02 in ≥ 9/10 | **PASS** (10/10, all 0.000) | **FAIL** (3/10; G(r) up to 0.23) |
| 3. Pinning closes the direction: G(turn, f=0.4) ≤ G(turn, f=0)/3 | **FAIL** (0.052 → 0.030) | **PASS** (0.025 → 0.000) |
| 4. Spearman(field, G(turn)) ≤ −0.7 | **FAIL** (−0.28) | **FAIL** (−0.62) |

**What happened.**
* **Flock: the turn stays open when the direction is pinned.** The field cuts the slow wandering of the heading by a factor of 15 (sd 1.5 → 0.10 rad). It barely changes the size of one-unit turns (sd 0.08 → 0.07), and the arrangement still adds G ≈ 0.03–0.06 to forecasting them. Polarization and spread are closed at every field. So the flock is "closed in its stable variables, open in its turn" with or without a neutral direction. The openness of the turn does not come from neutrality.
* **Oscillators: the stable variable r is open, and the seeds differ.** At weak fields the arrangement adds G = 0.04–0.23 to forecasting r. Strong pinning closes it. A post-hoc check (not pre-registered) shows why: G(r) follows the number of phase slips. Slips are times when one or two oscillators that are not locked to the group fall a whole cycle behind. Seed 1 has 151 slips at f = 0 (G 0.23), falling to 2 slips at f = 0.4 (G 0.01). Seed 0 goes from 24 slips (G 0.10) to 1 (G 0.00). The rank order of G(r) and slips is almost perfect over all ten runs. The field closes r by locking the stragglers, not by pinning a neutral direction.

**What this means.** The tidy rule "stable variables close, neutral directions stay open" is false in both systems. It fails in opposite ways: a turn that stays open in the flock even when pinned, and a stable variable that is open in the oscillators. What the open cases share (stated after the fact, so a hypothesis only) is **a few members that are not slaved to the collective**: fast front individuals in the flock and in real fish (e52), unlocked oscillators here. Whatever such members do is not yet written into the group's state, so it is "inside information" for any variable it affects. When every member is slaved, the level closes. This is a sharper and testable version of Section 7.6. The next experiment would control the number of free members directly (for example, the spread of natural frequencies in the oscillators, or a subset of informed agents in the flock), with no field.

Figures are in `results/e54_figures/`:
* `02_main_result.png`: G for the turn and for the stable variables against field strength;
* `04_pinning.png`: how strongly the field pins the direction;
* `03_test_checks.png`: calibration and the null control.

### e55: is a level open exactly where some of its members are not locked to it?

e54's open cases shared a few members not slaved to the collective. e55 tested that directly, with no field, by setting the number m of free members out of 30 to 0, 1, 2, 4 or 8:
* **Oscillators:** free members have natural frequencies far outside the locking range and drift through the group. The rest are tightly locked, with about one slip per run.
* **Flock:** free members also steer towards private goal directions that wander slowly. No one else sees these goals (informed individuals).

The test, block and arrangement features are unchanged from e54. Two extra models use only the free members' arrangement, or only the locked members'. Two seeds per m. The simulation regime was checked on dynamics only.

**Results: 5 of 10 PASS. Strong support in the oscillators; none in the flock.**

| Check | Flock (target: turn) | Oscillators (target: r) |
|---|---|---|
| 0. The test works (planted ≥ 8/10, null ≥ 8/10) | **PASS** (10/10, 10/10) | **FAIL** (planted 7/10, null 9/10) |
| 1. Spearman(m, G) ≥ 0.7 | **FAIL** (+0.39) | **FAIL** (+0.02) |
| 2. Mean G at m = 8 ≥ mean at m = 0 + 0.03 | **FAIL** (0.036 → 0.053) | **PASS** (0.005 → 0.258) |
| 3. G(free members' features) ≥ G(locked members') in ≥ 7/8 | **FAIL** (2/8) | **PASS** (8/8) |
| 4. Oscillators: G(r) ≤ 0.02 at m = 0 / flock: shape closed in ≥ 9/10 | **PASS** (10/10, all 0.000) | **PASS** (0.005, 0.005) |

**Oscillators: one loose member opens the level.**
* **Closed with nothing free:** with no free member, r is closed (G = 0.005 in both seeds), and so is the group's phase drift (0.000).
* **Opened by a single free member:** one free oscillator out of 30 opens r to G = 0.31–0.34, and that information sits mostly in the free oscillator's own phase (8/8).
* **Not a dose–response for r:** r does not keep opening as more members come loose. It jumps at m = 1 and then sits at 0.25–0.31. So Check 1 fails on the pre-registered target.
* **The phase drift does scale:** G rises steadily, 0.00 → 0.08 → 0.29 → 0.46 → 0.61 over m = 0, 1, 2, 4, 8. That variable was not the pre-registered target, so this is an observation, not a pass.
* **Calibration:** it failed in 3 runs. All three already had G(r) ≥ 0.25, where the expected rise, 0.05 × (1 − G), is about 0.035 and close to its noise. This is the same threshold problem as in e53.

**Flock: privately informed members do not behave like loose oscillators.**
* **Already open with nothing free:** with no informed member, the turn is already open (G = 0.036 in both seeds, as in e54).
* **Little effect of informed members:** adding up to 8 raises it only a little and not consistently (0.04–0.09).
* **The information is mostly in the others:** the uninformed members' arrangement carries more than the informed members' own in 6 of 8 runs.

A likely reason (stated after the fact): an informed agent's private pull spreads to its neighbours within one time unit, so by forecasting time it is written into the others' headings. The flock's baseline openness also has a source in every member: individual cruising speeds, which make each agent partly "free" in speed (the speed channel carried the turn in e52).

Polarization and spread stay closed throughout (all G = 0.000).

**What this means.** "Open where members are not locked" holds cleanly where "locked" is crisp. An oscillator is either phase-locked or drifting, and one drifting oscillator out of 30 opens the group, with the information sitting in that oscillator. In the flock, freedom is graded and contagious. Every member is partly free (its speed), and an informed member's freedom passes to its neighbours within the forecast horizon. There, "which members are free" is not a fixed set, and the test cannot locate the information in them. A next step would be to measure each member's lock to the group directly from data (for example, how much of its motion its neighbours explain) and ask whether openness tracks that, rather than assigning freedom by design.

Figures are in `results/e55_figures/`:
* `02_main_result.png`: G against the number of free members;
* `03_free_vs_locked.png`: whose arrangement carries the information;
* `04_test_checks.png`: calibration and the null control.

### e56: does openness follow how loosely members are measured to be locked?

e55 assigned freedom by design. e56 measures it. Each member's looseness is λ = 1 − |mean over time of exp(iφ)|, where φ is its heading or phase relative to the group's. It is computed on the fitted part of each run only: 0 means always aligned, 1 means no relation. The data:
* **Oscillators:** 24 new groups with randomly drawn coupling, frequency spread and noise.
* **Flocks:** 24 new flocks with randomly drawn noise, alignment, speed spread and 0–4 informed agents.
* **Real fish:** the 6 Puy et al. schools and the 9 zebrafish groups.

Besides the usual test, two models use the arrangement of only the k loosest or only the k tightest members (k = 3 in simulations, N/4 in fish).

**Results: 3 of 8 PASS.**

| Check | Result |
|---|---|
| S0. The test works in both simulated ensembles | **PASS.** Planted 19/24 and 23/24, null 24/24 and 23/24 |
| A1. Oscillators: Spearman(max λ, G(r)) ≥ 0.6 | **FAIL.** +0.19 |
| A2. Oscillators: Spearman(number of members with λ > 0.5, G(drift)) ≥ 0.6 | **FAIL.** +0.35, but 23 of 24 groups had no loose member, so the check had nothing to rank |
| A3. Oscillators: loosest k > tightest k where r is open | **PASS.** 4/4 |
| B1. Flocks: Spearman(max λ, G(turn)) ≥ 0.6 | **FAIL.** −0.13 |
| B2. Flocks: loosest k > tightest k where the turn is open (≥ 80%) | **FAIL.** 16/21 (76%) |
| C1. Fish: loosest k > tightest k in ≥ 11/15 | **FAIL.** 9/15 |
| C2. Fish: null clean in ≥ 12/15 | **PASS.** 14/15 |

**What happened.**
* **The random oscillator ensemble was poorly drawn.** Almost every drawn group was fully locked: 19 of 24 had max λ < 0.1, and only one had a member with λ > 0.5. So A1 and A2 had little variation to work with. This was a design error. I should have checked the spread of λ across draws, which is a dynamics-only check, before pre-registering.
* **Where the oscillators were open, the information was in the loosest members (4/4).** In the clearest case (run 8), one oscillator had λ = 0.65. The all-member features found nothing (G = 0.000), because one oscillator in 30 is diluted in a group average. The 3 loosest members' features alone gave G = 0.079.
* **Post hoc (not pre-registered):** with the loosest-member features as the measure, Spearman(max λ, G(r)) is +0.73. All four groups with G(r) ≥ 0.03 have max λ ≥ 0.18, and no group with max λ < 0.1 exceeds 0.065. In oscillators, measured looseness both predicts and locates openness, *provided the test looks at the loose members*.
* **Flocks: heading looseness says nothing.** Every flock agent stays aligned (λ 0.04–0.26), and λ does not predict how open the turn is. The loosest members carry more than the tightest in 76% of open flocks, close to the 80% bar but no more.
* **Real fish: no.** Loosest beat tightest in 9 of 15 groups, close to chance. In fish, either quarter of the school (loosest or tightest) carries much less than the whole school: the information is spread across the group, not held by a loose few.

**What this means.** e54–e56 together give a consistent picture:
* **Oscillators:** a member is either locked or drifting. Openness comes from drifting members and sits in them, whether they are assigned (e55) or measured (e56).
* **Flocks and real fish:** no member is "loose" in heading. The inside information is carried by graded, short-lived differences spread over many members (speed, position, momentary heading). Neither assigned nor measured looseness locates it.

So "open where members are loose" is a property of oscillator-like systems with discrete locking, not a general law of levels. The flock and fish result from e52 (front fish, speed) stands. It describes *where in the school* information sits, not *which individuals* are free.

Figures are in `results/e56_figures/`:
* `02_looseness_vs_openness.png`: max looseness against G for the random groups;
* `03_loosest_vs_tightest.png`: loosest against tightest members, including real fish;
* `04_test_checks.png`: calibration and the null control.

### e57: in oscillators, does measured looseness predict and locate openness? (e56 redone)

e56's oscillator ensemble had almost no loose groups. e57 redraws 30 random groups from a range chosen on dynamics alone (coupling 1.0–2.0, frequency spread 0.3–1.0, noise 0.2–0.5). A dynamics check gave about a third fully locked groups and the rest with 1–30 loose members. Learning from e56, the measure was fixed in advance as G from the arrangement of the **3 loosest members**, compared with the 3 tightest and with all 30.

**Results: 4 of 5 PASS.**

| Check | Result |
|---|---|
| E0. The test works (planted ≥ 24/30, null ≥ 24/30) | **PASS.** Planted 24/30, null 30/30 |
| E1. Spearman(max λ, G_loosest(r)) ≥ 0.6 | **PASS** (just). +0.61 |
| E2. Where r is open, loosest 3 > tightest 3 (≥ 80%) | **PASS.** 21/21 |
| E3. Max λ < 0.1: G_loosest(r) ≤ 0.02 (≥ 80%) | **PASS.** 9/10 |
| E4. Spearman(total λ, G_loosest(drift)) ≥ 0.6 | **FAIL.** +0.50 |

**What it shows.**
* **The relation is a threshold, not a slope.**
  * **Tightly locked groups are closed:** every group whose loosest member has λ < 0.1 has G(r) ≤ 0.05, and 9 of 10 are ≤ 0.02.
  * **Groups with a loose member are open:** every group with max λ ≥ 0.35 is open (G 0.05–0.54, all lower bounds > 0).
  * **No slope beyond that:** above the threshold, looser does not mean more open, which is why E1 only just passed.
* **The information sits in the loose members:** in all 21 open groups, the 3 loosest carry more than the 3 tightest, usually many times more (e.g. 0.54 against 0.02).
* **Looking at the loose members matters.** In several groups (4, 11, 16, 22), the 3 loosest members' features reveal two to a hundred times more than an average over all 30, which dilutes them.
* **Exceptions:**
  * **Two tightly locked groups:** groups 18 and 20 are open to the all-member average (0.11 and 0.05) but not to their loosest members. Something other than looseness opens them.
  * **Phase drift:** it opens with looseness but saturates (E4 fails at +0.50), plausibly because three members cannot represent many loose ones.

**What this means.** Combined with e55, the oscillator result is clear. A synchronized group is closed when every member is locked. One or more measurably loose members open it, and the inside information is in those members. In oscillators, "where is a level open" has a concrete answer that can be read from the data (find the loose members). e56 found that this does not carry over to flocks or real fish, where no member is loose in heading.

Figures are in `results/e57_figures/`:
* `02_main_result.png`: looseness against openness for the 30 groups;
* `03_test_checks.png`: calibration and the null control.

### e58: is a flock's or school's openness carried by its members' momentary innovations?

In flocks and fish, no member is loose in heading (e56). e58 tested whether freedom there is momentary instead.

**The measures.**
* **Innovation:** for each member and step, the part of its turn that its surroundings do not explain. The surroundings are its nearest neighbours' alignment, attraction and repulsion, the group's centre, walls, its own previous turn and its relative speed, fitted on the first 70% of the data.
* **Independence:** a member's share of turning that is innovation.
* **Innovation map:** members' recent innovations placed by position in the school.

**The data:** 24 new random flocks with 1–4 privately informed agents, the 6 Puy et al. schools and the 9 zebrafish groups.

**Results: 4 of 9 PASS.**

| Check | Result |
|---|---|
| V1. Independence finds the informed agents (AUC ≥ 0.8 in ≥ 20/24) | **PASS.** AUC = 1.00 in all 24 |
| S0. The test works in the flocks | **PASS.** Planted 22/24, null 24/24 |
| I1. Flocks: innovations carry ≥ half of the openness (≥ 80% of open flocks) | **FAIL.** 0/16; innovations G ≤ 0.035, all features 0.03–0.14 |
| I2. Flocks: innovations > raw own turning | **PASS.** 15/16 |
| I3. Flocks: the most independent members carry more than the least | **FAIL.** 12/16 (75%) |
| F1. Fish: innovations carry ≥ half of the openness | **FAIL.** 2/9 |
| F2. Fish: innovations > raw own turning (≥ 11/15) | **FAIL.** 10/15 |
| F3. Fish: independent > conforming members (≥ 11/15) | **FAIL.** 9/15 |
| F4. Fish: null clean | **PASS.** 15/15 |

**What it shows.**
* **The measure works, but the hypothesis does not.** Independence identifies every informed agent perfectly (AUC 1.00 in 24 of 24 flocks), so in flocks free members *can* be found from data. Yet their innovations carry little of the group's openness: G from innovations is at most 0.035, against 0.03–0.14 from the whole arrangement, never half. Raw own turning carries even less.
* **Real fish split by species.**
  * **Adult fish (Puy et al.):** innovations carry a real share of the openness, G 0.04–0.16, which is 23–64% of the arrangement's.
  * **Zebrafish:** innovations carry nothing.
  * **Independence:** in neither species does it locate the information (9/15).
* **Flock 13** gave a large negative G for the all-member model (−1.08). The ridge forecaster extrapolated badly on the test period. Other models on the same flock behaved normally.

**What this means.** Across e55–e58, flocks behave consistently. The openness of their turn is not carried by who is free, or by what anyone just did that was new. It is carried by the **arrangement itself**: who is where, heading which way, and how fast. The interaction rules turn that arrangement into the group's next turn, mechanically and within about one time unit. This points to a different, unifying idea (a hypothesis only): a level is open when some internal degrees of freedom are **slow**, persisting over the forecast horizon instead of relaxing quickly to follow the group's summary. That is Haken's "slaving" principle read as a test.
* **Oscillators:** locked members relax fast and are slaved; drifting ones persist, so the group is open where they are (e55, e57).
* **Flocks and schools:** the spatial arrangement persists for about a second, so the turn is open; shape and spread are slaved, so they are closed.

A next experiment would measure each system's internal relaxation time directly and ask whether openness tracks it.

Figures are in `results/e58_figures/`:
* `02_what_carries.png`: all features against innovations against raw turning, for every flock and fish group;
* `03_independence.png`: whether informed agents stand out, and whether independent members carry more;
* `04_test_checks.png`: calibration and the null control.

### e59: is a level open when its internal state is slow? (slaving, as a test)

e58 ended with a unifying hypothesis: a level is open when some internal degrees of freedom are slow, persisting over the forecast horizon instead of relaxing quickly to follow the group (Haken's slaving principle).

**The measure.** e59 measured each group's **internal memory time** τ, with no target involved:
* **Residual:** the members' arrangement features, minus what the group summary predicts.
* **Reduction:** the residual is reduced to its leading principal components (90% of variance, at most 10).
* **Memory time:** τ is the lag at which these components can no longer be predicted from themselves (R² < 1/e).

**The data:** new runs of e57's oscillator draws (30) and e58's flock draws (24), plus the 15 real fish groups. The test was G at one time unit.

**Results: 2 of 6 PASS. The hypothesis is not supported as measured, and the measure is the likely culprit.**

| Check | Result |
|---|---|
| T0. Null clean in ≥ 80% of 69 groups | **PASS.** 66/69 |
| T1. Oscillators: Spearman(τ, G(r)) ≥ 0.6 | **PASS.** +0.64 |
| T2. Flocks: Spearman(τ, G(turn)) ≥ 0.6 | **FAIL.** +0.57 |
| T3. Fish: Spearman(τ, G(turn)) ≥ 0.5 | **FAIL.** −0.59 (opposite sign) |
| T4. Pooled: Spearman(τ / horizon, G) ≥ 0.5 | **FAIL.** +0.10 |
| T5. τ < 0.3 horizons → closed (≥ 80%) | **FAIL.** 9/35 |

**What happened.**
* **Within a family, τ tracks openness weakly.** Oscillators +0.64 and flocks +0.57: groups whose internal state lingers a little longer are a little more open.
* **Across families it does not.**
  * **Oscillators:** every group's internal state is "forgotten" within 0.05–0.18 time units, far shorter than the one-unit horizon. Yet most of them are open (G up to 0.40), so "fast inside, closed outside" fails (9/35).
  * **Fish:** the adult fish have the shortest memory (0.14–0.31 s) and the largest openness (G 0.07–0.24). Zebrafish remember longer (0.7–1.0 s) and are less open. The relation runs backwards.
* **Why: the measure is dominated by fast, high-variance fluctuations.**
  * **Oscillators:** noise kicks dominate their internal state. Even at the shortest lag its R² is below 0.4 (`03_memory_curves.png`).
  * **Adult fish:** the 0.1 s heading estimates are probably dominated by tracking noise.
  * **What τ misses:** the slow component that carries information (a drifting oscillator's phase, a fish's place at the front) can be a small part of the variance, so a variance-weighted memory time cannot see it.

**What this means.** The slaving idea is not refuted, but this test of it failed: "how fast is the dominant internal fluctuation?" is the wrong question. The right one is "is there *any* slow internal direction the summary misses?" A fair test needs a measure that finds the slowest internal directions regardless of their variance, such as slow feature analysis on the residual. It still must not use the target. That is the natural e60.

Figures are in `results/e59_figures/`:
* `02_main_result.png`: τ against G per family and pooled;
* `03_memory_curves.png`: how fast each group's internal state is forgotten.

### e60: is a level open when there is any slow internal direction its summary misses?

e60 repeated e59 with the measure e59 lacked. **Slow feature analysis (SFA)** finds the slowest directions of the residual internal state regardless of their variance, without using the target:
* **Whitening:** the residual is whitened on the fitted part.
* **Directions:** they are ordered by how little they change from step to step.
* **Memory time:** τ_slow is how long the slowest one stays correlated with itself on the held-out part.

The data were new runs of the same 30 oscillator and 24 flock draws, plus the 15 fish groups. An extra test asked whether the 3 slowest directions carry more information than the 3 fastest.

**Results: 1 of 7 PASS. Slowness does not predict openness.**

| Check | Result |
|---|---|
| U0. Null clean | **PASS.** 66/69 |
| U1. Oscillators: Spearman(τ_slow, G) ≥ 0.6 | **FAIL.** +0.21 |
| U2. Flocks: ≥ 0.6 | **FAIL.** +0.05 |
| U3. Fish: ≥ 0.5 | **FAIL.** −0.61 |
| U4. Pooled: ≥ 0.5 | **FAIL.** 0.00 |
| U5. No slow direction → closed | **FAIL, empty.** No group lacks a slow direction |
| U6. 3 slowest > 3 fastest in open groups (≥ 80%) | **FAIL.** 35/46 (76%) |

**What it shows.**
* **Every group has slow internal directions, including closed ones.** Closed oscillator groups have slowest memory times of 0.4–0.8, the same as open ones. So having a slow internal direction is not sufficient for openness.
* **Fast internal directions carry nothing, anywhere.** The 3 fastest directions gave G ≈ 0.000 in nearly every group of every family. The slow end carries some of the information in oscillators (often half or more) and a little in flocks. In fish, neither end of the spectrum carries it; it sits in between. U6 fell just short (76%), mostly because in fish both ends carry nothing.
* **Fish go backwards again** (−0.61): the adult fish have the shortest slow memory and are the most open.

**What e54–e60 add up to.** Seven experiments looked for a rule that says, without looking at the target, where and when a level stays open:
* **Neutral directions** (e54): no.
* **Free members** (e55): in oscillators only.
* **Measured looseness** (e56, e57): yes in oscillators, where it both predicts and locates openness; not in flocks or fish.
* **Momentary innovations** (e58): they find informed agents perfectly, but do not carry the openness.
* **Slowness of the internal state** (e59, e60): no. Fast internal directions never matter, but slow ones exist in closed groups too.

The consistent conclusion: **openness is not a property of the internal dynamics alone.** It depends on whether some slow internal state is *coupled* to the group variable in question. That coupling is exactly what the closure test measures. In oscillators, looseness happens to identify the coupled states. In flocks and fish, no simple target-free property we tried does. The closure test itself, calibrated and null-checked, remains the tool; we found no shortcut around it.

Figures are in `results/e60_figures/`:
* `02_main_result.png`: slowest memory time against G;
* `03_slow_vs_fast.png`: how much the 3 slowest and the 3 fastest internal directions carry.

### e61: a new layer. Is a brain area's activity shaped by its neurons interacting?

The interaction share moves from animals to neurons:
* **Members:** neurons.
* **Group quantity:** the total spike count of one brain area in the next 50 ms.
* **Data:** Steinmetz et al. (2019), Neuropixels recordings from 39 sessions of mice doing a visual choice task, giving 221 areas with ≥ 30 neurons firing ≥ 0.5 Hz.
* **Twin:** one Poisson model per neuron, from its own history and the recorded causes, with no other neuron.
* **Recorded causes**, in three nested sets:
  * E0: the stimulus;
  * E1: + task events and behaviour (wheel, pupil, face, licks);
  * E2: + the activity of every other recorded area.
* **Virtual groups:** each neuron's trials are shuffled within the same stimulus condition.
* **Simulations with known answers** come first: independent, coupled, a hidden shared drive, and the same drive recorded.

**4 of 9 pre-registered predictions passed, and the real-data numbers are void.** The per-neuron fits had no step control, and for a few low-rate neurons they diverged: the twin predicted 403 spikes per 50 ms where the truth was 0.2. Real and virtual groups then both scored up to 0.98.

The simulations still taught two things:
* **An unrecorded shared drive is flagged as interaction** (S about 0.4), as expected. Once the drive is recorded, S falls to 0.
* **Weak random coupling leaves the population count untouched** (S ≤ 0.003; pairwise correlation 0.01). This is the springs lesson of Section 3 again: interaction exists, but it does not shape this quantity.

Figures are in `results/e61_figures/`.

### e62: brain areas with a repaired twin

e62 fixed e61's three faults:
1. **Safeguarded fits:** Newton steps with backtracking. Every one of 54,840 fits converged.
2. **A twin matched to the group law:** each neuron's model uses the same kinds of history features as the area's law (raw window counts and the latest one squared), plus logs. Without interaction, the projected share is 0 only if each neuron's errors are uncorrelated with the features the group law uses.
3. **A coupling strong enough to test**, set on dynamics only: pairwise correlation 0.02–0.04, typical of cortex.

e62 also adds **time in session** as a recorded cause: whole areas drift by up to 30% over a session.

**8 of 12 pre-registered predictions passed.**

| network (5 seeds) | S_proj | virtual |
| --- | --- | --- |
| independent | −0.002 to 0.000 | ≈ 0 |
| weakly coupled (corr 0.01) | 0.001 to 0.005 | ≈ 0 |
| strongly coupled (corr 0.02–0.05) | 0.011 to 0.022, all lower bounds > 0 | ≈ 0 |
| hidden shared drive | 0.33 to 0.37 | 0.05 to 0.09 |
| drive recorded | −0.002 to 0.002 | ≈ 0 |
| coupled + drive recorded | 0.029 to 0.051 | ≈ 0 |

Real areas (221):

| | S_proj, median [10%, 90%] |
| --- | --- |
| beyond the stimulus (E0) | 0.048 [0.005, 0.164] |
| + task, behaviour, session time (E1) | 0.040 [0.003, 0.131] |
| + other areas (E2) | 0.027 [0.001, 0.089] |
| virtual groups | 0.005 [−0.002, 0.046] |

* **Passed:**
  * N1, N3, N4 and N6: the independent, weak, hidden and recorded-drive networks behave as predicted.
  * R0: the fits converge.
  * R1: virtual groups are clean in 189 of 221 areas.
  * R3: recorded causes absorb part of the effect.
  * R4: the share beyond all recorded causes is positive (lower bound > 0) in 177 of 221 areas.
* **Failed:**
  * N2 and N7: realistic coupling is detected in 5 of 5 networks (lower bound > 0, virtual clean), but its share is 0.01–0.05, under the 0.05 bar I set.
  * N5: the repaired twin does not clean a hidden drive's virtual groups (0.05–0.09). With a slow shared drive, a neuron's own history is a noisy, non-linear clue to the drive, and the per-neuron model cannot fully capture it.
  * R2: 153 of 221 areas (69%) beat their virtual groups by 0.02 with the lower bound above, short of 80%.
* **Which areas:**
  * Hippocampus is about 0 (CA1 0.000, CA3 0.006, DG 0.009, median over sessions, E2).
  * Highest are striatum and basal ganglia (SNr 0.098, CP 0.080), thalamus (VPM 0.065), and frontal areas (ORB 0.061, MOs 0.043). Visual cortex is in between (VISp 0.033).
  * Larger recordings show more (correlation with log neuron count 0.33).
* **Caveat:** virtual groups are not clean in a few motor and somatosensory recordings (MOp, SSp, PO: 0.11–0.29), consistent with N5's leak through a slow drive.

**What this means.** For the first time on this layer, the method gives a calibrated answer. Mouse brain areas have a small interaction share at 50 ms: about 3% of what a no-interaction model gets wrong about the area's next activity is explained by coordination beyond the stimulus, task, behaviour, session drift and every other recorded area. This holds in about 80% of areas and is absent in hippocampus. The simulations set the scale and the limit:
* realistic recurrent coupling gives exactly this size (0.01–0.05);
* an unrecorded shared drive such as arousal can give much more (0.35), and some of it would survive.

**So the 3% is best read as an upper bound on what the neurons' interactions contribute to the area's 50 ms activity.** The method cannot by itself say how much is coupling and how much is a common input no one recorded. The neural layer is, by this measure, close to closed under its recorded causes. That fits the view that population activity at this scale is mostly driven from outside the local circuit.

Figures are in `results/e62_figures/`:
* `02_simulations.png`: the six kinds of network;
* `03_real_areas.png`: every area;
* `04_nested.png`: how recorded causes absorb the share.

### e63: is a brain area's population state a closed level?

e62 measured how much coordination shapes an area's activity. e63 asks the closure question of the same 221 areas, with e51's calibrated test.
* **Summary state (the level):** the area's top 3 principal components (of square-rooted, standardized 50 ms counts), with their history, the area's total activity and e62's recorded causes (stimulus, task events, behaviour, session time).
* **Question:** does anything else improve the forecast of the summary 50, 100 or 200 ms ahead?
  * **internal:** the rest of the area's population state (components 4–30);
  * **other areas:** the recent activity of every other recorded area;
  * **null:** the internal features of another trial (no information).
* **Calibration:** a planted signal worth 5% of the error must be found.

**4 of 7 pre-registered predictions passed.**

| G (top 3 components), median [10%, 90%] | 50 ms | 100 ms | 200 ms |
| --- | --- | --- | --- |
| internal | 0.012 [0.002, 0.031] | 0.006 [0.001, 0.024] | 0.003 [0.000, 0.017] |
| other areas | 0.011 [0.003, 0.027] | 0.009 [0.002, 0.022] | 0.005 [0.000, 0.016] |
| null | 0.000 | 0.000 | 0.000 |

* **Passed:**
  * C2: the null is exactly clean in 221 of 221 areas.
  * C4: closure grows with horizon.
  * C5: areas are open to other areas in 203 of 221.
  * C7: the stimulus response opens the level a little (median 0.011, against 0.007 before the stimulus).
* **Failed:**
  * C1: the planted signal is detected (lower bound > 0) in 218 of 221 areas, but its gain reaches 0.025 in only 176 (median 0.045). The test is sensitive, but in a fifth of the areas it recovers less than half of a 5% signal.
  * C3: only 55 of 221 areas are open inside by ≥ 0.02.
  * C6: areas with a larger interaction share are only weakly more open inside (Spearman 0.23, below the 0.3 bar).
* **Which areas:**
  * Frontal and limbic areas (ILA, PL, LS, SUB) and hippocampus (CA1 0.022) are the most open *to other areas*. CA1 is closed inside (0.007) and had no interaction share in e62: its summary is driven from outside, not shaped within.
  * Sensory thalamus and basal ganglia (VPL, VPM, SNr, GPe) are the least open to other areas.
* **Recording size matters:** larger recordings look more open inside (correlation with log neuron count 0.45). Internal closure is partly a statement about how many neurons were recorded.

**What this means.** A mouse brain area's 3-dimensional summary, together with its recorded causes, is almost a closed level. The rest of the population adds about 1% to its 50 ms forecast and 0.3% at 200 ms. Other areas add about as much. This agrees with e62: little of an area's activity is shaped internally beyond what its recorded causes explain. What openness there is splits evenly between the area's own detail and its neighbours, and it fades within 200 ms.

Figures are in `results/e63_figures/`:
* `02_checks.png`: the planted and null checks;
* `03_horizons.png`: openness against horizon;
* `04_by_area.png`: each area, internal against external;
* `05_link_and_periods.png`: the link to e62, and before, during and after the stimulus.

### e64: does a brain area's coordination survive switching off a major input?

*Followed up by e80:* an intervention test that was decisive, on fish. Removing vision (dark arena) takes zebrafish pairs and triads from 0.023 and 0.089 to zero, and making a virtual partner ignore the real fish lowers the share from 0.063 to 0.019.

e62 could not tell coordination between neurons apart from an input nobody recorded. A perturbation can. Inagaki, Chen et al. (2022, DANDI 000221) recorded the anterior lateral motor cortex (ALM) while optogenetically silencing the midbrain (PPN/MRN) that drives it, from 0.6 s before to 0.6 s after the go cue. The prediction: if ALM's coordination comes from that input, its interaction share falls when the input is off. 21 sessions qualified (≥ 20 units, ≥ 20 silenced trials). The test used e62's twin and law, cross-fitted over trials, with control trials subsampled to the silenced count.

**2 of 8 pre-registered predictions passed. The real-data numbers are void.**
* **Passed:**
  * R0: silencing works; it blunts ALM's response to the cue in 19 of 21 sessions.
  * A2: a simulated hidden input switched off is seen (5/5).
* **Failed: calibration.**
  * A1: simulated independent networks scored +0.04 in the light window and −0.1 before it, not 0 (0/5).
  * A3: switching off the cue input in a coupled network moved the share as much as a hidden input did (0/5).
  * R1: only 22 of 84 virtual-group cells were clean.
  * Some cross-fitted Poisson twins blew up on held-out trials.
* **Failed on real data, not interpretable:** R2 (median −0.057), R3 (−0.082, p = 0.33), R4.

**Why.** The light window straddled the go cue. Its large time-locked response, estimated from about 40 trials, dominated the share: a model that follows the transient slightly wrong looks like a shared input. The Poisson twins also overfitted at this sample size.

### e65: e64 redesigned on simulations only

The redesign was made using simulations alone (development seeds 10–19), before any real share in the new window was computed:
* **Window:** the light **before** the cue (0.6 s to 0 s), a steady delay state.
* **Twin:** a linear ridge model whose recorded cause is each neuron's cross-fitted average response for the instructed side.
* **Group law:** the same model on the ALM total.
* **Simulations:** fresh seeds 20–29.

**4 of 7 pre-registered predictions passed.**

| | pre-cue, control → silenced | baseline, control → silenced |
| --- | --- | --- |
| simulated independent (median) | −0.020 → −0.013 | −0.013 → −0.023 |
| simulated local coupling | −0.020 → −0.017 | −0.015 → −0.013 |
| simulated hidden input | +0.30 → −0.02 | +0.32 → +0.32 |
| ALM, median of 21 sessions | +0.001 → −0.012 (median change 0.000) | −0.015 → +0.000 (median change +0.017) |

* **Passed:**
  * B1: independent networks are calibrated (8/10 within ±0.03).
  * B2: a hidden input being switched off is found every time (10/10).
  * R2: the baseline difference is within 0.02 (+0.017).
  * R4: the difference-in-differences is negative (median −0.033, p = 0.013).
* **Failed:**
  * B3: local coupling stays within ±0.03 in 7/10, not 8 (noise of about ±0.03 per estimate).
  * R1: virtual groups are clean in 67/84 cells, just under 80%.
  * R3: there is no coordination to remove; the median control share before the cue is +0.001.

**R4's pass does not support its claim.** Split into its two halves, the effect sits entirely in the window **before** the light. There, the trials that were later silenced score higher in 18 of 21 sessions (p = 0.0007), although nothing has been silenced yet. During the light itself, the share changes in neither direction (10 of 21 lower, median 0.000). The difference-in-differences picked up a difference between trial sets, not an effect of silencing. Silencing also raised ALM's pre-cue firing (median ×1.14, 14 of 21 sessions).

**What this means.**
* During the delay, ALM shows no coordination beyond its recorded causes that this test can see: the median share is about 0, with a per-session noise of about ±0.03.
* The simulations show the test would have seen a hidden input of e62's size being switched off; nothing like that happened.
* This is consistent with e62's small shares (at most about 3%). However, a share that small is below what about 40 trials per condition can resolve. Whether the remaining coordination is local or inherited from the midbrain stays open.
* A decisive test needs more silenced trials (about 100 or more) during a steady period.

Figures are in `results/e64_figures/` and `results/e65_figures/`:
* `02_simulations.png`: calibration;
* `03_sessions.png`: each session, control against silenced.

### e66: a new domain. Is market volatility shaped by coordination between industries?

* **Group:** the market's daily volatility, the mean |return| of Kenneth French's 49 industry portfolios (1972 to September 2026, 13,801 days).
* **Twin:** each industry's own volatility history (yesterday, last week, last month, as in the standard HAR forecast) plus the day of the week. Everything is cross-fitted over 250-day blocks.
* **Virtual groups:** each industry shifted in time by its own random offset.
* **Replication:** 467 S&P 500 stocks (2013–2018), with the daily high-low range as the volatility measure.
* **Calibration:** simulated markets of 49 GARCH members.
* **Development:** on simulations only (seeds 100–102). It showed something new: a common driver is **not** counted as coordination when every member's own history reveals it. This was pre-registered as F4.

**6 of 10 pre-registered predictions passed.**

| | S [90%] | virtual |
| --- | --- | --- |
| simulated independent / correlated shocks | −0.002 to 0.000 | about −0.002 |
| simulated volatility spillover | +0.020 to +0.036 | about −0.001 |
| simulated common driver (loading 0.8) | 0.000 to +0.003 | about −0.002 |
| **49 industries, 1972–2026** | **−0.005 [−0.011, +0.002]** | −0.002 to 0.000 |
| 467 stocks, 2013–2018 | +0.090 [+0.039, +0.141] | +0.49 to +0.55 (broken) |

* **Passed:** all five simulation checks (F1–F5), and R3 (stocks, but void, see below).
* **Failed:**
  * R2: the real market shows no coordination (−0.005).
  * R5: neither half shows any either (1972–1997 −0.014, 1998–2026 0.000).
  * R4: no rise in turbulent years (Spearman 0.12). 1987, 2008 and 2020 score −0.01, +0.02 and +0.02.
  * R1: the stock virtual groups are not clean.
* **Why the stock replication is void (exploratory, after the run).** With only 5 years, the random shifts span 500–759 days. Every stock carries the market's volatility episodes, so averaging 467 shifted copies leaves a slow, smeared shared component. The group can forecast that component, but no single stock's own history can; the virtual groups score +0.5 even when fitted in-sample. Shifts need a series much longer than the shift range, as with the 54-year industries. *Redone in e79* with 27 years of daily prices in eight markets: virtual stock groups are clean (−0.002 to +0.004), and stocks show a share of 1.5–7.9%, larger than portfolios of the same region.

**What this means** (revised by e69: on log volatility there is a small positive share, 0.013–0.029)**.**
* The market's volatility is exactly what its industries' own volatility dynamics add up to. Knowing the market's past adds nothing beyond each industry's own past.
* Volatility is strongly shared across industries, but in the simulations that pattern is a common driver that each industry's history already reveals. It is not spillover from other industries' shocks, which the test detects at 0.02–0.04 (if spillover is present, it is smaller than the test can see).
* This is the same verdict as for brain areas (e62–e65): a level whose members each carry the shared signal is not shaped by their coordination.
* It also locates the boundary of the interaction share. The share counts a common input only to the extent that members' own histories reveal it poorly. Noisy spike counts reveal a shared drive poorly, so neurons were flagged (e62); volatility reveals it well, so markets are not.

Figures are in `results/e66_figures/`:
* `02_shares.png`: simulated and real markets;
* `03_years.png`: year by year.

### e67: one equation for the interaction share (the memory gap)

The e9–e66 results shared a pattern. S was high where the group's past reveals something no member's own past reveals (fish schools, neurons with a hidden drive), and zero where every member already carries the shared signal (markets). For linear twins and laws this can be made exact:
* A member's twin weights its past by the member's own **memory**, its autocovariance o(k) averaged over members.
* The group law weights the group's past by the **group's memory**, g(k).
* With interchangeable members, the summed twin forecast is the members' weights applied to the group's past. So

  S_pred = 1 − MSE*(g) / MSE(β(o); g),

  the loss from forecasting the group with its members' memory instead of its own.

With one lag this is the **memory-gap equation**: with ρ_m = o(1)/o(0) and ρ_g = g(1)/g(0),

  S₁ = (ρ_g − ρ_m)² / (1 − 2ρ_mρ_g + ρ_m²).

**7 of 8 pre-registered predictions passed.**
* **Passed:**
  * P1: in all 52 linear simulations, S_pred matches the measured S (largest gap 0.003).
  * P2: this holds equally for common input, all-to-all coupling and local coupling.
  * P3: the score peaks at an interior per-member signal share, and the peak moves to weaker signals in larger groups.
  * P4: the one-lag equation ranks the cells (Spearman 0.97).
  * P5: simulated GARCH markets and spiking networks match too (14/14 within 0.02; most within 0.001).
  * R2: the 221 brain areas match (Spearman 0.97, median gap 0.006).
  * R3: the equation also explains e62's Poisson-twin results (Spearman 0.93).
* **Failed:**
  * R1: industries. S_pred is +0.016, against a measured −0.003.
* **Why R1 failed (exploratory, after the run).**
  * Heterogeneity is not the reason: per-industry weights give 0.015, and equal-variance scaling 0.014.
  * The reason is heavy tails. On log volatility, measured S = 0.012 against S_pred 0.015. In raw |return|, a few crash days dominate the out-of-sample error, and the measured share falls below the long-run memory gap.
  * This also qualifies e66. On log volatility, the market shows a small positive share (about 0.01). That is post hoc, and a lead for a pre-registered test, not a result.

**What this means.**
* **One law covers every system we have measured.** The interaction share is the cost of forecasting the group with its members' memory instead of its own.
  * It is zero when the group remembers the way its members do, however correlated they are (correlated GARCH shocks: member correlation 0.22, S 0.000).
  * It is large when the group remembers longer than its members. That happens when a persistent shared signal is spread thinly over many members (the pooling peak), or when coupling feeds the group's own past back into everyone.
* **The equation predicts S from two cheap numbers** (member and group autocorrelation) without fitting any twin. On brain areas it reproduces the full Poisson-twin analysis (Spearman 0.93).
* **It also states a limit precisely.** S depends only on second-order statistics, so a common input and all-to-all coupling that give the same memories give the same S (P2). No analysis of S alone can separate them; that needs recorded causes or perturbation (e64, e65). The separation question has moved from "the method's weakness" to a theorem about what any such measure can see.

Figures are in `results/e67_figures/`:
* `02_predicted_vs_measured.png`: one equation, every system;
* `03_pooling_peak.png`: the pooling peak;
* `04_memories.png`: member against group memory.

### e68: does the memory-gap equation work on fish schools?

Fish are the hardest test. Their measured shares (e33 adult fish, e38 zebrafish) came from nonlinear twins (walls, persistent turning, speed) and a cubic law in polarization, a nonlinear group variable. The equation got neither:
* **Members:** each fish's heading as a unit vector, every 0.1 s.
* **Group:** the school's summed heading.
* **Forecast:** 1 s ahead, from the heading 1 s before and two older averages.
* **Compared with:** 8 adult-fish recordings and 9 zebrafish recordings, each with virtual groups (17 real, 17 virtual).

**3 of 4 pre-registered predictions passed.**
* **Passed:**
  * F1: every virtual group scores about 0 (17/17).
  * F2: every real school scores above its virtual groups (17/17; real 0.10 to 0.66).
  * F3: across all 34 groups, the equation ranks the measured shares (Spearman 0.76).
* **Failed:**
  * F4: within the 17 real schools, the ranking is reversed (Spearman −0.65). Adult fish have the highest measured shares (0.64–0.89) but the lowest predictions (0.10–0.37). Zebrafish are the reverse (measured 0.18–0.62, predicted 0.33–0.66).

**What this means.**
* With no twin and no school model, two memory curves tell real schools from virtual ones every time.
* They do not give the size of the share for polarization. Polarization is a nonlinear function of the headings; its share depends on how the school turns as a whole, which a linear forecast of the summed heading does not see. The equation's guarantee holds for linear group variables, and fish polarization is outside it.

Figures are in `results/e68_figures/`:
* `02_comparison.png`: predicted against measured, real and virtual.

### e69: a pre-registered test of the market lead (log volatility, new data)

e67's exploratory analysis found a small positive share in the market's log volatility (about 0.01), hidden in raw |return| by crash days. e69 tested it on data never analysed before: 40 value-weighted industries from 1936 to 1971 (9,713 days). It used e66's method unchanged, with log(|r| + 0.05) as volatility, 5 virtual groups, and e67's equation with the same HAR windows. Equal-weighted portfolios from 1972 to 2026 served as a replication.

**5 of 5 pre-registered predictions passed.**

| | log volatility S [90%] | equation | raw \|r\| S | virtual (log) |
| --- | --- | --- | --- | --- |
| **holdout: 40 industries 1936–1971** | **+0.029 [+0.017, +0.041]** | +0.028 | +0.012 | −0.004 to −0.001 |
| equal-weighted 1972–2026 | +0.017 [+0.010, +0.024] | +0.019 | −0.003 | −0.002 to +0.004 |
| value-weighted 1972–2026 (seen before) | +0.013 [+0.006, +0.020] | +0.015 | −0.005 | −0.001 to 0.000 |

* M1: the holdout shows a positive share (+0.029, lower bound +0.017).
* M2: the equation predicts it within 0.001.
* M3: all virtual groups are within ±0.004.
* M4: the equal-weighted replication is positive.
* M5: on raw |return| the share is smaller (+0.012).

**What this means.**
* **e66's verdict is revised.** Market volatility does carry a small coordination share: knowing the market's own volatility history improves a forecast of tomorrow's market volatility by 1.5–3%, beyond what all industries' own histories give. It was invisible in e66 because a few crash days dominate raw |return|.
* The equation predicted the size of the share on 36 years of data it had never seen.
* Like every share, this is mechanism-blind (e67). It may be spillover between industries or a slow common driver that each industry reveals poorly; S alone cannot tell.

Figures are in `results/e69_figures/`:
* `02_shares.png`: log and raw shares, the equation, and virtual groups.

*Revised by e76:* e66's construction leaks the test fold into the twin, which biases market shares low. Leak-free, the log-volatility shares are 0.001–0.005 higher and still match the equation (for example, equal-weighted 1972–2026: 0.019 against 0.019).

### e70 (exploratory): a memory-gap screening tool

`experiments/memgap.py` turns e67 into a tool. Given members' time series, it returns the predicted share, the two memories and a virtual floor, with no twin to fit:

    uv run python -m experiments.memgap data.csv --horizon 1 [--log]

The floor is the median of 5 random shifts, or of trial permutations for trial data. The tool uses the biased autocovariance, so very smooth series stay well defined. e70 ran it on every dataset in the lab, including flocks where no twin worked. Everything below is exploratory.

| domain | datasets | predicted share | floor |
| --- | --- | --- | --- |
| fish (adult, zebrafish) | 17 | 0.10–0.64 (median 0.33) | ≤ 0.003 |
| pigeons (flocks A, B, C) | 3 | 0.009–0.015 | 0.007–0.011 |
| ALM, delay before the cue | 21 | 0.002–0.069 (median 0.017) | ≤ 0.001 |
| industries (log volatility) | 2 | 0.015, 0.028 | 0.007, 0.022 |
| 467 stocks 2013–2018 | 1 | 0.117 | 0.152 |

* **ALM has a small coordination share before the cue.** The screen predicted it, and the full measurement on the same trials confirmed it: measured 0.014 against predicted 0.017, Spearman 0.94 across sessions, positive in 19 of 21. This uses all ~270 control trials per session; e65 used about 40 and could not resolve it. It is consistent with e62's brain-area shares and, like them, mechanism-blind.
* **Pigeons:** the screen is silent. At 1 s, each bird's heading (memory 0.99, smoothed GPS) already forecasts the flock; this agrees with e36's closed turn.
* **The floor matters for long-memory series.** In markets, slow drifts in shifted copies create spurious cross-covariance, so the equation's in-sample floor is high (0.022 on the 1936–1971 industries, where the cross-fitted virtual groups of e69 sat at 0). For stocks the floor exceeds the real value, so the screen is inconclusive, as e66 found with the full method. For such data, the screen flags candidates and a cross-fitted measurement decides.
* **Revision (Oct 30, found while building the twinshare package): the market floors were mostly an artifact of how the shifts were drawn.** `memgap.shift` draws each member's offset from the middle half of the record (a quarter to three quarters of its length). Every member then moves by roughly half the record, so slow regimes stay partly aligned across members and the shifted group still looks coordinated. With offsets spread over the whole circle (5% kept clear at each end, as e66's virtual groups did), the floors fall from 0.007 and 0.022 to 0.0007 and 0.0005 for the industries, and from 0.152 to 0.009 for the 467 stocks. The stock screen is therefore not inconclusive: it predicts 0.117 against a floor of 0.009, and only the full measurement can decide. The pigeon floors fall from 0.007–0.011 to 0.003–0.007, leaving flocks A and C slightly above them (0.010 and 0.015), so "silent" overstates it: faint, at about 1%. The fish floors (≤ 0.003) and every trial-permutation floor (ALM, e71, e73) are unaffected. `memgap.py` is hashed and used by e80, so it is left unchanged; e80's time-shifted virtual groups were clean, and the narrow range can only push a virtual group up. The twinshare package uses the whole-circle offsets. Reproduce: `PYTHONPATH=. uv run python twinshare/validation/e70_floors.py`.
* **Mistakes caught while building it.** The first version used a per-lag (unbiased) autocovariance. That gave impossible values (80) on the very smooth pigeon tracks. Within-trial shifts left trial-wide shared fluctuations in the ALM floor. An in-place residualization bug inflated ALM to 0.08. All three were fixed before the numbers above, and the ALM check against the full method is what caught the last one.

Figures are in `results/e70_figures/`:
* `01_screen.png`: every dataset, with floors and e67's brain areas.

### e71: ALM's small share, on unseen sessions and against silencing

e70's ALM finding was exploratory and on sessions analysed before. e71 tested two things:
* **Part A, confirmation.** The 19 ALM sessions of DANDI 000221 with ≥ 20 units that e64 and e65 never used (136–371 control trials each), with e70's pipeline unchanged.
* **Part B, silencing.** Does the share need the midbrain input? This uses the memory gap, which needs no fitting, in place of e65's noisy fitted twins:
  * the excess of S_pred over its trial-permutation floor;
  * silenced against control trials subsampled to the silenced count;
  * in the light window before the cue, and a baseline before the light;
  * scored as a difference-in-differences (DiD).

The design was developed on e64's simulated networks only, and the simulations here use fresh seeds.

**9 of 10 pre-registered predictions passed.**
* **Simulations (C1–C4):**
  * independent and coupled networks: |DiD| ≤ 0.03 (10/10 each);
  * a hidden input switched off: −0.15 to −0.33 (10/10);
  * a weak input (control excess 0.018), over 21 simulated sessions: 15/21 negative, p = 0.039. Detectable, but only just.
* **Part A, unseen sessions:**

| | median | positive |
| --- | --- | --- |
| A1 measured share | **+0.013** | 16/19 |
| A2 memory-gap prediction | +0.016 (Spearman with measured 0.86) | |
| A3 virtual groups (trials permuted per neuron) | −0.002 (all within ±0.005) | 1/19 |
| A4 beyond session drift | +0.014 | 17/19 |

* **Part B:**
  * B1: before the light, silenced and control trials do not differ (−0.0004).
  * **B2 failed.** The median DiD is +0.002 (10/21 negative, p = 0.67). The median control excess in the light window is 0.011, below the pre-registered 0.02, so by the reading fixed in advance the test is **inconclusive (underpowered)**, not evidence of independence. Real sessions are also much noisier than the simulations (DiD −0.07 to +0.08).

**What this means.**
* **Confirmed:** in the 1.5 s before movement, mouse motor cortex has a small coordination share: about 1.3% of its next 50 ms is predictable from the area's own history beyond every neuron's own history and side-specific response. It holds on unseen sessions, survives removing slow drift, and the memory-gap equation predicts which sessions have more.
* Whether it is local or inherited from the midbrain is still open. The memory gap removed the fitting noise, but ~40 silenced trials per session are still too few for a share this size.

Figures are in `results/e71_figures/`:
* `02_holdout.png`: prediction against measurement, drift and virtual controls;
* `03_silencing.png`: simulated and real DiD.

### e72: a quadratic memory gap for polarization

e68's failure (F4) had a specific diagnosis, written into e72 before any number was computed. The linear equation centres each heading, so it ignores the **level** of polarization. A no-interaction twin predicts that a tightly aligned school relaxes towards disorder, and that error is most of the measured share for polarized schools. e72 keeps every other assumption:
* **Twin forecast per fish:** its mean heading plus its own memory weights applied to its own past; no fish model is fitted.
* **Twin forecast of polarization squared:** exact for independent fish, (N + |Σmᵢ|² − Σ|mᵢ|²)/N².
* **Group law:** cubic in the school's own windowed polarization.
* **Score:** S_proj against the real polarization squared, cross-fitted.

**6 of 6 pre-registered predictions passed.**

| | quadratic gap S_q | measured S_proj (fitted fish twins) | linear gap (e68) |
| --- | --- | --- | --- |
| simulated independent walkers (10) | −0.013 to −0.003 | | |
| simulated aligning groups (10) | 0.51–0.87 | | |
| adult fish, real (8) | 0.80–0.93 | 0.64–0.89 | 0.10–0.37 |
| zebrafish, real (9) | 0.57–0.80 | 0.18–0.62 | 0.33–0.66 |
| virtual groups (17) | −0.10 to +0.04 (16/17 within ±0.05) | −0.10 to +0.05 | about 0 |

* Q4: every real school is above its virtual groups by ≥ 0.1 (17/17).
* **Q5: within real schools the ranking is now right** (Spearman +0.86, against −0.65 for the linear gap).
* Q6: the median size error is 0.140 (threshold 0.15), so this passed narrowly.

**What this means.**
* With only each fish's memory curve, the equation now gets fish schools nearly right in rank and roughly right in size. e68's failure was the missing level term, as diagnosed.
* The quadratic gap runs **above** the fitted twins, by about 0.03–0.2 in adult fish and 0.07–0.41 in zebrafish. That is expected: the fitted twins also explain some school-level motion through walls and persistent turning, which a memory-only twin cannot. So the gap is best read as an upper bound with the right ordering.
* The memory gap generalizes beyond sums: for a group variable that is a quadratic function of members (alignment, synchrony, energy), the twin's forecast stays exact under independence and needs only the member memories.

Figures are in `results/e72_figures/`:
* `02_comparison.png`: linear against quadratic gap, against the measured shares.

*Revised by e77:* the excess over the fitted twins came mostly from how the twins were scored, not from what they explain. Scored the same way (interleaved cross-fitting), the full fish twin is only 0.015–0.076 below the memory gap (median 0.031). The memory gap is close to an estimate, not a loose upper bound.

### e73: ALM's share in a second laboratory, and what a silencing test would need

e71 left one question open: does ALM's small share need its midbrain input? Settling it needs many silenced trials, so we searched public data first. **No public dataset has ≥ 100 steady-period silenced trials with ≥ 20 ALM units:**
* in DANDI 000009 (Guo 2017), thalamus silencing collapses ALM activity, so there is nothing left to measure;
* in DANDI 000011 (Li 2016), there are about 13 units and about 30 trials per condition;
* DANDI 000006 (Economo 2018) has no stimulation.

e73 therefore did the two things that were possible.
* **Part A, replication in a second laboratory.** 19 Economo et al. 2018 ALM sessions with ≥ 20 units (33–75 units, 196–336 trials), never analysed. The task, set-up and probes are all different. The pipeline is e71 Part A unchanged, over the whole 1.5 s delay.
* **Part B, power.** e71's simulated network with an input sized like ALM's share (control excess about 0.011), tested at 40–320 silenced trials per session. Power is for a 21-session experiment, and false positives come from coupled networks without input dependence.

**7 of 7 pre-registered predictions passed.**

| Economo 2018, 19 sessions | median | positive |
| --- | --- | --- |
| E1 measured share | **+0.026** | 19/19 |
| E2 memory-gap prediction | +0.029 (Spearman with measured 0.95) | |
| E3 virtual groups | −0.002 (all within ±0.005) | 1/19 |
| E4 beyond session drift | +0.025 | 19/19 |

| silenced trials per session | 40 | 80 | 160 | 320 |
| --- | --- | --- | --- | --- |
| power, ALM-sized input (P1, P2) | 0.74 | 1.00 | 1.00 | 1.00 |
| false positives, coupled networks (P3) | | | 0.01 | 0.02 |

**What this means.**
* **ALM's coordination share is a robust property of motor cortex before movement**, not of one dataset. It replicates in a second laboratory, and there it is about twice as large (0.026 against 0.013). The equation again ranks the sessions almost perfectly. It runs slightly above the measurement (median 0.029), as in e71.
* **Why e71's silencing test was inconclusive** (exploratory, after seeing P1). Simulated sessions are cleaner than real ones. e71's real difference-in-differences had a spread of 0.019 (interquartile) to 0.034 (standard deviation), against 0.013 in the simulations at 40 trials. Noise falls with the square root of trial count, and power reached 0.8 between 40 and 80 simulated trials. So a decisive experiment needs roughly **100–300 silenced trials per session**, against the 24–59 that exist. That is a concrete design for a new experiment, not something public data can answer.

Figures are in `results/e73_figures/`:
* `02_replication_power.png`: prediction against measurement in Economo sessions, and power against silenced trials.

### e74: the memory gap for nonlinear group variables

The memory gap was derived for a group variable that is a sum of its members. e72 extended it to polarization, which is quadratic. e74 takes the general case on markets, with industries' log volatility as the members.
* **Additive variables reduce exactly.** A variable G = Σ h(xᵢ), for any member transform h, is a sum of the transformed members. So its share is the linear memory gap on h(xᵢ). The test variable is **breadth**: the share of industries above their own 95th percentile, so h is an indicator.
* **Non-additive variables also count the same day.** The **worst industry** (the max) depends on how members move together on the same day, which no member's history shows.
  * Its twin is a Monte Carlo **memory twin**: each industry's own forecast plus a draw of its own past surprises. The draws come from days when its forecast was similar (the spread grows with the level), independently across industries.
  * A **joint twin** draws every industry at the rank its surprise had on the same past day. This keeps same-day dependence, so its share is the dynamic coordination only.
  * The Monte Carlo twin is not exact. Independent markets and virtual groups both show about 0.015, so every share is reported minus its virtual floor (the median of 5 virtual groups).

This was developed on e66's development seeds only. Confirmation used fresh seeds and three real sets: equal-weighted 1936–1971 (primary, never analysed), and value-weighted 1936–1971 and equal-weighted 1972–2026 (secondary, without predictions).

**10 of 10 pre-registered predictions passed.**

| share minus virtual floor (breadth's prediction is for the raw share) | mean | breadth | worst industry | worst, joint twin |
| --- | --- | --- | --- | --- |
| simulated independent (5) | 0.000 | 0.000 | −0.005 to +0.005 | −0.009 to +0.010 |
| simulated correlated shocks (5) | +0.001 to +0.004 | +0.002 to +0.005 | **+0.16 to +0.18** | −0.003 to +0.005 |
| simulated spillover (5) | +0.019 to +0.029 | +0.019 to +0.033 | +0.015 to +0.024 | +0.006 to +0.024 |
| simulated common factor (5) | +0.007 to +0.012 | +0.001 to +0.013 | **+0.32 to +0.42** | −0.008 to −0.003 |
| **equal-weighted 1936–1971 (primary)** | **+0.032** | **+0.024** (pred +0.024) | **+0.28** | +0.038 |
| value-weighted 1936–1971 | +0.032 | +0.025 (pred +0.025) | +0.20 | +0.021 |
| equal-weighted 1972–2026 | +0.017 | +0.006 (pred +0.016) | +0.15 | −0.014 |

* N1: the additive reduction holds in 40/40 simulated runs (largest gap 0.008).
* N2: the Monte Carlo twin agrees with the linear twin on the mean (19/20, largest gap 0.010).
* N3–N6: the max separates same-day from dynamic coordination as designed. Correlated shocks and a common factor appear only in the worst industry, and vanish against the joint twin. Spillover appears in every variable.
* R1, R2: on the primary data, real virtual groups are clean and breadth matches its prediction (+0.022 measured against +0.024).
* R3: the worst industry's share is dominated by same-day co-movement (+0.28).
* **R4 passed by 0.001 and does not replicate.** On the primary data the joint-twin max is 0.038, against 0.032 + 0.005 for the mean. Both secondary sets go the other way (0.021 and −0.014 against 0.032 and 0.017).

**What this means.**
* **The equation now has a general form.**
  * For any additive group variable (mean, breadth, counts, fractions above a threshold), the share is the memory gap on the transformed members, with no new theory.
  * For a non-additive variable, the share is a dynamic part plus a same-day part. The dynamic part is the memory gap's, about the size of the average's. The same-day part is the difference between the independent and joint twins.
* **In markets, the worst industry is mostly same-day co-movement.** Industries crash together on the day (same-day part 0.17–0.24), but a crash does not carry extra coordination into the next days beyond what the average already shows. There is **no evidence for contagion of extremes**: R4's pass is within noise, and it reverses in both secondary sets.
* **One miss to note.** Breadth on equal-weighted 1972–2026 measured 0.006 against a prediction of 0.016 (no prediction was registered there). The fit was within 0.003 on both 1936–1971 sets, so the reduction holds in the clean case. Rare-event indicators over a long, changing period may need a longer forecast history than HAR gives.

Figures are in `results/e74_figures/`:
* `02_excess.png`: share by group variable in simulated and real markets;
* `03_additive.png`: the additive reduction, prediction against measurement.

*Revised by e75:* with an exact twin, the worst industry's shares on these sets are 0.16–0.22. The floor-corrected 0.28 on the primary set was 0.07 too high.

*Revised by e76:* breadth's miss on equal-weighted 1972–2026 came from a leak in the measurement. Leak-free, it is 0.015 against a prediction of 0.016.

### e75: an exact twin for non-additive group variables

e74's Monte Carlo twin was approximate: independent markets scored about 0.015, so its shares had to be read against a virtual floor. e75 looked for a twin that needs no correction. It was developed on e66's development seeds only.
* **Why e74's twin was biased.** Under independence, the exact twin is E[G | every member's own history], and then the share is 0. An oracle twin that knew each simulated industry's true volatility scored −0.001 to +0.002. e74's twin conditioned each industry on its forecast level alone, but the group's history (yesterday's worst industry) still reveals members' recent shocks.
* **The volatility twin.** It models each industry where volatility lives, in returns:
  * its scale is a cross-fitted regression of |return| on its own decaying averages of past |returns| (three decays), floored at 0.25 × its median;
  * its standardized shocks are drawn from its own out-of-fold days;
  * log volatility is rebuilt from the two.

  A first version on squared returns was exact on long series. On 2,800-day series, though, some fitted variances collapsed, so it was replaced before the hash.
* **Variables.** The worst industry, and a **systemic day**: at least 20% of industries above their own 95th percentile, a threshold of a sum. Each was scored against the independent and the joint twin, with nothing subtracted.
* **Data.** Fresh simulation seeds. The primary real set was value-weighted industries 1926–1935 (37 industries, 2,824 days, including 1929, never analysed), with equal-weighted 1926–1935 as the replication and e74's three sets for comparison.
* **A bug, logged.** In the first run, e74's set named "primary" overwrote this experiment's primary in the code, so R1–R4 were scored on the wrong set. The key was renamed, the correction hashed with predictions unchanged, and the experiment rerun. The first run is kept in `results/e75_exact_twin_run1*`. The replication set was correct in both runs and fails R2–R4 either way.

**5 of 10 pre-registered predictions passed: all five on simulations, none on real data.**

| | worst industry S | its joint twin | systemic day S | its joint twin |
| --- | --- | --- | --- | --- |
| simulated independent (5) | +0.001 to +0.002 | ≤ +0.003 | ≈ 0 | ≈ 0 |
| simulated correlated shocks (5) | +0.16 to +0.20 | ≤ +0.002 | +0.057 to +0.060 | ≤ +0.006 |
| simulated spillover (5) | +0.009 to +0.015 | +0.008 to +0.013 | ≈ 0 | ≈ 0 |
| simulated common factor (5) | +0.30 to +0.40 | +0.005 to +0.022 | +0.033 to +0.057 | ≤ +0.010 |
| **value-weighted 1926–1935 (primary)** | **+0.206** | +0.048 | **+0.004 [−0.036, +0.018]** | −0.011 |
| equal-weighted 1926–1935 | +0.094 | +0.005 | +0.001 [−0.015, +0.008] | −0.006 |
| equal-weighted 1936–1971 | +0.209 | +0.014 | +0.046 [+0.035, +0.057] | +0.002 |
| value-weighted 1936–1971 | +0.217 | +0.011 | +0.037 [+0.025, +0.049] | +0.002 |
| equal-weighted 1972–2026 | +0.161 | +0.008 | +0.029 [+0.020, +0.037] | +0.002 |

* X1–X5 passed. The twin is exact under independence with nothing subtracted (5/5), and virtual groups are clean (200/200 within ±0.007). Same-day dependence is caught and isolated, and e74's twin read +0.018 to +0.027 where the truth is 0.
* R1 failed. One systemic-day virtual draw on the primary set reached −0.026; the threshold was 0.02.
* R2 failed. The same-day part of the worst industry was 0.159 on the primary set but 0.089 on the replication.
* R3 failed. Systemic days showed no share in 1926–1935 (intervals spanning 0 widely). In all three longer sets they were 0.03–0.05 with lower bounds above 0.
* R4 failed. Breadth missed its memory-gap prediction by about 0.03 on both 1926–1935 sets (−0.005 against 0.024; 0.023 against 0.051). The mean's reduction held there (0.022 against 0.026; 0.053 against 0.056).
* R5 failed narrowly. e74's floor-corrected max was 0.069 too high on its primary set, and within 0.03 on the other two.

**What this means.**
* **The method part worked.** For non-additive group variables there is now a twin that is exact, with no virtual-floor correction. The memory twin should model members in the variable where their randomness lives (here returns), not in a transformed variable.
* **In real markets, the worst industry is mostly same-day co-movement.** Its share is 0.16–0.22 in the long series and 0.09–0.21 in 1926–1935, and the dynamic part is small (0.005–0.048).
* **Systemic days are a coordination signal in long series** (0.03–0.05 from 1936 on), with almost no dynamic part. In 1926–1935 we cannot tell: 2,800 days of a 6.5%-frequency event are too few, so R3 is unresolved by power rather than refuted.
* **Breadth's exact reduction is fragile on real data.** It held in simulation (40/40) and on both 1936–1971 sets. It missed on 1972–2026 (e74) and on both 1926–1935 sets, always with the measured share below the prediction. An extreme-day indicator is a rare, noisy member series. The in-sample memory gap does not pay the cost of estimating its forecasts, and the cross-fitted measurement does. For rare-event members the memory gap is an upper bound, not an estimate.

Figures are in `results/e75_figures/`:
* `02_shares.png`: worst industry and systemic days, both twins and virtual groups;
* `03_bias.png`: e74's twin against the volatility twin where the truth is 0.

*Revised by e76:* breadth's miss was a leak in the measurement, not a limit of the memory gap. Leak-free, breadth matches its prediction.

### e76: a leak in the cross-fitted share, its size, and the memory gap on fresh markets

e75 left breadth below its memory-gap prediction on 3 of 5 real sets, so the next step was to find out why. An exploratory decomposition on those already-seen US sets ruled out the obvious suspects: pooled against per-member memory, OLS against ridge, and the day-of-week terms each moved the share by at most 0.008. What it found was a **leak in the measurement**:
* The cross-fitted share fits the projection for test fold f on the twin's out-of-fold forecasts from the other folds.
* Those forecasts come from member models fitted on folds that include f. So the twin has partly seen the test days, looks better than it is, and the share comes out **too low**.
* The fix: for test fold f, fit every member model on the other folds, apply it to every day, and fit the projection on those forecasts.
* Leak-free, breadth matches its prediction on every seen set (exploratory, recomputed after the run with e76's code):

| US industries, log volatility (seen) | mean: leaky → leak-free (memory gap) | breadth: leaky → leak-free (memory gap) |
| --- | --- | --- |
| equal-weighted 1936–1971 | +0.030 → +0.032 (+0.030) | +0.022 → +0.026 (+0.024) |
| value-weighted 1936–1971 | +0.029 → +0.031 (+0.028) | +0.022 → +0.028 (+0.025) |
| equal-weighted 1972–2026 | +0.017 → +0.019 (+0.019) | +0.005 → +0.015 (+0.016) |
| value-weighted 1926–1935 | +0.022 → +0.027 (+0.026) | −0.005 → +0.019 (+0.024) |
| equal-weighted 1926–1935 | +0.053 → +0.058 (+0.056) | +0.023 → +0.045 (+0.051) |

**Affected:** the market shares (e66, e69, e74, e75), e67's measured share (its simulations and the 221 brain areas), and every result scored with it (ALM in e70, e71 and e73). e64 and e65 use the same construction but were not reanalysed.

**Not affected:** e61 and e62 (one train/test split), the fish twins, and every memory-gap prediction, which involves no fitting. The leak can only flatter the twin, so affected shares were too low, never too high.

e76 was pre-registered in three parts:
* **A, simulations:** fresh seeds of e66's markets. Development used seeds 100–102 only.
* **B, fresh markets:** daily value-weighted returns of 25 size × book-to-market portfolios in Japan, Europe and Asia Pacific ex Japan, 1990–2026 (9,458 days, French library, never analysed), measured on log volatility with mean and breadth.
* **C, reanalysis:** e67's 66 simulations, the 221 brain areas and the 38 ALM sessions, each measured with e67's construction and the leak-free one. No leak-free values for these had been computed before the hash.

**10 of 12 pre-registered predictions passed.**

| simulated markets, 14,000 days (5 seeds each) | mean: leaky → leak-free (memory gap) | breadth: leaky → leak-free (memory gap) |
| --- | --- | --- |
| independent | −0.003..−0.001 → −0.001..0.000 (0.000) | −0.003..0.000 → −0.001..0.000 (0.000) |
| correlated shocks | 0.000..+0.003 → +0.002..+0.004 (+0.002..+0.004) | −0.001..+0.003 → +0.001..+0.004 (+0.002..+0.004) |
| spillover | +0.018..+0.028 → +0.020..+0.030 (+0.021..+0.030) | +0.017..+0.028 → +0.019..+0.029 (+0.019..+0.028) |
| common factor | +0.003..+0.008 → +0.008..+0.010 (+0.009..+0.010) | +0.004..+0.009 → +0.007..+0.013 (+0.008..+0.010) |

| fresh markets, 1990–2026 | mean: leak-free [90%] | memory gap | leaky | breadth: leak-free [90%] | memory gap | leaky |
| --- | --- | --- | --- | --- | --- | --- |
| Japan | +0.002 [0.000, +0.005] | +0.002 | +0.001 | +0.004 [0.000, +0.008] | +0.005 | +0.003 |
| Europe | +0.002 [0.000, +0.004] | +0.002 | −0.001 | −0.002 [−0.007, +0.004] | +0.002 | **−0.015** |
| Asia Pacific ex Japan | **+0.012 [+0.005, +0.018]** | +0.012 | +0.008 | +0.009 [−0.002, +0.019] | +0.010 | +0.003 |

* **Passed:**
  * A1: the leak lowers S in 39 of 40 simulated cells.
  * A2: leak-free is unbiased under independence (all ten cells within 0.0013 of 0).
  * A3: leak-free matches the memory gap within 0.003 in all 30 dependent cells.
  * A4: e66's construction reads below 0 in all five independent markets.
  * B1: on fresh markets the memory gap predicts the leak-free share within 0.004, for mean and breadth in every region.
  * B2: virtual groups are clean (all within 0.0015).
  * B4: the leak is larger for breadth than for the mean in all three regions. It is largest in Europe, where breadth goes from −0.015 to −0.002.
  * C1: the brain areas barely change (median +0.0005, 178/221 up, Spearman with the memory gap 0.975 before and after).
  * C3: ALM's share stands and rises slightly (median 0.018 → 0.019, positive in 37/38). The leak-free virtual sessions are within −0.004..+0.001.
  * C4: e67's simulations still meet P1 (52/52).
* **Failed:**
  * B3: volatility coordination is small abroad. Japan and Europe show 0.002, with lower bounds at 0; only Asia Pacific (0.012) cleared the bar. The memory gap predicted all three sizes.
  * C2: for the brain areas, removing the leak moved the measured share slightly *further* from the memory gap (median gap 0.0061 against 0.0057). The change is within noise: the leak barely touches trial data.

**What this means.**
* **Breadth's "failure" in e74 and e75 was the measurement's, not the theory's.** For additive group variables the memory gap is an estimate, not an upper bound, including for rare-event members.
* **The leak matters for markets and hardly anywhere else.** In markets the shares were too low by about 0.001–0.005 for the mean and up to 0.024 for breadth. In brain areas, ALM and e67's simulations the changes are within ±0.006 and mostly below 0.002, so no conclusion there changes. A likely reason, not tested: market member models fitted on a fold learn that fold's volatility level, which leaks slow information; residualized, stationary trial data have no such level to leak.
* **The memory gap is now confirmed on three fresh markets and two new continents**, at sizes from −0.002 to 0.012, with nothing fitted.
* **Coordination in volatility varies by market.** It is clear in US industries (0.015–0.03) and Asia Pacific (0.012), and small in Japan and Europe (0.002). The memory gap says why: there, the market remembers little that its members don't.
* **Lesson:** cross-fitting has to cover every fitted stage. A stage fitted on another stage's out-of-fold predictions still sees the test fold.

Figures are in `results/e76_figures/`:
* `02_against_gap.png`: measured against the memory gap, leaky and leak-free;
* `03_regions.png`: the three fresh markets.

### e77: why the quadratic memory gap ran above the fitted fish twins

e72's quadratic memory gap ranked real fish schools correctly but ran above the published shares of the fitted fish twins (e33, e38), by a median of 0.14 and up to 0.41. The roadmap read this as "what fuller twins explain" and asked for a correction. Development on simulated schools only (e32's fish model fitted to adult recording 0, run with several interaction rules and as loners) did not support that reading:
* **The memory twin was never above the true twin.** It came within +0.03 in most schools and fell far below in a school locked onto delayed neighbours (0.01 against 0.995). There, each fish's own past already carries the school's heading.
* **A linear twin from each fish's own walls and turns scored lower, not higher**, because in a school a fish's position reveals the school's, so that twin absorbs interaction. It was dropped.
* **The fitted full twin did not absorb interaction** even when the simulated interaction rule differed from the model's.
* **On stationary simulated schools**, training on the first 70% and testing on the last 30% gives the same share as interleaved cross-fitting.

So nothing in simulation puts the memory gap above the fitted twin. e77 tested the two explanations left, head to head on the real schools:
* **Split:** e33 and e38 trained on the first 70% of each recording and tested on the last 30%, while e72 cross-fits over interleaved 60 s blocks.
* **Twin:** real fish reveal more to the full twin (walls, persistent turning) than to their heading memory.

Every share is on e72's target (polarization squared), law and exact independent-fish formula. Only the member forecasts and the split change:
* memory twin, or full fish twin (refitted on the training part only, simulated 1 s ahead with 32 replicas);
* interleaved or chronological split.

**9 of 10 pre-registered predictions passed.**

| real schools | memory twin, interleaved (e72) | full twin, interleaved | full twin, 70/30 split | published (e33, e38) |
| --- | --- | --- | --- | --- |
| adult fish (8) | 0.80–0.93 | 0.72–0.91 | 0.63–0.91 | 0.64–0.89 |
| zebrafish (9) | 0.57–0.80 | 0.54–0.77 | 0.04–0.71 | 0.18–0.62 |
| virtual groups (17) | −0.10 to +0.04 | −0.09 to +0.07 | | −0.10 to +0.05 |

e72's excess, decomposed into medians over the 17 real schools:

| part of the 0.140 excess | median |
| --- | --- |
| twin: memory twin minus full twin, same split | +0.031 (0.015–0.076, all 17 positive) |
| split: full twin interleaved minus 70/30 | **+0.068** (up to 0.50) |
| law and target: full twin 70/30 minus published | +0.016 |

* **Passed:**
  * A1–A5 on fresh simulation seeds: loners within 0.009 of 0; stationary splits agree within 0.026; the bridge holds; the memory gap is never more than 0.032 above the true twin; the fitted twin matches the true one in 19/20.
  * B1: on the same 70/30 split, the full twin on polarization squared reproduces the published shares (median gap 0.031).
  * B2: virtual groups are clean (17/17).
  * B3: the chronological split lowers the share (median 0.068).
  * B4: what the fuller twin explains beyond memory is small (median 0.031).
* **Failed narrowly:**
  * B5: scored on the 70/30 split, the memory gap is a median 0.074 from the published values, not ≤ 0.07. The split correction halves e72's 0.140, and the remaining 0.03 is the twin content.

**What this means.**
* **The memory gap was not overshooting.** About half of e72's excess was the published shares' chronological split, which is sensitive to schools drifting within a recording. In 10-minute zebrafish recordings it moved one share from 0.54 to 0.04. About a tenth was the polarization target, and about a fifth was the fuller twin's extra knowledge of walls and turning (the medians don't add exactly).
* **The correction for fuller twins is about −0.03.** Under the same fitting, the full twin is 0.015–0.076 below the memory gap in every school, with rank correlation 0.98. For fish, the quadratic memory gap minus about 0.03 estimates what a full individual model would give.
* **The fish shares are higher than published, and more stable.** Cross-fitted over interleaved blocks, the fitted twins give 0.72–0.91 for adult fish and 0.54–0.77 for zebrafish (published 0.64–0.89 and 0.18–0.62). The verdicts (real above virtual, every school) are unchanged. For the zebrafish, the "small and variable" share reported in e38 was largely the split.
* **A limit of the memory twin, found in development.** When fish lock onto the school's heading, each fish's own past carries the group's. The memory gap then misses interaction that a model of the isolated fish counts (0.01 against 0.995 in one simulated school). The memory gap measures what the group adds beyond its members' records; a full twin measures what the members would do alone. They differ when members are locked.
* **Lesson:** with drifting recordings, a single chronological split measures transfer to a later period, not the share. Cross-fit over interleaved blocks (as e66 onward do), and score every stage without the test blocks (e76).

Figures are in `results/e77_figures/`:
* `02_decomposition.png`: each real school under each twin and split;
* `03_simulations.png`: simulated schools against the true twin.

### e78: do the fish closure verdicts depend on testing the last 30%?

e77 showed that a chronological split can move a drifting school's share a long way (zebrafish 60/3: 0.54 to 0.04). The closure and influence experiments e51–e60 all tested on the last 30% of each recording. They also fitted every reference stage on the first 70%: feature scaling, place maps, looseness and the own-rule innovation model. e78 reran all seven experiments with their own code, scored by their own pre-registered predictions, with the held-out 30% moved to the **first** or the **middle** of each recording. Every fitting stage moved with it.

The hashed files were not edited. In each worker process, the real-data functions were recompiled from their own source with the 70% expressions replaced, and the shared split helper was swapped. As a machinery check, the original segment reproduced every published raw number exactly (difference 0.0 in all seven experiments).

Not audited:
* e48–e50, superseded by e51;
* e61–e63, which hold out interleaved trial blocks;
* e64–e65, superseded by e71;
* e66 onward, cross-fitted and audited in e76 and e77.

**10 of 11 pre-registered predictions passed.** Every claim the white paper makes from these experiments survived both rotations:

| claim | published (last 30%) | first 30% | middle 30% |
| --- | --- | --- | --- |
| next turn open (e51 K3), G(turn) | 0.06–0.21 | 0.09–0.25 | 0.08–0.21 |
| shape closed: G(shape) < 0.05 | 6/6 | 5/6 (one 0.066) | 6/6 |
| motion closed: G(motion) < 0.03 | 6/6 | 6/6 | 6/6 |
| front steers (e52 D3, e53 Z3) | pass, pass | pass, pass | pass, pass |
| learned map beats configuration, real > virtual (e52 D1, D4) | pass | pass | pass |
| schools more self-contained than virtual (e51 K6) | pass | pass | pass |
| pigeons closed at 1 s (e53 P1 fails) | 0/3 | 0/3 | 0/3 |
| fish: slower state, smaller G(turn) (e59 T3, e60 U3 rho) | −0.59, −0.61 | −0.67, −0.57 | −0.74, −0.77 |

Of the 24 real-data verdicts, 20 matched the published ones with the first segment tested and 22 with the middle segment.

* **Flipped verdicts.**
  * First segment: e52 D2 (gain beyond the configuration, 5/6), e53 Z1 and Z2, and e58 F2 all *passed*.
  * Middle segment: the pigeon map checks P3 and P4 swapped, on maps that are nearly zero.
* **Failed (my classification error):** R1 listed e51 K1 and K2 as simulation checks. They plant a signal in, and add null features to, the real recordings, so they moved with the segment:
  * K1 (calibration), passed as published, would fail in both rotations (spread 9/12 and 8/12, needed 10);
  * K2 (null), failed as published, would pass in both (10/12 and 12/12).
  The simulation-only checks in e52 and e56–e60 did not move.

**What this means.**
* **The fish-school picture does not depend on which part of the recording is tested.** A school is closed in its shape and motion and open in its next turn, which is carried by fast fish at the front. That held at the start, the middle and the end of every recording.
* **Zebrafish turning strength does depend on it.** Zebrafish turning is predictable from inside the group (G ≥ 0.03, lower bound > 0) in 8 of 9 groups early in the recordings, 3 of 9 at the end (published) and 2 of 9 in the middle. Real > virtual: 9/9, 6/9, 5/9. e53's own calibration, which failed as published (planted signal seen in 16/24), passes in both rotations (20/24). e53 called the zebrafish strength "unconfirmed, a lack of power". It is better described as time-dependent: early in the 10-minute recordings it is clear. This matches e77's drifting zebrafish groups.
* **e51's calibration is marginal.** Its planted-signal check would pass in only 8–10 of 12 groups in the other segments. The headline values (G(turn) ≥ 0.08 everywhere) are far above the 5% signal it calibrates, so no claim rests on that margin.

Figures are in `results/e78_figures/`:
* `02_verdicts.png`: every real-data verdict under the three test segments;
* `03_e51_values.png`: e51's G for turn, shape and motion in each real school and segment.

### e79: individual stocks in eight markets

Every market result so far used portfolios: 49 US industries (e66–e69) and 25 size × book-to-market portfolios per region (e76). A portfolio averages away most of each firm's own noise, so its volatility history reveals the market's shared driver well. The share counts a common driver only as far as members' own histories reveal it poorly (e66). So individual stocks, whose volatility is mostly their own noise, should show more of the market's volatility as coordination. The one stock set analysed before (467 S&P 500 stocks, 2013–2018, e66) was void, because five years is too short for time-shifted virtual groups.

* **Data:** today's constituents of eight indices (Wikipedia, fetched 2026-10-30 UTC; `data/stocks/constituents.json`, hashed): S&P 100, Nikkei 225, FTSE 100, DAX, CAC 40, Hang Seng, S&P/ASX 50 and S&P/TSX 60. Prices are Yahoo Finance adjusted daily closes, 2000-01-03 to 2026-10-23 (6,680–6,888 days).
* **Members:** the constituents priced on at least 97% of the market's days, at most 40 (a seeded subset): 17 (Hong Kong) to 40. Daily |return| was capped at 30% against data errors.
* **Survivors only:** firms that left the index or were delisted since 2000 are absent.
* **Pipeline:** e76's, unchanged. Members are log(|r| + 0.05); the group variables are the mean and breadth; the share is the leak-free cross-fitted one; the memory gap uses HAR windows; there are 5 virtual groups per market.
* **Comparators:** the same pipeline over the same dates on the region's portfolios (US: 49 industries; Japan; Europe for the UK, Germany and France; Asia Pacific ex Japan for Hong Kong and Australia). Canada has none.
* **Development:** simulations only (400 GARCH stocks, groups of 40 stocks or 25 portfolios of 16). With a strong common volatility factor, stocks scored 0.006–0.010 and portfolios 0.000–0.001. With a moderate one the two were similar.

**8 of 9 pre-registered predictions passed.**

| market | stocks | mean log volatility: S [90%] | memory gap | breadth: S (memory gap) | portfolios of the region, same dates |
| --- | --- | --- | --- | --- | --- |
| US (S&P 100) | 40 | **+0.056 [+0.034, +0.078]** | +0.052 | +0.081 (+0.079) | +0.026 (49 industries) |
| Japan (Nikkei 225) | 40 | **+0.015 [+0.005, +0.029]** | +0.015 | +0.066 (+0.068) | +0.003 |
| UK (FTSE 100) | 40 | **+0.045 [+0.028, +0.059]** | +0.038 | +0.080 (+0.079) | +0.002 (Europe) |
| Germany (DAX) | 25 | **+0.023 [+0.012, +0.034]** | +0.021 | +0.039 (+0.042) | +0.002 (Europe) |
| France (CAC 40) | 29 | **+0.028 [+0.016, +0.039]** | +0.025 | +0.039 (+0.037) | +0.002 (Europe) |
| Hong Kong (Hang Seng) | 17 | +0.008 [−0.001, +0.022] | +0.008 | +0.041 (+0.041) | +0.013 (Asia Pacific) |
| Australia (S&P/ASX 50) | 31 | **+0.045 [+0.019, +0.074]** | +0.039 | +0.080 (+0.079) | +0.013 (Asia Pacific) |
| Canada (S&P/TSX 60) | 36 | **+0.079 [+0.034, +0.117]** | +0.070 | +0.131 (+0.120) | |

Virtual stock groups: −0.002 to +0.004 (all 40).

* **Passed:**
  * S1: on simulations, the memory gap is within 0.005 in 30/30 dependent cells (worst 0.003).
  * S3: with a strong factor, stocks > portfolios in 5/5.
  * M1: the memory gap holds for stocks within 0.01 in every market (largest gap Canada, +0.008).
  * M2: virtual groups are clean (40/40).
  * M3: stocks coordinate (7/8; Hong Kong's interval includes 0).
  * M4: stocks > portfolios in 6/7 comparisons (not Hong Kong).
  * M5: breadth matches its memory gap within 0.015 in 8/8 markets.
  * M6: the memory gap ranks the eight markets perfectly (Spearman 1.00).
* **Failed narrowly:** S2. Simulated nulls were held to 0.003. Independent markets stayed within 0.0025, but one simulated virtual group reached 0.0055. The tolerance came from two development seeds; the real virtual groups were within ±0.004.

**What this means.**
* **Coordination in market volatility is much larger among firms than among portfolios.** The market's own history improves a forecast of its stocks' mean volatility by 1.5% (Japan) to 7.9% (Canada) beyond every stock's own history. The region's portfolios give 0.2–2.6% over the same 27 years. In Europe the contrast is a factor of 10–20 (UK, German and French stocks 2.3–4.5%, Europe's portfolios 0.2%).
* This is the mechanism found in e66, now on real data. A shared driver counts as coordination only where members' own histories reveal it poorly. A portfolio's history reveals the market's state almost as well as the market's own does; a single firm's does not.
* **The memory gap predicts it, with no model fitted.** It is within 0.008 of the measured share in every market, for the mean and for breadth, and orders the eight markets exactly. Read through the equation: firms forget faster than their market.
* **Limits.** The stocks are survivors, chosen from today's index members, and Hong Kong has only 17 with full history. The share cannot say whether the shared component is spillover between firms or a common driver they reveal poorly (Section 10.3 of the white paper). It says how much a firm-by-firm forecast misses.

Figures are in `results/e79_figures/`:
* `02_markets.png`: stocks, memory gap and portfolios in each market;
* `03_against_gap.png`: measured against predicted, stocks, portfolios and simulations.

### e80: switching interaction off

Every result before this one was observational. Here the experimenter removes or weakens interaction, and the question is whether the share follows. Two public datasets, both new to this project (species corrected on 3 November 2026; the analysis is unchanged, see the note below):
* **Light and dark** (Liu & Burbano 2025, Zenodo 15351451): zebrafish (*Danio rerio*), 9–10 recordings each of pairs and triads in a bright arena and a dark one lit only by infrared, which zebrafish cannot see (about 10.5 min, 33 Hz). Darkness removes vision; the lateral line remains. Ten fish were also recorded alone in each light, which gives two things no earlier dataset had: groups of single fish that certainly never interacted (synthetic groups), and a twin built from the measured behaviour of a fish without a partner (the iso twin).
* **Virtual reality** (Escobedo et al. 2026, figshare 30539666): rummy-nose tetras (*Hemigrammus rhodostomus*), a real fish with a virtual conspecific that responds to it (C1; C2, a faster clone), the same C2 clone made independent of the real fish (C3, one-way interaction), and pairs of real fish (2R). Continuous intervals of at least 30 s; 6–21 experiments per condition.
* **Measurement:** member variable planar velocity at 10 Hz; group variable the summed velocity; horizon 0.5 s; e67's windows; cross-fitted over 30 s blocks with purged training rows (nothing fitted sees test rows or their lags). Two twins: the member twin (each fish's own past, weights fitted on the members) and, in the light data, the iso twin (weights fitted on single fish recorded alone in the same light). Memory gaps from both.
* **Development:** simulated tetras only (speed relaxation, wall avoidance in a round arena, persistent noise, attraction and alignment with a 0.2 s reaction delay; all-range, short-range only, or one-way). Without a reaction delay, strongly coupled fish lock together and the member twin's share falls (e77's limit); the delay restores a share that grows with coupling, and the 0.5 s horizon gave the clearest one.

**15 of 15 pre-registered predictions passed.**

| condition | recordings | S, member twin [90%] | S, iso twin [90%] |
| --- | --- | --- | --- |
| pairs, bright | 9 | **+0.023 [+0.019, +0.028]** | +0.087 [+0.055, +0.121] |
| pairs, dark | 10 | −0.001 [−0.001, −0.000] | +0.027 [+0.007, +0.050] |
| triads, bright | 9 | **+0.089 [+0.076, +0.099]** | +0.200 [+0.134, +0.247] |
| triads, dark | 10 | +0.001 [−0.001, +0.003] | +0.007 [+0.001, +0.012] |
| synthetic groups of single fish | 40 | −0.003 to +0.001 | −0.001 to +0.038 |

| VR condition | experiments | S, member twin [90%] |
| --- | --- | --- |
| 2R: two real fish | 6 | +0.031 [+0.023, +0.040] |
| C1: virtual fish responds | 8 | +0.013 [+0.010, +0.015] |
| C2: faster clone responds | 21 | **+0.063 [+0.058, +0.066]** |
| C3: the same clone ignores the fish | 11 | **+0.019 [+0.012, +0.025]** |
| C2 − C3 | | +0.044 [+0.036, +0.052] |

* **Passed:**
  * S1–S4 (simulations, fresh seeds): nulls within 0.004 (member) and 0.007 (iso) in 80 groups; all-range above short-range coupling (p < 0.0001, both twins, pairs and triads); mutual above one-way (0.116 against 0.056); memory gap within 0.01 in 99%.
  * L1: synthetic groups of real single fish are clean: member twin within 0.003 (40/40), iso twin within 0.02 in 85% (largest 0.038).
  * L2: time-shifted virtual groups within 0.009 (38/38).
  * L3: bright pairs and triads coordinate: lower bounds +0.019 and +0.076, against at most +0.001 for synthetic groups.
  * L4: **the causal test**: in the dark the share falls, every bright recording above every dark one (Mann–Whitney p = 0.0001 for pairs and for triads).
  * L5: the same with the twin from fish recorded alone (p = 0.003 pairs, 0.0001 triads).
  * L6: memory gap within 0.01 of the measured share in 97% of recordings.
  * L7: the dark share is less than half the bright one (it is about zero).
  * V1: VR virtual groups clean in 95% (one 38 s experiment scored −0.18).
  * V2: **reciprocity**: the same virtual clone gives 0.063 when it responds to the real fish and 0.019 when it does not (difference +0.044 [+0.036, +0.052]).
  * V3: one-way interaction is still detected (C3 lower bound +0.012).
  * V4: two real fish coordinate (+0.031).

**What this means.**
* **When the experimenter switches interaction off, the share goes to zero.** In the dark, pairs and triads score −0.001 and +0.001 on the member twin, the same as groups of fish that were never together. In the light the same kind of group scores 0.023 (pairs) and 0.089 (triads). Every bright recording scores above every dark one. This is the first interventional validation of the measure: the share tracks interaction that was manipulated, not only interaction that was inferred.
* **Halving interaction lowers it.** With the virtual clone's behaviour held fixed, making it ignore the real fish (one-way instead of mutual) lowers the share from 0.063 to 0.019. One-way interaction is still counted, as it should be.
* **The twin built from fish alone agrees.** Using single fish recorded alone (a measured, not modelled, no-interaction twin), the bright–dark contrast holds (0.087 against 0.027 for pairs, 0.200 against 0.007 for triads). It is larger than the member twin's because the iso twin also counts how a partner changes a fish's own movement. It is not zero in the dark, and synthetic groups reach 0.038 on it, because fish differ from each other.
* **The memory gap holds here too**: within 0.01 in 37 of 38 recordings, with no twin fitted.
* **Limits.** Darkness also changes how a fish swims alone; the member twin is fitted within each condition, so this does not enter the comparison, but the iso twin is sensitive to it. The C1 clone (slower, differently parameterised) scored below C3, so the share depends on the partner's kinematics as well as on reciprocity; only C2 against C3 holds kinematics fixed. The simulations used for development were generic, not fitted to these fish.
* **Correction (3 November 2026).** The light and dark recordings were first described here, in the code and in the papers as rummy-nose tetras. Liu & Burbano's fish are zebrafish (*Danio rerio*); only the virtual-reality fish are tetras. The analysis uses positions and velocities only, with no species-specific parameter, so every number above is unchanged. The descriptions in `experiments/intervention_data.py`, `e80_intervention.py` and `e80_figures.py` were corrected, with the new hashes logged.

Figures are in `results/e80_figures/`:
* `02_light_dark.png`: per-recording shares, both twins, bright and dark, with synthetic groups;
* `03_virtual_reality.png`: shares with a virtual fish that responds or ignores;
* `04_against_gap.png`: measured against the memory gap.

### e81: the interaction share against the Diebold–Yilmaz spillover index

An economist's first question: does the share tell me anything the standard spillover index does not? The Diebold–Yilmaz (2012) total spillover index fits a VAR to the members and reports the share of their forecast-error variance due to each other's shocks. We computed both, with mean pairwise correlation as a baseline, on e79's eight stock panels in rolling 500-day windows stepped monthly (294–304 windows per market). The share is its memory gap. A placebo share comes from a virtual market of time-shifted firms.

**7 of 10 pre-registered predictions passed.**

* **Known answers (simulations, D1–D4, all passed).** The two measures see different things:

| simulated market (40 members) | S | DY (%) |
| --- | --- | --- |
| independent | +0.000 to +0.001 | 0.7 |
| simultaneous correlated shocks, no spillover | +0.001 to +0.003 | 44–46 |
| volatility spillover between members | **+0.022 to +0.028** | **0.6** |
| common volatility factor | +0.009 to +0.010 | 71–74 |

  DY registers members being hit at the same moment and misses dynamic spillover; development showed this at VAR orders 1, 2, 5 and 22 and on a smoothed proxy. S does the reverse. In 500-day windows DY is inflated by estimation (independent firms 16) and cannot tell spillover (14.7) from independence. S is biased up too (median 0.005–0.009) but keeps spillover apart (0.021). With 400 members S takes 3 ms per window against 66 ms.
* **Real markets.** S and DY are not the same measurement (R1): their correlation across windows ranges from −0.82 (Japan) to +0.64 (Canada).
* **Next month's market volatility (R2–R4, failed).** Neither index improves a HAR forecast. Adding S to HAR + DY helped in 1 of 8 markets; adding DY to HAR + S helped in 2 of 8. The placebo did not help either (R5 passed, 7/8).
* **Next month's correlation between firms (R6, passed).** Adding S to HAR + DY lowered the out-of-sample error in all 8 markets: Germany 11.3%, France 12.9%, US 5.4%, Australia 4.1%, UK 3.3%, Hong Kong 1.4%, Japan 0.5%, Canada 0.3%. DY alone beyond HAR helped in 5 of 8 and hurt in the UK and Australia (−9%).
  * *Exploratory, after the run:* the placebo beyond HAR + DY gave −1.2% to +3.0%. That is small except in Australia (+3.0%, against +4.1% for S), so the gains in Germany, France, the US and the UK are not just smoothness.
  * The pooled in-sample coefficient is **negative**: −0.019 [−0.031, −0.009] per standard deviation of S. Windows with a higher share are followed by *lower* correlation between firms. The direction was not predicted.

**What this means.**
* **Against the standard tool, the share is a different instrument, not a better forecaster.** In simulations it sees spillover and ignores simultaneous shocks; DY does the opposite. That is the clearest single argument for the share in finance: DY's "spillover" is mostly contemporaneous correlation.
* **Neither forecasts volatility.** Claims that either index predicts market volatility beyond its own history are not supported here.
* **The share carries information about future diversification.** It improves forecasts of next month's correlation between firms in every market, beyond DY and the market's volatility. Its sign is negative, which needs explanation before it can be used: a possible reading is that coordination built over two years precedes a period in which firms move more independently. This was found, not predicted, and needs a fresh pre-registered test.
  * *Revision (e82):* this did not hold up. Most of the gain came from not controlling for the current correlation between firms, which is persistent. With that control, gains remained only in France, Germany and Japan. On six fresh markets, adding the share helped in 2 of 6, and adding it without the control, as in this test, also helped in 2 of 6. Only a small pooled negative association survives.
* **Limits.** The volatility proxy is daily |return|; Diebold and Yilmaz used intraday range, which is less noisy and may help DY. Firms are survivors (e79).

Figures are in `results/e81_figures/`:
* `02_known_answers.png`: S and DY on simulated markets;
* `03_over_time.png`: S, DY and next month's volatility over 2002–2026, each market;
* `04_forecast_gains.png`: out-of-sample gains for volatility and correlation.

### e82: the correlation forecast, confirmed on fresh markets? No

e81's one positive real-market result was that the share improved forecasts of next month's correlation between firms in all eight markets, with a negative sign. e82 tested it again with the direction stated in advance.

**Discovery (e81's markets, exploratory, before hashing).** e81 never controlled for the *current* correlation between firms, over the last 21, 63 and 500 days. Correlation is persistent, so the share could have been standing in for it.
* With the control, the share's out-of-sample gain fell from +0.3–12.9% to: France +6.4%, Germany +2.8%, Japan +2.1%, about zero in the other five.
* The coefficient stayed negative in 7 of 8 markets. The placebo's coefficients split 4/4.

**Confirmation.** Fresh markets and firms that e79–e81 never touched, under e79's panel rule unchanged (at least 15 firms with prices from 2000):
* 14 indices were fetched (`experiments/stock_data_fresh.py`; the code was hashed before the prices finished downloading, and the data were hashed after).
* Only six qualified: Sweden, India, Korea, New Zealand, Brazil, and S&P 500 firms outside the S&P 100. Most European indices have too few members with 26 years of history.
* The model is HAR + DY + current correlation; the predictions F1–F6 are listed in the docstring.

**2 of 6 predictions passed.**

| | Discovery (8 markets) | Fresh (6 markets, scored) |
| --- | --- | --- |
| Pooled coefficient of S, per s.d. | −0.015 [−0.020, −0.008] | −0.008 [−0.011, −0.003] (F1 pass) |
| Placebo pooled coefficient | +0.002 [−0.003, +0.005] | +0.002 [−0.002, +0.007] (F4 pass) |
| Markets with a negative coefficient | 7/8 | 3/6 (F2 fail) |
| Out-of-sample gain > 0, with correlation control | 4/8 | 2/6: Sweden +3.0%, New Zealand +0.8% (F3 fail) |
| S's gain beats the placebo's | — | 3/6 (F5 fail) |
| e81's test, no correlation control | 8/8 | 2/6 (F6 fail) |

**What this means.**
* **e81's correlation result does not replicate.** It does not replicate even in e81's own form (F6). As a forecasting tool for diversification, the share fails.
* **What survives is a small pooled association.** Higher share, slightly lower correlation next month, beyond current correlation. It is about half the size in fresh markets, the placebo shows nothing, and it is carried by a few markets: France, Germany, Japan and Sweden.
* **Net position for an economist (e81 + e82).** The share is not a better forecaster than the standard tools, of either volatility or correlation. Its case in finance rests on measuring something different: in simulations it sees dynamic spillover that the Diebold–Yilmaz index misses (e81, D1–D4), and it predicts its own size from memory alone (e79).
* **Limits.** Only six fresh markets qualified, all survivors. Global shocks are shared across markets, so markets are not independent replications.

Figures are in `results/e82_figures/`:
* `02_coefficients.png`: the share's and placebo's coefficients per market, discovery and fresh;
* `03_forecast_gains.png`: out-of-sample gains with the correlation control, and without it (crosses).

### e83: a dose–response in 50-fish schools? No, because darkness did not switch interaction off

e80 switched vision off in pairs and triads, and the share fell to zero. e83 asked for a graded effect in large groups:
* **Data:** Lafoux et al. (2023), schools of about 50 rummy-nose tetras, 50 minutes each at one of eight fixed light levels (0–900 lux), two replicates (`experiments/lafoux_data.py`).
* **Exclusion:** one recording was excluded under a rule set before any analysis, because its metadata said 0.3 while its name said 1.0. That leaves 15.

**Method: the memory gap from broken tracks.** FastTrack gives 140–1,500 identities per recording for 50 fish, with median tracks of 4–80 s and 5–23% of frames missing. A twin from each fish's own history is impossible, but the memory gap needs only short-lag statistics:
* each fish's autocovariance, from frames within a track;
* the cross-covariance between fish, from pairs of tracks that coexist in time (so they must be different fish);
* the school's autocovariance g = n·o + n(n−1)·c, with n = 50.

In development, a first version counted a fish before and after a track break as two fish and gave shares above 1; restricting cross pairs to coexisting tracks fixed it. On simulated 50-fish schools with tracks broken as in the data, the estimator recovered the full-identity share within 0.011. It read up to 0.007 for loners, and up to 0.003 for time-shifted schools.

**4 of 9 predictions passed.**
* **Simulations:**
  * S1 passed (accuracy, 0.011);
  * S3 passed (order by visual range, 6/6);
  * S2 failed narrowly: one loner school read 0.0073 under the heaviest fragmentation, against a limit of 0.005.
* **Real schools:**

| light (lux) | 0 | 45 | 90 | 135 | 270 | 450 | 675 | 900 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S, replicates | 0.138, 0.233 | 0.165, 0.278 | 0.099, 0.201 | 0.173, 0.112 | 0.228, 0.171 | 0.190, 0.143 | 0.144, 0.111 | 0.204 |

  * Time-shifted schools: 0.0000–0.0013 (R3 and R4 passed).
  * No dose–response: Spearman −0.14 (R1 failed). Darkness was not below the lit schools (R2, R5 and R6 failed).

**Why (exploratory, after the run).**
* **Not duplicate tracks.** Pairs of tracks under 0.3 body lengths apart are 0.01% of pairs, and dropping all pairs under 3 body lengths leaves the dark share unchanged (0.14, 0.27).
* **Distance-dependent coordination in the dark.** The velocity correlation between dark fish is 0.28–0.43 under 1 body length, about 0.2 at 3–10, 0.06 at 10–20 and 0 beyond. Lit fish show 0.38–0.52, 0.18–0.28, about 0 and about 0. A camera or water-flow artifact would not fall with distance.
* **The school-wide movement is behavioural.** It is smooth (autocorrelation 0.5–0.9 at 1 s, about 0 at 5 s) and contains no high-frequency jitter.
* **What light changes is dispersion.** 9–30% of pairs are over 20 body lengths apart in the dark, against 2–4% when lit, and this matches Lafoux's interindividual distance.

**What this means.**
* **The test was uninformative, not a counterexample.** Fish in "darkness" (0 ± 0.1 lux, a projector's black) still coordinate with nearby fish, by lateral line or residual light. Lafoux's order parameters miss this because it is not school-wide order. The share picks it up, as an interaction measure should. But with no condition free of interaction, a dose–response cannot be tested in these data.
* **Contrast with e80.** Liu & Burbano's infrared arena removed interaction in zebrafish pairs and triads (shares around 0). Rummy-nose tetras in a large tank at 0.1 lux did not lose it. The two differ in species, group size and kind of darkness, so the contrast does not isolate any one of them.
* **Method gain.** The share can be measured from short, broken tracks of large groups, with no identities kept. This was validated in simulation; real large groups give large shares, and shifted controls give zero.
* **Next:** a large-group intervention that removes interaction outright, such as true infrared darkness or a blocked lateral line, or the within-session darkness-to-light ramps in Lafoux's continuous set, though their darkness has the same limitation.

Figures are in `results/e83_figures/`:
* `02_dose_response.png`: the share at each light level (dots) and time-shifted schools (crosses);
* `03_simulations.png`: the broken-track estimator against full identities, and the share by visual range in simulation.

### e84: can memory find primes? Faintly, but divisibility is a hundred times better

**The question.** A search does not need to predict every prime; it would gain if it knew where to look. Can the memory framework find prime-rich places ahead of time, beyond what standard filters already give? This is a forecasting test along the number line, not a claim that primes interact: their structure is arithmetic.

**Setup** (`experiments/e84_primes.py`, `experiments/prime_data.py`; no data to download, an exact segmented sieve checked against Miller–Rabin):
* **Blocks** of 30,030 = 2·3·5·7·11·13 consecutive integers, aligned so every block has the same divisibility by 2–13.
* **Members:** primes per block ending in 1, 3, 7 and 9 (every prime above 5 ends in one of these). **Group:** their total.
* **Informed baseline:** trial division by every prime ≤ 101, and the prime number theorem's density among the survivors. No primality test needed.
* **Memory:** forecast each block's surplus over the baseline from the blocks before it (window means over 1, 5 and 22 blocks), with coefficients frozen from the development range, and also refitted within each range.
* **Search value:** primes per primality test in the 10% of blocks a score ranks highest, against testing every survivor. For scale: an oracle that knows each block's true count (the ceiling for any way of choosing blocks), and simply trial-dividing up to 1000 instead of 101.
* **Controls:** "sieve-random" primes (each survivor prime by chance at the baseline rate), and the same with a planted slowly varying rate (memory should find it).
* **Ranges:** developed on 10⁹, then the code was hashed; held out 10¹² and 10¹⁶, 5,000 blocks each.

**9 of 9 predictions passed.**

| | 10⁹ (development) | 10¹² | 10¹⁶ |
| --- | --- | --- | --- |
| primality tests per prime found, trial division to 101 | 2.48 | 3.29 | 4.39 |
| block variance / Poisson: primes vs sieve-random | 0.43 vs 0.58 | 0.56 vs 0.71 | 0.71 vs 0.79 |
| lag-1 correlation of neighbouring blocks | −0.064 | −0.041 | −0.015 |
| search gain from memory (frozen) | +0.24% | +0.32% | +0.10% |
| oracle ceiling for choosing blocks | +3.0% | +4.0% | +5.2% |
| trial division to 1000 instead of 101 | +46% | +47% | +47% |
| twinshare, digit classes | +0.0003, none | −0.0017, none | −0.0002, none |

* **Controls (C1, C2):** sieve-random primes show no memory and no coordination. The planted signal is detected by twinshare (S 0.054–0.069) and by memory search (+3.7–4.4%, over 20 s.d.). So the test had power.
* **There is real structure beyond divisibility (P2, P3).** Primes are more evenly spread than divisibility alone makes them, and a prime-rich block tends to be followed by a slightly poorer one. Both fade as numbers grow, as the Hardy–Littlewood conjectures predict (Montgomery and Soundararajan: counts in short intervals are sub-Poisson, with neighbours negatively correlated).
* **It is useless for search (P4–P6).** Memory improves primes per test by at most 0.3%. Even perfect knowledge of every block's count would gain only 4–5%, because block counts barely vary. One more cheap divisibility filter gains 47%.
* **twinshare (P7):** "No coordination detected" in both held-out ranges.

**What this means.**
* **The equation found something true but not useful.** The negative memory between neighbouring blocks is genuine number theory, which the framework recovered blind. But it is a fraction of a percent.
* **Choosing blocks is the wrong level.** Primes per block vary by a few percent at most; what matters is which individual numbers survive divisibility, and that is what sieves already exploit. A useful "where to look" method would have to beat trial division per unit of work, and memory of past counts cannot.
* **For the method:** a clean negative with working controls. The share and memory gap stay at zero where nothing coordinates, and detect a planted signal of a few percent.

Figures are in `results/e84_figures/`:
* `02_search_gain.png`: primes per primality test gained by memory, the oracle and deeper trial division, per range;
* `03_block_memory.png`: autocorrelation of block surpluses, primes against sieve-random;
* `04_twinshare.png`: interaction shares of the digit classes, with the no-coordination bars.

### e85: memory on top of every filter? Nothing left, even before paying for it

**The question.** e84 tested memory on top of a shallow filter (trial division to 101). Would memory add something if every standard filter were applied first and memory came last? And does it still pay when a search is charged for the information memory needs (a block's count is known only if the block was tested)?

**Setup** (`experiments/e85_prime_filters.py`; e84's blocks, baseline and forecast):
* **Filter depths:** trial division by every prime ≤ D, for D = 101, 10³, 10⁴, 10⁵, 10⁶ (all below √x, so each still leaves composites).
* **Memory, free:** e84's best case, where every earlier block's true count is known.
* **Memory, charged:** a sequential search that knows only the blocks it tested. It tests the first 100 blocks to warm up, then tests a block only if its forecast is in the top 10% so far.
* **Oracle:** each block's true count, the ceiling for any way of choosing blocks.
* **Controls at depth 10⁴:** sieve-random primes, and e84's planted slow signal, with coefficients learned on a simulated development series.
* **Ranges:** developed on 10¹⁴; held out 10¹⁷ and 10¹⁸ (5,000 blocks each).

**8 of 8 predictions passed.**

| depth D | tests per prime (10¹⁷ / 10¹⁸) | memory, free | memory, charged | oracle ceiling |
| --- | --- | --- | --- | --- |
| 101 | 4.66 / 4.93 | +0.07% / +0.04% | +0.22% / −0.13% | 5.4% / 5.6% |
| 1,000 | 3.17 / 3.35 | +0.04% / −0.03% | +0.10% / −0.05% | 5.3% / 5.5% |
| 10,000 | 2.38 / 2.52 | +0.08% / −0.02% | +0.09% / +0.10% | 4.9% / 5.1% |
| 100,000 | 1.91 / 2.02 | +0.04% / −0.02% | +0.09% / +0.16% | 4.5% / 4.7% |
| 1,000,000 | 1.61 / 1.69 | +0.06% / −0.03% | +0.08% / +0.07% | 4.0% / 4.2% |

* **Memory adds nothing at any depth (H3–H5).** Every memory gain is within 1.7 random s.d. of zero (noise about 0.1%). The mean free gain over the deep filters is +0.02%.
* **Each filter step is worth 19–47% (H1, H2).** Every tenfold deeper filter saves more tests than even the oracle could, and deeper filters shrink the oracle's ceiling. They absorb the block-to-block variation that memory would need. The lag-1 correlation e84 found (−0.03 at depth 101 here) is gone by depth 10⁴.
* **The test had power (C1, C2).** A planted 3% slow signal is found by the charged search too: +2.2% and +1.7%, about 10 s.d. Sieve-random primes give nothing.
* **A bonus (H6).** Near √x the baseline itself needs Buchstab's function: the survivors of trial division to 10⁶ are not distributed as Mertens' product says. The predicted correction matched to 0.0004 (0.9909 against 0.9905, and 0.9952 against 0.9948).

**What this means.** e84's faint memory is a trace of divisibility by primes beyond the filter, so stacking filters first leaves memory nothing to work with. The one pattern it found was real number theory, and the method stays at zero where nothing is left while detecting a planted signal. The proposal "filter with everything known, then add memory at the end" is now tested: it gives no advantage, with or without paying for the information.

Figures are in `results/e85_figures/`:
* `02_memory_by_depth.png`: memory's search gain (free and charged) and the oracle ceiling at each filter depth;
* `03_filters.png`: tests per prime by depth, and the baseline's Buchstab drift.

### e86: the Ising magnet. The share peaks at the phase transition and sharpens with size

**The question.** Does the share behave like a physical marker of collective behaviour? The 2D Ising model has an exactly known critical temperature, Tc = 2.269.

**Setup** (`experiments/e86_ising.py`): heat-bath dynamics on L × L periodic lattices. Members are the spins, the group is the magnetization, and the share is the memory gap (windows 1, 5, 22 sweeps). The temperature runs from 1.6 to 10. Controls: virtual lattices (each spin shifted in time), and J = 0 (independent spins). Developed on L = 8 and 16 with seed 0; the final run used L = 16, 32 and 64, seed 1, with 40,000 sweeps.

**4 of 5 predictions passed.**
* **Vanishes at both extremes (I2):** 0.08 in the frozen ordered phase (T 1.6), and 0.05–0.06 at T 10, where spins are nearly independent.
* **A peak at the transition that sharpens with size (I3, I4):** the maximum is 0.68 at T 2.50 for L 16, 0.79 at T 2.35 for L 32, and 0.89 at T 2.30 for L 64. The peak rises and moves toward Tc as the lattice grows, the finite-size signature of a quantity tied to the phase transition. The curve tracks the magnetic susceptibility, the textbook marker, closely.
* **Size-independent far from Tc (I5):** within 0.013 of L 16 at T ≥ 3.4, as short-range physics should be.
* **Failed (I1):** the virtual-lattice control reached 0.094 at L 64, T 2.30 (and 0.054 at L 32, T 2.25), against a limit of 0.02. Right at Tc a large lattice relaxes so slowly that 40,000 sweeps hold only a few independent swings of the magnetization, and randomly shifted copies of the spins still partly line up (the record-length effect behind the e70 floors). The real share there, 0.89, is far above that floor, but the control as pre-registered failed. J = 0 stayed at 0.0003 everywhere.

**What this means.** The share is a measure of collective dynamics with the textbook behaviour of a phase-transition marker. It is near zero when nothing moves together (frozen order) and when nothing interacts (high temperature), and largest at criticality, more so in larger systems. Near criticality its virtual-group floor needs records much longer than the slowest relaxation time.

Figure: `results/e86_figures/02_share_vs_temperature.png` (share and susceptibility against temperature, per lattice size).

### e87: RANDU. The share is blind to a famous bad generator, and catches a flaw classical tests miss

**The question.** RANDU (x_{n+1} = 65539 x_n mod 2³¹) was IBM's 1960s generator. All its triples of consecutive outputs lie on 15 planes. Does the share notice?

**Setup** (`experiments/e87_randu.py`): eight interleaved output streams are the members, and their sum is the group; twinshare's full analysis runs on 20,000 steps. Generators: PCG64 and MINSTD (good), RANDU, and a "leaky" control in which a stream occasionally copies the previous output of the next stream (a buffer bug). Classical check: a 20³-cell chi-square test on a million triples. Developed on seeds 1–2; the final run used seeds 3–7.

**5 of 5 predictions passed.**
* **The share is blind to RANDU:** S between −0.0011 and −0.0001 and "No coordination detected" in every seed. RANDU's flaw is arithmetic modulo 2³¹, which leaves no linear correlation for linear forecasts to use.
* **The triple test catches RANDU:** p underflows to 0 in every seed.
* **The share catches the leak:** S 0.009–0.011, "Coordination detected", in every seed.
* **The triple test misses the leak:** p 0.01–0.6.
* **Good generators:** both tools stay silent.

**What this means.** The share and classical uniformity tests see different things. The share detects dependence between streams over time, while a static distribution test detects modular structure. Neither replaces the other. This is a clean, documented blind spot: the share sees only what linear forecasts can see.

Figure: `results/e87_figures/02_two_tools.png`.

### e88: group size from 2 to 10,000. The share saturates; exact theory for the mean-field case

**The question.** Under one fixed interaction rule, does the share grow, vanish or level off as groups grow?

**Setup** (`experiments/e88_group_size.py`): agents align their 2D velocity with neighbours (v' = 0.9[(1 − w) v + w · neighbours' mean] + noise, w = 0.5) and move in a periodic square at fixed density. Three neighbourhood rules: mean-field (everyone), topological (the 7 nearest, as in starlings), and metric (within distance 1.5). Group sizes run from 2 to 10,000. For mean-field the memory gap has an exact formula: the group momentum is an AR(0.9) process, and each member is a mixture of that shared mode and a fast deviation whose memory depends on N. Developed with seed 0 up to N 1,000; the final run used seed 1 up to N 10,000.

**6 of 6 predictions passed.**
* **Theory confirmed:** for mean-field, the measured share is within 0.011 of the exact value at every size (0.142 at N 2, 0.301 at 10, 0.5155 at 10,000).
* **A steep rise, then saturation:** the share climbs from 0.14 at N 2 to about N 30, then levels off. Mean-field reaches 0.50–0.52; the topological rule 0.37–0.40 and the metric rule 0.36–0.39 at N 100–10,000. It varies by at most 0.023 across N 1,000–10,000.
* **Why it saturates:** the group's memory is the same at every size (lag-1 = 0.90); what changes is how much of it each member carries, and that settles once a neighbourhood is full.
* **Rules:** all three coincide at N 2–3, where everyone sees everyone. Local interaction saturates 0.12–0.14 below mean-field.
* **Controls:** virtual groups ≤ 0.0007.

**What this means.** The share is an intensive property of the interaction rule, not of group size: neither more members nor fewer inflate it, once a neighbourhood is complete. This answers one of the lab's standing theory questions.

Figure: `results/e88_figures/02_share_vs_size.png`.

### e89: time reversal. The share does not know which way time runs; influence does

**The question.** Run a record backwards. Does the share change, and can the method tell cause from effect?

**Setup** (`experiments/e89_time_reversal.py`):
* **Common driver:** 10 members with a hidden shared driver and no interaction.
* **Leader:** member 0 drives the other nine.
* **Real markets:** daily log volatility of 30 US industries, developed on 1936–1971 and held out on 1972–2026.

Measures, each computed forwards and on the reversed record: the memory gap; twinshare's cross-fitted share; and each member's influence (how much its history improves the group's forecast).

**5 of 5 predictions passed.**
* **The memory gap is identical both ways,** to within 4·10⁻¹⁵. This is a theorem: it is built from autocovariances, which read the same forwards and backwards.
* **The share is nearly identical too** (differences at most 0.006, every dataset). It measures how much coordination there is, not its direction.
* **Influence carries the direction.** The leader's influence is 0.62 forwards and 0.06–0.07 reversed, a tenfold drop in every seed. With a common driver, no member has influence either way, so the method distinguishes "one member drives the others" from "an outside cause drives everyone".
* **Markets 1972–2026:** the share is 0.021 forwards and 0.019 reversed; no industry leads either way (influence ≤ 0.002).

**What this means.** A reviewer's natural worry, "is the share just correlation?", has a precise answer. The share alone is time-symmetric, like any second-order measure; the influence decomposition is not, and it identifies leaders only in the true direction of time.

Figure: `results/e89_figures/02_forward_vs_reversed.png`.

### e90: Perth petrol. Influence names the price leader before collusion, and sees no leader once it is a calendar rule

**The question.** Can the method help detect tacit collusion? Byrne and de Roos (2019, *AER*) documented, from public data, how BP initiated tacit collusion in Perth petrol: a price cycle that BP (and Caltex before 2006) led in 2004–2008; Wednesday jumps by BP, followed by the market on Thursday, in 2009–2011; and from mid-2012 a calendar rule, everyone jumping on Thursday and cutting 2 cents a day, with margins up 74%.

**Setup** (`experiments/e90_petrol.py`, data `experiments/fuelwatch_data.py`):
* **Data:** FuelWatch daily prices for every Western Australian station, 2001–2026 (CC BY 4.0), and daily terminal gate prices from 2003.
* **Members:** brands, each its Perth metro stations' mean unleaded price minus the wholesale price (7–10 brands with at least 10 stations).
* **Recorded causes:** the day's change in wholesale price, and weekday.
* **Periods,** fixed from the published account before any share was computed: A 2004–2008 (cycle), B 2009–2011 (Wednesday leader), C 2012–2015 (Thursday rule); D–F 2016–2026, never analysed in the literature (exploratory).
* **Development on simulations only:** a station-level model of 8 firms. A leader whose rivals follow a day later stands out by influence (0.46). Firms that all jump on a fixed weekday show no leader, because the weekday explains them. A fixed-weekday leader with only 20% of its stations is also invisible. Virtual groups had to be shifted by whole weeks, over records of whole weeks. Day-level shifts moved weekly cycles off their weekday and read up to 0.39.

**6 of 7 predictions passed.**
* **Simulations (P1–P3):** all as developed on fresh seeds. Leader 0.38–0.43 (0.13–0.19 when leadership is shared); a Thursday rule no leader (≤ 0.005); Wednesday leadership 0.49–0.72 with the whole network, ≤ 0.021 with 20% of it.
* **Perth 2004–2008 (P5):** BP has the largest influence of all brands, **0.215**. The method was told nothing about who leads. Peak, a 20-station independent chain, is second (0.164). The published account does not mention it, and this is unexplained.
* **Perth 2012–2015 (P6, P7):** no brand has influence above 0.034, and BP's fell from 0.215 to 0.001. A calendar rule shows no leader.
* **Failed (P4): virtual groups in three real periods.** Shifted brands gave shares up to 0.11 (B), influence up to 0.16 (E) and shares up to 0.19 (F), against limits of 0.02 and 0.04. These periods contain regime changes: the 2009–2011 transition, 2020 and 2022 price shocks, and rebranding. Shifting a member moves those regimes too. The controls were clean in A, C and D (virtual influence ≤ 0.039). So B, E and F, including the apparent leaders there (Peak and Gull in B, Puma in E, Atlas and Liberty in F), are not interpretable.

**Not predicted (exploratory).**
* **The share:** 0.29 in 2004–2008 (memory gap 0.13; jumps are far from linear), and 0.088 [0.026, 0.142] in 2012–2015 against virtual groups ≤ −0.016. In simulation, a pure calendar rule read 0. The real Thursday rule therefore carries some day-to-day coordination beyond the calendar, without any leader.
* **The standard screens,** on the same data: mean margin 3.6, 6.0 and 9.3 cents per litre in A, B and C, and coefficient of variation 0.73, 0.59 and 0.37. Both flag the collusive period clearly, as the literature found.

**What this means.** Influence and the classical screens see different stages of tacit collusion:
* Influence names the firm that leads prices, from public data and without being told, during the phase in which BP built the market's coordination.
* Once collusion is a calendar rule, nobody leads. Influence goes silent, and margins and low price variance are what reveal it.

The method is therefore a **leadership detector**, complementary to margin and variance screens, not a collusion detector on its own. Its controls need records without regime changes.

Figure: `results/e90_figures/02_influence.png`.

### e91: Sydney petrol. Coordination is seen, but leaders who take turns and are followed within the day are not named

**The question.** e90 found a single leader whose rivals followed a day later. Can the method name leaders when they share the role? Byrne, de Roos, Lewis, Marx and Wu studied Sydney with the same public data (FuelCheck, August 2016 to December 2017). They found 18 market-wide restorations, about one a month. The five majors (BP, Caltex, Woolworths, Coles Express, 7-Eleven) led and coordinated them by station-level price signals and took turns: each signalled before 17–39% of restorations. In their worked example BP jumped first at noon, Woolworths and Caltex within two hours, 7-Eleven by evening and Coles the next day. Smaller chains followed.

**Development showed the limit before any real share was computed.** In simulation, a single leader followed a day later stands out (0.19–0.68). But the same leader followed by most majors within the same day reads about 0.04, and rotating leadership at most 0.11. Even the five majors taken as one block add almost nothing (0.003). A member's history adds to the group's own past only if that member moves at least one time step ahead of most of the group, and does so consistently. When several large members jump together, the group average already shows the restoration. So the experiment was pre-registered as a test of this limit: coordination should be seen, but no leader named.

**Setup** (`experiments/e91_sydney.py`, data `experiments/fuelcheck_data.py`):
* **Data:** every unleaded (U91) price change at every NSW station (FuelCheck, CC BY 4.0), and the Australian Institute of Petroleum's daily Sydney terminal gate price.
* **Members:** brands, each the mean of its Greater Sydney stations' end-of-day prices minus the wholesale price (12 brands with at least 10 stations, 11 to 132 each). September 2016 to December 2017, with August as a warm-up.
* **Recorded causes, design, virtual groups:** as e90.
* **Detection line:** on a 70-week record, virtual groups in development reached influence 0.135 (Perth's 190 weeks: 0.045), so the line for naming a leader was set at 0.15. Perth's BP (0.215) would clear it.

**5 of 6 predictions passed.**
* **Simulations (P1–P3):** independent firms show no share or influence; a single leader followed a day later reads 0.40–0.45; rotating or same-day leadership is seen as coordination (S 0.38–0.47) with no firm above 0.105.
* **Sydney (P5):** S = **0.143** [0.101, 0.186]; its virtual groups read ≤ 0.001. Coordination is clearly seen.
* **Sydney (P6):** no brand reaches the line. The largest is Caltex Woolworths at 0.083, which is level with Sydney's own virtual-group maximum (0.086). The method names no leader, as predicted.
* **Failed (P4):** one of 65 virtual groups, in the simulated same-day-leader market, read S 0.078 against a limit of 0.06. Sydney's own virtual groups all passed.

**Not predicted (exploratory).** The majors as a block add 0.009 and the other brands −0.003. The mean margin was 8.4 cents per litre with a coefficient of variation of 0.63, between Perth's competitive and collusive periods.

**What this means.** Together, e90 and e91 bound what influence can do on daily price data:
* It names a leader that moves at least a day before most of the market and does so consistently (Perth, 2004–2008).
* It does not name leaders that take turns, or whose rivals follow within hours (Sydney, 2016–2017). The share still shows the coordination.
* Finer time resolution (hourly data) would separate same-day following, but not rotation. A rotating club of leaders needs another tool, for example event-by-event ordering of price jumps.

Figure: `results/e91_figures/02_influence.png`.

### e92: Superfluid vortices. The share sees Onsager clusters in 20 vortices, but not reliably in 8

**The question.** In a flat superfluid (a thin Bose-Einstein condensate or helium film) the flow is carried by quantized vortices of circulation +1 or −1. Their motion follows the point-vortex equations, the long-distance limit of the Gross-Pitaevskii equation. Onsager (1949) predicted that at high energy same-sign vortices gather into large clusters, and two experiments observed this in 2019 (Gauthier et al.; Johnstone et al., *Science* 364). Told only each vortex's velocity, does the share see the cluster state as collective motion, and stay at zero when vortices don't interact? If so, it is a model-free order parameter for vortex-tracking data.

**Setup** (`experiments/e92_vortices.py`, simulation only):
* **Model:** 20 vortices (10 of each sign) in a disk with wall images, softened cores and weak thermal diffusion; 2,000 frames. Also 8 vortices over 500 frames, closer to what an experiment tracks.
* **Preparations,** from low to high energy: +/− pairs (dipoles), random positions, and two same-sign clusters of decreasing spread (0.35 to 0.08). A non-interacting control keeps only each vortex's own wall image.
* **Members:** each vortex's frame-to-frame velocity (x and y separately; S is their mean). No recorded causes. Imaging jitter of a third of a core size was added in a second reading.
* **Development** (seeds 0 and 10–13) chose velocities over positions: positions read S 0.12 with no interaction, because smooth precession fools the forecast.

**4 of 6 predictions passed.**
* **No interaction (P1):** S within ±0.015, with or without jitter.
* **Clusters (P3, P4):** S 0.89–0.95 at spread 0.25 and 0.97–1.00 tighter; every cluster state above every dipole and random state on every seed.
* **Jitter (P5):** tight clusters keep S 0.81–0.97.
* **Failed (P2):** one time-shifted control, in the tightest clusters, read 0.034 against a limit of 0.02 (all others ≤ 0.029 for 8 vortices and ≤ 0.009 otherwise).
* **Failed (P6):** with 8 vortices over 500 frames, random states read 0.74 and 0.60 on two of three seeds, against 0.89–1.00 for clusters. In development they had read 0.03–0.19. With so few vortices, a chance configuration can move as coherently as a cluster.

**Not predicted (exploratory).**
* **Energy:** across all 20-vortex states, S ranks with energy per vortex at 0.90. Below random-state energies it is erratic, though: dipole and random states range from 0.14 to 0.71.
* **Memory gap:** S_pred tracks S within about 0.05 everywhere. The share here is carried by the members' and the group's memory alone, so it could be computed from autocovariances of tracked velocities.
* **Jitter in small groups:** it pulls clusters down to 0.34–0.74.

**What this means.**
* In a 20-vortex superfluid, the share is a clean, model-free detector of Onsager clustering: zero without interaction, near 1 for clusters, from velocities alone, robust to modest imaging error.
* It is not a thermometer at low energy, and with about 8 vortices it cannot tell clustering from chance.
* A real test needs in-situ (non-destructive) tracking of about 20 or more vortices over hundreds of frames, with sub-core position precision. Many condensate experiments image destructively, one shot per run, which gives no time series. So the data step depends on finding a lab that tracks vortices in real time.

Figure: `results/e92_figures/02_share_vs_energy.png`.

---

## 5. Equations

**Global layer (fixed size, e13).** With z = (φ, L) observed every τ = 10
agent steps:

  z_{t+1} = z_t + f(z_t, η) + σ(z_t, η) ξ_t,  ξ ~ N(0, R)

f is a cubic polynomial in (φ, L) times a quadratic in noise η, and σ is the
exponential of the same basis. It reproduces mean order at 13/13 held-out
noise levels and, with the corrected noise estimator, the transition. Its
coefficients depend on N (e14).

**Field level (universal across size, e20–e22).** Cells of side 5 (≈ 100
agents), density ρ and momentum m (sum of headings / 100). Each step, a cell
sends the fraction c·max(±u_k, 0) + D of its agents, and their momentum, to
each neighbour (u = m/ρ). Then the local update is

  m ← T(m) + (a₁ + Δ + a₂ρ + a₃|m|² + a₄ρ²) m + a₅ ∇²m + a₆ ∇ρ + σ_m ξ

with c, D ≥ 0, a₁…a₆ and σ_m fitted at N = 1600 by least squares (at noise
3.0: c = 0.156 against the free-flight value 0.2; a = −0.248, +0.246,
−0.256, +0.055, +0.020, −0.091). Δ = +0.05 is the single calibrated offset
(e21). Global order is φ = |Σm| / Σρ. Nothing in the law refers to N.

---

## 6. Lessons for the theory

1. **Positive Ψ (or synergy) is not evidence of emergence from interaction.**
   Persistent independent parts produce it. Always run a no-interaction
   control.
2. **Closure is necessary but trivially satisfied by aggregates.** Averages of
   independent parts are closed (as for an ideal gas). Add an interaction
   share: how much of G the non-interacting baseline cannot explain.
3. **The right C is usually more than the obvious order parameter.** Here,
   global plus local alignment. When closure fails, the failure points to
   the missing variable (local order), and a confirmatory test on fresh data
   settles it.
4. **A closed layer need not have a universal law.** The global flock layer
   is closed at every size, but its law depends on size. Universality lives
   at a fixed physical scale, and the upper layer inherits its law from that
   scale plus the system's boundary.
5. **Macro laws must respect micro conservation laws.** A law that
   predicts one step well but does not move momentum with the agents decays
   to disorder when iterated (e17).
6. **A macro variable must average over enough parts.** With ~25 parts per
   cell, counting noise biases every estimator tried; with ~100, the law
   works (e17–e20). This suggests a minimum scale for a layer to exist.
7. **Pre-registration matters.** It caught the Ψ false positive (e8), a
   noise-estimator bug (e13), and four plausible-looking laws that did not
   generalize (e14, e15, e17, e18).

A sharper version of the starting hypothesis, consistent with all of the
above:

> A layer is emergent when (i) its variables are averages over enough parts
> to be well defined, (ii) it is closed, (iii) its law is not explained by
> non-interacting aggregation, and (iv) its law respects the conservation laws
> of the level below. Such a law can be universal only at a fixed physical
> scale. Larger-scale layers inherit their laws from it together with the
> system's size and boundary.

---

## 7. Open problems and next steps

* **Noise universality.** One field law with η as an input instead of 13
  separate laws.
* **The over-ordering below the transition** and the size scaling of
  fluctuations (field noise may need the density dependence predicted by
  counting statistics).
* **Estimator bias.** A validated way to fit laws from few-part averages
  (e19's hidden-state approach failed validation; a working version would
  remove the calibrated offset Δ).
* **Other systems.** Run the confirmed pipeline (interaction share, closure,
  equation, unseen-size prediction) on Gray-Scott chemistry or a different
  flocking model.
* **Lean.** Formalize the approximate version: an ε-closed coarse-graining
  has an approximate macro law whose iterates stay close for a bounded time.

## 8. Reproducing

```bash
uv run python run.py interaction bands local equation     # closure results (~20 min)
uv run python run.py regional bigcells calibrated fresh   # size results (~50 min first run, then cached)
uv run python run.py walkers                              # interaction share vs Psi, closed form (~20 s)
uv run python run.py twin noise flocktwin isolation       # stress tests of S (~2 min)
uv run python run.py curieweiss ising kuramoto            # transitions and oscillators (~1 min)
uv run python run.py fish fish2 fish3                     # real fish schools (needs data/puy2024/)
uv run python run.py kuramoto2 pigeons pigeons2 pigeons3  # S_proj on Kuramoto; pigeons (needs data/pigeons/D2/)
uv run python run.py zebrafish                            # zebrafish groups (needs data/zebrafish/trajectories/)
uv run python run.py elimination elimination2             # multi-scale elimination of rules (~6 min each)
uv run python run.py elimination3                         # e41 + figures in results/e41_figures (~8 min)
uv run python run.py elimination4                         # e42 + figures in results/e42_figures (~11 min)
uv run python run.py elimination5                         # e43 + figures in results/e43_figures (~13 min)
uv run python -m experiments.e43b_virtual_diagnostic      # e43b forecast-bias diagnostic (~5 s)
uv run python run.py elimination6                         # e44 + figures in results/e44_figures (~15 min)
uv run python run.py spectrum                             # e45 interaction-share spectrum + figures (~10 s)
uv run python run.py spectrum2                            # e46 spectrum with estimated twins + real fish (~15 s)
uv run python run.py layers                               # e47 closed layers + figures (~20 s)
uv run python run.py fishlayers                           # e48 group-level layer in real fish (~2 min)
uv run python -m experiments.e48b_fish_layer_diagnostic   # e48b exploratory diagnostic (~30 s)
uv run python run.py schoollayer                          # e49 group size + school shape layer (~3 min)
uv run python run.py configuration                        # e50 configuration closure + leadership (~5 s)
uv run python -m experiments.e50b_leadership_diagnostic   # e50b exploratory diagnostic (~2 s)
uv run python run.py trusted                              # e51 calibrated closure test + figures (~45 s)
uv run python run.py influence                            # e52 learned influence map + figures (~30 s)
uv run python run.py influence2                           # e53 replication: zebrafish + pigeons (~30 s)
uv run python run.py neutral                              # e54 pinning test: flock + oscillators (~20 s)
uv run python run.py free                                 # e55 free members: flock + oscillators (~25 s)
uv run python run.py lock                                 # e56 measured looseness: sims + real fish (~1 min)
uv run python run.py lock2                                # e57 measured looseness, oscillators redone (~10 s)
uv run python run.py innovation                           # e58 momentary innovations: flocks + real fish (~1 min)
uv run python run.py slaving                              # e59 internal memory time vs openness (~1 min)
uv run python run.py slow                                 # e60 slow feature analysis vs openness (~1 min)
uv run python run.py neurons neurons2                     # e61, e62 mouse brain areas (needs data/steinmetz/; ~15 min each)
uv run python run.py areaclosure                          # e63 closure of brain areas (needs e62 results; ~4 min)
uv run python run.py silencing silencing2                 # e64, e65 midbrain silencing (needs data/inagaki/; ~1 min)
uv run python run.py markets                              # e66 market volatility (needs data/finance/; ~10 s)
uv run python run.py memorygap                            # e67 memory-gap equation across domains (~3 min)
uv run python run.py fishgap                              # e68 the equation on fish schools (needs fish data)
uv run python run.py logvol                               # e69 log market volatility, 1936-1971 holdout (needs data/finance/)
uv run python run.py screen                               # e70 exploratory memory-gap screen of every dataset
uv run python run.py almholdout                           # e71 ALM share on unseen sessions, and silencing (needs DANDI 000221)
uv run python run.py polgap                               # e72 quadratic memory gap for fish polarization (needs fish data)
uv run python run.py almrep                               # e73 ALM share in Economo 2018, and silencing power (needs DANDI 000006)
uv run python run.py nonlinear                            # e74 memory gap for breadth and the worst industry (needs data/finance/)
uv run python run.py exacttwin                            # e75 exact twin for the worst industry and systemic days (needs data/finance/)
uv run python run.py leak                                 # e76 leak audit, fresh markets, reanalysis (needs data/finance/, DANDI; ~5 min)
uv run python run.py twingap                              # e77 why the fish memory gap ran above the fitted twins (needs fish data; ~20 min)
uv run python run.py rotate                               # e78 do the fish closure verdicts depend on the test segment? (needs fish, pigeon data; ~16 min)
uv run python run.py stocks                               # e79 individual stocks in eight markets (needs data/stocks/: python -m experiments.stock_data constituents, prices; ~30 s)
uv run python run.py ising                                # e86 the Ising magnet across its phase transition (no data needed; ~18 min)
uv run python run.py randu                                # e87 RANDU and a leaky generator (no data needed; ~80 s)
uv run python run.py groupsize                            # e88 group size 2 to 10,000 under one alignment rule (no data needed; ~11 min)
uv run python run.py reversal                             # e89 time reversal: common driver, leader, markets (needs data/finance/; ~12 s)
uv run python run.py primefilters                         # e85 memory on top of deeper filters, 10^17 and 10^18 (no data needed; ~70 s, 3.5 GB memory)
uv run python run.py primes                               # e84 can memory find primes? 10^12 and 10^16 (no data needed; ~25 s)
uv run python run.py lightdose                            # e83 the share of 50-fish schools at eight light levels (needs data/lafoux/; ~4 min)
uv run python run.py correlation                          # e82 the correlation forecast on fresh markets (needs data/stocks_fresh/; ~15 s)
uv run python run.py spillover                            # e81 the share against the Diebold-Yilmaz spillover index (needs data/stocks/; ~10 s)
uv run python run.py intervention                         # e80 switching interaction off: zebrafish in light and dark, virtual fish (needs data/intervention/; ~40 s)
uv run python -m experiments.memgap data.csv --horizon 1  # the screening tool on any CSV (rows = time, columns = members)
uv run python -m lab.build                                # visual lab: open lab/site/index.html
uv run --with markdown python whitepaper/build.py         # white paper PDF (v2.1)
uv run --with markdown python manuscript/build.py         # peer-review manuscript PDF (record from lab/catalog.py)
lean lean/InteractionShare.lean                           # Newton's third law => interaction share 0
uv run pytest                                     # 15 tests
~/.elan/bin/lean lean/Emergence.lean              # formal proofs
```

Figures are written to `results/`; the most informative ones are
`e9_interaction_closure.png`, `e12_local_order_closure.png`,
`e13_macro_equation_calibrated.png`, `e16_regional_order.png`,
`e21_calibrated_field.png` and `e22_fresh_size.png`.
