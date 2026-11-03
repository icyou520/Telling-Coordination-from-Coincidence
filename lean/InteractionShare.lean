/-!
# Interaction share: when is a macro law *not* made by interaction?

Micro dynamics `F : X → X` (parts interacting), its no-interaction twin
`F₀ : X → X` (same parts, interactions switched off), coarse-graining
`C : X → Z`. The twin's exact aggregation law is `G₀` with `C (F₀ x) = G₀ (C x)`.

The interaction share is `S = 1 - MSE(G) / MSE(G₀)` for the best macro law
`G` of `F`. It is zero exactly when `G₀` is already a law of `F`. Part 1 shows
this happens whenever interactions do not move the macro variable, and that
the law is then unique. Part 2 shows that pairwise forces obeying Newton's
third law, linear or not, never move the sum of the parts: the centre of
mass of interacting walkers has the twin's law, so its interaction share is
0 at every coupling strength (experiment e23 is the linear special case).

Part 2 holds in any commutative ring (`Lean.Grind.CommRing`, which Mathlib
instantiates for ℝ, ℚ, ℂ; core Lean for ℤ), with no assumption on the
characteristic. Core Lean 4 only. Check with:  `lean lean/InteractionShare.lean`
-/

namespace InteractionShare

/-! ## Part 1: interaction-free macro variables keep the twin's law -/

/-- Interactions never change the macro variable's next value. -/
def InteractionFree {X Z : Type} (F F₀ : X → X) (C : X → Z) : Prop :=
  ∀ x, C (F x) = C (F₀ x)

def iter {X : Type} (F : X → X) : Nat → X → X
  | 0, x => x
  | n + 1, x => iter F n (F x)

/-- The twin's aggregation law is an exact macro law of the interacting system. -/
theorem twin_law_exact {X Z : Type} (F F₀ : X → X) (C : X → Z) (G₀ : Z → Z)
    (hfree : InteractionFree F F₀ C) (h₀ : ∀ x, C (F₀ x) = G₀ (C x)) :
    ∀ x, C (F x) = G₀ (C x) :=
  fun x => (hfree x).trans (h₀ x)

/-- ... on every time horizon. -/
theorem twin_law_exact_iter {X Z : Type} (F F₀ : X → X) (C : X → Z) (G₀ : Z → Z)
    (hfree : InteractionFree F F₀ C) (h₀ : ∀ x, C (F₀ x) = G₀ (C x)) :
    ∀ n x, C (iter F n x) = iter G₀ n (C x) := by
  intro n
  induction n with
  | zero => intro x; rfl
  | succ n ih =>
    intro x
    show C (iter F n (F x)) = iter G₀ n (G₀ (C x))
    rw [ih, twin_law_exact F F₀ C G₀ hfree h₀ x]

/-- Uniqueness: if every macro value is realized, any exact macro law of the
interacting system *is* the twin's law. Nothing is left for interaction to
explain, so the interaction share is zero. -/
theorem law_is_twin_law {X Z : Type} (F F₀ : X → X) (C : X → Z) (G G₀ : Z → Z)
    (hfree : InteractionFree F F₀ C) (h₀ : ∀ x, C (F₀ x) = G₀ (C x))
    (hG : ∀ x, C (F x) = G (C x)) (hsurj : ∀ z, ∃ x, C x = z) :
    ∀ z, G z = G₀ z := by
  intro z
  obtain ⟨x, rfl⟩ := hsurj z
  rw [← hG x, ← h₀ x, hfree x]

/-- Contrapositive: if the interacting law differs from the twin's anywhere,
interactions must move the macro variable somewhere. -/
theorem interaction_needed {X Z : Type} (F F₀ : X → X) (C : X → Z) (G G₀ : Z → Z)
    (h₀ : ∀ x, C (F₀ x) = G₀ (C x)) (hG : ∀ x, C (F x) = G (C x))
    (hsurj : ∀ z, ∃ x, C x = z) (hdiff : ∃ z, G z ≠ G₀ z) :
    ¬ InteractionFree F F₀ C :=
  fun hfree => by
    obtain ⟨z, hz⟩ := hdiff
    exact hz (law_is_twin_law F F₀ C G G₀ hfree h₀ hG hsurj z)

/-! ## Part 2: Newton's third law makes the centre of mass interaction-free -/

section Ring

variable {α : Type} [Lean.Grind.CommRing α]

/-- `Σ_{a ∈ xs} f a`. -/
def sumMap (f : α → α) : List α → α
  | [] => 0
  | a :: l => f a + sumMap f l

theorem sumMap_zero (l : List α) : sumMap (fun _ => (0 : α)) l = 0 := by
  induction l with
  | nil => rfl
  | cons a l ih => simp only [sumMap, ih]; grind

theorem sumMap_add (f g : α → α) (l : List α) :
    sumMap (fun b => f b + g b) l = sumMap f l + sumMap g l := by
  induction l with
  | nil => simp only [sumMap]; grind
  | cons a l ih => simp only [sumMap, ih]; grind

theorem sumMap_mul (c : α) (l : List α) : sumMap (fun b => c * b) l = c * sumMap id l := by
  induction l with
  | nil => simp only [sumMap]; grind
  | cons a l ih => simp only [sumMap, ih, id]; grind

theorem sumMap_map (f h : α → α) (l : List α) :
    sumMap f (l.map h) = sumMap (fun a => f (h a)) l := by
  induction l with
  | nil => rfl
  | cons a l ih => simp only [List.map, sumMap, ih]

theorem sumMap_congr (f g : α → α) (l : List α) (h : ∀ a, f a = g a) :
    sumMap f l = sumMap g l := by
  rw [funext h]

/-- Total pairwise force `Σ_i Σ_j φ(x_i, x_j)`. -/
def pairSum (φ : α → α → α) (xs : List α) : α :=
  sumMap (fun a => sumMap (fun b => φ a b) xs) xs

theorem pairSum_cons (φ : α → α → α) (a : α) (xs : List α) :
    pairSum φ (a :: xs) =
      φ a a + sumMap (fun b => φ a b + φ b a) xs + pairSum φ xs := by
  unfold pairSum
  simp only [sumMap]
  rw [sumMap_add (fun a' => φ a' a) (fun a' => sumMap (fun b => φ a' b) xs) xs,
      sumMap_add (fun b => φ a b) (fun b => φ b a) xs]
  grind

/-- Action equals reaction, and no self-force: pairwise forces cancel in
total. (Over ℝ, `hself` follows from `hφ`; it is stated separately so that
no assumption on the characteristic is needed.) -/
theorem pairSum_antisymm (φ : α → α → α) (hφ : ∀ a b, φ a b = -φ b a)
    (hself : ∀ a, φ a a = 0) (xs : List α) : pairSum φ xs = 0 := by
  induction xs with
  | nil => rfl
  | cons a xs ih =>
    rw [pairSum_cons, ih, hself,
        sumMap_congr _ (fun _ => (0 : α)) xs (fun b => by rw [hφ a b]; grind),
        sumMap_zero]
    grind

/-- Interacting walkers: each moves by its own linear law plus the pairwise
forces from all others (noise adds identically to both systems and is
omitted). -/
def step (c : α) (φ : α → α → α) (xs : List α) : List α :=
  xs.map (fun a => c * a + sumMap (fun b => φ a b) xs)

/-- The no-interaction twin. -/
def twin (c : α) (xs : List α) : List α := xs.map (fun a => c * a)

/-- Total position (the centre of mass up to the factor 1/N). -/
def total (xs : List α) : α := sumMap id xs

theorem twin_law (c : α) (xs : List α) : total (twin c xs) = c * total xs := by
  simp only [total, twin, sumMap_map, id]
  exact sumMap_mul c xs

/-- **Centre of mass is interaction-free** for any Newton's-third-law force. -/
theorem com_interaction_free (c : α) (φ : α → α → α) (hφ : ∀ a b, φ a b = -φ b a)
    (hself : ∀ a, φ a a = 0) : InteractionFree (step c φ) (twin c) total := by
  intro xs
  simp only [total, step, twin, sumMap_map, id]
  rw [sumMap_add, sumMap_mul]
  have := pairSum_antisymm φ hφ hself xs
  unfold pairSum at this
  rw [this]
  grind

/-- Hence the centre of mass of interacting walkers follows exactly the
no-interaction law `z ↦ c z`, on every horizon: interaction share zero. -/
theorem com_twin_law (c : α) (φ : α → α → α) (hφ : ∀ a b, φ a b = -φ b a)
    (hself : ∀ a, φ a a = 0) :
    ∀ n xs, total (iter (step c φ) n xs) = iter (fun z => c * z) n (total xs) :=
  twin_law_exact_iter _ _ _ _ (com_interaction_free c φ hφ hself) (twin_law c)

/-- The spring force of e23, `φ(a, b) = γ (b - a)`, qualifies. -/
theorem spring_antisymm (γ : α) : ∀ a b, γ * (b - a) = -(γ * (a - b)) := by
  intro a b; grind

theorem spring_self (γ : α) : ∀ a, γ * (a - a) = 0 := by
  intro a; grind

end Ring

/-! ## Part 3: total turning, and eliminating a whole family of rules (e39)

Each part has a state in `β` (position, heading, ...). In one step its heading
changes by its own turn `own a` plus pairwise terms `φ a b` from the others.
If `φ` is antisymmetric, the group's total turn equals the twin's. Metric
alignment, `φ a b = s a b * w (θ b - θ a)` with a symmetric neighbourhood `s`
(within a fixed radius) and an odd response `w` (e.g. `sin`), qualifies. So
one observation of a total turn that differs from the twin's rules out
every such rule at once, whatever the radius, response or strength. -/

section Turning

variable {α β : Type} [Lean.Grind.CommRing α]

def sumB (f : β → α) : List β → α
  | [] => 0
  | a :: l => f a + sumB f l

theorem sumB_zero (l : List β) : sumB (fun _ => (0 : α)) l = 0 := by
  induction l with
  | nil => rfl
  | cons a l ih => simp only [sumB, ih]; grind

theorem sumB_add (f g : β → α) (l : List β) :
    sumB (fun b => f b + g b) l = sumB f l + sumB g l := by
  induction l with
  | nil => simp only [sumB]; grind
  | cons a l ih => simp only [sumB, ih]; grind

theorem sumB_congr (f g : β → α) (l : List β) (h : ∀ a, f a = g a) :
    sumB f l = sumB g l := by
  rw [funext h]

def pairSumB (φ : β → β → α) (xs : List β) : α :=
  sumB (fun a => sumB (fun b => φ a b) xs) xs

theorem pairSumB_cons (φ : β → β → α) (a : β) (xs : List β) :
    pairSumB φ (a :: xs) =
      φ a a + sumB (fun b => φ a b + φ b a) xs + pairSumB φ xs := by
  unfold pairSumB
  simp only [sumB]
  rw [sumB_add (fun a' => φ a' a) (fun a' => sumB (fun b => φ a' b) xs) xs,
      sumB_add (fun b => φ a b) (fun b => φ b a) xs]
  grind

theorem pairSumB_antisymm (φ : β → β → α) (hφ : ∀ a b, φ a b = -φ b a)
    (hself : ∀ a, φ a a = 0) (xs : List β) : pairSumB φ xs = 0 := by
  induction xs with
  | nil => rfl
  | cons a xs ih =>
    rw [pairSumB_cons, ih, hself,
        sumB_congr _ (fun _ => (0 : α)) xs (fun b => by rw [hφ a b]; grind),
        sumB_zero]
    grind

/-- The group's total turn in one step. -/
def turnTotal (own : β → α) (φ : β → β → α) (xs : List β) : α :=
  sumB (fun a => own a + sumB (fun b => φ a b) xs) xs

/-- Antisymmetric pairwise turning never changes the total turn: it equals the
twin's, `Σ own`. -/
theorem total_turn_twin (own : β → α) (φ : β → β → α) (hφ : ∀ a b, φ a b = -φ b a)
    (hself : ∀ a, φ a a = 0) (xs : List β) : turnTotal own φ xs = sumB own xs := by
  unfold turnTotal
  rw [sumB_add]
  have := pairSumB_antisymm φ hφ hself xs
  unfold pairSumB at this
  rw [this]
  grind

/-- Symmetric neighbourhood times odd response is antisymmetric. -/
theorem sym_odd_antisymm (s : β → β → α) (w : α → α) (g : β → α)
    (hs : ∀ a b, s a b = s b a) (hw : ∀ x, w (-x) = -w x) :
    ∀ a b, s a b * w (g b - g a) = -(s b a * w (g a - g b)) := by
  intro a b
  have h : g a - g b = -(g b - g a) := by grind
  rw [hs a b, h, hw]
  grind

/-- **Family-level elimination.** If the observed total turn differs from the
twin's for one configuration, no metric-alignment rule (any symmetric
neighbourhood, any odd response with `w 0 = 0`, any strength folded into `w`)
can have produced the data. -/
theorem eliminate_symmetric_alignment (own : β → α) (g : β → α) (obs : List β → α)
    (xs : List β) (hobs : obs xs ≠ sumB own xs) :
    ¬ ∃ (s : β → β → α) (w : α → α), (∀ a b, s a b = s b a) ∧ (∀ x, w (-x) = -w x) ∧
        w 0 = 0 ∧ ∀ ys, obs ys = turnTotal own (fun a b => s a b * w (g b - g a)) ys := by
  intro ⟨s, w, hs, hw, hw0, hmodel⟩
  apply hobs
  rw [hmodel xs]
  apply total_turn_twin own _ (sym_odd_antisymm s w g hs hw)
  intro a
  have h : g a - g a = 0 := by grind
  rw [h, hw0]
  grind

end Turning

/-! ## Part 4: which collective variables can interactions not touch? (e45)

Linear dynamics `x ↦ A x` with interactions, and the twin `x ↦ A₀ x`, so
`A = A₀ + Δ` with `Δ` the interaction operator. A collective variable is a
weighted sum `y = Σ_i w_i x_i`. If `w` is a left null vector of `Δ`
(`Σ_i w_i Δ_ij = 0` for every `j`), the full model and the twin predict the
same next value of `y` from every state, so its interaction share is 0.
Newton's third law (columns of `Δ` sum to zero) is the case `w = 1`, the
centre of mass; for one-way coupling the blind variable is a *weighted*
average, which e45 must find from data. -/

section Spectrum

variable {α β : Type} [Lean.Grind.CommRing α]

theorem sumB_mul_left (c : α) (f : β → α) (l : List β) :
    c * sumB f l = sumB (fun b => c * f b) l := by
  induction l with
  | nil => simp only [sumB]; grind
  | cons a l ih => simp only [sumB, ← ih]; grind

theorem sumB_mul_right (c : α) (f : β → α) (l : List β) :
    sumB f l * c = sumB (fun b => f b * c) l := by
  induction l with
  | nil => simp only [sumB]; grind
  | cons a l ih => simp only [sumB, ← ih]; grind

theorem sumB_swap (f : β → β → α) (l₁ l₂ : List β) :
    sumB (fun i => sumB (fun j => f i j) l₂) l₁ = sumB (fun j => sumB (fun i => f i j) l₁) l₂ := by
  induction l₁ with
  | nil => simp only [sumB]; exact (sumB_zero l₂).symm
  | cons a l ih => simp only [sumB]; rw [ih, sumB_add]

/-- One linear step `(M x)_i = Σ_j M_ij x_j` over the parts `idx`. -/
def linStep (M : β → β → α) (idx : List β) (x : β → α) : β → α :=
  fun i => sumB (fun j => M i j * x j) idx

/-- The collective variable `Σ_i w_i y_i`. -/
def collective (w : β → α) (idx : List β) (y : β → α) : α :=
  sumB (fun i => w i * y i) idx

theorem collective_linStep (w : β → α) (M : β → β → α) (idx : List β) (x : β → α) :
    collective w idx (linStep M idx x) =
      sumB (fun j => sumB (fun i => w i * M i j) idx * x j) idx := by
  unfold collective linStep
  rw [sumB_congr _ (fun i => sumB (fun j => w i * (M i j * x j)) idx) idx
        (fun i => sumB_mul_left _ _ _), sumB_swap]
  apply sumB_congr
  intro j
  rw [sumB_mul_right]
  apply sumB_congr
  intro i
  grind

/-- **Left null vectors of the interaction operator are interaction-blind.**
The full model and its no-interaction twin give the same prediction of the
collective variable `w` from every state. -/
theorem left_null_blind (w : β → α) (A A₀ Δ : β → β → α) (idx : List β)
    (hA : ∀ i j, A i j = A₀ i j + Δ i j) (hnull : ∀ j, sumB (fun i => w i * Δ i j) idx = 0)
    (x : β → α) : collective w idx (linStep A idx x) = collective w idx (linStep A₀ idx x) := by
  rw [collective_linStep, collective_linStep]
  apply sumB_congr
  intro j
  have h : sumB (fun i => w i * A i j) idx =
      sumB (fun i => w i * A₀ i j) idx + sumB (fun i => w i * Δ i j) idx := by
    rw [← sumB_add]
    apply sumB_congr
    intro i
    rw [hA]
    grind
  rw [h, hnull j]
  grind

/-- Newton's third law in matrix form (every column of `Δ` sums to zero): the
plain total, i.e. the centre of mass, is blind. -/
theorem columns_zero_total_blind (A A₀ Δ : β → β → α) (idx : List β)
    (hA : ∀ i j, A i j = A₀ i j + Δ i j) (hcol : ∀ j, sumB (fun i => Δ i j) idx = 0)
    (x : β → α) :
    collective (fun _ => 1) idx (linStep A idx x) = collective (fun _ => 1) idx (linStep A₀ idx x) := by
  apply left_null_blind _ A A₀ Δ idx hA
  intro j
  rw [sumB_congr _ (fun i => Δ i j) idx (fun i => by grind)]
  exact hcol j

end Spectrum

/-! ## Part 5: closed layers (e47)

A coarse-graining `C : X → Z` is a *closed* description of `F` when the next
coarse state depends on the current coarse state only: `C (F x) = G (C x)`.
Then the parts can be forgotten on every horizon, and a closed description of
a closed description is a closed description of the parts, so layers stack.
For linear dynamics, a block of collective variables `W` (rows `W k`) is
closed when it is left-invariant, `Wᵀ A = M Wᵀ`. -/

def Closed {X Z : Type} (F : X → X) (C : X → Z) (G : Z → Z) : Prop :=
  ∀ x, C (F x) = G (C x)

theorem closed_iter {X Z : Type} (F : X → X) (C : X → Z) (G : Z → Z) (h : Closed F C G) :
    ∀ n x, C (iter F n x) = iter G n (C x) := by
  intro n
  induction n with
  | zero => intro x; rfl
  | succ n ih =>
    intro x
    show C (iter F n (F x)) = iter G n (G (C x))
    rw [ih, h x]

/-- **Layers stack.** -/
theorem closed_comp {X Y Z : Type} (F : X → X) (C : X → Y) (G : Y → Y) (D : Y → Z) (H : Z → Z)
    (h₁ : Closed F C G) (h₂ : Closed G D H) : Closed F (fun x => D (C x)) H := by
  intro x
  show D (C (F x)) = H (D (C x))
  rw [h₁ x, h₂ (C x)]

section Layers

variable {α β γ : Type} [Lean.Grind.CommRing α]

theorem sumB_swap₂ (f : β → γ → α) (l₁ : List β) (l₂ : List γ) :
    sumB (fun i => sumB (fun j => f i j) l₂) l₁ = sumB (fun j => sumB (fun i => f i j) l₁) l₂ := by
  induction l₁ with
  | nil => simp only [sumB]; exact (sumB_zero l₂).symm
  | cons a l ih => simp only [sumB]; rw [ih, sumB_add]

/-- **A left-invariant block is closed:** if `Σ_i W_ki A_ij = Σ_l M_kl W_lj`,
the block's next value is `M` applied to its current value, from every state. -/
theorem left_invariant_closed (W : γ → β → α) (A : β → β → α) (M : γ → γ → α)
    (idx : List β) (lidx : List γ)
    (hinv : ∀ k j, sumB (fun i => W k i * A i j) idx = sumB (fun l => M k l * W l j) lidx)
    (x : β → α) (k : γ) :
    collective (W k) idx (linStep A idx x) = sumB (fun l => M k l * collective (W l) idx x) lidx := by
  rw [collective_linStep]
  rw [sumB_congr _ (fun j => sumB (fun l => M k l * W l j * x j) lidx) idx (fun j => by
        rw [hinv k j, sumB_mul_right])]
  rw [sumB_swap₂]
  apply sumB_congr
  intro l
  unfold collective
  rw [sumB_mul_left]
  apply sumB_congr
  intro j
  grind

end Layers

/-! ## Sanity checks on ℤ -/

/-- Springs on a concrete configuration: the pairwise forces sum to zero. -/
example : pairSum (fun (a b : Int) => 3 * (b - a)) [1, 5, -7, 2] = 0 := by decide

/-- A force that breaks Newton's third law *does* move the centre of mass, so
the hypothesis is needed. -/
def pull (a _ : Int) : Int := -a

example : ¬ InteractionFree (step (1 : Int) pull) (twin 1) total := by
  intro h
  have := h [1, 2]
  revert this
  decide

/-- A one-sided (topological-like) neighbourhood breaks the symmetry and does
move the total turn, so the family theorem says nothing about such rules. -/
example : turnTotal (fun (_ : Int) => (0 : Int))
    (fun a b => (if a < b then 1 else 0) * (b - a)) [0, 1] ≠ 0 := by decide

/-- One-way coupling on the cycle 0 → 1 → 2 → 0 with weights 1, 1, 2
(`Δ = -L`, `L` = out-degree minus weights). Its left null vector is
`π = (2, 2, 1)`: the π-weighted total is blind, the plain total is not. -/
def dirΔ : Nat → Nat → Int
  | 0, 0 => -1 | 0, 1 => 1
  | 1, 1 => -1 | 1, 2 => 1
  | 2, 2 => -2 | 2, 0 => 2
  | _, _ => 0

def dirπ : Nat → Int
  | 2 => 1
  | _ => 2

example : ∀ j ∈ [0, 1, 2], sumB (fun i => dirπ i * dirΔ i j) [0, 1, 2] = 0 := by decide

example : collective dirπ [0, 1, 2] (linStep dirΔ [0, 1, 2] (fun i => if i = 0 then 1 else 0)) = 0 := by
  decide

example : collective (fun _ => (1 : Int)) [0, 1, 2]
    (linStep dirΔ [0, 1, 2] (fun i => if i = 0 then 1 else 0)) ≠ 0 := by decide

end InteractionShare
