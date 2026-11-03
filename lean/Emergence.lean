/-!
# Exact emergence: when does an autonomous macro law exist?

Micro dynamics `F : X → X`, coarse-graining `C : X → Z`.
A macro law is a `G : Z → Z` solving the cross-scale consistency equation
`C (F x) = G (C x)` for every micro state `x`.

Core Lean 4 only (no Mathlib). Check with:  `lean lean/Emergence.lean`
-/

namespace Emergence

/-- Micro states that look identical at the macro level still look identical
after one step of the micro dynamics. -/
def Closed {X Z : Type} (F : X → X) (C : X → Z) : Prop :=
  ∀ x y, C x = C y → C (F x) = C (F y)

/-- **Main theorem.** An exact macro law exists iff the coarse-graining is closed. -/
theorem macro_law_iff_closed {X Z : Type} (F : X → X) (C : X → Z) :
    (∃ G : Z → Z, ∀ x, C (F x) = G (C x)) ↔ Closed F C := by
  constructor
  · intro ⟨G, hG⟩ x y hxy
    rw [hG x, hG y, hxy]
  · intro h
    classical
    refine ⟨fun z => if hz : ∃ x, C x = z then C (F (Classical.choose hz)) else z, ?_⟩
    intro x
    have hz : ∃ x', C x' = C x := ⟨x, rfl⟩
    dsimp only
    rw [dite_eq_left hz]
    exact h x (Classical.choose hz) (Classical.choose_spec hz).symm

/-- `n` steps of the micro dynamics. -/
def iter {X : Type} (F : X → X) : Nat → X → X
  | 0, x => x
  | n + 1, x => iter F n (F x)

/-- A closed macro layer stays closed on every coarser time scale. -/
theorem closed_iter {X Z : Type} (F : X → X) (C : X → Z) (h : Closed F C) :
    ∀ n, Closed (iter F n) C := by
  intro n
  induction n with
  | zero => intro x y hxy; exact hxy
  | succ n ih => intro x y hxy; exact ih (F x) (F y) (h x y hxy)

/-- Coarse-graining twice: if `C₁` is closed under `F` with macro law `G₁`,
and `C₂` is closed under `G₁`, then the composite `C₂ ∘ C₁` is closed under `F`.
Layers of emergence can be stacked. -/
theorem closed_comp {X Y Z : Type} (F : X → X) (C₁ : X → Y) (G₁ : Y → Y) (C₂ : Y → Z)
    (h₁ : ∀ x, C₁ (F x) = G₁ (C₁ x)) (h₂ : Closed G₁ C₂) :
    Closed F (fun x => C₂ (C₁ x)) := by
  intro x y hxy
  show C₂ (C₁ (F x)) = C₂ (C₁ (F y))
  rw [h₁ x, h₁ y]
  exact h₂ (C₁ x) (C₁ y) hxy

/-! ## The motivating example: two bits that both flip every step -/

def microStep (s : Bool × Bool) : Bool × Bool := (!s.1, !s.2)

def macroState (s : Bool × Bool) : Bool := s.1 != s.2

theorem emergentRule (s : Bool × Bool) : macroState (microStep s) = macroState s := by
  cases s with
  | mk a b => cases a <;> cases b <;> decide

theorem flip_closed : Closed microStep macroState := by
  intro x y hxy
  rw [emergentRule x, emergentRule y, hxy]

/-! ## A macro variable that is *not* closed: popcount of a 3-bit counter -/

def counterStep (x : Nat) : Nat := (x + 1) % 8

def popcount (x : Nat) : Nat := x % 2 + x / 2 % 2 + x / 4 % 2

/-- States 1 (001) and 2 (010) both have one bit set, but their successors
2 (010) and 3 (011) do not: no macro law for popcount exists. -/
theorem popcount_not_closed : ¬ Closed counterStep popcount := by
  intro h
  have h12 := h 1 2 (by decide)
  exact absurd h12 (by decide)

theorem no_popcount_law : ¬ ∃ G : Nat → Nat, ∀ x, popcount (counterStep x) = G (popcount x) :=
  fun hG => popcount_not_closed ((macro_law_iff_closed counterStep popcount).mp hG)

end Emergence
