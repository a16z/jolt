/-
NOTE: This file is taken from
https://github.com/Verified-zkEVM/ArkLib. We include it here
because of a Lean4 version mismatch with ArkLib. Once that's
resolved, we can remove this file.
-/

/-
Copyright (c) 2024 ArkLib Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Quang Dao
-/

import MathLib

/-!
  # The BN254 scalar prime field
-/

namespace BN254

@[reducible]
def scalarFieldSize : Nat :=
  21888242871839275222246405745257275088548364400416034343698204186575808495617

abbrev ScalarField := ZMod scalarFieldSize

theorem ScalarField_is_prime : Nat.Prime scalarFieldSize := by sorry

instance : Fact (Nat.Prime scalarFieldSize) := ⟨ScalarField_is_prime⟩

instance : Field ScalarField := ZMod.instField scalarFieldSize

end BN254
