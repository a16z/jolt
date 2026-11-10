# Spartan

🚧 These docs are under construction 🚧

👷If you are urgently interested in this specific page, open a Github issue and we'll try to expedite it.👷

## Outer row groups

The canonical equality-conditional R1CS in `jolt-claims` has 20 RV64 rows,
partitioned into two groups of 10. Field-inline adds ten rows, giving two
groups of 15. The packed stopping row occupies the last RV64 slot, before
the field-inline suffix. The uni-skip domain and first-round degree remain
10 and 27 for RV64, and 15 and 42 for field-inline.

The optimized outer kernel splits the stopping row's second factor into
low limb `LookupOutput - UnexpandedPC` and high limb `1 - CanEnd`.
It extracts the guard in the existing trace pass and folds one additional
coefficient. The kernel retains nine extended nodes for RV64 and fourteen
for field-inline; it allocates no new trace-sized table for the guard.
The reference kernel and BlindFold consume the same canonical matrices.
