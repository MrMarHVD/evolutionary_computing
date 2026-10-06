# Mutation study

Status: complete. Completed runs: 12/12.
Paired complete seeds used below: [42, 43]. Planned: [42, 43].
SMOKE TEST ONLY: no scientific performance conclusion.

Mean +/- sample standard deviation across independent seeds; lower fitness is better.
Unfinished or unmatched seeds are excluded from comparisons, never padded forward.

| Condition | n | Mean fitness | Sample SD | Success | Failure |
|---|---:|---:|---:|---:|---:|
| gaussian_p0.1_s0.05 | 2 | 6.078152 | 0.001415730673097238 | 0% | 0% |
| gaussian_p0.1_s0.15 | 2 | 6.078511 | 0.001923084690960707 | 0% | 0% |
| uniform_step_p0.1_s0.05 | 2 | 6.078097 | 0.0015908450741693994 | 0% | 0% |
| uniform_step_p0.1_s0.15 | 2 | 6.077914 | 0.0012681537415721628 | 0% | 0% |
| uniform_reset_p0.1 | 2 | 6.077396 | 0.002641838719753284 | 0% | 0% |
| random_search | 2 | 6.078428 | 0.0018049980318286514 | 0% | 0% |

Fitness is 3D final core-to-target distance plus terrain/fall penalties (fixed for all conditions).
The assignment's default is planar distance; explain and justify this alternative in Methods.
Inspect convergence before selecting the final budget. Extend all conditions equally if still improving.
Video representatives are each run's best controller; they are not additional independent runs.
