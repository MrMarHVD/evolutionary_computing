# Mutation study

Status: complete. Completed runs: 25/25.
Paired complete seeds used below: [42, 43, 44, 45, 46]. Planned: [42, 43, 44, 45, 46].
Results describe this fixed body, arena and budget; they do not establish a generally best mutation.

Mean +/- sample standard deviation across independent seeds; lower fitness is better.
Unfinished or unmatched seeds are excluded from comparisons, never padded forward.

| Condition | n | Mean fitness | Sample SD | Success | Failure |
|---|---:|---:|---:|---:|---:|
| gaussian_p0.1_s0.05 | 5 | 5.236235 | 0.14684355409636402 | 0% | 0% |
| gaussian_p0.1_s0.15 | 5 | 5.138970 | 0.13997921963371374 | 0% | 0% |
| gaussian_p0.1_s0.45 | 5 | 5.020299 | 0.028463429744471466 | 0% | 0% |
| uniform_reset_p0.1 | 5 | 5.345059 | 0.038930300711331474 | 0% | 0% |
| random_search | 5 | 5.596959 | 0.05650972431199357 | 0% | 0% |

Fitness is 3D final core-to-target distance plus terrain/fall penalties (fixed for all conditions).
The assignment's default is planar distance; explain and justify this alternative in Methods.
Inspect convergence before selecting the final budget. Extend all conditions equally if still improving.
Video representatives are each run's best controller; they are not additional independent runs.
