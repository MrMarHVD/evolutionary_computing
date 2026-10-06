# Report figure captions

## Figure 1: Mutation strength and operator comparison

Performance of three Gaussian mutation strengths, uniform-reset mutation and random search for a fixed snake body and arena. (a) Mean best-so-far fitness against search evaluations; shaded bands show sample standard deviation across five independent seeds. (b) Final best fitness for each seed, with grey lines connecting matched seeds across conditions; black squares and error bars show the mean and sample standard deviation. Each run used 32 initial controllers and 200 generations of 32 evaluations (6,432 search evaluations total). Evolutionary conditions used per-gene mutation probability 0.1 with the remaining EA settings held fixed. All conditions began with the same initial population for a given seed. Lower fitness is better. Fitness is final 3D core-to-target distance plus terrain/fall penalties; all final winners were penalty-free, so their final fitness equals distance in metres. No run reached the target. Error bands are between-seed spread, not confidence intervals.

Suggested discussion: Gaussian mutation with sigma 0.45 achieved the lowest mean final fitness (5.020 ± 0.028) and outperformed the smaller strengths in four of five matched seeds. Every evolutionary condition outperformed random search in all five seeds. The best individual controller came from sigma 0.15 (4.941), distinguishing the best single run from the best mean setting. Some runs continued improving near the budget limit, so the curves do not establish full convergence. These observations apply to the tested body, arena, seeds and budget; no statistical significance or universal optimality is claimed.

## Figure 2: Genetic diversity during search

Mean per-gene population standard deviation against search evaluations for the same five conditions. Bold lines show means across five independent seeds and faint lines show individual runs. The vertical axis is logarithmic. The metric describes the surviving population; for random search it describes the retained best controllers, not the distribution of every newly sampled controller. Greater values indicate a wider spread of neural-network weights, not necessarily greater behavioural diversity.

Suggested discussion: The final mean diversity was 0.022, 0.073 and 0.198 for Gaussian strengths 0.05, 0.15 and 0.45 respectively. Stronger Gaussian mutation maintained more genetic variation and achieved better mean fitness in this experiment. Uniform reset maintained still greater variation but performed worse than the Gaussian conditions, so greater genetic diversity alone did not guarantee better fitness. This association does not by itself establish why performance differed. Reset replaces weights from U(-5,5), whereas Gaussian mutation perturbs existing weights; their comparison changes locality and mutation magnitude as well as distribution shape.

## Use in the report

- Use Figure 1 as the main results figure. Include Figure 2 when discussing mutation strength and diversity; otherwise it can be supplementary.
- PDF and SVG are vector exports; PNG is 300 dpi. Use the vector versions when possible.
- Success and reported failure rates were identical across conditions (both zero); state them in prose rather than adding empty bar charts. All saved population genotype counts were 32, so that plot was omitted.
- Regenerate from this script with the project's Python environment: `python make_figures.py`. Paths resolve relative to the script. It reads committed histories, merges interrupted/resumed attempts and verifies equal budgets without running simulations.
