# Report template (aim for 2 pages, write it in your own words)

**Title:** Does Adam need less tuning than SGD? A from-scratch C implementation and experiments

## 1. Question (3-4 sentences)
What claim from the Adam paper are you testing, and why does it matter in practice?

## 2. Method (about 1/2 page)
- The network and the three optimizers you implemented in C
- Dataset, split, and how the best setting was chosen (validation) and reported (test)
- Experiment grid: learning rates, batch sizes, seeds, epochs
- How you checked correctness (gradient check, scikit-learn comparison)

## 3. Results (about 1 page)
- Insert `fig1_lr_sensitivity.png`, then 2-3 sentences on what it shows
- Insert `fig2_convergence.png`, then 2-3 sentences on speed
- The best-setting table from `results/summary.md`
- Diverged runs: which optimizer, which settings

## 4. Discussion (about 1/3 page)
- Does the claim hold? Which part yes, which part no?
- One thing that surprised you, and a guess at why

## 5. Limitations and next steps (3-4 bullets)
- Small data and network, few seeds
- What you would test next (for example MNIST, deeper networks, learning-rate schedules)

## References
Kingma & Ba (2015), Adam: A Method for Stochastic Optimization.
