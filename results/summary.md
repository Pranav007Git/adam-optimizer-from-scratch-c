# Results summary

## Best setting per optimizer (chosen on validation accuracy, reported on test accuracy)

| Optimizer | Learning rate | Batch size | Test acc (mean ± std over seeds) | Epochs to reach 97% val acc |
|---|---|---|---|---|
| SGD | 1 | 64 | 0.9759 ± 0.0016 | not reached |
| SGD + Momentum | 0.1 | 64 | 0.9750 ± 0.0000 | 21 |
| Adam | 0.01 | 64 | 0.9796 ± 0.0042 | 10 |

## How many learning rates work? (mean test accuracy over batch sizes and seeds ≥ 90%)

| Optimizer | Learning rates that work | Count |
|---|---|---|
| SGD | 0.1, 1 | 2 of 5 |
| SGD + Momentum | 0.01, 0.1 | 2 of 5 |
| Adam | 0.001, 0.01, 0.1, 1 | 4 of 5 |

## Diverged runs (loss became NaN or infinity)

| Optimizer | Diverged runs | Total runs |
|---|---|---|
| SGD | 0 | 45 |
| SGD + Momentum | 3 | 45 |
| Adam | 0 | 45 |

## Mean test accuracy by learning rate (averaged over batch sizes and seeds)

| Learning rate | SGD | SGD + Momentum | Adam |
|---|---|---|---|
| 0.0001 | 0.1296 | 0.4512 | 0.6272 |
| 0.001 | 0.4540 | 0.8861 | 0.9605 |
| 0.01 | 0.8883 | 0.9667 | 0.9790 |
| 0.1 | 0.9679 | 0.9438 | 0.9698 |
| 1 | 0.9179 | 0.5935 | 0.9559 |
