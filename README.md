# Do Adam's claims hold up? A neural network written from scratch in C

I implemented a small neural network in plain C (forward pass, backpropagation, and three
optimizers: SGD, SGD with momentum, Adam) and used it from Python to test a claim from the
Adam paper (Kingma & Ba, 2015): that Adam trains faster and is easier to tune than plain SGD.

**Research question:** across different learning rates, batch sizes and random seeds, does
Adam really need less tuning than SGD, and where does each method break?

## Files

| File | What it does |
|---|---|
| `src/nn.c`, `src/nn.h` | The neural network in C: forward pass, backprop, SGD / Momentum / Adam, gradient check |
| `nn.py` | Python wrapper (ctypes). Builds the C library automatically with `gcc` |
| `data.py` | Loads the dataset (scikit-learn digits, or MNIST) and makes train / val / test splits |
| `validate.py` | Proves the C code is correct (gradient check + comparison with scikit-learn) |
| `run_experiments.py` | Runs the experiment grid and saves `results/runs.csv` and `results/curves.csv` |
| `analyze.py` | Makes the plots and `results/summary.md` |
| `REPORT_TEMPLATE.md` | Structure for your 2-page write-up |

## Run it

```bash
pip install -r requirements.txt
python validate.py          # both checks should say PASS
python run_experiments.py   # 135 runs, about 40 seconds on a laptop
python analyze.py           # plots + summary table in results/
```

You need a C compiler (`gcc`). Linux: `sudo apt install build-essential`. macOS: `xcode-select --install`.
Windows: use WSL (easiest) or MinGW-w64.

For MNIST instead of the small digits dataset (needs internet the first time):

```bash
python run_experiments.py --dataset mnist --n-total 10000 --hidden 128 --epochs 20 --out results_mnist
python analyze.py --results results_mnist
```

## How it works (short)

- Network: fully connected, ReLU hidden layers, softmax output, cross-entropy loss, He initialisation.
- Optimizers: SGD, SGD with momentum (0.9), Adam (beta1 = 0.9, beta2 = 0.999, eps = 1e-8, bias correction).
- Experiment grid (defaults): 3 optimizers x 5 learning rates (1e-4 ... 1) x 3 batch sizes (16, 64, 256) x 3 seeds, 30 epochs.
- Data is split 60 / 20 / 20 into train / validation / test. The best setting of each optimizer is chosen on the
  **validation** set and reported on the **test** set, so the test set never influences any choice.
- Seeds change the weight initialisation and the batch order. The data split stays fixed.

## Correctness checks

1. **Gradient check:** backprop gradients match numerical gradients (worst relative error about 1e-8; limit is 1e-6).
2. **Reference check:** same architecture and Adam settings in scikit-learn gives a test accuracy within 1 point
   of the C network (96.4% vs 96.9% on one run).

## Results (default settings, digits dataset)

Run `python analyze.py` to regenerate everything in `results/` (`fig1_lr_sensitivity.png`, `fig2_convergence.png`, `summary.md`).

- **Tuning range:** Adam reached at least 90% mean test accuracy at 4 of the 5 learning rates. SGD and SGD with momentum did so at 2 of 5.
- **Low learning rates:** at 1e-3, Adam averaged about 96% test accuracy; SGD only about 45%.
- **Best accuracy is similar:** with the best learning rate, all three land between 97.5% and 98%. A well-tuned SGD matches Adam here.
- **Speed:** at its best setting, Adam reached 97% validation accuracy in about 10 epochs, momentum in about 21, and plain SGD did not reach it within 30.
- **Where it breaks:** SGD with momentum diverged (loss became NaN) in all 3 runs at learning rate 1 with batch size 16.

So the "easier to tune" claim holds in this setup, while the claim that Adam gives clearly better final accuracy does not.

## Limitations (say these in your report)

- One small dataset (1,797 images, 8x8 pixels), one small network (one hidden layer of 64 units), 3 seeds per setting. Results may differ on larger problems.
- No learning-rate schedules, weight decay or other optimizers.
- The C code is written for clarity, not speed (no BLAS, no multithreading).

## Reference

D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015. https://arxiv.org/abs/1412.6980
