/* A small fully connected neural network written from scratch in C.
 * ReLU hidden layers, softmax + cross-entropy output, mini-batch training with
 * SGD, SGD with momentum, or Adam (Kingma & Ba, 2015). No external libraries. */

#include "nn.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NN_PI 3.14159265358979323846

struct Net {
    int n_layers;               /* number of entries in sizes, including the input layer */
    int sizes[NN_MAX_LAYERS];
    int max_batch;
    int n_params;
    long t;                     /* Adam step counter */
    uint64_t rng;

    /* all parameters live in one flat array so the optimizers are simple loops */
    double *params, *grads, *m, *v;

    /* layer l (1 .. n_layers-1) maps sizes[l-1] -> sizes[l] */
    double *W[NN_MAX_LAYERS], *b[NN_MAX_LAYERS];     /* views into params */
    double *gW[NN_MAX_LAYERS], *gb[NN_MAX_LAYERS];   /* views into grads  */

    double *A[NN_MAX_LAYERS];   /* activations (A[0] points at the input batch) */
    double *Z[NN_MAX_LAYERS];   /* pre-activations */
    double *dZ[NN_MAX_LAYERS];  /* gradient of the loss w.r.t. Z */
};

/* ---------- random numbers (splitmix64 + Box-Muller) ---------- */

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static double rand_uniform(Net *n) { /* in (0, 1) */
    return ((double)(splitmix64(&n->rng) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

static double rand_normal(Net *n) {
    double u1 = rand_uniform(n), u2 = rand_uniform(n);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * NN_PI * u2);
}

/* ---------- create / free ---------- */

Net *net_create(const int *sizes, int n_layers, int max_batch, unsigned int seed) {
    if (!sizes || n_layers < 2 || n_layers > NN_MAX_LAYERS || max_batch < 1) return NULL;
    for (int l = 0; l < n_layers; l++)
        if (sizes[l] < 1) return NULL;

    Net *n = (Net *)calloc(1, sizeof(Net));
    if (!n) return NULL;
    n->n_layers = n_layers;
    n->max_batch = max_batch;
    n->rng = (uint64_t)seed * 0x2545F4914F6CDD1DULL + 12345ULL;
    memcpy(n->sizes, sizes, (size_t)n_layers * sizeof(int));

    int total = 0;
    for (int l = 1; l < n_layers; l++) total += sizes[l - 1] * sizes[l] + sizes[l];
    n->n_params = total;

    n->params = (double *)calloc((size_t)total, sizeof(double));
    n->grads  = (double *)calloc((size_t)total, sizeof(double));
    n->m      = (double *)calloc((size_t)total, sizeof(double));
    n->v      = (double *)calloc((size_t)total, sizeof(double));
    int ok = n->params && n->grads && n->m && n->v;

    int offset = 0;
    for (int l = 1; l < n_layers && ok; l++) {
        int in = sizes[l - 1], out = sizes[l];
        n->W[l]  = n->params + offset;
        n->gW[l] = n->grads + offset;
        offset += in * out;
        n->b[l]  = n->params + offset;
        n->gb[l] = n->grads + offset;
        offset += out;

        double std = sqrt(2.0 / (double)in); /* He initialisation, suited to ReLU */
        for (int i = 0; i < in * out; i++) n->W[l][i] = std * rand_normal(n);
        /* biases start at 0 */

        size_t cells = (size_t)max_batch * (size_t)out;
        n->A[l]  = (double *)calloc(cells, sizeof(double));
        n->Z[l]  = (double *)calloc(cells, sizeof(double));
        n->dZ[l] = (double *)calloc(cells, sizeof(double));
        if (!n->A[l] || !n->Z[l] || !n->dZ[l]) ok = 0;
    }
    if (!ok) { net_free(n); return NULL; }
    return n;
}

void net_free(Net *n) {
    if (!n) return;
    for (int l = 1; l < n->n_layers; l++) {
        free(n->A[l]);
        free(n->Z[l]);
        free(n->dZ[l]);
    }
    free(n->params);
    free(n->grads);
    free(n->m);
    free(n->v);
    free(n);
}

int net_num_params(const Net *n) { return n ? n->n_params : 0; }

/* ---------- forward pass ---------- */

static void forward(Net *n, const double *X, int B) {
    n->A[0] = (double *)X;
    int last = n->n_layers - 1;

    for (int l = 1; l <= last; l++) {
        int in = n->sizes[l - 1], out = n->sizes[l];
        const double *Ain = n->A[l - 1];
        double *Z = n->Z[l];
        const double *W = n->W[l];

        for (int i = 0; i < B; i++)
            for (int j = 0; j < out; j++) Z[i * out + j] = n->b[l][j];

        /* Z = Ain * W + b   (i-k-j loop order keeps memory access sequential) */
        for (int i = 0; i < B; i++)
            for (int k = 0; k < in; k++) {
                double a = Ain[i * in + k];
                const double *Wk = W + k * out;
                double *Zi = Z + i * out;
                for (int j = 0; j < out; j++) Zi[j] += a * Wk[j];
            }

        if (l < last) {
            for (int i = 0; i < B * out; i++) n->A[l][i] = Z[i] > 0.0 ? Z[i] : 0.0; /* ReLU */
        } else {
            for (int i = 0; i < B; i++) { /* softmax, subtracting the max for stability */
                const double *z = Z + i * out;
                double *p = n->A[l] + i * out;
                double mx = z[0];
                for (int j = 1; j < out; j++) if (z[j] > mx) mx = z[j];
                double sum = 0.0;
                for (int j = 0; j < out; j++) { p[j] = exp(z[j] - mx); sum += p[j]; }
                for (int j = 0; j < out; j++) p[j] /= sum;
            }
        }
    }
}

/* mean cross-entropy of the probabilities currently stored in the output layer */
static double mean_loss(const Net *n, const int *y, int B) {
    int out = n->sizes[n->n_layers - 1];
    const double *P = n->A[n->n_layers - 1];
    double s = 0.0;
    for (int i = 0; i < B; i++) {
        double p = P[i * out + y[i]];
        if (p < 1e-12) p = 1e-12; /* avoid log(0); a NaN passes through so divergence stays visible */
        s += -log(p);
    }
    return s / (double)B;
}

/* ---------- backward pass ---------- */

static void backward(Net *n, const int *y, int B) {
    int last = n->n_layers - 1;
    int out_last = n->sizes[last];

    /* softmax + cross-entropy: dL/dZ = (p - onehot(y)) / B */
    for (int i = 0; i < B; i++)
        for (int j = 0; j < out_last; j++) {
            double p = n->A[last][i * out_last + j];
            n->dZ[last][i * out_last + j] = (p - (j == y[i] ? 1.0 : 0.0)) / (double)B;
        }

    for (int l = last; l >= 1; l--) {
        int in = n->sizes[l - 1], out = n->sizes[l];
        const double *dZ = n->dZ[l];
        const double *Ain = n->A[l - 1];
        double *gW = n->gW[l], *gb = n->gb[l];

        memset(gW, 0, (size_t)in * (size_t)out * sizeof(double));
        memset(gb, 0, (size_t)out * sizeof(double));

        for (int i = 0; i < B; i++) {
            for (int j = 0; j < out; j++) gb[j] += dZ[i * out + j];
            for (int k = 0; k < in; k++) {          /* gW = Ain^T * dZ */
                double a = Ain[i * in + k];
                double *gWk = gW + k * out;
                const double *dZi = dZ + i * out;
                for (int j = 0; j < out; j++) gWk[j] += a * dZi[j];
            }
        }

        if (l > 1) { /* pass the gradient to the previous layer through W and the ReLU */
            const double *W = n->W[l];
            for (int i = 0; i < B; i++)
                for (int k = 0; k < in; k++) {
                    double s = 0.0;
                    const double *Wk = W + k * out;
                    const double *dZi = dZ + i * out;
                    for (int j = 0; j < out; j++) s += dZi[j] * Wk[j];
                    n->dZ[l - 1][i * in + k] = (n->Z[l - 1][i * in + k] > 0.0) ? s : 0.0;
                }
        }
    }
}

/* ---------- optimizers ---------- */

static void apply_update(Net *n, int opt, double lr, double b1, double b2, double eps) {
    int N = n->n_params;
    double *p = n->params, *g = n->grads, *m = n->m, *v = n->v;

    if (opt == OPT_SGD) {
        for (int i = 0; i < N; i++) p[i] -= lr * g[i];
    } else if (opt == OPT_MOMENTUM) {
        for (int i = 0; i < N; i++) {
            m[i] = b1 * m[i] + g[i];
            p[i] -= lr * m[i];
        }
    } else { /* Adam, Algorithm 1 of Kingma & Ba with bias-corrected moments */
        n->t += 1;
        double c1 = 1.0 - pow(b1, (double)n->t);
        double c2 = 1.0 - pow(b2, (double)n->t);
        for (int i = 0; i < N; i++) {
            m[i] = b1 * m[i] + (1.0 - b1) * g[i];
            v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
            p[i] -= lr * (m[i] / c1) / (sqrt(v[i] / c2) + eps);
        }
    }
}

double net_train_batch(Net *n, const double *X, const int *y, int B,
                       int optimizer, double lr, double beta1, double beta2, double eps) {
    if (!n || !X || !y || B < 1 || B > n->max_batch) return -1.0;
    if (optimizer < OPT_SGD || optimizer > OPT_ADAM) return -1.0;
    forward(n, X, B);
    double loss = mean_loss(n, y, B);
    backward(n, y, B);
    apply_update(n, optimizer, lr, beta1, beta2, eps);
    return loss;
}

/* ---------- evaluation ---------- */

static int argmax_row(const double *row, int len) {
    int best = 0;
    for (int j = 1; j < len; j++) if (row[j] > row[best]) best = j;
    return best;
}

double net_evaluate(Net *n, const double *X, const int *y, int N, double *accuracy) {
    if (!n || !X || !y || N < 1) return -1.0;
    int d = n->sizes[0], c = n->sizes[n->n_layers - 1];
    double total = 0.0;
    int correct = 0;
    for (int start = 0; start < N; start += n->max_batch) {
        int B = N - start < n->max_batch ? N - start : n->max_batch;
        forward(n, X + (size_t)start * (size_t)d, B);
        total += mean_loss(n, y + start, B) * (double)B;
        for (int i = 0; i < B; i++)
            if (argmax_row(n->A[n->n_layers - 1] + i * c, c) == y[start + i]) correct++;
    }
    if (accuracy) *accuracy = (double)correct / (double)N;
    return total / (double)N;
}

void net_predict(Net *n, const double *X, int N, int *out) {
    if (!n || !X || !out || N < 1) return;
    int d = n->sizes[0], c = n->sizes[n->n_layers - 1];
    for (int start = 0; start < N; start += n->max_batch) {
        int B = N - start < n->max_batch ? N - start : n->max_batch;
        forward(n, X + (size_t)start * (size_t)d, B);
        for (int i = 0; i < B; i++)
            out[start + i] = argmax_row(n->A[n->n_layers - 1] + i * c, c);
    }
}

/* ---------- gradient check ---------- */

double net_grad_check(Net *n, const double *X, const int *y, int B, double h) {
    if (!n || !X || !y || B < 1 || B > n->max_batch || h <= 0.0) return -1.0;

    forward(n, X, B);
    backward(n, y, B);

    int N = n->n_params;
    double *analytic = (double *)malloc((size_t)N * sizeof(double));
    if (!analytic) return -1.0;
    memcpy(analytic, n->grads, (size_t)N * sizeof(double));

    double worst = 0.0;
    for (int i = 0; i < N; i++) {
        double orig = n->params[i];

        n->params[i] = orig + h;
        forward(n, X, B);
        double lp = mean_loss(n, y, B);

        n->params[i] = orig - h;
        forward(n, X, B);
        double lm = mean_loss(n, y, B);

        n->params[i] = orig;

        double numeric = (lp - lm) / (2.0 * h);
        double diff = fabs(numeric - analytic[i]);
        double denom = fabs(numeric) + fabs(analytic[i]);
        /* when both gradients are essentially zero, compare the absolute difference */
        double rel = denom < 1e-6 ? diff : diff / denom;
        if (rel > worst) worst = rel;
    }
    free(analytic);
    return worst;
}
