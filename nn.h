#ifndef NN_H
#define NN_H

#ifdef _WIN32
#define NN_API __declspec(dllexport)
#else
#define NN_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NN_MAX_LAYERS 8

/* optimizer ids */
enum { OPT_SGD = 0, OPT_MOMENTUM = 1, OPT_ADAM = 2 };

typedef struct Net Net;

/* Create a fully connected network.
 *   sizes    : layer sizes, e.g. {64, 64, 10} = 64 inputs, one hidden layer of 64, 10 classes
 *   n_layers : number of entries in sizes (2 .. NN_MAX_LAYERS)
 *   max_batch: largest batch that will ever be passed in
 *   seed     : seed for weight initialisation (He initialisation)
 * Hidden layers use ReLU, the output layer uses softmax + cross-entropy loss.
 * Returns NULL on bad arguments or out of memory. */
NN_API Net *net_create(const int *sizes, int n_layers, int max_batch, unsigned int seed);
NN_API void net_free(Net *net);
NN_API int net_num_params(const Net *net);

/* One training step on one mini-batch (forward, backward, parameter update).
 * X is batch x sizes[0], row-major. y holds class labels.
 * optimizer: OPT_SGD, OPT_MOMENTUM (beta1 = momentum), OPT_ADAM (beta1, beta2, eps).
 * Use one Net for one optimizer only (momentum and Adam keep state inside the Net).
 * Returns the mean loss on the batch BEFORE the update, or -1.0 on bad arguments. */
NN_API double net_train_batch(Net *net, const double *X, const int *y, int batch,
                              int optimizer, double lr, double beta1, double beta2, double eps);

/* Mean loss over n samples. If accuracy is not NULL, also stores the accuracy. */
NN_API double net_evaluate(Net *net, const double *X, const int *y, int n, double *accuracy);

/* Writes the predicted class of each of the n samples into out. */
NN_API void net_predict(Net *net, const double *X, int n, int *out);

/* Compares backpropagation gradients with numerical gradients (central differences, step h).
 * Returns the worst relative error over all parameters. Values below ~1e-6 mean backprop is correct. */
NN_API double net_grad_check(Net *net, const double *X, const int *y, int batch, double h);

#ifdef __cplusplus
}
#endif

#endif
