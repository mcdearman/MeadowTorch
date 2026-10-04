/* C ABI over libtorch for the Meadow `Torch` package.
 *
 * Only int64_t, double, pointers and NUL-terminated strings cross the boundary.
 * A tensor handle owns a reference to its tensor and must be released with
 * mt_free, or by the scope it was made in (see mt_scope_enter). A function that fails returns NULL (handles), 0 (numbers) or does
 * nothing (void), and records a message readable with mt_last_error.
 */
#ifndef MEADOW_TORCH_H
#define MEADOW_TORCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mt_tensor_s *mt_tensor;
typedef struct mt_optim_s *mt_optim;
typedef struct mt_weights_s *mt_weights;

/* dtype codes */
enum { MT_FLOAT32 = 0, MT_FLOAT64 = 1, MT_INT64 = 2, MT_BOOL = 3 };
/* device codes */
enum { MT_CPU = 0, MT_CUDA = 1, MT_MPS = 2 };

/* --- errors ----------------------------------------------------------------- */
/* The message of the last failed call on this thread, or NULL if none. */
const char *mt_last_error(void);
/* Its length in bytes, 0 if none. */
int64_t mt_last_error_len(void);
void mt_clear_error(void);

/* --- process-wide state ----------------------------------------------------- */
void mt_manual_seed(int64_t seed);
int64_t mt_cuda_available(void);
int64_t mt_mps_available(void);
/* Returns the previous setting. */
int64_t mt_set_grad_enabled(int64_t enabled);
/* Tensor handles alive right now; for leak checks in tests. */
int64_t mt_live_tensors(void);

/* --- creation --------------------------------------------------------------- */
mt_tensor mt_zeros(const int64_t *shape, int64_t ndim, int64_t dtype, int64_t device);
mt_tensor mt_ones(const int64_t *shape, int64_t ndim, int64_t dtype, int64_t device);
mt_tensor mt_full(const int64_t *shape, int64_t ndim, double value, int64_t dtype, int64_t device);
mt_tensor mt_randn(const int64_t *shape, int64_t ndim, int64_t device);
mt_tensor mt_rand(const int64_t *shape, int64_t ndim, int64_t device);
mt_tensor mt_arange(int64_t start, int64_t end, int64_t step, int64_t device);
mt_tensor mt_scalar_f64(double value, int64_t dtype, int64_t device);
/* Copies `len` values; `len` must equal the product of `shape`. */
mt_tensor mt_from_f64(const double *data, int64_t len, const int64_t *shape, int64_t ndim, int64_t dtype, int64_t device);
mt_tensor mt_from_i64(const int64_t *data, int64_t len, const int64_t *shape, int64_t ndim, int64_t device);

void mt_free(mt_tensor t);

/* --- scopes ----------------------------------------------------------------- */
/* A handle made while a scope is open belongs to the innermost one, and
 * mt_scope_exit frees every handle the scope still owns. mt_keep hands a handle
 * to the enclosing scope, or to nobody if there is none, so it outlives the
 * exit. Scopes are per thread. mt_scope_enter returns the new depth. */
int64_t mt_scope_enter(void);
void mt_scope_exit(void);
/* Handles the innermost scope owns; 0 outside every scope. */
int64_t mt_scope_size(void);
void mt_keep(mt_tensor t);

/* --- inspection ------------------------------------------------------------- */
int64_t mt_ndim(mt_tensor t);
int64_t mt_size(mt_tensor t, int64_t dim);
int64_t mt_numel(mt_tensor t);
int64_t mt_dtype(mt_tensor t);
int64_t mt_device(mt_tensor t);
int64_t mt_requires_grad(mt_tensor t);
double mt_item_f64(mt_tensor t);
int64_t mt_item_i64(mt_tensor t);
/* Copy up to `cap` elements in row-major order; returns how many were written. */
int64_t mt_copy_f64(mt_tensor t, double *out, int64_t cap);
int64_t mt_copy_i64(mt_tensor t, int64_t *out, int64_t cap);
/* Write the printed form into `out` (not NUL-terminated beyond `cap`); returns
 * the full length, so a caller can size a buffer by calling with cap = 0. */
int64_t mt_to_string(mt_tensor t, char *out, int64_t cap);

/* --- conversion ------------------------------------------------------------- */
mt_tensor mt_to_device(mt_tensor t, int64_t device);
mt_tensor mt_to_dtype(mt_tensor t, int64_t dtype);
mt_tensor mt_clone(mt_tensor t);
mt_tensor mt_detach(mt_tensor t);

/* --- shape ------------------------------------------------------------------ */
mt_tensor mt_reshape(mt_tensor t, const int64_t *shape, int64_t ndim);
mt_tensor mt_transpose(mt_tensor t, int64_t dim0, int64_t dim1);
mt_tensor mt_squeeze(mt_tensor t, int64_t dim);
mt_tensor mt_unsqueeze(mt_tensor t, int64_t dim);
mt_tensor mt_cat(const mt_tensor *ts, int64_t n, int64_t dim);
mt_tensor mt_stack(const mt_tensor *ts, int64_t n, int64_t dim);
mt_tensor mt_index_select(mt_tensor t, int64_t dim, mt_tensor index);
mt_tensor mt_narrow(mt_tensor t, int64_t dim, int64_t start, int64_t length);

/* --- arithmetic (broadcasting) ---------------------------------------------- */
mt_tensor mt_add(mt_tensor a, mt_tensor b);
mt_tensor mt_sub(mt_tensor a, mt_tensor b);
mt_tensor mt_mul(mt_tensor a, mt_tensor b);
mt_tensor mt_div(mt_tensor a, mt_tensor b);
mt_tensor mt_matmul(mt_tensor a, mt_tensor b);
mt_tensor mt_add_scalar(mt_tensor a, double s);
mt_tensor mt_mul_scalar(mt_tensor a, double s);
mt_tensor mt_pow_scalar(mt_tensor a, double exponent);

/* --- unary ------------------------------------------------------------------ */
mt_tensor mt_neg(mt_tensor a);
mt_tensor mt_exp(mt_tensor a);
mt_tensor mt_log(mt_tensor a);
mt_tensor mt_sqrt(mt_tensor a);
mt_tensor mt_abs(mt_tensor a);
mt_tensor mt_tanh(mt_tensor a);
mt_tensor mt_sigmoid(mt_tensor a);
mt_tensor mt_relu(mt_tensor a);
mt_tensor mt_gelu(mt_tensor a);
mt_tensor mt_sin(mt_tensor a);
mt_tensor mt_cos(mt_tensor a);

/* --- reductions ------------------------------------------------------------- */
mt_tensor mt_sum(mt_tensor a);
mt_tensor mt_mean(mt_tensor a);
mt_tensor mt_sum_dim(mt_tensor a, int64_t dim, int64_t keepdim);
mt_tensor mt_mean_dim(mt_tensor a, int64_t dim, int64_t keepdim);
mt_tensor mt_argmax(mt_tensor a, int64_t dim, int64_t keepdim);
mt_tensor mt_softmax(mt_tensor a, int64_t dim);
mt_tensor mt_log_softmax(mt_tensor a, int64_t dim);

/* --- comparison ------------------------------------------------------------- */
mt_tensor mt_eq(mt_tensor a, mt_tensor b);
int64_t mt_allclose(mt_tensor a, mt_tensor b, double rtol, double atol);

/* --- nn functional ---------------------------------------------------------- */
/* `bias` may be NULL. */
mt_tensor mt_linear(mt_tensor input, mt_tensor weight, mt_tensor bias);
mt_tensor mt_embedding(mt_tensor weight, mt_tensor indices);
mt_tensor mt_layer_norm(mt_tensor input, mt_tensor weight, mt_tensor bias, double eps);
mt_tensor mt_dropout(mt_tensor input, double p, int64_t training);
/* `logits` is (N, C), `target` is int64 (N); mean reduction. */
mt_tensor mt_cross_entropy(mt_tensor logits, mt_tensor target);
mt_tensor mt_mse_loss(mt_tensor input, mt_tensor target);
/* Scaled dot-product attention over the last two dimensions, (..., T, D).
 * With `causal`, a position attends only to itself and earlier ones. */
mt_tensor mt_attention(mt_tensor query, mt_tensor key, mt_tensor value, int64_t causal);

/* --- autograd --------------------------------------------------------------- */
void mt_set_requires_grad(mt_tensor t, int64_t requires_grad);
void mt_backward(mt_tensor loss);
/* NULL without an error if the tensor has no gradient yet. */
mt_tensor mt_grad(mt_tensor t);
void mt_zero_grad(mt_tensor t);
/* t += alpha * other, in place and outside the graph. */
void mt_add_inplace(mt_tensor t, mt_tensor other, double alpha);

/* --- optimizers ------------------------------------------------------------- */
mt_optim mt_sgd(const mt_tensor *params, int64_t n, double lr, double momentum, double weight_decay);
mt_optim mt_adamw(const mt_tensor *params, int64_t n, double lr, double beta1, double beta2, double eps, double weight_decay);
void mt_optim_step(mt_optim o);
void mt_optim_zero_grad(mt_optim o);
void mt_optim_free(mt_optim o);

/* --- weight files ----------------------------------------------------------- */
/* Open a .safetensors file. Only its header is read; mt_weights_get reads one
 * tensor's data. */
mt_weights mt_weights_open(const char *path);
int64_t mt_weights_count(mt_weights w);
/* The name of tensor `i`, valid until the file is closed; NULL if out of range. */
const char *mt_weights_name(mt_weights w, int64_t i);
/* The named tensor, on the CPU. A floating-point tensor is answered as
 * float32 unless stored as float64, and an integer one as int64. */
mt_tensor mt_weights_get(mt_weights w, const char *name);
void mt_weights_free(mt_weights w);

#ifdef __cplusplus
}
#endif

#endif
