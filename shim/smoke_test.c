/* Exercises the shim from plain C, the same way Meadow's FFI will. */
#include "meadow_torch.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      failures++;                                                              \
      const char *e = mt_last_error();                                         \
      printf("FAIL %s:%d: %s%s%s\n", __FILE__, __LINE__, #cond, e ? " -- " : "", e ? e : ""); \
    }                                                                          \
  } while (0)

static int close_to(double a, double b) { return fabs(a - b) < 1e-4; }

static void test_arithmetic(void) {
  int64_t shape[] = {2, 3};
  double a_data[] = {1, 2, 3, 4, 5, 6};
  mt_tensor a = mt_from_f64(a_data, 6, shape, 2, MT_FLOAT32, MT_CPU);
  mt_tensor b = mt_ones(shape, 2, MT_FLOAT32, MT_CPU);
  mt_tensor c = mt_add(a, b);
  mt_tensor s = mt_sum(c);
  CHECK(a && b && c && s);
  CHECK(mt_ndim(c) == 2 && mt_size(c, 0) == 2 && mt_size(c, 1) == 3 && mt_numel(c) == 6);
  CHECK(mt_dtype(c) == MT_FLOAT32 && mt_device(c) == MT_CPU);
  CHECK(close_to(mt_item_f64(s), 27.0));

  double out[6];
  CHECK(mt_copy_f64(c, out, 6) == 6);
  CHECK(close_to(out[0], 2.0) && close_to(out[5], 7.0));

  mt_tensor at = mt_transpose(a, 0, 1);
  mt_tensor m = mt_matmul(a, at);
  CHECK(mt_size(m, 0) == 2 && mt_size(m, 1) == 2);
  CHECK(mt_copy_f64(m, out, 4) == 4);
  CHECK(close_to(out[0], 14.0) && close_to(out[1], 32.0) && close_to(out[3], 77.0));

  char text[256];
  int64_t n = mt_to_string(a, text, sizeof text);
  CHECK(n > 0 && memchr(text, '6', (size_t)(n < 256 ? n : 256)) != NULL);

  mt_free(a); mt_free(b); mt_free(c); mt_free(s); mt_free(at); mt_free(m);
}

static void test_errors(void) {
  int64_t s23[] = {2, 3}, s45[] = {4, 5};
  mt_tensor a = mt_ones(s23, 2, MT_FLOAT32, MT_CPU);
  mt_tensor b = mt_ones(s45, 2, MT_FLOAT32, MT_CPU);
  mt_clear_error();
  CHECK(mt_last_error() == NULL);
  mt_tensor bad = mt_matmul(a, b);
  const char *e = mt_last_error();
  if (bad != NULL || e == NULL || mt_last_error_len() != (int64_t)strlen(e)) {
    failures++;
    printf("FAIL: a shape mismatch should return NULL and record an error\n");
  }
  mt_clear_error();
  CHECK(mt_last_error() == NULL && mt_last_error_len() == 0);
  CHECK(mt_add(NULL, a) == NULL);
  mt_clear_error();
  mt_free(a); mt_free(b);
}

static void test_autograd(void) {
  /* d/dx sum(x^2) = 2x */
  int64_t shape[] = {3};
  double data[] = {1, 2, 3};
  mt_tensor x = mt_from_f64(data, 3, shape, 1, MT_FLOAT32, MT_CPU);
  mt_set_requires_grad(x, 1);
  CHECK(mt_requires_grad(x) == 1);
  CHECK(mt_grad(x) == NULL && mt_last_error() == NULL);
  mt_tensor sq = mt_mul(x, x);
  mt_tensor loss = mt_sum(sq);
  mt_backward(loss);
  mt_tensor g = mt_grad(x);
  double out[3];
  CHECK(g && mt_copy_f64(g, out, 3) == 3);
  CHECK(close_to(out[0], 2.0) && close_to(out[1], 4.0) && close_to(out[2], 6.0));
  mt_free(x); mt_free(sq); mt_free(loss); mt_free(g);
}

/* Fit y = 2x + 1 with a linear layer; the loss must fall and the weights land. */
static void test_training(mt_optim (*make)(const mt_tensor *, int64_t), int steps) {
  mt_manual_seed(0);
  int64_t xs_shape[] = {8, 1}, w_shape[] = {1, 1}, b_shape[] = {1};
  double xs[8], ys[8];
  for (int i = 0; i < 8; i++) { xs[i] = i / 4.0 - 1.0; ys[i] = 2.0 * xs[i] + 1.0; }
  mt_tensor x = mt_from_f64(xs, 8, xs_shape, 2, MT_FLOAT32, MT_CPU);
  mt_tensor y = mt_from_f64(ys, 8, xs_shape, 2, MT_FLOAT32, MT_CPU);
  mt_tensor w = mt_zeros(w_shape, 2, MT_FLOAT32, MT_CPU);
  mt_tensor b = mt_zeros(b_shape, 1, MT_FLOAT32, MT_CPU);
  mt_set_requires_grad(w, 1);
  mt_set_requires_grad(b, 1);
  mt_tensor params[] = {w, b};
  mt_optim opt = make(params, 2);
  CHECK(opt != NULL);

  double first = 0, last = 0;
  for (int step = 0; step < steps; step++) {
    mt_tensor pred = mt_linear(x, w, b);
    mt_tensor loss = mt_mse_loss(pred, y);
    mt_optim_zero_grad(opt);
    mt_backward(loss);
    mt_optim_step(opt);
    last = mt_item_f64(loss);
    if (step == 0) first = last;
    mt_free(pred); mt_free(loss);
  }
  CHECK(last < first * 1e-3);
  CHECK(close_to(mt_item_f64(w), 2.0) && close_to(mt_item_f64(b), 1.0));
  CHECK(mt_last_error() == NULL);
  mt_optim_free(opt);
  mt_free(x); mt_free(y); mt_free(w); mt_free(b);
}

static mt_optim make_sgd(const mt_tensor *p, int64_t n) { return mt_sgd(p, n, 0.3, 0.0, 0.0); }
static mt_optim make_adamw(const mt_tensor *p, int64_t n) { return mt_adamw(p, n, 0.05, 0.9, 0.999, 1e-8, 0.0); }

static void test_nn(void) {
  /* cross entropy of uniform logits over 4 classes is ln 4 */
  int64_t shape[] = {2, 4}, tshape[] = {2};
  int64_t targets[] = {1, 3};
  mt_tensor logits = mt_zeros(shape, 2, MT_FLOAT32, MT_CPU);
  mt_tensor target = mt_from_i64(targets, 2, tshape, 1, MT_CPU);
  mt_tensor loss = mt_cross_entropy(logits, target);
  CHECK(loss && close_to(mt_item_f64(loss), log(4.0)));

  /* an ignored position does not count: only the first row is averaged */
  int64_t some[] = {1, -100};
  mt_tensor partial = mt_from_i64(some, 2, tshape, 1, MT_CPU);
  mt_tensor ignoring = mt_cross_entropy_ignoring(logits, partial, -100);
  CHECK(ignoring && close_to(mt_item_f64(ignoring), log(4.0)));
  CHECK(mt_mps_allocated() >= 0);
  mt_free(partial); mt_free(ignoring);

  mt_tensor table = mt_arange(0, 12, 1, MT_CPU);
  int64_t t43[] = {4, 3};
  mt_tensor table2 = mt_reshape(table, t43, 2);
  mt_tensor tablef = mt_to_dtype(table2, MT_FLOAT32);
  mt_tensor emb = mt_embedding(tablef, target);
  double out[6];
  CHECK(emb && mt_size(emb, 0) == 2 && mt_size(emb, 1) == 3 && mt_copy_f64(emb, out, 6) == 6);
  CHECK(close_to(out[0], 3.0) && close_to(out[3], 9.0));

  mt_tensor am = mt_argmax(tablef, 1, 0);
  int64_t idx[4];
  CHECK(am && mt_copy_i64(am, idx, 4) == 4 && idx[0] == 2 && idx[3] == 2);

  mt_free(logits); mt_free(target); mt_free(loss); mt_free(table); mt_free(table2);
  mt_free(tablef); mt_free(emb); mt_free(am);
}

static void test_scopes(void) {
  int64_t shape[] = {2};
  mt_tensor outside = mt_ones(shape, 1, MT_FLOAT32, MT_CPU);
  int64_t before = mt_live_tensors();
  CHECK(mt_scope_enter() == 1);
  mt_tensor a = mt_add(outside, outside);
  CHECK(mt_scope_enter() == 2);
  mt_tensor b = mt_add(a, a);
  mt_tensor kept = mt_add(b, b);
  mt_tensor freed = mt_add(b, b);
  mt_free(freed);
  mt_keep(kept);
  mt_scope_exit(); /* frees b; kept now belongs to the outer scope */
  CHECK(mt_live_tensors() == before + 2);
  CHECK(close_to(mt_item_f64(mt_sum(kept)), 16.0));
  mt_keep(kept);
  mt_scope_exit(); /* frees a and the sum; kept belongs to nobody */
  CHECK(mt_live_tensors() == before + 1);
  CHECK(mt_numel(kept) == 2 && mt_last_error() == NULL);
  mt_scope_exit();
  CHECK(mt_last_error() != NULL);
  mt_clear_error();
  mt_free(kept); mt_free(outside);
}

static void test_math(void) {
  /* attention with identical keys averages the values each position may see */
  int64_t qshape[] = {1, 3, 2}, vshape[] = {1, 3, 1};
  double v_data[] = {3, 6, 9};
  mt_tensor q = mt_zeros(qshape, 3, MT_FLOAT32, MT_CPU);
  mt_tensor v = mt_from_f64(v_data, 3, vshape, 3, MT_FLOAT32, MT_CPU);
  mt_tensor causal = mt_attention(q, q, v, 1);
  mt_tensor full = mt_attention(q, q, v, 0);
  double out[3];
  CHECK(causal && mt_copy_f64(causal, out, 3) == 3);
  CHECK(close_to(out[0], 3.0) && close_to(out[1], 4.5) && close_to(out[2], 6.0));
  CHECK(full && mt_copy_f64(full, out, 3) == 3 && close_to(out[0], 6.0) && close_to(out[2], 6.0));
  mt_tensor z = mt_zeros(vshape, 3, MT_FLOAT32, MT_CPU);
  mt_tensor s = mt_sin(z), c = mt_cos(z);
  CHECK(close_to(mt_item_f64(mt_sum(s)), 0.0) && close_to(mt_item_f64(mt_sum(c)), 3.0));
}

static void test_solve(void) {
  /* [[2, 0], [0, 4]] X = [[2], [8]] has X = [[1], [2]] */
  int64_t a_shape[] = {2, 2}, b_shape[] = {2, 1};
  double a_data[] = {2, 0, 0, 4}, b_data[] = {2, 8}, out[2];
  mt_tensor a = mt_from_f64(a_data, 4, a_shape, 2, MT_FLOAT32, MT_CPU);
  mt_tensor b = mt_from_f64(b_data, 2, b_shape, 2, MT_FLOAT32, MT_CPU);
  mt_tensor x = mt_solve(a, b);
  CHECK(x && mt_copy_f64(x, out, 2) == 2 && close_to(out[0], 1.0) && close_to(out[1], 2.0));
  mt_free(a); mt_free(b); mt_free(x);
}

static void test_weights(void) {
  mt_weights w = mt_weights_open("tests/tiny.safetensors");
  CHECK(w != NULL && mt_weights_count(w) == 3);
  CHECK(w && strcmp(mt_weights_name(w, 0), "layer.weight") == 0 && mt_weights_name(w, 3) == NULL);
  mt_tensor m = mt_weights_get(w, "layer.weight");
  double out[4];
  CHECK(m && mt_dtype(m) == MT_FLOAT32 && mt_size(m, 0) == 2 && mt_copy_f64(m, out, 4) == 4);
  CHECK(close_to(out[0], 1.5) && close_to(out[1], -2.0) && close_to(out[2], 0.25) && close_to(out[3], 4.0));
  mt_tensor ids = mt_weights_get(w, "ids");
  int64_t iout[3];
  CHECK(ids && mt_dtype(ids) == MT_INT64 && mt_copy_i64(ids, iout, 3) == 3 && iout[1] == -8 && iout[2] == 9);
  mt_tensor scale = mt_weights_get(w, "scale");
  CHECK(scale && mt_ndim(scale) == 0 && close_to(mt_item_f64(scale), 3.5));
  CHECK(mt_last_error() == NULL);
  CHECK(mt_weights_get(w, "missing") == NULL && mt_last_error() != NULL);
  mt_clear_error();
  CHECK(mt_weights_open("tests/no-such-file.safetensors") == NULL && mt_last_error() != NULL);
  mt_clear_error();
  CHECK(mt_weights_open("shim/build.sh") == NULL && mt_last_error() != NULL);
  mt_clear_error();
  mt_free(m); mt_free(ids); mt_free(scale);
  mt_weights_free(w);
}

static void test_mps(void) {
  if (!mt_mps_available()) { printf("mps: not available, skipped\n"); return; }
  int64_t shape[] = {2, 2};
  mt_tensor a = mt_ones(shape, 2, MT_FLOAT32, MT_MPS);
  mt_tensor s = mt_sum(a);
  CHECK(a && mt_device(a) == MT_MPS && close_to(mt_item_f64(s), 4.0));
  /* data crosses to and from the device intact */
  double in[] = {1, 2, 3, 4}, out[4] = {0};
  mt_tensor b = mt_from_f64(in, 4, shape, 2, MT_FLOAT32, MT_MPS);
  CHECK(b && mt_copy_f64(b, out, 4) == 4 && close_to(out[0], 1.0) && close_to(out[3], 4.0));
  mt_tensor c = mt_to_device(a, MT_CPU);
  mt_tensor d = mt_to_device(c, MT_MPS);
  CHECK(mt_copy_f64(d, out, 4) == 4 && close_to(out[0], 1.0) && close_to(out[3], 1.0));
  mt_free(a); mt_free(s); mt_free(b); mt_free(c); mt_free(d);
}

static void test_cuda(void) {
  if (!mt_cuda_available()) { printf("cuda: not available, skipped\n"); return; }
  int64_t shape[] = {2, 2};
  double in[] = {1, 2, 3, 4}, out[4] = {0};
  mt_tensor a = mt_from_f64(in, 4, shape, 2, MT_FLOAT32, MT_CUDA);
  mt_tensor b = mt_matmul(a, a);
  CHECK(a && b && mt_device(b) == MT_CUDA && mt_copy_f64(b, out, 4) == 4);
  CHECK(close_to(out[0], 7.0) && close_to(out[3], 22.0));
  mt_set_requires_grad(a, 1);
  mt_tensor sq = mt_mul(a, a);
  mt_tensor loss = mt_sum(sq);
  mt_backward(loss);
  mt_tensor g = mt_grad(a);
  CHECK(g && mt_copy_f64(g, out, 4) == 4 && close_to(out[0], 2.0) && close_to(out[3], 8.0));
  printf("cuda: ok\n");
  mt_free(a); mt_free(b); mt_free(sq); mt_free(loss); mt_free(g);
}

int main(void) {
  test_arithmetic();
  test_errors();
  test_autograd();
  test_training(make_sgd, 200);
  test_training(make_adamw, 600);
  test_nn();
  test_scopes();
  mt_scope_enter();
  test_math();
  mt_scope_exit();
  test_weights();
  test_solve();
  test_mps();
  test_cuda();
  if (mt_live_tensors() != 0) {
    failures++;
    printf("FAIL: %lld tensor handles leaked\n", (long long)mt_live_tensors());
  }
  printf(failures ? "%d FAILED\n" : "all shim tests passed\n", failures);
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
