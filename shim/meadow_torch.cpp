#include "meadow_torch.h"

#include <torch/torch.h>

#include <atomic>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

struct mt_tensor_s {
  torch::Tensor t;
  // Index of the scope that owns the handle, or -1 if none does.
  int64_t scope;
};

struct mt_optim_s {
  std::unique_ptr<torch::optim::Optimizer> o;
};

namespace {

thread_local std::string last_error;
thread_local bool has_error = false;
std::atomic<int64_t> live_tensors{0};
// Handles made while a scope is open belong to the innermost one and are freed
// when it exits, unless freed or kept first.
thread_local std::vector<std::unordered_set<mt_tensor>> scopes;

void set_error(const char *what) {
  last_error = what;
  has_error = true;
}

// Run `body`, turning any C++ exception into a recorded error and `fallback`.
template <typename R, typename F> R guard(R fallback, F &&body) {
  try {
    return body();
  } catch (const c10::Error &e) {
    // what() carries a C++ backtrace, which is noise on the Meadow side.
    set_error(e.what_without_backtrace());
  } catch (const std::exception &e) {
    set_error(e.what());
  } catch (...) {
    set_error("unknown C++ exception");
  }
  return fallback;
}

template <typename F> void guard_void(F &&body) {
  guard<int>(0, [&] {
    body();
    return 0;
  });
}

mt_tensor wrap(torch::Tensor t) {
  live_tensors.fetch_add(1);
  auto *h = new mt_tensor_s{std::move(t), static_cast<int64_t>(scopes.size()) - 1};
  if (h->scope >= 0) scopes.back().insert(h);
  return h;
}

template <typename F> mt_tensor guard_tensor(F &&body) {
  return guard<mt_tensor>(nullptr, [&] { return wrap(body()); });
}

const torch::Tensor &get(mt_tensor t) {
  if (t == nullptr) throw std::invalid_argument("null tensor handle");
  return t->t;
}

torch::Dtype dtype_of(int64_t code) {
  switch (code) {
  case MT_FLOAT32: return torch::kFloat32;
  case MT_FLOAT64: return torch::kFloat64;
  case MT_INT64: return torch::kInt64;
  case MT_BOOL: return torch::kBool;
  }
  throw std::invalid_argument("unknown dtype code " + std::to_string(code));
}

int64_t dtype_code(torch::Dtype d) {
  switch (d) {
  case torch::kFloat32: return MT_FLOAT32;
  case torch::kFloat64: return MT_FLOAT64;
  case torch::kInt64: return MT_INT64;
  case torch::kBool: return MT_BOOL;
  default: throw std::invalid_argument("tensor has a dtype the shim does not expose");
  }
}

torch::Device device_of(int64_t code) {
  switch (code) {
  case MT_CPU: return torch::Device(torch::kCPU);
  case MT_CUDA: return torch::Device(torch::kCUDA);
  case MT_MPS: return torch::Device(torch::kMPS);
  }
  throw std::invalid_argument("unknown device code " + std::to_string(code));
}

int64_t device_code(const torch::Device &d) {
  if (d.is_cpu()) return MT_CPU;
  if (d.is_cuda()) return MT_CUDA;
  if (d.is_mps()) return MT_MPS;
  throw std::invalid_argument("tensor is on a device the shim does not expose");
}

torch::IntArrayRef shape_of(const int64_t *shape, int64_t ndim) {
  if (ndim < 0 || (ndim > 0 && shape == nullptr)) throw std::invalid_argument("bad shape");
  return torch::IntArrayRef(shape, static_cast<size_t>(ndim));
}

torch::TensorOptions options(int64_t dtype, int64_t device) {
  return torch::TensorOptions().dtype(dtype_of(dtype)).device(device_of(device));
}

void check_len(int64_t len, torch::IntArrayRef shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  if (len != n) {
    throw std::invalid_argument("data has " + std::to_string(len) + " elements but the shape holds " +
                                std::to_string(n));
  }
}

std::vector<torch::Tensor> list_of(const mt_tensor *ts, int64_t n) {
  if (n < 0 || (n > 0 && ts == nullptr)) throw std::invalid_argument("bad tensor list");
  std::vector<torch::Tensor> out;
  out.reserve(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; i++) out.push_back(get(ts[i]));
  return out;
}

template <typename T> int64_t copy_out(mt_tensor t, T *out, int64_t cap, torch::Dtype dtype) {
  return guard<int64_t>(0, [&]() -> int64_t {
    if (cap <= 0) return 0;
    if (out == nullptr) throw std::invalid_argument("null output buffer");
    // Device first, then dtype: MPS has no float64 to convert on.
    torch::Tensor flat = get(t).detach().to(torch::kCPU).to(dtype).contiguous().flatten();
    int64_t n = std::min<int64_t>(cap, flat.numel());
    std::memcpy(out, flat.data_ptr<T>(), static_cast<size_t>(n) * sizeof(T));
    return n;
  });
}

} // namespace

#define MT_BINARY(name, expr)                                                                      \
  mt_tensor name(mt_tensor a, mt_tensor b) {                                                       \
    return guard_tensor([&] {                                                                      \
      const torch::Tensor &x = get(a);                                                             \
      const torch::Tensor &y = get(b);                                                             \
      return expr;                                                                                 \
    });                                                                                            \
  }

#define MT_UNARY(name, expr)                                                                       \
  mt_tensor name(mt_tensor a) {                                                                    \
    return guard_tensor([&] {                                                                      \
      const torch::Tensor &x = get(a);                                                             \
      return expr;                                                                                 \
    });                                                                                            \
  }

extern "C" {

// --- errors ------------------------------------------------------------------

const char *mt_last_error(void) { return has_error ? last_error.c_str() : nullptr; }
int64_t mt_last_error_len(void) { return has_error ? static_cast<int64_t>(last_error.size()) : 0; }
void mt_clear_error(void) {
  has_error = false;
  last_error.clear();
}

// --- process-wide state ------------------------------------------------------

void mt_manual_seed(int64_t seed) {
  guard_void([&] { torch::manual_seed(static_cast<uint64_t>(seed)); });
}
int64_t mt_cuda_available(void) {
  return guard<int64_t>(0, [] { return torch::cuda::is_available() ? 1 : 0; });
}
int64_t mt_mps_available(void) {
  return guard<int64_t>(0, [] { return torch::mps::is_available() ? 1 : 0; });
}
int64_t mt_set_grad_enabled(int64_t enabled) {
  return guard<int64_t>(0, [&] {
    int64_t previous = torch::GradMode::is_enabled() ? 1 : 0;
    torch::GradMode::set_enabled(enabled != 0);
    return previous;
  });
}
int64_t mt_live_tensors(void) { return live_tensors.load(); }

// --- creation ----------------------------------------------------------------

mt_tensor mt_zeros(const int64_t *shape, int64_t ndim, int64_t dtype, int64_t device) {
  return guard_tensor([&] { return torch::zeros(shape_of(shape, ndim), options(dtype, device)); });
}
mt_tensor mt_ones(const int64_t *shape, int64_t ndim, int64_t dtype, int64_t device) {
  return guard_tensor([&] { return torch::ones(shape_of(shape, ndim), options(dtype, device)); });
}
mt_tensor mt_full(const int64_t *shape, int64_t ndim, double value, int64_t dtype, int64_t device) {
  return guard_tensor([&] { return torch::full(shape_of(shape, ndim), value, options(dtype, device)); });
}
mt_tensor mt_randn(const int64_t *shape, int64_t ndim, int64_t device) {
  // Sampled on the CPU so a seed gives the same values on every device.
  return guard_tensor([&] { return torch::randn(shape_of(shape, ndim)).to(device_of(device)); });
}
mt_tensor mt_rand(const int64_t *shape, int64_t ndim, int64_t device) {
  return guard_tensor([&] { return torch::rand(shape_of(shape, ndim)).to(device_of(device)); });
}
mt_tensor mt_arange(int64_t start, int64_t end, int64_t step, int64_t device) {
  return guard_tensor([&] { return torch::arange(start, end, step, options(MT_INT64, device)); });
}
mt_tensor mt_scalar_f64(double value, int64_t dtype, int64_t device) {
  return guard_tensor([&] { return torch::scalar_tensor(value, options(dtype, device)); });
}
mt_tensor mt_from_f64(const double *data, int64_t len, const int64_t *shape, int64_t ndim, int64_t dtype,
                      int64_t device) {
  return guard_tensor([&] {
    auto dims = shape_of(shape, ndim);
    check_len(len, dims);
    torch::Tensor t = torch::empty(dims, torch::kFloat64);
    if (len > 0) std::memcpy(t.data_ptr<double>(), data, static_cast<size_t>(len) * sizeof(double));
    // Dtype on the CPU, then device: MPS has no float64 to convert from.
    return t.to(dtype_of(dtype)).to(device_of(device));
  });
}
mt_tensor mt_from_i64(const int64_t *data, int64_t len, const int64_t *shape, int64_t ndim, int64_t device) {
  return guard_tensor([&] {
    auto dims = shape_of(shape, ndim);
    check_len(len, dims);
    torch::Tensor t = torch::empty(dims, torch::kInt64);
    if (len > 0) std::memcpy(t.data_ptr<int64_t>(), data, static_cast<size_t>(len) * sizeof(int64_t));
    return t.to(device_of(device));
  });
}

void mt_free(mt_tensor t) {
  if (t == nullptr) return;
  if (t->scope >= 0) scopes[static_cast<size_t>(t->scope)].erase(t);
  live_tensors.fetch_sub(1);
  delete t;
}

// --- scopes ------------------------------------------------------------------

int64_t mt_scope_enter(void) {
  scopes.emplace_back();
  return static_cast<int64_t>(scopes.size());
}
void mt_scope_exit(void) {
  if (scopes.empty()) {
    set_error("mt_scope_exit without a matching mt_scope_enter");
    return;
  }
  for (mt_tensor h : scopes.back()) {
    live_tensors.fetch_sub(1);
    delete h;
  }
  scopes.pop_back();
}
int64_t mt_scope_size(void) { return scopes.empty() ? 0 : static_cast<int64_t>(scopes.back().size()); }
void mt_keep(mt_tensor t) {
  if (t == nullptr || t->scope < 0) return;
  scopes[static_cast<size_t>(t->scope)].erase(t);
  t->scope -= 1;
  if (t->scope >= 0) scopes[static_cast<size_t>(t->scope)].insert(t);
}

// --- inspection --------------------------------------------------------------

int64_t mt_ndim(mt_tensor t) {
  return guard<int64_t>(0, [&] { return get(t).dim(); });
}
int64_t mt_size(mt_tensor t, int64_t dim) {
  return guard<int64_t>(0, [&] { return get(t).size(dim); });
}
int64_t mt_numel(mt_tensor t) {
  return guard<int64_t>(0, [&] { return static_cast<int64_t>(get(t).numel()); });
}
int64_t mt_dtype(mt_tensor t) {
  return guard<int64_t>(0, [&] { return dtype_code(get(t).scalar_type()); });
}
int64_t mt_device(mt_tensor t) {
  return guard<int64_t>(0, [&] { return device_code(get(t).device()); });
}
int64_t mt_requires_grad(mt_tensor t) {
  return guard<int64_t>(0, [&] { return get(t).requires_grad() ? 1 : 0; });
}
double mt_item_f64(mt_tensor t) {
  return guard<double>(0.0, [&] { return get(t).item<double>(); });
}
int64_t mt_item_i64(mt_tensor t) {
  return guard<int64_t>(0, [&] { return get(t).item<int64_t>(); });
}
int64_t mt_copy_f64(mt_tensor t, double *out, int64_t cap) { return copy_out<double>(t, out, cap, torch::kFloat64); }
int64_t mt_copy_i64(mt_tensor t, int64_t *out, int64_t cap) { return copy_out<int64_t>(t, out, cap, torch::kInt64); }
int64_t mt_to_string(mt_tensor t, char *out, int64_t cap) {
  return guard<int64_t>(0, [&] {
    std::ostringstream s;
    s << get(t);
    std::string text = s.str();
    if (out != nullptr && cap > 0) {
      std::memcpy(out, text.data(), std::min<size_t>(static_cast<size_t>(cap), text.size()));
    }
    return static_cast<int64_t>(text.size());
  });
}

// --- conversion --------------------------------------------------------------

mt_tensor mt_to_device(mt_tensor t, int64_t device) {
  return guard_tensor([&] { return get(t).to(device_of(device)); });
}
mt_tensor mt_to_dtype(mt_tensor t, int64_t dtype) {
  return guard_tensor([&] { return get(t).to(dtype_of(dtype)); });
}
MT_UNARY(mt_clone, x.clone())
MT_UNARY(mt_detach, x.detach())

// --- shape -------------------------------------------------------------------

mt_tensor mt_reshape(mt_tensor t, const int64_t *shape, int64_t ndim) {
  return guard_tensor([&] { return get(t).reshape(shape_of(shape, ndim)); });
}
mt_tensor mt_transpose(mt_tensor t, int64_t dim0, int64_t dim1) {
  return guard_tensor([&] { return get(t).transpose(dim0, dim1); });
}
mt_tensor mt_squeeze(mt_tensor t, int64_t dim) {
  return guard_tensor([&] { return get(t).squeeze(dim); });
}
mt_tensor mt_unsqueeze(mt_tensor t, int64_t dim) {
  return guard_tensor([&] { return get(t).unsqueeze(dim); });
}
mt_tensor mt_cat(const mt_tensor *ts, int64_t n, int64_t dim) {
  return guard_tensor([&] { return torch::cat(list_of(ts, n), dim); });
}
mt_tensor mt_stack(const mt_tensor *ts, int64_t n, int64_t dim) {
  return guard_tensor([&] { return torch::stack(list_of(ts, n), dim); });
}
mt_tensor mt_index_select(mt_tensor t, int64_t dim, mt_tensor index) {
  return guard_tensor([&] { return get(t).index_select(dim, get(index)); });
}
mt_tensor mt_narrow(mt_tensor t, int64_t dim, int64_t start, int64_t length) {
  return guard_tensor([&] { return get(t).narrow(dim, start, length); });
}

// --- arithmetic --------------------------------------------------------------

MT_BINARY(mt_add, x + y)
MT_BINARY(mt_sub, x - y)
MT_BINARY(mt_mul, x * y)
MT_BINARY(mt_div, x / y)
MT_BINARY(mt_matmul, torch::matmul(x, y))
mt_tensor mt_add_scalar(mt_tensor a, double s) {
  return guard_tensor([&] { return get(a) + s; });
}
mt_tensor mt_mul_scalar(mt_tensor a, double s) {
  return guard_tensor([&] { return get(a) * s; });
}
mt_tensor mt_pow_scalar(mt_tensor a, double exponent) {
  return guard_tensor([&] { return get(a).pow(exponent); });
}

// --- unary -------------------------------------------------------------------

MT_UNARY(mt_neg, -x)
MT_UNARY(mt_exp, x.exp())
MT_UNARY(mt_log, x.log())
MT_UNARY(mt_sqrt, x.sqrt())
MT_UNARY(mt_abs, x.abs())
MT_UNARY(mt_tanh, x.tanh())
MT_UNARY(mt_sigmoid, x.sigmoid())
MT_UNARY(mt_relu, x.relu())
MT_UNARY(mt_gelu, torch::gelu(x))

// --- reductions --------------------------------------------------------------

MT_UNARY(mt_sum, x.sum())
MT_UNARY(mt_mean, x.mean())
mt_tensor mt_sum_dim(mt_tensor a, int64_t dim, int64_t keepdim) {
  return guard_tensor([&] { return get(a).sum(dim, keepdim != 0); });
}
mt_tensor mt_mean_dim(mt_tensor a, int64_t dim, int64_t keepdim) {
  return guard_tensor([&] { return get(a).mean(dim, keepdim != 0); });
}
mt_tensor mt_argmax(mt_tensor a, int64_t dim, int64_t keepdim) {
  return guard_tensor([&] { return get(a).argmax(dim, keepdim != 0); });
}
mt_tensor mt_softmax(mt_tensor a, int64_t dim) {
  return guard_tensor([&] { return torch::softmax(get(a), dim); });
}
mt_tensor mt_log_softmax(mt_tensor a, int64_t dim) {
  return guard_tensor([&] { return torch::log_softmax(get(a), dim); });
}

// --- comparison --------------------------------------------------------------

MT_BINARY(mt_eq, x == y)
int64_t mt_allclose(mt_tensor a, mt_tensor b, double rtol, double atol) {
  return guard<int64_t>(0, [&] { return torch::allclose(get(a), get(b), rtol, atol) ? 1 : 0; });
}

// --- nn functional -----------------------------------------------------------

mt_tensor mt_linear(mt_tensor input, mt_tensor weight, mt_tensor bias) {
  return guard_tensor([&] {
    return torch::nn::functional::linear(get(input), get(weight), bias ? bias->t : torch::Tensor());
  });
}
mt_tensor mt_embedding(mt_tensor weight, mt_tensor indices) {
  return guard_tensor([&] { return torch::embedding(get(weight), get(indices)); });
}
mt_tensor mt_layer_norm(mt_tensor input, mt_tensor weight, mt_tensor bias, double eps) {
  return guard_tensor([&] {
    const torch::Tensor &w = get(weight);
    return torch::layer_norm(get(input), w.sizes(), w, get(bias), eps);
  });
}
mt_tensor mt_dropout(mt_tensor input, double p, int64_t training) {
  return guard_tensor([&] { return torch::dropout(get(input), p, training != 0); });
}
mt_tensor mt_cross_entropy(mt_tensor logits, mt_tensor target) {
  return guard_tensor([&] { return torch::nn::functional::cross_entropy(get(logits), get(target)); });
}
mt_tensor mt_mse_loss(mt_tensor input, mt_tensor target) {
  return guard_tensor([&] { return torch::mse_loss(get(input), get(target)); });
}

// --- autograd ----------------------------------------------------------------

void mt_set_requires_grad(mt_tensor t, int64_t requires_grad) {
  guard_void([&] {
    torch::Tensor x = get(t);
    x.set_requires_grad(requires_grad != 0);
  });
}
void mt_backward(mt_tensor loss) {
  guard_void([&] { get(loss).backward(); });
}
mt_tensor mt_grad(mt_tensor t) {
  return guard<mt_tensor>(nullptr, [&]() -> mt_tensor {
    torch::Tensor g = get(t).grad();
    return g.defined() ? wrap(g) : nullptr;
  });
}
void mt_zero_grad(mt_tensor t) {
  guard_void([&] {
    torch::Tensor g = get(t).grad();
    if (g.defined()) g.zero_();
  });
}
void mt_add_inplace(mt_tensor t, mt_tensor other, double alpha) {
  guard_void([&] {
    torch::NoGradGuard no_grad;
    torch::Tensor x = get(t);
    x.add_(get(other), alpha);
  });
}

// --- optimizers --------------------------------------------------------------

mt_optim mt_sgd(const mt_tensor *params, int64_t n, double lr, double momentum, double weight_decay) {
  return guard<mt_optim>(nullptr, [&] {
    auto opts = torch::optim::SGDOptions(lr).momentum(momentum).weight_decay(weight_decay);
    return new mt_optim_s{std::make_unique<torch::optim::SGD>(list_of(params, n), opts)};
  });
}
mt_optim mt_adamw(const mt_tensor *params, int64_t n, double lr, double beta1, double beta2, double eps,
                  double weight_decay) {
  return guard<mt_optim>(nullptr, [&] {
    auto opts = torch::optim::AdamWOptions(lr).betas({beta1, beta2}).eps(eps).weight_decay(weight_decay);
    return new mt_optim_s{std::make_unique<torch::optim::AdamW>(list_of(params, n), opts)};
  });
}
void mt_optim_step(mt_optim o) {
  guard_void([&] {
    if (o == nullptr) throw std::invalid_argument("null optimizer handle");
    o->o->step();
  });
}
void mt_optim_zero_grad(mt_optim o) {
  guard_void([&] {
    if (o == nullptr) throw std::invalid_argument("null optimizer handle");
    o->o->zero_grad();
  });
}
void mt_optim_free(mt_optim o) { delete o; }

} // extern "C"
