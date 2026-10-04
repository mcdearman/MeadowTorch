#include "meadow_torch.h"

#include <torch/torch.h>

#include <atomic>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
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

struct weight_entry {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  uint64_t begin = 0;
  uint64_t end = 0;
};

struct mt_weights_s {
  std::string path;
  // Where tensor data starts in the file: after the length and the header.
  uint64_t data_start = 0;
  std::vector<weight_entry> entries;
  std::unordered_map<std::string, size_t> index;
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

// Just enough JSON for a safetensors header: an object of
// name -> {"dtype": string, "shape": [int], "data_offsets": [int, int]}.
struct HeaderParser {
  const std::string &s;
  size_t i = 0;

  [[noreturn]] void fail(const char *what) {
    throw std::invalid_argument(std::string("bad safetensors header: ") + what + " at byte " + std::to_string(i));
  }
  void ws() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) i++;
  }
  char peek() {
    ws();
    if (i >= s.size()) fail("unexpected end");
    return s[i];
  }
  void expect(char c) {
    if (peek() != c) fail("unexpected character");
    i++;
  }
  bool accept(char c) {
    if (peek() != c) return false;
    i++;
    return true;
  }
  std::string string() {
    expect('"');
    std::string out;
    while (true) {
      if (i >= s.size()) fail("unterminated string");
      char c = s[i++];
      if (c == '"') return out;
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (i >= s.size()) fail("unterminated escape");
      char e = s[i++];
      switch (e) {
      case 'n': out.push_back('\n'); break;
      case 't': out.push_back('\t'); break;
      case 'r': out.push_back('\r'); break;
      case 'b': out.push_back('\b'); break;
      case 'f': out.push_back('\f'); break;
      case 'u': fail("\\u escapes in names are not supported");
      default: out.push_back(e);
      }
    }
  }
  int64_t integer() {
    ws();
    size_t start = i;
    if (i < s.size() && s[i] == '-') i++;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') i++;
    if (start == i) fail("expected a number");
    return std::stoll(s.substr(start, i - start));
  }
  std::vector<int64_t> integers() {
    std::vector<int64_t> out;
    expect('[');
    if (accept(']')) return out;
    do out.push_back(integer());
    while (accept(','));
    expect(']');
    return out;
  }
  void skip_value() {
    char c = peek();
    if (c == '"') {
      string();
    } else if (c == '{') {
      i++;
      if (accept('}')) return;
      do {
        string();
        expect(':');
        skip_value();
      } while (accept(','));
      expect('}');
    } else if (c == '[') {
      i++;
      if (accept(']')) return;
      do skip_value();
      while (accept(','));
      expect(']');
    } else {
      while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']') i++;
    }
  }
  weight_entry entry(std::string name) {
    weight_entry e;
    e.name = std::move(name);
    expect('{');
    do {
      std::string key = string();
      expect(':');
      if (key == "dtype") {
        e.dtype = string();
      } else if (key == "shape") {
        e.shape = integers();
      } else if (key == "data_offsets") {
        auto offsets = integers();
        if (offsets.size() != 2 || offsets[0] < 0 || offsets[1] < offsets[0]) fail("bad data_offsets");
        e.begin = static_cast<uint64_t>(offsets[0]);
        e.end = static_cast<uint64_t>(offsets[1]);
      } else {
        skip_value();
      }
    } while (accept(','));
    expect('}');
    return e;
  }
};

torch::Dtype stored_dtype(const std::string &name) {
  static const std::unordered_map<std::string, torch::Dtype> known = {
      {"F16", torch::kFloat16}, {"BF16", torch::kBFloat16}, {"F32", torch::kFloat32}, {"F64", torch::kFloat64},
      {"I8", torch::kInt8},     {"I16", torch::kInt16},     {"I32", torch::kInt32},   {"I64", torch::kInt64},
      {"U8", torch::kUInt8},    {"BOOL", torch::kBool},
  };
  auto it = known.find(name);
  if (it == known.end()) throw std::invalid_argument("unsupported safetensors dtype " + name);
  return it->second;
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
MT_UNARY(mt_sin, x.sin())
MT_UNARY(mt_cos, x.cos())

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

mt_tensor mt_attention(mt_tensor query, mt_tensor key, mt_tensor value, int64_t causal) {
  return guard_tensor([&] {
    return torch::scaled_dot_product_attention(get(query), get(key), get(value), {}, 0.0, causal != 0);
  });
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

// --- weight files ------------------------------------------------------------

mt_weights mt_weights_open(const char *path) {
  return guard<mt_weights>(nullptr, [&] {
    if (path == nullptr) throw std::invalid_argument("null path");
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::invalid_argument(std::string("cannot open ") + path);
    unsigned char raw[8];
    file.read(reinterpret_cast<char *>(raw), 8);
    if (!file) throw std::invalid_argument(std::string(path) + " is too short to be a safetensors file");
    uint64_t header_len = 0;
    for (int b = 7; b >= 0; b--) header_len = (header_len << 8) | raw[b];
    if (header_len > (uint64_t{1} << 30)) throw std::invalid_argument(std::string(path) + " is not a safetensors file");
    std::string header(static_cast<size_t>(header_len), '\0');
    file.read(header.data(), static_cast<std::streamsize>(header_len));
    if (!file) throw std::invalid_argument(std::string(path) + " ends inside its header");

    auto w = std::make_unique<mt_weights_s>();
    w->path = path;
    w->data_start = 8 + header_len;
    HeaderParser p{header};
    p.expect('{');
    if (!p.accept('}')) {
      do {
        std::string name = p.string();
        p.expect(':');
        if (name == "__metadata__") {
          p.skip_value();
        } else {
          w->index[name] = w->entries.size();
          w->entries.push_back(p.entry(name));
        }
      } while (p.accept(','));
      p.expect('}');
    }
    return w.release();
  });
}
int64_t mt_weights_count(mt_weights w) { return w ? static_cast<int64_t>(w->entries.size()) : 0; }
const char *mt_weights_name(mt_weights w, int64_t i) {
  if (w == nullptr || i < 0 || i >= static_cast<int64_t>(w->entries.size())) return nullptr;
  return w->entries[static_cast<size_t>(i)].name.c_str();
}
mt_tensor mt_weights_get(mt_weights w, const char *name) {
  return guard_tensor([&] {
    if (w == nullptr || name == nullptr) throw std::invalid_argument("null weights handle or name");
    auto found = w->index.find(name);
    if (found == w->index.end()) throw std::invalid_argument(w->path + " has no tensor named " + name);
    const weight_entry &e = w->entries[found->second];
    torch::Tensor t = torch::empty(e.shape, stored_dtype(e.dtype));
    uint64_t bytes = e.end - e.begin;
    if (bytes != t.nbytes()) throw std::invalid_argument(std::string(name) + ": the data does not fit its shape");
    std::ifstream file(w->path, std::ios::binary);
    file.seekg(static_cast<std::streamoff>(w->data_start + e.begin));
    file.read(static_cast<char *>(t.data_ptr()), static_cast<std::streamsize>(bytes));
    if (!file) throw std::invalid_argument(w->path + " ends inside " + name);
    if (t.is_floating_point()) return t.scalar_type() == torch::kFloat64 ? t : t.to(torch::kFloat32);
    return t.scalar_type() == torch::kBool ? t : t.to(torch::kInt64);
  });
}
void mt_weights_free(mt_weights w) { delete w; }

} // extern "C"
