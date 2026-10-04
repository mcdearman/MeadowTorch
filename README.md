# MeadowTorch

PyTorch for [Meadow](https://github.com/mcdearman/meadow): tensors, autograd,
optimizers and the GPU, through a small C shim over libtorch.

```meadow
use Torch as T
use Std.Exn (toResult)

fun main () =
  println (toResult (\() -> T.run (\() -> T.scope (\() ->
    let x = T.requireGrad (T.fromFloats [3] [1.0, 2.0, 3.0]) in
    let _ = T.backward (T.sum (T.mul x x)) in
    match T.grad x with
    | Just g -> T.toFloats g
    | None -> []))))
```

```
Ok([2.0, 4.0, 6.0])
```

## Requirements

- A Meadow whose `Std.Ffi` names its arguments `Arg.Int`, `Arg.Float`,
  `Arg.String` and `Arg.Ptr` (commit 0fcd21d or later). That is newer than the
  `0.1.0-alpha` release, so for now it means a build of Meadow from source.
  MeadowTorch 0.3.0 is the last version for the older `Arg.IntArg` names.
- libtorch 2.x. No Python is needed.
- A C++20 compiler.
- macOS or Linux. Tested on aarch64 macOS (CPU and MPS) and on x86-64 Linux
  with CUDA, there against the libtorch inside a Python install of PyTorch 2.11.

## Setup

The Meadow code calls a shared library, `libmeadow_torch`, which you build once
from a checkout of this repository:

```sh
git clone https://github.com/mcdearman/MeadowTorch && cd MeadowTorch
shim/build.sh install       # builds, then copies to ~/.local/lib
```

`build.sh` finds libtorch through `LIBTORCH` (a directory with `include/` and
`lib/`), or else in `~/.local/libtorch`. On aarch64 macOS it downloads libtorch
there if it is missing; on Linux, download the build for your CUDA version from
pytorch.org and set `LIBTORCH`. `PREFIX` replaces `~/.local` for both.

Then depend on the package:

```toml
[dependencies]
Torch = { git = "https://github.com/mcdearman/MeadowTorch", version = "0.4.1" }

[profile.debug]
threads = 1

[profile.release]
threads = 1
```

At run time `T.run` opens the first of: `$MEADOW_TORCH_LIB` if set; otherwise
`shim/build/` under the working directory, then `~/.local/lib`.

## Things to know

- **Tensors are not garbage collected.** A tensor is a handle to C memory. Do
  the work inside `T.scope`: every tensor made in it is freed when it returns,
  except those passed to `T.keep`, which pass to the enclosing scope. A tensor
  used after its scope has exited, or freed twice, crashes in C.
- **One OS thread.** libtorch's grad mode and the shim's scopes are per OS
  thread, and a Meadow green thread can move between workers once a program
  spawns. Set `threads = 1` as above, and do not use Torch from two green
  threads at once. libtorch still uses its own threads for the math.
- **Errors raise.** A call libtorch rejects raises its message through
  `Std.Exn`; `toResult` turns it into an `Err`.
- **Calls are unchecked.** `Std.Ffi` does not check a call against the C
  signature, so the shim and `src/Lib.mw` must change together.

## What is covered

Creation (`zeros`, `ones`, `full`, `randn`, `rand`, `arange`, `scalar`,
`fromFloats`, `fromInts`), inspection and reading back, device and dtype
conversion, shape operations, broadcasting arithmetic, reductions, `linear`,
`embedding`, `layerNorm`, `dropout`, `attention`, `crossEntropy` (with ignored targets), `mseLoss`,
autograd, the SGD and AdamW optimizers, and reading tensors from a
`.safetensors` file (`openWeights`, `weight`). `src/Lib.mw` is the reference.

Not yet: saving weights, operator overloading on tensors, and anything else in
libtorch that the shim does not expose.

## Developing

```sh
shim/build.sh test          # build the shim and run its C smoke test
meadow test                 # the Meadow tests, against shim/build/
meadow run examples/Xor     # a two-layer network learns XOR
```
