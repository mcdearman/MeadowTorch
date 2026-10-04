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

- A Meadow with `Std.Ffi`. It is newer than the `0.1.0-alpha` release, so for
  now that means a build of Meadow from source.
- libtorch 2.x: an unpacked libtorch download, or a Python install of `torch`.
- A C++20 compiler.
- macOS or Linux. Tested on aarch64 macOS only.

## Setup

The Meadow code calls a shared library, `libmeadow_torch`, which you build once
from a checkout of this repository:

```sh
git clone https://github.com/mcdearman/MeadowTorch && cd MeadowTorch
shim/build.sh install       # builds, then copies to ~/.local/lib
```

`build.sh` finds libtorch through `LIBTORCH` (a directory with `include/` and
`lib/`), or else uses the `torch` that `$PYTHON` (default `python3`) imports.
`PREFIX` changes where `install` copies to.

Then depend on the package:

```toml
[dependencies]
Torch = { git = "https://github.com/mcdearman/MeadowTorch", version = "0.1.0" }

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
`embedding`, `layerNorm`, `dropout`, `crossEntropy`, `mseLoss`, autograd, and
the SGD and AdamW optimizers. `src/Lib.mw` is the reference.

Not yet: loading pretrained models, operator overloading on tensors, and
anything else in libtorch that the shim does not expose.

## Developing

```sh
shim/build.sh test          # build the shim and run its C smoke test
meadow test                 # the Meadow tests, against shim/build/
meadow run examples/Xor     # a two-layer network learns XOR
```
