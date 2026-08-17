# Test layout

`unit/` is the required, deterministic standard-library suite. It checks the
public compiler surface, instruction selection, binary output, error behavior,
and source lowering without an external WebAssembly engine.

`scripts/` contains executable integration probes written as package users
would write them. `test_scripts.py` smoke-tests their compile-only path. The
scripts use Wasmtime only when it is separately installed; pass
`--require-engine` to make a missing engine an error.

Run the full required suite from the repository root:

```console
./test-local.sh
```

Run the probes directly:

```console
python tests/scripts/wasm_examples.py --require-engine
python tests/scripts/numeric_runtime.py --require-engine
python tests/scripts/stdlib_runtime.py --require-engine
python tests/scripts/array_runtime.py --require-engine
python tests/scripts/interop_runtime.py --require-engine
python tests/scripts/sections_runtime.py --require-engine
python tests/scripts/allocator_runtime.py --require-engine
python tests/scripts/benchmark.py --require-engine
python tests/scripts/limitations.py --require-engine
```

The scripts are not packaged in the PyPI distribution.
