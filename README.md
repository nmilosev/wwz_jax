# wwz-jax

A high-performance JAX implementation of the Weighted Wavelet Z-transform (WWZ), designed for efficient time-series analysis in astronomy and beyond.

This implementation provides a significant speedup over traditional CPU-based versions (`libwwz`) by leveraging JAX's JIT compilation and vectorization capabilities. It also includes a compatibility mode to exactly replicate the specific heuristics and historical "leaks" found in the original `libwwz` library.

## Features

- **JAX Optimized**: Fully JIT-compilable and vectorized using `jax.vmap` for execution on CPUs, GPUs, and TPUs.
- **Compatibility Mode**: Optionally replicate `libwwz`'s specific weight thresholding and sequential `dvarw` accumulation (heuristics that are often required to match legacy results).
- **Double Precision**: Explicitly enables `jax_enable_x64` for numerical stability and matching standard astronomical toolkits.
- **Batched Execution**: Easily scalable to large grids of time-delays ($\tau$) and frequencies.

## Installation

Ensure you have JAX installed. Follow the [official JAX installation guide](https://github.com/google/jax#installation) for your specific hardware (CPU/GPU/TPU).

```bash
pip install jax numpy
```

Then simply include `wwz_jax.py` in your project.

## libwwz Compatibility

The `libwwz` library contains several implementation-specific details:
1. **Weight Thresholding**: Weights below $10^{-9}$ are truncated to zero.
2. **dvarw Leak**: The weighted variance accumulation (`dvarw`) leaks between frequency iterations within the same $\tau$ bin.
3. **Normalization Heuristics**: Specific handling for low `Neff` and small determinants.

`wwz-jax` implements these quirks behind the `use_libwwz_heuristics` flag, allowing for 1:1 numerical validation against legacy code while offering a "clean" path for new studies.

## Comparison

See `example_data/compare_wwz.py` for a benchmarking and validation script against `libwwz` and Sebastian Kiehlmann's version. In testing, `wwz-jax` typically achieves orders of magnitude speedup while maintaining high correlation ($>0.999$) with legacy outputs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgement

This project was built as part of Deliverable for UNDP project "Harnessing AI for the common good – facilitating an AI-friendly ecosystem in Serbia"
