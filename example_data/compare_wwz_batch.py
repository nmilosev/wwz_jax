import pandas as pd
import numpy as np
import sys
import time
import os

# Add paths to sys.path
sys.path.append(os.path.abspath('libwwz'))
sys.path.append(os.path.abspath('wwz'))

import libwwz.wwz as libwwz_core
from wwz_jax import wwz_jax_core
import jax.numpy as jnp

def compare_for_object(obj_id, df, ntau, ngrid, fmin, fmax, decay_constant):
    # Extract data for one filter (e.g., 'z')
    mask = (df['object_id'] == obj_id) & (df['filter'] == 'z')
    data = df[mask].sort_values('mjd')
    
    times = data['mjd'].values
    flux = data['psMag'].values
    
    if len(times) < 10:
        return None

    # libwwz frequencies
    freqs = np.linspace(fmin, fmax, ngrid)
    fstep = freqs[1] - freqs[0]
    
    # Run libwwz
    start_time = time.time()
    lib_results = libwwz_core.wwt(times, flux, ntau, [fmin, fmax - fstep*0.5, fstep], decay_constant, method='linear', parallel=False)
    lib_duration = time.time() - start_time
    
    # Run JAX
    taus = np.linspace(times[0], times[-1], ntau)
    start_time = time.time()
    jax_wwz, jax_wwa, jax_neff = wwz_jax_core(jnp.array(times), jnp.array(flux), jnp.array(taus), jnp.array(freqs), decay_constant)
    jax_wwz.block_until_ready()
    jax_duration = time.time() - start_time
    
    # Metrics
    lib_wwz_flat = lib_results[2].flatten()
    jax_wwz_flat = np.array(jax_wwz).flatten()
    
    corr = np.corrcoef(lib_wwz_flat, jax_wwz_flat)[0, 1]
    max_diff = np.max(np.abs(lib_wwz_flat - jax_wwz_flat))
    
    return {
        "obj_id": obj_id,
        "n_points": len(times),
        "lib_dur": lib_duration,
        "jax_dur": jax_duration,
        "corr": corr,
        "max_diff": max_diff
    }

def main():
    df = pd.read_parquet('dp1_n4bands.parquet')
    all_ids = df['object_id'].unique()
    
    # Test on ALL objects
    test_ids = all_ids
    
    ntau = 50
    ngrid = 100
    fmin = 1.0 / 100.0
    fmax = 1.0 / 2.0
    decay_constant = 1.0 / (8.0 * np.pi**2)
    
    results = []
    print(f"Starting comparison for {len(test_ids)} objects...")
    print(f"{'Object ID':<20} | {'N':<5} | {'Corr':<10} | {'Max Diff':<10} | {'Speedup':<8}")
    print("-" * 65)
    
    # Warmup JAX
    dummy_t = jnp.zeros(10)
    dummy_f = jnp.zeros(10)
    dummy_taus = jnp.zeros(10)
    dummy_freqs = jnp.zeros(10)
    _ = wwz_jax_core(dummy_t, dummy_f, dummy_taus, dummy_freqs, decay_constant)

    for i, obj_id in enumerate(test_ids):
        res = compare_for_object(obj_id, df, ntau, ngrid, fmin, fmax, decay_constant)
        if res:
            speedup = res['lib_dur'] / res['jax_dur']
            # Only print every 10 objects to reduce noise
            if i % 10 == 0:
                print(f"{res['obj_id']:<20} | {res['n_points']:<5} | {res['corr']:<10.6f} | {res['max_diff']:<10.4f} | {speedup:<8.1f}x")
            results.append(res)
            
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('comparison_results.csv', index=False)
        
        avg_corr = results_df['corr'].mean()
        min_corr = results_df['corr'].min()
        avg_speedup = (results_df['lib_dur'] / results_df['jax_dur']).mean()
        
        print("-" * 65)
        print(f"Average Correlation: {avg_corr:.6f}")
        print(f"Minimum Correlation: {min_corr:.6f}")
        print(f"Average Speedup:     {avg_speedup:.1f}x")
        print(f"Results saved to comparison_results.csv")

if __name__ == "__main__":
    main()
