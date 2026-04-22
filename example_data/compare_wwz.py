import pandas as pd
import numpy as np
import sys
import time
import os

# Add paths to sys.path
sys.path.append(os.path.abspath('libwwz'))
sys.path.append(os.path.abspath('wwz'))

# Import libwwz (assuming it's a package or we can import from the file)
# The structure is libwwz/libwwz/wwz.py
# So we need to add libwwz/ as path and then 'import libwwz.wwz as libwwz_wwz'
import libwwz.wwz as libwwz_core

# Import wwz (Kiehlmann's version)
# The structure is wwz/wwz.py
import wwz as sk_wwz

# Import JAX version
from wwz_jax import wwz_jax_core
import jax.numpy as jnp

def main():
    # 1. Load data
    df = pd.read_parquet('dp1_n4bands.parquet')
    obj_id = df['object_id'].iloc[0]
    print(f"Using object ID: {obj_id}")
    
    # Extract data for one filter (e.g., 'z')
    mask = (df['object_id'] == obj_id) & (df['filter'] == 'z')
    data = df[mask].sort_values('mjd')
    
    times = data['mjd'].values
    flux = data['psMag'].values
    flux_err = data['psMagErr'].values
    
    print(f"Number of data points: {len(times)}")
    
    # 2. Configure WWZ
    ntau = 100
    ngrid = 400
    # minfq and maxfq from run_all_ids_csv_1.py: MinFq=12.0, MaxFq=2.0
    # But wait, in the script: MinFq=12.0, MaxFq=2.0. 
    # Usually frequency is 1/period.
    # Let's check typical values.
    # libwwz.wwt expects freq_params = [freq_low, freq_high, freq_step]
    fmin = 1.0 / 100.0 # period of 100 days
    fmax = 1.0 / 2.0   # period of 2 days
    fstep = (fmax - fmin) / ngrid
    
    # libwwz parameters
    # wwt(timestamps, magnitudes, time_divisions, freq_params, decay_constant, method='linear', parallel=True)
    decay_constant = 1.0 / (8.0 * np.pi**2)
    
    # Matching frequencies
    freqs = np.linspace(fmin, fmax, ngrid)
    fstep = freqs[1] - freqs[0]
    
    print(f"Running libwwz...")
    start_time = time.time()
    # Note: libwwz.wwz.make_freq uses np.arange(freq_low, freq_high + freq_steps, freq_steps)
    # To get exactly 100 frequencies, we need to pass [fmin, fmax, fstep] carefully or just use ngrid
    # Wait, libwwz.wwt calls make_freq(freq_params[0], freq_params[1], freq_params[2])
    # Let's adjust libwwz freq_params to match freqs.
    lib_results = libwwz_core.wwt(times, flux, ntau, [fmin, fmax - fstep*0.5, fstep], decay_constant, method='linear', parallel=False)
    lib_duration = time.time() - start_time
    print(f"libwwz duration: {lib_duration:.4f}s")
    
    # lib_results has shape (6, ntau, nfreq)
    # index 2 is WWZ power
    
    # 3. Running Sebastian Kiehlmann's version (wwz)
    print(f"Running sk_wwz...")
    start_time = time.time()
    wwz_obj = sk_wwz.WWZ(times, flux, flux_err)
    wwz_obj.set_freq(freqs)
    # get_tau(self, t_min=None, t_max=None, n_div=8, n_bins=None, dtau=None)
    taus = np.linspace(times[0], times[-1], ntau)
    wwz_obj.set_tau(taus)
    wwz_obj.transform(c=decay_constant)
    sk_duration = time.time() - start_time
    print(f"sk_wwz duration: {sk_duration:.4f}s")

    # 4. Running JAX version
    print(f"Running wwz_jax...")
    # Warmup JIT
    _ = wwz_jax_core(jnp.array(times), jnp.array(flux), jnp.array(taus), jnp.array(freqs), decay_constant)
    
    start_time = time.time()
    jax_wwz, jax_wwa, jax_neff = wwz_jax_core(jnp.array(times), jnp.array(flux), jnp.array(taus), jnp.array(freqs), decay_constant)
    # Trigger execution (JAX is async)
    jax_wwz.block_until_ready()
    jax_duration = time.time() - start_time
    print(f"wwz_jax duration: {jax_duration:.4f}s")
    
    # Compare outputs
    print(f"libwwz WWZ shape: {lib_results[2].shape}")
    print(f"sk_wwz WWZ shape: {wwz_obj.wwz.shape}")
    print(f"jax_wwz WWZ shape: {jax_wwz.shape}")
    
    # Compare values (sk vs jax)
    diff_sk_jax = np.abs(wwz_obj.wwz - np.array(jax_wwz))
    print(f"Max absolute difference (sk vs jax): {np.max(diff_sk_jax)}")
    print(f"Mean absolute difference (sk vs jax): {np.mean(diff_sk_jax)}")
    
    # Compare values (lib vs jax)
    diff_lib_jax = np.abs(lib_results[2] - np.array(jax_wwz))
    print(f"Max absolute difference WWZ (lib vs jax): {np.max(diff_lib_jax)}")
    print(f"Mean absolute difference WWZ (lib vs jax): {np.mean(diff_lib_jax)}")

    # Compare WWA (Amplitude)
    diff_wwa_lib_jax = np.abs(lib_results[3] - np.array(jax_wwa))
    print(f"Max absolute difference WWA (lib vs jax): {np.max(diff_wwa_lib_jax)}")

    # Compare Neff (Effective Number)
    diff_neff_lib_jax = np.abs(lib_results[5] - np.array(jax_neff))
    print(f"Max absolute difference Neff (lib vs jax): {np.max(diff_neff_lib_jax)}")
    
    # Correlation check (do they move together?)
    corr = np.corrcoef(lib_results[2].flatten(), np.array(jax_wwz).flatten())[0, 1]
    print(f"Correlation between libwwz and jax_wwz: {corr:.6f}")

if __name__ == "__main__":
    main()
