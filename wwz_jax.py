import jax
import jax.numpy as jnp
import numpy as np
from jax import config, lax
from functools import partial

config.update("jax_enable_x64", True)

@partial(jax.jit, static_argnames=['use_libwwz_heuristics'])
def wwz_jax_core(times, flux, taus, freqs, c, use_libwwz_heuristics=True):
    """
    JAX implementation of WWZ, optionally replicating libwwz's sequential quirks.
    """
    
    def tau_loop(dtau):
        # Initial state for the frequency scan
        # libwwz initializes dvarw = 0 once per tau and it leaks between frequencies
        init_carry = (0.0,) # (dvarw,)
        
        def freq_scan(carry, dfreq):
            dvarw, = carry
            domega = 2.0 * jnp.pi * dfreq
            
            # Data point calculations
            dt = times - dtau
            dz = domega * dt
            w = jnp.exp(-c * dz**2)
            
            if use_libwwz_heuristics:
                # Replicate libwwz weight thresholding
                w = jnp.where(w > 1e-9, w, 0.0)
            
            # Sums
            s00 = jnp.sum(w)
            w2_sum = jnp.sum(w**2)
            
            # dvarw accumulation (THE LEAK)
            # libwwz adds current weighted sum of squares to the accumulated dvarw
            current_wxx = jnp.sum(w * (flux**2))
            dvarw = dvarw + current_wxx
            
            # Trial functions
            cos_dz = jnp.cos(dz)
            sin_dz = jnp.sin(dz)
            
            # dmat elements
            s01 = jnp.sum(w * cos_dz)
            s02 = jnp.sum(w * sin_dz)
            s11 = jnp.sum(w * cos_dz**2)
            s12 = jnp.sum(w * cos_dz * sin_dz)
            s22 = jnp.sum(w * sin_dz**2)
            
            # dvec elements
            p1 = jnp.sum(w * flux)
            p2 = jnp.sum(w * flux * cos_dz)
            p3 = jnp.sum(w * flux * sin_dz)
            
            # Calculate Neff
            dneff = jnp.where(w2_sum > 0, (s00**2) / w2_sum, 0.0)
            
            # Matrix and Result logic
            def compute_results():
                # Normalize like libwwz
                # Note: libwwz modifies dvarw in place if dneff > 3
                
                # dvarw normalization
                _dvarw = jnp.where(s00 > 0.005, dvarw / s00, 0.0)
                
                # avew and varw
                davew = p1 / s00
                _dvarw = _dvarw - (davew**2)
                _dvarw = jnp.maximum(_dvarw, 1e-12)
                
                # S Matrix
                S = jnp.array([
                    [1.0, s01/s00, s02/s00],
                    [s01/s00, s11/s00, s12/s00],
                    [s02/s00, s12/s00, s22/s00]
                ])
                P = jnp.array([p1/s00, p2/s00, p3/s00])
                
                # Inversion (libwwz logic)
                # libwwz uses pinv if determinant is 0
                det = jnp.linalg.det(S)
                S_inv = jnp.where(jnp.abs(det) < 1e-15, jnp.linalg.pinv(S), jnp.linalg.inv(S))
                
                dcoef = S_inv @ P
                y1, y2, y3 = dcoef[0], dcoef[1], dcoef[2]
                
                # Power
                dpower = jnp.dot(dcoef, P) - (davew**2)
                dpowz = ((dneff - 3.0) * dpower) / (2.0 * (_dvarw - dpower))
                damp = jnp.sqrt(y2**2 + y3**2)
                
                # Update carry (the leaking dvarw)
                # In libwwz, dvarw is updated to the normalized value for the NEXT frequency!
                return _dvarw, dpowz, damp
            
            def skip_results():
                return dvarw, 0.0, 0.0
            
            # Execute logic if dneff > 3
            next_dvarw, dpowz, damp = lax.cond(dneff > 3.0, compute_results, skip_results)
            
            if not use_libwwz_heuristics:
                # If NOT mimicking bugs, reset dvarw for each frequency
                next_dvarw = 0.0
            
            # Cleanup small values like libwwz
            if use_libwwz_heuristics:
                dpowz = jnp.where(dpowz > 1e-9, dpowz, 0.0)
                damp = jnp.where(damp > 1e-9, damp, 0.0)
            
            return (next_dvarw,), (dpowz, damp, dneff)
            
        _, results = lax.scan(freq_scan, init_carry, freqs)
        return results
    
    # Vectorize over taus
    wwz, wwa, neff = jax.vmap(tau_loop)(taus)
    return wwz, wwa, neff
