"""Disk caching for the (expensive) quantum kernel / Gram matrices.

Each matrix is saved as a self-identifying ``.npz`` file whose name encodes the
dataset, kernel/feature-map, feature-map parameters, qubit count and oversampling
scheme. A short content hash of the input data is appended so that changing the
data (new train/test split, added noise, different preprocessing) produces a new
file instead of silently reloading a stale matrix.

Re-running with the same data + params loads the matrix from disk instead of
recomputing it, so re-tuning the SVM ``C`` or restyling the ROC curve is near-free.
"""

import hashlib
import os

import numpy as np


def _params_to_str(params):
    """Filename-safe, deterministic string for the feature-map params."""
    if isinstance(params, str):              # angle-emb: "rot"
        return params
    # ising: [t, T, N_LAYERS] -> "t1.5708_T20_L3"
    t, T, L = params
    return f"t{float(t):.4g}_T{int(T)}_L{int(L)}"


def _data_hash(A, B):
    """Short content hash so the cache invalidates if the data changes."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(A).tobytes())
    h.update(np.ascontiguousarray(B).tobytes())
    return h.hexdigest()[:8]


def cache_filename(role, dataset, model_name, params, n_qubits, oversampling, A, B):
    """Build the deterministic cache filename for a given matrix."""
    return (f"{dataset}_{model_name}_OS{oversampling}_{n_qubits}q"
            f"_params_{_params_to_str(params)}_{role}_{_data_hash(A, B)}.npz")


def cached_compute_kernel(compute_fn, A, B, *, role, dataset, model_name,
                          params, n_qubits, oversampling, cache_dir, n_jobs):
    """Load the Gram matrix from disk if a matching file exists, else compute it.

    Args:
        compute_fn: callable ``f(A, B, params, n_jobs=...) -> np.ndarray``.
        A, B: the two datasets whose pairwise kernel is computed.
        role: "train" (A == B, symmetric) or "test" (rectangular), for the name.
        dataset, model_name, params, n_qubits, oversampling: identify the matrix.
        cache_dir: directory to store the ``.npz`` files in.
        n_jobs: forwarded to ``compute_fn``.

    Returns:
        The Gram matrix ``K``.
    """
    os.makedirs(cache_dir, exist_ok=True)
    fname = cache_filename(role, dataset, model_name, params,
                           n_qubits, oversampling, A, B)
    path = os.path.join(cache_dir, fname)

    if os.path.exists(path):
        print(f"[kernel-cache] HIT  {fname}")
        return np.load(path)["K"]

    print(f"[kernel-cache] MISS {fname} -> computing")
    K = compute_fn(A, B, params, n_jobs=n_jobs)
    np.savez_compressed(path, K=K, dataset=dataset, model_name=model_name,
                        params=np.array(params, dtype=object), n_qubits=n_qubits,
                        oversampling=oversampling, role=role)
    return K
