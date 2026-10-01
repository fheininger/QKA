"""Fast exact quantum-kernel evaluation and kernel-target-alignment search.

The compute-uncompute fidelity kernel used throughout this project,

    K(a, b) = |<0| U(b)^dag U(a) |0>|^2,   U(x) = H_evo(x) . SEL,

is the squared overlap of two *feature states* ``|phi(x)> = U(x)|0>``. Building
the Gram matrix therefore does NOT require n(n+1)/2 circuit simulations: it needs
n state preparations and one matrix product,

    Phi = feature_states(X, params)          # (n, 2**n_qubits)
    K   = |Phi.conj() @ Phi.T|^2

which is ~560x faster than the per-entry joblib path it replaces (measured: one
n=2000 Gram, 1040 s across 30 cores -> 1.86 s single process) and is *exact*
rather than shot-sampled, so the resulting Gram is PSD by construction.

For the alignment search there is a second identity. Reparametrising by the
per-Trotter-step angle ``tau = t/T``, the state after k Trotter steps at fixed
tau IS the state for ``(t = k*tau, T = k)``. One pass with ``qml.Snapshot``
therefore yields the entire T family for the price of a single state prep.

Alignment runs on the exact statevector: at shots=3000 the shot noise on the KTA
value (~6e-4) is amplified by the old finite-difference step (h=1e-3) into a
spurious gradient of magnitude ~0.6, i.e. the old search was a random walk. Pass
``emulate_shots=`` to reproduce shot-sampled matrices where that is wanted.

Standalone on purpose: do NOT import QSVM_template or pdac (they have no
__main__ guard and run the whole training pipeline on import). Everything those
scripts keep as a module global is passed in as a keyword argument here.

The Ising couplings live on nearest-neighbour qubit pairs, so the number of
features the map can load is the number of such pairs. A linear CHAIN of
n_qubits qubits has only n_qubits-1 of them, which silently dropped the last
feature whenever n_feat == n_qubits (7 PCA components on 7 qubits -> the 7th
never entered the circuit). ``TOPOLOGY = "ring"`` closes the chain with the pair
(n_qubits-1, 0) so all n_qubits features are loaded. See ``coupling_pairs``.

Verified against the production ``kernel_circ`` by ``verify_fast_kernel.py``.
"""

import json
import os
import time
from contextlib import nullcontext
from functools import lru_cache

import numpy as np
import pennylane as qml
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from threadpoolctl import threadpool_limits
except ImportError:                                  # pragma: no cover
    threadpool_limits = None

SEED = 42
DEFAULT_CHUNK = 512

# Qubit topology the data features are loaded onto; see ``coupling_pairs``.
# "ring" is the default because "chain" wastes one feature at n_feat == n_qubits.
# CHANGING THIS CHANGES THE KERNEL: Grams, KTA landscapes and every downstream
# AUC produced under "chain" are not comparable to "ring" ones. Kept switchable
# so earlier results stay reproducible -- set ``kernel_align.TOPOLOGY = "chain"``
# (or pass ``topology="chain"``) to recover them.
TOPOLOGY = "ring"

# Feature map used by the fast kernel path; see ``FEATURE_MAPS`` below.
# "ising" is the original SEL + Trotterised Heisenberg map behind every result
# to date. CHANGING THIS CHANGES THE KERNEL: Grams, KTA landscapes and every
# downstream AUC produced under "ising" are not comparable to "reup" ones. Kept
# switchable so earlier results stay reproducible -- the default is and stays
# "ising"; set ``kernel_align.FEATURE_MAP = "reup"`` (or pass
# ``feature_map="reup"``) to opt into the re-uploading map, and flip it back to
# "ising" to recover the original kernel bit-exactly.
FEATURE_MAP = "ising"

# Coarse geometric ladder used when snapshotting every T would blow the memory
# budget. Dense at small T where the Trotter error actually varies.
T_LADDER = (1, 2, 3, 4, 6, 8, 12, 16, 20, 28, 40)


# --------------------------------------------------------------------------
# devices and circuit pieces
# --------------------------------------------------------------------------
def blas_limit(n_threads):
    """Temporarily raise the BLAS thread count for the dense O(n^2)/O(n^3) work.

    Both scripts pin OMP/MKL/OPENBLAS to 1 thread at import, which is correct for
    the 30-process joblib pool but leaves the Gram matmul and the KTA centering
    single-threaded -- and those now dominate the alignment cost. threadpoolctl
    can raise a limit set by environment variable at runtime; the env vars
    themselves are left alone so the joblib paths are unaffected.
    """
    if not n_threads or threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=int(n_threads), user_api="blas")


@lru_cache(maxsize=None)
def _device(n_qubits):
    """Analytic statevector device.

    ``default.qubit`` on purpose: it broadcasts natively and is ~2.7x faster
    than ``lightning.qubit`` here (n=2000: 1.86 s vs 5.11 s). Do not swap in the
    scripts' own ``dev``, which is shot-based and carries an extra idle wire.
    """
    return qml.device("default.qubit", wires=n_qubits, shots=None)


def max_loadable(n_qubits, topology=None):
    """How many features the feature map can load on ``n_qubits`` qubits.

    One feature per nearest-neighbour coupling: ``n_qubits - 1`` on an open
    chain, ``n_qubits`` once the chain is closed into a ring. Below 3 qubits the
    ring degenerates -- the wrap pair (1, 0) is the pair (0, 1) again, so the
    second feature would only add to the first angle instead of opening a new
    direction -- and ring falls back to chain.
    """
    topo = TOPOLOGY if topology is None else topology
    if topo not in ("ring", "chain"):
        raise ValueError(f"topology must be 'ring' or 'chain', got {topo!r}")
    n_qubits = int(n_qubits)
    return n_qubits if (topo == "ring" and n_qubits >= 3) else n_qubits - 1


def qubits_for(n_feat, topology=None):
    """Smallest circuit width that loads all ``n_feat`` features.

    Chain needs a spare wire (``n_feat + 1``) because its last coupling is
    missing; ring needs none. Both scripts hard-coded the ``+ 1`` workaround in
    some dataset branches -- call this instead so the spare wire disappears when
    the ring closes it out.
    """
    n_feat = int(n_feat)
    n = 2
    while max_loadable(n, topology) < n_feat:
        n += 1
    return n


def coupling_pairs(n_qubits, n_feat, topology=None):
    """Wire pair carrying feature ``j``, for every loadable feature.

    Feature ``j`` drives the XX+YY+ZZ coupling on ``(j, (j+1) % n_qubits)``. The
    modulo is the whole difference between the two topologies: on a chain the
    last pair would be out of range and the feature is dropped, on a ring it
    wraps onto qubit 0. Features beyond ``max_loadable`` are still dropped --
    that is a real capacity limit, not the off-by-one bug.

    Returns:
        list of ``(wire_a, wire_b)``; ``len(...)`` is the old ``nload``.
    """
    nload = min(int(n_feat), max_loadable(n_qubits, topology))
    return [(j, (j + 1) % int(n_qubits)) for j in range(nload)]


@lru_cache(maxsize=None)
def sel_weights(n_layers, n_qubits, seed=SEED):
    """StronglyEntanglingLayers weights, identical to those in the scripts."""
    rng = np.random.default_rng(seed)
    w = rng.normal(loc=0, scale=1, size=(int(n_layers), int(n_qubits), 3))
    w.flags.writeable = False           # cached: never let a caller mutate it
    return w


def hamiltonian_feature_map(x, params, wires, n_feat, initial_state=True, seed=SEED,
                           topology=None):
    """Haar-like initial state followed by Trotterised Ising XYZ evolution.

    Broadcast-safe version of ``Hamiltonian_feature_map`` in QSVM_template.py.
    The original takes ``nload = min(len(x), n_qubits - 1)`` and indexes ``x[j]``,
    which silently addresses the BATCH axis when ``x`` is 2-D; ``n_feat`` is
    therefore explicit here and the feature axis is always the last one.

    Args:
        x: (n_feat,) or (batch, n_feat) input data.
        params: [t, T, n_layers]. T and n_layers are truncated with ``int()``,
            matching ``kernel_entry_param`` in the scripts.
        wires: wires to act on.
        n_feat: number of input features (NOT ``len(x)``).
        initial_state: whether to apply the SEL block.
        seed: SEL weight seed.
        topology: "ring", "chain", or None for the module default ``TOPOLOGY``.
    """
    n_qubits = len(wires)
    t, T, n_layers = float(params[0]), int(params[1]), int(params[2])

    if initial_state:
        qml.StronglyEntanglingLayers(sel_weights(n_layers, n_qubits, seed), wires=wires)

    # coupling_pairs already keeps every wire index in range, so no inner guard.
    pairs = coupling_pairs(n_qubits, n_feat, topology)
    for _ in range(T):
        for j, (wa, wb) in enumerate(pairs):
            theta = (t / T) * x[..., j]
            qml.IsingXX(2 * theta, wires=[wires[wa], wires[wb]])
            qml.IsingYY(2 * theta, wires=[wires[wa], wires[wb]])
            qml.IsingZZ(2 * theta, wires=[wires[wa], wires[wb]])


@lru_cache(maxsize=None)
def sel_block_weights(n_layers, n_qubits, seed=SEED, block=0):
    """SEL weights for the ``block``-th entangler of the re-uploading map.

    Seeded with the sequence ``[seed, block]`` so the blocks are deterministic
    and mutually independent streams. Deliberately NOT ``sel_weights`` shifted:
    ``block`` is a seed-sequence entry, so ``sel_block_weights(..., block=0)``
    differs from ``sel_weights(...)`` -- the initial state of the re-uploading
    map keeps using ``sel_weights`` directly, which is what makes its R = 1
    case bit-identical to the ising map.
    """
    rng = np.random.default_rng([int(seed), int(block)])
    w = rng.normal(loc=0, scale=1, size=(int(n_layers), int(n_qubits), 3))
    w.flags.writeable = False           # cached: never let a caller mutate it
    return w


def reuploading_feature_map(x, params, wires, n_feat, initial_state=True, seed=SEED,
                            topology=None):
    """SEL initial state + R data layers interleaved with fresh SEL blocks.

        U(x) = V_R(x) . W_{R-1} . ... . W_1 . V_1(x) . SEL_init

    Each ``V_r`` is one Heisenberg data layer at angle ``(t/R) * x_j`` per
    coupling pair -- the r-th re-upload of the SAME data. The ``W_r`` blocks in
    between carry block-specific weights (``sel_block_weights``), and they are
    what breaks the shift invariance of the ising map: ``U(y)^dag U(x)`` is no
    longer a function of ``x - y``, so the fidelity kernel need not be radial
    in any fixed metric, unlike the ising map.

    No ``W`` block follows the last data layer: a trailing unitary cancels in
    ``|<phi(y)|phi(x)>|^2`` and would be dead weight in a fidelity kernel.

    Args:
        x: (n_feat,) or (batch, n_feat) input data; feature axis last.
        params: [t, R, n_layers] -- total angle t split over R re-uploads.
            At R = 1 this is exactly ``hamiltonian_feature_map`` with
            params [t, 1, n_layers] (the regression anchor).
        wires / n_feat / initial_state / seed / topology: as in
            ``hamiltonian_feature_map``.
    """
    n_qubits = len(wires)
    t, R, n_layers = float(params[0]), int(params[1]), int(params[2])

    if initial_state:
        qml.StronglyEntanglingLayers(sel_weights(n_layers, n_qubits, seed), wires=wires)

    pairs = coupling_pairs(n_qubits, n_feat, topology)
    for r in range(R):
        if r > 0:
            qml.StronglyEntanglingLayers(
                sel_block_weights(n_layers, n_qubits, seed, r), wires=wires)
        for j, (wa, wb) in enumerate(pairs):
            theta = (t / R) * x[..., j]
            qml.IsingXX(2 * theta, wires=[wires[wa], wires[wb]])
            qml.IsingYY(2 * theta, wires=[wires[wa], wires[wb]])
            qml.IsingZZ(2 * theta, wires=[wires[wa], wires[wb]])


# Registry behind the ``feature_map=`` kwarg. ``None`` resolves to the module
# default ``FEATURE_MAP`` at call time, mirroring the ``topology`` pattern.
FEATURE_MAPS = {
    "ising": hamiltonian_feature_map,
    "reup": reuploading_feature_map,
}


def _resolve_feature_map(name):
    name = FEATURE_MAP if name is None else name
    if name not in FEATURE_MAPS:
        raise ValueError(
            f"feature_map must be one of {sorted(FEATURE_MAPS)}, got {name!r}")
    return name


@lru_cache(maxsize=None)
def _state_qnode(n_qubits, feature_map="ising"):
    """QNode returning the feature state, broadcast over the batch axis."""
    fmap = FEATURE_MAPS[feature_map]

    @qml.qnode(_device(n_qubits), diff_method=None)
    def _qn(x, params, n_feat, seed, topology):
        fmap(x, params, range(n_qubits), n_feat,
             initial_state=True, seed=seed, topology=topology)
        return qml.state()
    return _qn


@lru_cache(maxsize=None)
def _snapshot_qnode(n_qubits):
    """QNode snapshotting the state after each requested Trotter step count."""
    @qml.qnode(_device(n_qubits), diff_method=None)
    def _qn(x, tau, T_max, n_layers, n_feat, snap_at, seed, topology):
        wires = range(n_qubits)
        qml.StronglyEntanglingLayers(sel_weights(n_layers, n_qubits, seed), wires=wires)
        pairs = coupling_pairs(n_qubits, n_feat, topology)
        for k in range(int(T_max)):
            for j, (wa, wb) in enumerate(pairs):
                theta = tau * x[..., j]
                qml.IsingXX(2 * theta, wires=[wires[wa], wires[wb]])
                qml.IsingYY(2 * theta, wires=[wires[wa], wires[wb]])
                qml.IsingZZ(2 * theta, wires=[wires[wa], wires[wb]])
            if (k + 1) in snap_at:
                qml.Snapshot(f"T{k + 1}")
        return qml.state()
    return _qn


# --------------------------------------------------------------------------
# feature states and Gram matrices
# --------------------------------------------------------------------------
def feature_states(X, params, n_qubits, *, n_feat=None, seed=SEED, chunk=DEFAULT_CHUNK,
                   topology=None, feature_map=None):
    """Feature states |phi(x)> for every row of X.

    ``feature_map`` picks the circuit from ``FEATURE_MAPS`` ("ising"/"reup");
    None uses the module default ``FEATURE_MAP``.

    Returns:
        (n, 2**n_qubits) complex array.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if n_feat is None:
        n_feat = X.shape[-1]
    qn = _state_qnode(n_qubits, _resolve_feature_map(feature_map))

    out = []
    for lo in range(0, len(X), chunk):
        # atleast_2d: a single-row chunk comes back as a bare (2**q,) vector.
        out.append(np.atleast_2d(np.asarray(
            qn(X[lo:lo + chunk], params, n_feat, seed, topology))))
    return np.concatenate(out, axis=0)


def trotter_state_family(X, tau, T_values, n_layers, n_qubits, *, n_feat=None,
                         seed=SEED, chunk=DEFAULT_CHUNK, topology=None):
    """Feature states for every T in ``T_values`` at fixed per-step angle tau.

    ISING MAP ONLY: the identity below is a property of repeated identical
    Trotter layers and does not hold for the "reup" feature map, whose
    interleaved entangler blocks make the state after k data layers a different
    circuit from any single-parameter family member.

    The state after k Trotter steps at step-angle tau is exactly the state for
    ``params=[tau*k, k, n_layers]``, so one pass with snapshots covers the whole
    T family. Verified to 0.0 error against separate ``feature_states`` calls.

    Returns:
        dict {T: (n, 2**n_qubits) complex array}.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if n_feat is None:
        n_feat = X.shape[-1]
    T_values = sorted({int(T) for T in T_values})
    snap_at = frozenset(T_values)
    T_max = max(T_values)
    qn = _snapshot_qnode(n_qubits)

    parts = {T: [] for T in T_values}
    for lo in range(0, len(X), chunk):
        snaps = qml.snapshots(qn)(X[lo:lo + chunk], tau, T_max, n_layers,
                                  n_feat, snap_at, seed, topology)
        for T in T_values:
            parts[T].append(np.atleast_2d(np.asarray(snaps[f"T{T}"])))
    return {T: np.concatenate(v, axis=0) for T, v in parts.items()}


def gram_from_states(Phi_A, Phi_B=None, *, emulate_shots=None, rng=None):
    """Gram matrix from feature states: K = |Phi_A.conj() @ Phi_B.T|^2.

    Args:
        Phi_A: (n, d) feature states.
        Phi_B: (m, d) feature states, or None for the symmetric case.
        emulate_shots: if given, resample each entry as Binomial(m, K)/m. The
            production estimator reads P(|0...0>) off ``m`` shots, whose count is
            exactly Binomial(m, K), so this reproduces the old shot-sampled
            matrices in distribution at no extra cost.
        rng: numpy Generator for the shot emulation.
    """
    Phi_A = np.atleast_2d(Phi_A)
    symmetric = Phi_B is None
    Phi_B = Phi_A if symmetric else np.atleast_2d(Phi_B)

    K = np.abs(Phi_A.conj() @ Phi_B.T) ** 2
    np.clip(K, 0.0, 1.0, out=K)
    if symmetric:
        K = 0.5 * (K + K.T)
        np.fill_diagonal(K, 1.0)

    if emulate_shots:
        rng = np.random.default_rng(SEED) if rng is None else rng
        m = int(emulate_shots)
        if symmetric:
            iu = np.triu_indices_from(K, k=1)
            sampled = rng.binomial(m, K[iu]) / m
            K[iu] = sampled
            K[(iu[1], iu[0])] = sampled
        else:
            K = rng.binomial(m, K) / m
    return K


def compute_kernel_exact(A, B, params, n_jobs=None, *, n_qubits, n_feat=None,
                         seed=SEED, chunk=DEFAULT_CHUNK, emulate_shots=None, rng=None,
                         topology=None, feature_map=None):
    """Drop-in replacement for ``compute_kernel_param``.

    ``n_jobs`` is accepted and ignored: there is no process pool in this path.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if n_feat is None:
        n_feat = A.shape[-1]

    kw = dict(n_feat=n_feat, seed=seed, chunk=chunk, topology=topology,
              feature_map=feature_map)
    Phi_A = feature_states(A, params, n_qubits, **kw)
    if A.shape == B.shape and np.array_equal(A, B):
        return gram_from_states(Phi_A, emulate_shots=emulate_shots, rng=rng)
    Phi_B = feature_states(B, params, n_qubits, **kw)
    return gram_from_states(Phi_A, Phi_B, emulate_shots=emulate_shots, rng=rng)


def make_exact_kernel_fn(n_qubits, *, seed=SEED, chunk=DEFAULT_CHUNK,
                         emulate_shots=None, rng=None, topology=None,
                         feature_map=None):
    """Closure ``f(A, B, params, n_jobs=None) -> K``.

    Rebinding the *name* ``compute_kernel_param`` in a script to this redirects
    every call site at once, since Python resolves globals at call time.
    """
    def _compute_kernel(A, B, params, n_jobs=None):
        return compute_kernel_exact(A, B, params, n_jobs, n_qubits=n_qubits,
                                    seed=seed, chunk=chunk,
                                    emulate_shots=emulate_shots, rng=rng,
                                    topology=topology, feature_map=feature_map)
    return _compute_kernel


# --------------------------------------------------------------------------
# kernel-target alignment
# --------------------------------------------------------------------------
def centered_labels(Y, rescale_class_labels=True):
    """Labels mapped to +/-1, optionally class-rebalanced, then mean-centered.

    Centering happens AFTER the 1/sqrt(n_pos) rescale, which in general leaves
    mean(y) != 0, so the two steps do not commute.
    """
    y = np.asarray(Y).ravel()
    uniq = set(np.unique(y).tolist())
    if uniq <= {0, 1}:
        y = np.where(y == 1, 1.0, -1.0)
    elif uniq <= {-1, 1}:
        y = y.astype(float)
    else:
        raise ValueError(
            f"target_alignment is defined for binary labels only, got {sorted(uniq)}. "
            "The +/-1 formulation is meaningless for multi-class targets."
        )

    if rescale_class_labels:
        n_pos = np.count_nonzero(y == 1)
        n_neg = np.count_nonzero(y == -1)
        if n_pos > 0 and n_neg > 0:
            y = np.where(y == 1, 1.0 / np.sqrt(n_pos), -1.0 / np.sqrt(n_neg))
    return y - y.mean()


def target_alignment(Y, K, rescale_class_labels=True):
    """Centered kernel-target alignment (Cortes et al. 2012), in O(n^2).

    With H the centering matrix and Tc = H outer(y,y) H = outer(y_c, y_c):

        <Kc, Tc>_F = y_c^T K y_c      (H is idempotent, so K needs no centering)
        ||Tc||_F   = y_c . y_c        (Tc is rank-1)

    so only ||Kc||_F needs the centered matrix, and that is a rank-1 update
    rather than two O(n^3) GEMMs. Agrees with the dense form to ~4e-17.

    Returns 0.0 when ||Kc||_F collapses: as the evolution time goes to zero
    K -> all-ones and the dense form divides by its own 1e-12 epsilon, which is
    numerically meaningless rather than a genuine alignment of zero.
    """
    yc = centered_labels(Y, rescale_class_labels)
    K = np.asarray(K, dtype=float)

    num = float(yc @ (K @ yc))
    Kc = K - K.mean(axis=1, keepdims=True) - K.mean(axis=0, keepdims=True) + K.mean()
    kc_norm = float(np.sqrt(np.einsum("ij,ij->", Kc, Kc)))
    if kc_norm < 1e-9:
        return 0.0
    return num / (kc_norm * float(yc @ yc) + 1e-12)


def target_alignment_dense(Y, K, rescale_class_labels=True):
    """O(n^3) reference implementation, kept as the test oracle only."""
    y = np.asarray(Y).ravel()
    if set(np.unique(y).tolist()) <= {0, 1}:
        y = np.where(y == 1, 1.0, -1.0)
    else:
        y = y.astype(float)
    if rescale_class_labels:
        n_pos = np.count_nonzero(y == 1)
        n_neg = np.count_nonzero(y == -1)
        if n_pos > 0 and n_neg > 0:
            y = np.where(y == 1, 1.0 / np.sqrt(n_pos), -1.0 / np.sqrt(n_neg))
    T = np.outer(y, y)
    n = len(y)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Tc = H @ T @ H
    return np.sum(Kc * Tc) / (np.sqrt(np.sum(Kc * Kc) * np.sum(Tc * Tc)) + 1e-12)


def kernel_diagnostics(K):
    """Off-diagonal statistics used to reject degenerate corners of the search.

    A fidelity Gram degenerates in two directions: tau -> 0 drives K -> all-ones,
    and large evolution time drives K -> 2^-n_qubits (exponential concentration).
    Both crush the off-diagonal variance while KTA becomes noise, so the search
    must be constrained by ``offdiag_var``, not by KTA alone.
    """
    K = np.asarray(K, dtype=float)
    off = K[~np.eye(len(K), dtype=bool)]
    return {
        "offdiag_mean": float(off.mean()),
        "offdiag_var": float(off.var()),
        "offdiag_min": float(off.min()),
        "offdiag_max": float(off.max()),
    }


# --------------------------------------------------------------------------
# alignment search
# --------------------------------------------------------------------------
def tau_grid_from_data(X, n_qubits, *, n_feat=None, max_T=40, angle_lo=1e-3,
                       angle_hi=2 * np.pi, n_tau=32, topology=None):
    """Geometric grid of per-step angles tau, scaled to the data.

    The gate angle is ``2 * tau * x_j``, so what matters is the accumulated angle
    ``tau * T * scale`` with ``scale = median(|x|)`` over the loaded features.
    The grid is chosen so that the union over T of ``t = tau*T`` covers the
    scripts' own ``t`` range of [0, 2*pi] in accumulated-angle terms.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if n_feat is None:
        n_feat = X.shape[-1]
    nload = len(coupling_pairs(n_qubits, n_feat, topology))
    scale = float(np.median(np.abs(X[:, :nload])))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return np.geomspace(angle_lo / (scale * max_T), angle_hi / scale, n_tau)


def _choose_T_values(n_samples, n_qubits, min_T, max_T, mem_budget_bytes):
    """Every T in range if the snapshots fit, else the coarse geometric ladder."""
    T_all = list(range(int(min_T), int(max_T) + 1))
    per_T = n_samples * (2 ** n_qubits) * 16          # complex128
    if per_T * len(T_all) <= mem_budget_bytes:
        return T_all
    return [T for T in T_LADDER if min_T <= T <= max_T] or [min_T]


def align_landscape(X, y, *, n_qubits, tau_grid, T_values, n_layers, n_feat=None,
                    seed=SEED, chunk=DEFAULT_CHUNK, rescale_class_labels=True,
                    period_t=2 * np.pi, cache=None, blas_threads=None, verbose=True,
                    topology=None):
    """KTA over the (tau, T) grid. One snapshot pass per tau covers all T.

    Cells whose total evolution time ``t = tau*T`` exceeds ``period_t`` are left
    as NaN: the scripts clip/wrap t into [0, period_t] before building the
    circuit, so those points are not reachable by the original search either.

    Returns:
        dict with "tau" (nt,), "T" (nT,), "t" (nt,nT), "kta" (nt,nT),
        "offdiag_var" (nt,nT) and "n_evals".
    """
    tau_grid = np.asarray(tau_grid, dtype=float)
    T_values = [int(T) for T in T_values]
    nt, nT = len(tau_grid), len(T_values)

    t_arr = tau_grid[:, None] * np.array(T_values, dtype=float)[None, :]
    kta = np.full((nt, nT), np.nan)
    var = np.full((nt, nT), np.nan)
    n_evals = 0
    t_start = time.perf_counter()

    with blas_limit(blas_threads):
        for i, tau in enumerate(tau_grid):
            reachable = [T for T in T_values if tau * T <= period_t]
            if not reachable:
                continue
            fam = trotter_state_family(X, tau, reachable, n_layers, n_qubits,
                                       n_feat=n_feat, seed=seed, chunk=chunk,
                                       topology=topology)
            for T in reachable:
                j = T_values.index(T)
                K = gram_from_states(fam[T])
                kta[i, j] = target_alignment(y, K, rescale_class_labels)
                var[i, j] = kernel_diagnostics(K)["offdiag_var"]
                n_evals += 1
                if cache is not None:
                    cache[_cache_key(tau * T, T, n_layers)] = kta[i, j]
            del fam
            if verbose and (i == 0 or (i + 1) % max(1, nt // 8) == 0):
                best = np.nanmax(kta[:i + 1]) if np.any(np.isfinite(kta[:i + 1])) else np.nan
                print(f"  [landscape] tau {i + 1:3d}/{nt} ({tau:.3g}) | "
                      f"best KTA so far {best:.4f} | {time.perf_counter() - t_start:.0f} s",
                      flush=True)

    return {"tau": tau_grid, "T": np.array(T_values), "t": t_arr,
            "kta": kta, "offdiag_var": var, "n_evals": n_evals}


def _cache_key(t, T, L):
    """Exact key -- do NOT round: Brent probes values differing in the last ulps
    and those should miss the cache rather than collide."""
    return (np.float64(t).tobytes(), int(T), int(L))


def _kta_at(X, y, t, T, L, *, n_qubits, n_feat, seed, chunk,
            rescale_class_labels, cache, blas_threads=None, topology=None):
    """Memoised KTA at a single (t, T, L). Exact because the objective is."""
    key = _cache_key(t, T, L)
    if cache is not None and key in cache:
        return cache[key]
    with blas_limit(blas_threads):
        K = compute_kernel_exact(X, X, [t, T, L], n_qubits=n_qubits, n_feat=n_feat,
                                 seed=seed, chunk=chunk, topology=topology)
        val = target_alignment(y, K, rescale_class_labels)
    if cache is not None:
        cache[key] = val
    return val


def refine_tau(X, y, *, n_qubits, T, tau_lo, tau_hi, n_layers, n_feat=None,
               seed=SEED, chunk=DEFAULT_CHUNK, rescale_class_labels=True,
               max_iter=25, cache=None, blas_threads=None, topology=None):
    """Bounded Brent on tau at fixed T.

    Legitimate only because the objective is now deterministic; against the old
    shot-sampled objective a line search of this kind cannot converge.

    Returns:
        (best_tau, best_kta)
    """
    def neg(tau):
        return -_kta_at(X, y, float(tau) * T, T, n_layers, n_qubits=n_qubits,
                        n_feat=n_feat, seed=seed, chunk=chunk,
                        rescale_class_labels=rescale_class_labels, cache=cache,
                        blas_threads=blas_threads, topology=topology)

    res = minimize_scalar(neg, bounds=(float(tau_lo), float(tau_hi)),
                          method="bounded", options={"maxiter": max_iter})
    return float(res.x), float(-res.fun)


def search_alignment(
    xs_train, y_train, *,
    n_qubits,
    n_layers=3,
    n_feat=None,
    seed=SEED,
    tau_grid=None,
    n_tau=32,
    T_values=None,
    min_T=1,
    max_T=40,
    max_samples=None,
    subsample_seed=42,
    t_transform="clip",
    period_t=2 * np.pi,
    refine=True,
    L_values=None,
    min_offdiag_var=1e-6,
    rescale_class_labels=True,
    chunk=DEFAULT_CHUNK,
    blas_threads=8,
    topology=None,
    mem_budget_bytes=2 * 1024 ** 3,
    dataset="",
    model_name="ising_emb",
    out_dir=None,
    save_fig=True,
    verbose=True,
):
    """Search [t, T, L] for maximum centered kernel-target alignment.

    Replaces ``apply_alignment``. Scans the (tau, T) landscape exhaustively, then
    refines tau at the winning T, instead of taking 40 noisy gradient/hill-climb
    steps. ``L`` is NOT searched -- see below.

    Args:
        n_qubits: circuit width (7 in QSVM_template.py, 5 in pdac.py).
        n_layers: SEL depth, held FIXED. It only selects a data-independent
            initial state ``V_L|0>``: numpy fills C-order so ``weights(L)[:L']``
            == ``weights(L')`` (the states are a nested random sequence), and SEL
            with iid N(0,1) angles is near a unitary 2-design after ~2 layers at
            these widths. KTA(L) is therefore an iid draw with no structure, and
            argmax over L in 1..10 -- what the old hill-climb did -- is
            maximisation bias: it inflates the reported KTA by roughly the
            expected max of 10 draws and does not transfer off the subsample.
        max_samples: subsample size, or None to use all of xs_train. The fast
            path removes the reason to subsample, and using everything drops the
            bias the old StratifiedShuffleSplit(train_size=500) accepted.
        t_transform: "clip" (QSVM_template) or "wrap" (pdac), applied to the
            reported t so the returned params match each script's convention.
        L_values: optional replicate axis. Re-scores the winning (tau, T) at
            these SEL depths and reports mean/std -- turning the bias source
            above into an honest error bar.
        topology: "ring" (default), "chain", or None for the module-level
            ``TOPOLOGY``. A chain loads only ``n_qubits - 1`` features; anything
            beyond that is invisible to the kernel and cannot be aligned for.
        min_offdiag_var: reject cells whose Gram has collapsed. A fidelity Gram
            degenerates to all-ones as tau -> 0 and to 2^-n_qubits at large t;
            KTA is meaningless in both corners, so the maximum must be taken
            subject to this floor rather than over the raw surface.

    Returns:
        dict with "opt_params" [t, T, L], "best_kta", "landscape", "refined",
        "replicates", "n_used", "n_evals", "wall_s" and "meta".
    """
    t0 = time.perf_counter()
    X = np.atleast_2d(np.asarray(xs_train, dtype=float))
    y = np.asarray(y_train).ravel()

    if max_samples is not None and max_samples < len(X):
        sss = StratifiedShuffleSplit(n_splits=1, train_size=int(max_samples),
                                     random_state=subsample_seed)
        idx, _ = next(sss.split(X, y))
        X, y = X[idx], y[idx]
    if n_feat is None:
        n_feat = X.shape[-1]

    topology = TOPOLOGY if topology is None else topology
    nload = len(coupling_pairs(n_qubits, n_feat, topology))

    if tau_grid is None:
        tau_grid = tau_grid_from_data(X, n_qubits, n_feat=n_feat, max_T=max_T,
                                      angle_hi=period_t, n_tau=n_tau,
                                      topology=topology)
    if T_values is None:
        T_values = _choose_T_values(len(X), n_qubits, min_T, max_T, mem_budget_bytes)

    # The Gram and its centered copy are both n x n dense float64.
    gram_gb = 2 * len(X) ** 2 * 8 / 1024 ** 3
    if verbose:
        print(f"[align] n={len(X)} n_feat={n_feat} n_qubits={n_qubits} L={n_layers} "
              f"| {topology}: {nload}/{n_feat} features loaded "
              f"| {len(tau_grid)} tau x {len(T_values)} T "
              f"| tau in [{tau_grid[0]:.3g}, {tau_grid[-1]:.3g}] "
              f"| ~{gram_gb:.2f} GB peak per cell", flush=True)
    if nload < n_feat:
        print(f"[align] WARNING: {n_feat - nload} of {n_feat} feature(s) never enter the "
              f"circuit -- only {nload} couplings exist on {n_qubits} qubits with "
              f"topology={topology!r}. Raise n_qubits (or use topology='ring') before "
              "reading anything into the alignment result.", flush=True)
    if gram_gb > 4.0:
        print(f"[align] WARNING: {gram_gb:.1f} GB per Gram at n={len(X)}. The kernel is "
              "cheap now but the O(n^2) alignment algebra is not -- pass "
              "max_samples=~4000 unless you specifically need the full set.", flush=True)

    cache = {}
    land = align_landscape(X, y, n_qubits=n_qubits, tau_grid=tau_grid,
                           T_values=T_values, n_layers=n_layers, n_feat=n_feat,
                           seed=seed, chunk=chunk,
                           rescale_class_labels=rescale_class_labels,
                           period_t=period_t, cache=cache,
                           blas_threads=blas_threads, verbose=verbose,
                           topology=topology)

    # --- select subject to the concentration floor ---
    kta, var = land["kta"], land["offdiag_var"]
    healthy = np.isfinite(kta) & (var >= min_offdiag_var)
    if not healthy.any():
        healthy = np.isfinite(kta)
        if verbose:
            print(f"[align] WARNING: no cell clears offdiag_var >= {min_offdiag_var:g}; "
                  "the Gram is degenerate everywhere on this grid.")
    masked = np.where(healthy, kta, -np.inf)
    i, j = np.unravel_index(int(np.argmax(masked)), masked.shape)
    best_tau, best_T = float(land["tau"][i]), int(land["T"][j])
    best_kta = float(kta[i, j])
    if verbose:
        print(f"[align] grid best: KTA={best_kta:.4f} at tau={best_tau:.4g}, T={best_T} "
              f"(t={best_tau * best_T:.4f}, offdiag_var={var[i, j]:.2e})")

    # --- local refine on tau at the winning T ---
    refined = None
    if refine and len(tau_grid) > 1:
        lo = tau_grid[max(i - 1, 0)]
        hi = min(tau_grid[min(i + 1, len(tau_grid) - 1)], period_t / best_T)
        if hi > lo:
            r_tau, r_kta = refine_tau(X, y, n_qubits=n_qubits, T=best_T,
                                      tau_lo=lo, tau_hi=hi, n_layers=n_layers,
                                      n_feat=n_feat, seed=seed, chunk=chunk,
                                      rescale_class_labels=rescale_class_labels,
                                      cache=cache, blas_threads=blas_threads,
                                      topology=topology)
            refined = {"tau": r_tau, "kta": r_kta, "bracket": [float(lo), float(hi)]}
            if r_kta > best_kta:
                best_tau, best_kta = r_tau, r_kta
            if verbose:
                print(f"[align] refined: KTA={r_kta:.4f} at tau={r_tau:.6g} "
                      f"(t={r_tau * best_T:.6f})")

    # --- L as a replicate axis, not a search axis ---
    replicates = None
    if L_values:
        vals = []
        for L in L_values:
            vals.append(_kta_at(X, y, best_tau * best_T, best_T, int(L),
                                n_qubits=n_qubits, n_feat=n_feat, seed=seed,
                                chunk=chunk,
                                rescale_class_labels=rescale_class_labels,
                                cache=cache, blas_threads=blas_threads,
                                topology=topology))
        replicates = {"L": [int(L) for L in L_values], "kta": [float(v) for v in vals],
                      "mean": float(np.mean(vals)), "std": float(np.std(vals))}
        if verbose:
            print(f"[align] L replicates {replicates['L']}: "
                  f"KTA = {replicates['mean']:.4f} +/- {replicates['std']:.4f} "
                  "(spread over initial states, NOT a quantity to maximise)")

    # --- report t under each script's own convention, rounded for filenames ---
    t_best = best_tau * best_T
    t_best = np.clip(t_best, 0.0, period_t) if t_transform == "clip" else t_best % period_t
    opt_params = [round(float(t_best), 6), int(best_T), int(n_layers)]

    wall = time.perf_counter() - t0
    if verbose:
        print(f"[align] best KTA {best_kta:.4f} at t={opt_params[0]}, "
              f"T={opt_params[1]}, L={opt_params[2]} "
              f"| {land['n_evals']} grid + {len(cache) - land['n_evals']} refine evals "
              f"| {wall:.1f} s")

    res = {
        "opt_params": opt_params,
        "best_kta": best_kta,
        "landscape": land,
        "refined": refined,
        "replicates": replicates,
        "n_used": int(len(X)),
        "n_evals": int(land["n_evals"]),
        "wall_s": float(wall),
        "meta": {"dataset": dataset, "model_name": model_name, "n_qubits": int(n_qubits),
                 "n_layers": int(n_layers), "n_feat": int(n_feat), "seed": int(seed),
                 "topology": topology, "n_loaded": int(nload),
                 "t_transform": t_transform, "period_t": float(period_t),
                 "min_offdiag_var": float(min_offdiag_var),
                 "max_samples": None if max_samples is None else int(max_samples)},
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        stem = f"{dataset}_{model_name}_{n_qubits}q_L{n_layers}_{topology}_align"
        save_result(res, os.path.join(out_dir, stem + ".json"))
        if save_fig:
            plot_alignment_landscape(res, os.path.join(out_dir, stem + ".png"))
        if verbose:
            print(f"[align] wrote {os.path.join(out_dir, stem)}.{{json,png}}")
    return res


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def plot_alignment_landscape(res, path):
    """Heatmap of KTA over the (tau, T) grid, with the selected cell marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    land = res["landscape"]
    tau, T, kta = np.asarray(land["tau"]), np.asarray(land["T"]), np.asarray(land["kta"])

    fig, ax = plt.subplots(figsize=(9, 5))
    mesh = ax.pcolormesh(tau, T, np.ma.masked_invalid(kta).T, shading="nearest",
                         cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="centered KTA")
    ax.set_xscale("log")
    ax.set_xlabel(r"per-Trotter-step angle  $\tau = t/T$")
    ax.set_ylabel("Trotter steps $T$")

    t_opt, T_opt = res["opt_params"][0], res["opt_params"][1]
    ax.plot(t_opt / T_opt, T_opt, "r*", markersize=16, markeredgecolor="white")
    meta = res["meta"]
    ax.set_title(f"{meta['dataset']} {meta['model_name']} "
                 f"({meta['n_qubits']}q, L={meta['n_layers']}, n={res['n_used']}, "
                 f"{meta.get('topology', 'chain')}: "
                 f"{meta.get('n_loaded', meta['n_qubits'] - 1)}/{meta['n_feat']} feat)\n"
                 f"best KTA={res['best_kta']:.4f} at t={t_opt:g}, T={T_opt} "
                 f"(white: t > {meta['period_t']:.3g}, unreachable)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_result(res, path):
    """Write the search result as JSON (numpy arrays become nested lists)."""
    def _plain(o):
        if isinstance(o, np.ndarray):
            return [_plain(v) for v in o.tolist()]
        if isinstance(o, dict):
            return {k: _plain(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_plain(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o

    with open(path, "w") as fh:
        json.dump(_plain(res), fh, indent=2)


def load_result(path):
    """Read back a saved search result, restoring the landscape arrays."""
    with open(path) as fh:
        res = json.load(fh)
    land = res.get("landscape")
    if land:
        for k in ("tau", "T", "t", "kta", "offdiag_var"):
            if k in land:
                land[k] = np.array(land[k], dtype=float)
    return res
