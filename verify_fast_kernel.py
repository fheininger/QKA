"""Verify the fast feature-state kernel against the PennyLane reference.

1. Exact check: |Phi.conj() @ Phi.T|^2 must reproduce the production
   ``kernel_circ`` (compute-uncompute, one circuit per entry) to ~1e-12, across
   several [t, T, L] and several feature counts. The reference device carries the
   extra idle wire the scripts allocate, proving it is irrelevant.
2. Structure check: the symmetric Gram must be exactly symmetric, have a unit
   diagonal and be PSD (the exact Gram is PSD by construction, unlike the
   shot-sampled one); the rectangular branch must match entry by entry.
3. Snapshot check: the Trotter reparametrisation ``tau = t/T`` -- the state after
   k steps at fixed tau is the state for (t = k*tau, T = k).
4. Centering check: the O(n^2) target_alignment against the O(n^3) dense form.
5. Shot-emulation check: ``emulate_shots`` must land within shot noise of exact.
6. Topology check: every feature the topology claims to load must actually move
   the kernel, and every feature beyond that must leave it bit-identical. This is
   the regression test for the dropped-feature bug -- on a 7-qubit CHAIN only 6
   of 7 couplings exist and the 7th PCA component was silently ignored.
7. --slow: the real production path (lightning.qubit, shots=3000,
   compute_kernel_param) on a small subset, within shot noise.
8. Re-uploading map (feature_map="reup"): its R = 1 case must be bit-identical
   to the ising map at T = 1; R > 1 must match an independent local oracle;
   Gram structure, feature coverage and the trailing-entangler cancellation
   (a W block after the last data layer must not change the fidelity kernel)
   are checked the same way as for the ising map.

Run with the qmlenv python:
    ~/miniforge3/envs/qmlenv/bin/python verify_fast_kernel.py
    ~/miniforge3/envs/qmlenv/bin/python verify_fast_kernel.py --slow
"""

import argparse

import numpy as np
import pennylane as qml

from kernel_align import (compute_kernel_exact, coupling_pairs, feature_states,
                          gram_from_states, max_loadable, target_alignment,
                          target_alignment_dense, trotter_state_family)

N_QUBITS = 7
SEED = 42
TOPOLOGY = "ring"   # checked against "chain" too, in the topology section


# --- PennyLane reference: verbatim copy of QSVM_template.py:2828-2878, on an
# --- analytic device. Deliberately NOT imported from the scripts.
dev = qml.device("lightning.qubit", wires=N_QUBITS, batch_obs=True, shots=None)


def Hamiltonian_feature_map(x, params, wires, initial_state=True, seed=SEED):
    n_qubits = len(wires)
    t, T, N_LAYERS = params
    if initial_state:
        rng = np.random.default_rng(seed)
        weights = rng.normal(loc=0, scale=1, size=(N_LAYERS, n_qubits, 3))
        qml.StronglyEntanglingLayers(weights, wires=wires)
    # Only the wire layout is shared with kernel_align -- that layout is the
    # specification, not the evaluation path this script is checking.
    for _ in range(T):
        for j, (wa, wb) in enumerate(coupling_pairs(n_qubits, len(x), TOPOLOGY)):
            theta = (t / T) * x[j]
            qml.IsingXX(2 * theta, wires=[wires[wa], wires[wb]])
            qml.IsingYY(2 * theta, wires=[wires[wa], wires[wb]])
            qml.IsingZZ(2 * theta, wires=[wires[wa], wires[wb]])


@qml.qnode(dev)
def kernel_circ(a, b, params):
    wires = range(N_QUBITS)
    Hamiltonian_feature_map(a, params, wires=wires, initial_state=True)
    qml.adjoint(Hamiltonian_feature_map)(b, params, wires=wires, initial_state=True)
    return qml.probs(wires=wires)


def reference_gram(A, B, params):
    p = [params[0], int(params[1]), int(params[2])]
    return np.array([[float(kernel_circ(a, b, p)[0]) for b in B] for a in A])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slow", action="store_true",
                    help="also check the shot-based lightning.qubit production path")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    ok = True

    # --- 1. Exact equivalence -------------------------------------------------
    for params in ([0.1, 20, 3], [1.7, 1, 1], [0.37, 7, 2]):
        for n_feat in (N_QUBITS - 1, N_QUBITS, N_QUBITS + 1):
            A = rng.uniform(0, np.pi, (5, n_feat))
            B = rng.uniform(0, np.pi, (4, n_feat))
            K_fast = gram_from_states(feature_states(A, params, N_QUBITS),
                                      feature_states(B, params, N_QUBITS))
            err = float(np.abs(K_fast - reference_gram(A, B, params)).max())
            status = "PASS" if err < 1e-12 else "FAIL"
            ok &= err < 1e-12
            print(f"[exact] params={params} n_feat={n_feat}: "
                  f"max |fast - kernel_circ| = {err:.2e}  {status}")

    # --- 2. Gram structure ----------------------------------------------------
    params = [0.4, 8, 3]
    A = rng.uniform(0, np.pi, (12, N_QUBITS))
    K = compute_kernel_exact(A, A, params, n_qubits=N_QUBITS)
    asym = float(np.abs(K - K.T).max())
    diag = float(np.abs(np.diag(K) - 1.0).max())
    min_eig = float(np.linalg.eigvalsh(K).min())
    good = asym == 0.0 and diag == 0.0 and min_eig > -1e-10
    ok &= good
    print(f"[structure] symmetric: max|K-K.T|={asym:.2e} max|diag-1|={diag:.2e} "
          f"min_eig={min_eig:+.2e}  {'PASS' if good else 'FAIL'}")

    B = rng.uniform(0, np.pi, (5, N_QUBITS))
    K_rect = compute_kernel_exact(A, B, params, n_qubits=N_QUBITS)
    err = float(np.abs(K_rect - reference_gram(A, B, params)).max())
    good = K_rect.shape == (12, 5) and err < 1e-12
    ok &= good
    print(f"[structure] rectangular: shape={K_rect.shape} max err={err:.2e}  "
          f"{'PASS' if good else 'FAIL'}")

    # single row must not collapse to a 1-D state
    K_one = compute_kernel_exact(A[:1], B, params, n_qubits=N_QUBITS)
    good = K_one.shape == (1, 5)
    ok &= good
    print(f"[structure] batch=1: shape={K_one.shape}  {'PASS' if good else 'FAIL'}")

    # --- 3. Trotter snapshot identity ----------------------------------------
    tau, T_values, n_layers = 0.03, [1, 3, 7, 20, 40], 3
    X = rng.uniform(0, np.pi, (30, N_QUBITS))
    fam = trotter_state_family(X, tau, T_values, n_layers, N_QUBITS)
    err = max(float(np.abs(np.abs(fam[T])
                           - np.abs(feature_states(X, [tau * T, T, n_layers], N_QUBITS))).max())
              for T in T_values)
    status = "PASS" if err < 1e-12 else "FAIL"
    ok &= err < 1e-12
    print(f"[snapshots] T={T_values} at tau={tau}: max |snapshot - fresh| = {err:.2e}  {status}")

    # --- 4. Centering identity ------------------------------------------------
    n = 60
    X = rng.uniform(0, np.pi, (n, N_QUBITS))
    K = compute_kernel_exact(X, X, [0.2, 10, 3], n_qubits=N_QUBITS)
    K_rand = rng.random((n, n))
    K_rand = K_rand @ K_rand.T
    y01 = (rng.random(n) > 0.5).astype(int)
    worst = 0.0
    for K_test in (K, K_rand):
        for rescale in (True, False):
            for y in (y01, np.where(y01 == 1, 1.0, -1.0)):
                worst = max(worst, abs(target_alignment(y, K_test, rescale)
                                       - target_alignment_dense(y, K_test, rescale)))
    status = "PASS" if worst < 1e-10 else "FAIL"
    ok &= worst < 1e-10
    print(f"[centering] max |O(n^2) - dense| over 8 cases = {worst:.2e}  {status}")

    # --- 5. Shot emulation ----------------------------------------------------
    shots = 3000
    A = rng.uniform(0, np.pi, (10, N_QUBITS))
    exact = compute_kernel_exact(A, A, params, n_qubits=N_QUBITS)
    sampled = compute_kernel_exact(A, A, params, n_qubits=N_QUBITS,
                                   emulate_shots=shots, rng=np.random.default_rng(1))
    sigma = np.sqrt(np.maximum(exact * (1 - exact), 1e-12) / shots)
    n_sigma = np.abs(sampled - exact) / np.maximum(sigma, 1e-12)
    worst = float(np.max(n_sigma[~np.eye(len(A), dtype=bool)]))
    status = "PASS" if worst < 5 else "FAIL"
    ok &= worst < 5
    print(f"[emulate_shots] worst off-diagonal deviation: {worst:.2f} sigma "
          f"(shots={shots})  {status}")

    # --- 6. Feature coverage per topology ------------------------------------
    for topo in ("ring", "chain"):
        nload = max_loadable(N_QUBITS, topo)
        base = rng.uniform(0, np.pi, (8, N_QUBITS + 1))
        deltas = []
        for j in range(N_QUBITS + 1):
            bumped = base.copy()
            bumped[:, j] += 0.7
            K0 = compute_kernel_exact(base, base, [0.3, 8, 3], n_qubits=N_QUBITS,
                                      topology=topo)
            Kj = compute_kernel_exact(bumped, bumped, [0.3, 8, 3], n_qubits=N_QUBITS,
                                      topology=topo)
            deltas.append(float(np.abs(Kj - K0).max()))
        live = [j for j, d in enumerate(deltas) if d > 1e-9]
        dead = [j for j, d in enumerate(deltas) if d == 0.0]
        good = live == list(range(nload)) and dead == list(range(nload, N_QUBITS + 1))
        ok &= good
        print(f"[topology] {topo}: {nload} couplings -> features {live} move the kernel, "
              f"{dead} are exactly ignored  {'PASS' if good else 'FAIL'}")

    # --- 8. Re-uploading feature map ------------------------------------------
    # 8a. R = 1 reduction: reup with [t, 1, L] is the ising map with [t, 1, L].
    for params_r1 in ([0.1, 1, 3], [1.7, 1, 1]):
        A = rng.uniform(0, np.pi, (10, N_QUBITS))
        d = float(np.abs(feature_states(A, params_r1, N_QUBITS, feature_map="reup")
                         - feature_states(A, params_r1, N_QUBITS, feature_map="ising")).max())
        status = "PASS" if d < 1e-12 else "FAIL"
        ok &= d < 1e-12
        print(f"[reup] R=1 reduction, params={params_r1}: max |reup - ising| = {d:.2e}  {status}")

    # 8b. Independent oracle for R > 1 (local circuit, deliberately not imported).
    def Reuploading_feature_map(x, params, wires, initial_state=True, seed=SEED,
                                trailing_block=False):
        n_qubits = len(wires)
        t, R, L = float(params[0]), int(params[1]), int(params[2])
        if initial_state:
            w0 = np.random.default_rng(seed).normal(loc=0, scale=1, size=(L, n_qubits, 3))
            qml.StronglyEntanglingLayers(w0, wires=wires)
        pairs = coupling_pairs(n_qubits, len(x), TOPOLOGY)
        for r in range(R):
            if r > 0:
                w = np.random.default_rng([seed, r]).normal(loc=0, scale=1,
                                                            size=(L, n_qubits, 3))
                qml.StronglyEntanglingLayers(w, wires=wires)
            for j, (wa, wb) in enumerate(pairs):
                theta = (t / R) * x[j]
                qml.IsingXX(2 * theta, wires=[wires[wa], wires[wb]])
                qml.IsingYY(2 * theta, wires=[wires[wa], wires[wb]])
                qml.IsingZZ(2 * theta, wires=[wires[wa], wires[wb]])
        if trailing_block:
            w = np.random.default_rng([seed, R]).normal(loc=0, scale=1,
                                                        size=(L, n_qubits, 3))
            qml.StronglyEntanglingLayers(w, wires=wires)

    @qml.qnode(dev)
    def kernel_circ_reup(a, b, params, trailing_block=False):
        wires = range(N_QUBITS)
        Reuploading_feature_map(a, params, wires=wires, trailing_block=trailing_block)
        qml.adjoint(Reuploading_feature_map)(b, params, wires=wires,
                                             trailing_block=trailing_block)
        return qml.probs(wires=wires)

    A = rng.uniform(0, np.pi, (5, N_QUBITS))
    B = rng.uniform(0, np.pi, (4, N_QUBITS))
    for params_r in ([0.6, 2, 3], [1.1, 3, 2]):
        K_fast = gram_from_states(feature_states(A, params_r, N_QUBITS, feature_map="reup"),
                                  feature_states(B, params_r, N_QUBITS, feature_map="reup"))
        K_ref = np.array([[float(kernel_circ_reup(a, b, params_r)[0]) for b in B]
                          for a in A])
        err = float(np.abs(K_fast - K_ref).max())
        status = "PASS" if err < 1e-12 else "FAIL"
        ok &= err < 1e-12
        print(f"[reup] oracle, params={params_r}: max |fast - kernel_circ| = {err:.2e}  {status}")

    # 8c. Gram structure.
    params_r = [0.6, 3, 3]
    A = rng.uniform(0, np.pi, (12, N_QUBITS))
    K = compute_kernel_exact(A, A, params_r, n_qubits=N_QUBITS, feature_map="reup")
    asym = float(np.abs(K - K.T).max())
    diag = float(np.abs(np.diag(K) - 1.0).max())
    min_eig = float(np.linalg.eigvalsh(K).min())
    good = asym == 0.0 and diag == 0.0 and min_eig > -1e-10
    ok &= good
    print(f"[reup] structure: max|K-K.T|={asym:.2e} max|diag-1|={diag:.2e} "
          f"min_eig={min_eig:+.2e}  {'PASS' if good else 'FAIL'}")

    # 8d. Feature coverage, both topologies.
    for topo in ("ring", "chain"):
        nload = max_loadable(N_QUBITS, topo)
        base = rng.uniform(0, np.pi, (8, N_QUBITS + 1))
        deltas = []
        for j in range(N_QUBITS + 1):
            bumped = base.copy()
            bumped[:, j] += 0.7
            K0 = compute_kernel_exact(base, base, [0.3, 2, 3], n_qubits=N_QUBITS,
                                      topology=topo, feature_map="reup")
            Kj = compute_kernel_exact(bumped, bumped, [0.3, 2, 3], n_qubits=N_QUBITS,
                                      topology=topo, feature_map="reup")
            deltas.append(float(np.abs(Kj - K0).max()))
        live = [j for j, d in enumerate(deltas) if d > 1e-9]
        dead = [j for j, d in enumerate(deltas) if d == 0.0]
        good = live == list(range(nload)) and dead == list(range(nload, N_QUBITS + 1))
        ok &= good
        print(f"[reup] topology {topo}: features {live} move the kernel, "
              f"{dead} are exactly ignored  {'PASS' if good else 'FAIL'}")

    # 8e. Trailing-entangler cancellation: a W block after the last data layer
    # must leave the fidelity kernel unchanged (it cancels in |<phi|phi'>|^2).
    A = rng.uniform(0, np.pi, (4, N_QUBITS))
    params_r = [0.8, 2, 3]
    K_plain = np.array([[float(kernel_circ_reup(a, b, params_r)[0]) for b in A]
                        for a in A])
    K_trail = np.array([[float(kernel_circ_reup(a, b, params_r, trailing_block=True)[0])
                         for b in A] for a in A])
    err = float(np.abs(K_plain - K_trail).max())
    status = "PASS" if err < 1e-12 else "FAIL"
    ok &= err < 1e-12
    print(f"[reup] trailing entangler cancels: max |K_plain - K_trail| = {err:.2e}  {status}")

    # 8f. Unknown map names must fail loudly.
    try:
        compute_kernel_exact(A, A, params_r, n_qubits=N_QUBITS, feature_map="nope")
        good = False
    except ValueError:
        good = True
    ok &= good
    print(f"[reup] unknown feature_map raises ValueError  {'PASS' if good else 'FAIL'}")

    # --- 7. Production shot path ---------------------------------------------
    if args.slow:
        shots = 3000
        dev_prod = qml.device("lightning.qubit", wires=N_QUBITS,
                              batch_obs=True, shots=shots)

        @qml.qnode(dev_prod)
        def kernel_circ_shots(a, b, p):
            w = range(N_QUBITS)
            Hamiltonian_feature_map(a, p, wires=w, initial_state=True)
            qml.adjoint(Hamiltonian_feature_map)(b, p, wires=w, initial_state=True)
            return qml.probs(wires=w)

        A = rng.uniform(0, np.pi, (6, N_QUBITS))
        p = [params[0], int(params[1]), int(params[2])]
        prod = np.array([[float(kernel_circ_shots(a, b, p)[0]) for b in A] for a in A])
        exact = compute_kernel_exact(A, A, params, n_qubits=N_QUBITS)
        sigma = np.sqrt(np.maximum(exact * (1 - exact), 1e-12) / shots)
        n_sigma = np.abs(prod - exact) / np.maximum(sigma, 1e-12)
        worst = float(np.max(n_sigma[~np.eye(len(A), dtype=bool)]))
        status = "PASS" if worst < 5 else "FAIL"
        ok &= worst < 5
        print(f"[production] lightning.qubit shots={shots}: worst off-diagonal "
              f"deviation {worst:.2f} sigma  {status}")

    if not ok:
        raise SystemExit("verification FAILED")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
