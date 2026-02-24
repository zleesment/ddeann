"""
Sanity checks for deann.DynamicNaiveKde

Expected API:
  dyn = deann.DynamicNaiveKde(bandwidth=..., kernel="gaussian"|"exponential"|"laplacian")
  dyn.fit(X)                    # X: (n,d) float32/float64 contiguous
  ids = dyn.insert(X_new)        # returns 1D array of int_t ids (len=m)
  dyn.erase(ids_to_delete)       # ids: 1D int_t
  Z = dyn.query(Q)               # Q: (m,d) or (d,) -> returns (m,) float array

"""

import numpy as np
import deann


def kde_reference(X: np.ndarray, Q: np.ndarray, h: float, kernel: str) -> np.ndarray:
    """
    Reference KDE: mean_i K(||q-x_i|| / h) or mean_i exp(-||q-x_i||^2 / h^2) depending on your kernels.
    This matches DEANN's kernels as implemented in your C++ (NaiveKde).

    Notes:
      - "gaussian": exp(-||q-x||^2 / (2 h^2))  (common)
      - "exponential": exp(-||q-x||^2 / (h^2)) (sometimes used)
      - "laplacian": exp(-||q-x||_1 / h)
    Your project uses names EXPONENTIAL / GAUSSIAN / LAPLACIAN. If your exact constants differ,
    adjust below to match. For a sanity check, relative changes with insert/erase still validate logic.
    """
    if Q.ndim == 1:
        Q = Q[None, :]
    # distances
    diffs = Q[:, None, :] - X[None, :, :]
    if kernel == "laplacian":
        dist = np.sum(np.abs(diffs), axis=2)
        K = np.exp(-dist / h)
    elif kernel == "gaussian":
        dist2 = np.sum(diffs * diffs, axis=2)
        K = np.exp(-0.5 * dist2 / (h * h))
    elif kernel == "exponential":
        dist2 = np.sum(diffs * diffs, axis=2)
        K = np.exp(-dist2 / (h * h))
    else:
        raise ValueError(f"unknown kernel: {kernel}")
    return K.mean(axis=1)


def assert_close(name: str, a: np.ndarray, b: np.ndarray, rtol=1e-5, atol=1e-6):
    a = np.asarray(a)
    b = np.asarray(b)
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        max_abs = np.max(np.abs(a - b))
        max_rel = np.max(np.abs(a - b) / (np.abs(b) + 1e-12))
        raise AssertionError(
            f"{name} FAILED\n"
            f"  max_abs={max_abs:.3e}, max_rel={max_rel:.3e}\n"
            f"  a[:5]={a[:5]}\n"
            f"  b[:5]={b[:5]}"
        )
    print(f"{name}: OK")

def run_one(dtype, kernel: str):
    rng = np.random.default_rng(0)
    n, d, m = 50, 4, 20
    h = 0.7

    X0 = np.ascontiguousarray(rng.normal(size=(n, d)).astype(dtype))
    Q  = np.ascontiguousarray(rng.normal(size=(m, d)).astype(dtype))
    Xins = np.ascontiguousarray(rng.normal(size=(10, d)).astype(dtype))

    # Ground truth: existing static NaiveKde
    static = deann.NaiveKde(h, kernel)
    static.fit(X0)
    Z0_static, _ = static.query(Q)

    # Dynamic baseline
    dyn = deann.DynamicNaiveKde(h, kernel)
    dyn.fit(X0)
    Z0_dyn = dyn.query(Q)

    tol_r = 2e-5 if dtype == np.float32 else 1e-7
    tol_a = 2e-6 if dtype == np.float32 else 1e-9

    assert_close(f"{dtype.__name__} {kernel} baseline (dyn vs static)", Z0_dyn, Z0_static,
                 rtol=tol_r, atol=tol_a)

    # Insert
    ids = dyn.insert(Xins)
    X1 = np.ascontiguousarray(np.vstack([X0, Xins]).astype(dtype))

    static1 = deann.NaiveKde(h, kernel)
    static1.fit(X1)
    Z1_static, _ = static1.query(Q)

    Z1_dyn = dyn.query(Q)
    assert_close(f"{dtype.__name__} {kernel} after insert (dyn vs static)", Z1_dyn, Z1_static,
                 rtol=tol_r, atol=tol_a)

    # Erase half of inserted points
    ids_del = np.ascontiguousarray(ids[:5])
    dyn.erase(ids_del)

    X2 = np.ascontiguousarray(np.vstack([X0, Xins[5:]]).astype(dtype))
    static2 = deann.NaiveKde(h, kernel)
    static2.fit(X2)
    Z2_static, _ = static2.query(Q)

    Z2_dyn = dyn.query(Q)
    assert_close(f"{dtype.__name__} {kernel} after erase (dyn vs static)", Z2_dyn, Z2_static,
                 rtol=tol_r, atol=tol_a)

    # Single query shape
    z_single_dyn = np.asarray(dyn.query(Q[0])).reshape(-1)
    z_single_static, _ = static2.query(Q[0])
    z_single_static = np.asarray(z_single_static).reshape(-1)
    assert_close(f"{dtype.__name__} {kernel} single-query (dyn vs static)",
                 z_single_dyn, z_single_static,
                 rtol=tol_r, atol=tol_a)

def main():
    for kernel in ["gaussian", "exponential", "laplacian"]:
        run_one(np.float32, kernel)
        run_one(np.float64, kernel)
    print("\nAll DynamicNaiveKde sanity checks passed (vs deann.NaiveKde ground truth).")

if __name__ == "__main__":
    main()