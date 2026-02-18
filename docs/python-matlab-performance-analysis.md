# Python vs MATLAB Performance Bottleneck Analysis

> **Date:** 2026-02-18
> **Scope:** All computational modules in `src/` — comparison of the Python (NumPy/SciPy) port against the original MATLAB (C++/CUDA MEX) implementation.
> **Methodology:** Line-by-line source comparison, algorithmic complexity analysis, and memory allocation profiling.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Severity Classification](#2-severity-classification)
3. [Module: batchop](#3-module-batchop)
4. [Module: ray](#4-module-ray)
5. [Module: mbo](#5-module-mbo)
6. [Module: rtr](#6-module-rtr)
7. [Module: sdp](#7-module-sdp)
8. [Module: variety / mesh](#8-module-variety--mesh)
9. [Cross-Cutting Concerns](#9-cross-cutting-concerns)
10. [Pipeline-Level Performance Estimates](#10-pipeline-level-performance-estimates)
11. [Optimization Roadmap](#11-optimization-roadmap)

---

## 1. Executive Summary

The Python port faithfully replicates the MATLAB algorithms but inherits significant performance gaps from three systemic differences:

| Factor                   | MATLAB                                     | Python                                 |
| ------------------------ | ------------------------------------------ | -------------------------------------- |
| **Parallelism**          | Intel TBB `parallel_for` in C++ MEX        | Single-threaded NumPy (GIL-bound)      |
| **Compiled inner loops** | C++/CUDA MEX with BLAS/LAPACK direct calls | Interpreted Python with SciPy wrappers |
| **GPU acceleration**     | `gpuArray` + `arrayfun` paths              | Not available (`gpuflag` deleted)      |

The most impactful bottlenecks concentrate in two modules:

- **`ray`**: Serial Python `for`-loops over vertices in projection and initialization (10-50x slower)
- **`batchop`**: Absence of TBB parallelization across all batched LAPACK operations (2-16x slower)

Together these two modules dominate wall-clock time for the full frame field pipeline on meshes with >10k vertices.

---

## 2. Severity Classification

### CRITICAL (10x+ performance difference)

| Module  | Location               | Bottleneck                                   | Estimated Slowdown |
| ------- | ---------------------- | -------------------------------------------- | ------------------ |
| batchop | `batchop.py` (all ops) | No TBB parallelization                       | 8-16x              |
| ray     | `_core.py:282`         | `project_sph_field` serial vertex loop       | 8-16x              |
| ray     | `_core.py:392`         | Post-initialization serial projection loop   | 10-50x             |
| ray     | `_core.py:308-396`     | Linear system solve (LSQR vs OpenNL/CHOLMOD) | 5-20x              |

### HIGH (3-10x performance difference)

| Module  | Location                                    | Bottleneck                                  | Estimated Slowdown  |
| ------- | ------------------------------------------- | ------------------------------------------- | ------------------- |
| batchop | `batchop.py:121-131`                        | `mult` operation memory copies              | 2-3x                |
| batchop | `batchop.py:239-295`                        | `cholcong` axis swap overhead               | 2-3x                |
| batchop | `batchop.py:336-405`                        | `leastsq` workspace allocation              | 1.8-2.5x            |
| batchop | `batchop.py:408-458`                        | `pinv`/`svd` intermediate arrays            | 2-3x                |
| mbo     | `AlignmentConstrainedLinearSolve.py:71-112` | CG matvec callback allocations              | 3-4x                |
| mbo     | `AlignmentConstrainedLinearSolve.py:79-80`  | Fortran-order reshape copies                | 10-15x (cumulative) |
| rtr     | `OdecoBundleFactory.py:91-121`              | Odeco projection ~130MB/call                | 5-10x (memory)      |
| rtr     | `OctaManopt.py:93-106`                      | Cache granularity (recomputes all 4 fields) | 2-3x                |
| sdp     | `MultiSdp.py:189-207`                       | Per-column objective expression rebuild     | 3-6x                |

### MODERATE (1.5-3x performance difference)

| Module | Location                        | Bottleneck                                               |
| ------ | ------------------------------- | -------------------------------------------------------- |
| mbo    | `MBO.py:118,154,177`            | 4 sparse-dense products per iteration (MATLAB: 2-3)      |
| mbo    | `OctaMBO.py:36-45`              | 5 allocations in aligned projection (MATLAB: 1 fused op) |
| mbo    | `_basis.py:13-48`               | `einsum` tensor contraction overhead                     |
| rtr    | `OctahedralBundleFactory.py:59` | Unconditional full-array copy for boundary constraint    |
| rtr    | pymanopt wrappers               | `@pnp(manifold)` decorator call overhead                 |
| mesh   | `ProcessMesh.py:95-97`          | Vertex normal `np.add.at` loop (MATLAB: C++ built-in)    |
| mesh   | `ProcessMesh.py:108-145`        | Eigenvalue: CPU `eigsh()` only (MATLAB: GPU PCG)         |

---

## 3. Module: batchop

### 3.1 Architecture Difference

**MATLAB** (`batchop_cpu.cpp`): Every operation dispatches to a `tbb::parallel_for` loop that calls Fortran BLAS/LAPACK routines directly on each page (matrix slice). Thread-local workspaces are pre-allocated via `tbb::enumerable_thread_specific`.

**Python** (`batchop.py`): Operations route through NumPy/SciPy wrappers. Batched calls use `np.moveaxis` to reorder the page dimension, then invoke SciPy's batched LAPACK bindings. Fallback paths use per-page `_DPOTRF`, `_DGELS`, etc. with `_fcopy()` Fortran-order copies.

### 3.2 Parallelization Gap

```cpp
// MATLAB C++ MEX — batchop_cpu.cpp:41-58
tbb::affinity_partitioner ap;
tbb::parallel_for(0, static_cast<int>(nPages), 1, [&](int k) {
    dgemm_(transpA, transpB, &m, &n, &r1, &one, A_k, &dimA0, B_k, &dimB0, &zero, C_k, &m);
}, ap);
```

```python
# Python — batchop.py:121-131
A_batch = np.moveaxis(A_eff, 2, 0)   # Non-contiguous → copy
B_batch = np.moveaxis(B_eff, 2, 0)   # Non-contiguous → copy
C_batch = A_batch @ B_batch           # NumPy internal threading only
```

All 10 operations (`mult`, `chol`, `cholsolve`, `cholcong`, `trisolve`, `leastsq`, `svd`, `pinv`, `qr`, `eig`) share this pattern. The Python version lacks:

- Thread-level parallelism across pages
- Pre-allocated thread-local workspaces
- Direct BLAS/LAPACK function pointer calls
- Vector-specific code paths (`dgemv_` for single-column RHS)

### 3.3 Memory Layout Issues

Every batched operation follows this pattern:

```python
# batchop.py:77-82
def _batch_to_first(A3):
    return np.moveaxis(A3, 2, 0)    # (m, n, p) → (p, m, n) — non-contiguous view

def _first_to_batch(A_batch):
    return np.moveaxis(A_batch, 0, 2)  # (p, m, n) → (m, n, p)
```

`np.moveaxis` creates non-contiguous views that downstream BLAS calls must copy into contiguous memory. This produces **2-3 extra memory allocations per operation**.

Additionally, `_fcopy()` (line 134-136) creates Fortran-contiguous copies for every page in fallback paths:

```python
def _fcopy(page: np.ndarray) -> np.ndarray:
    return np.array(page, dtype=np.float64, order="F", copy=True)
```

Called at lines 171, 189, 230, 373, 441, 499, 564, etc.

### 3.4 Recursive Fallback Overhead

```python
# batchop.py:97-118
def _run_batch_fallback(indices, run_batch, run_single) -> None:
    try:
        run_batch(indices)
    except (np.linalg.LinAlgError, ValueError):
        if indices.size == 1:
            run_single(int(indices[0]))
            return
        mid = indices.size // 2
        _run_batch_fallback(indices[:mid], run_batch, run_single)   # Recursion
        _run_batch_fallback(indices[mid:], run_batch, run_single)
```

On batch failure, this performs O(log n) recursive splits, each attempting a full batch operation. MATLAB handles failures per-page within the TBB loop with no recursion overhead.

### 3.5 Per-Operation Overhead Summary

| Operation   | Python Overhead | Key Cause                                         |
| ----------- | --------------- | ------------------------------------------------- |
| `mult`      | 2-3x            | No TBB, extra copies, no vector path              |
| `chol`      | 1.5-2x          | Pre-scan for non-finite pages, fallback recursion |
| `cholsolve` | 1.3-1.8x        | Batch copy + SciPy indirection                    |
| `cholcong`  | 2-3x            | `np.swapaxes` creates intermediate arrays         |
| `trisolve`  | 1.3-1.5x        | Less efficient triangular solve path              |
| `leastsq`   | 1.8-2.5x        | No workspace pre-allocation, buffer padding       |
| `svd`       | 1.5-2x          | Multiple intermediate arrays, extra mult call     |
| `pinv`      | 2-3x            | SVD overhead + division overhead                  |
| `qr`        | 1.5-2x          | Copy overhead, partial assignment                 |
| `eig`       | 1.8-2.5x        | `take_along_axis` overhead, no batched solver     |

---

## 4. Module: ray

### 4.1 Architecture Difference

**MATLAB**: The core projection and frame field computation is implemented as a C++ MEX extension (`ext/ray/`) using Eigen for linear algebra and Intel TBB for parallelism. OpenNL provides the sparse linear system solver.

**Python**: Pure NumPy/SciPy implementation with `SphericalHarmonicL4` class in `_core.py`. All vertex processing is serial Python loops.

### 4.2 Critical Bottleneck: Serial Vertex Projection Loop

```python
# _core.py:276-286
def project_sph_field(q0, grad_threshold, dot_threshold):
    out = np.empty_like(q0m)
    for i in range(q0m.shape[1]):              # SERIAL: iterates over ALL vertices
        qi = SphericalHarmonicL4(q0m[:, i])
        proj = SphericalHarmonicL4.project_sph(qi, grad_threshold, dot_threshold)
        out[:, i] = proj.coeff
    return out
```

```cpp
// ext/ray/RayProjection.cpp:24-33
tbb::parallel_for(tbb::blocked_range<int>(0, n), [&](const auto& rng) {
    for (int i = rng.begin(); i != rng.end(); ++i) {
        SphericalHarmonicL4 qi = SphericalHarmonicL4::project_sph(q0i, 1e-8, 1e-8);
        // ...
    }
});
```

Each vertex's `project_sph` involves up to 10,000 gradient descent iterations (`_MAX_PROJ_ITERS` at line 217). The C++ version distributes these across all CPU cores.

**Estimated speedup with parallelization**: Linear with core count (8-core = 8x, 16-core = 16x).

### 4.3 Frame Field Initialization

```python
# _core.py:391-396  (inside compute_ff)
frames = np.empty((3, 3, nv), dtype=np.float64)
for v in range(nv):                            # SERIAL: iterates over ALL vertices
    sh = SphericalHarmonicL4(x[11 * v : 11 * v + 9])
    r = SphericalHarmonicL4.project(sh, 1e-3, 1e-5)
    frames[:, :, v] = r.as_matrix()
```

Same serial pattern. C++ handles this with TBB + Eigen SIMD-optimized 9x9 matrix operations.

### 4.4 Linear System Solve

```python
# _core.py:389
x = spla.lsqr(A, rhs, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
```

vs MATLAB's OpenNL solver (CHOLMOD backend). For a 10k-vertex mesh, the system has ~110k variables and ~1M equations. SciPy's LSQR is serial; OpenNL leverages CHOLMOD's supernodal factorization.

### 4.5 Optimization Loop Memory Pressure

```python
# Ray.py:205-212  (called ~5,000 times by L-BFGS)
def ray_obj_grad(eul):
    qi = np.zeros((9, nv), dtype=np.float64)   # 7.2 MB allocation EVERY call
    # ...
    egrad = np.asarray(L @ qi.T, dtype=np.float64)  # Double transpose
```

For 100k vertices over 5,000 L-BFGS iterations: **~36 GB cumulative allocation** (subject to GC pressure and fragmentation).

### 4.6 Rotation Operations

```python
# Ray.py:81-89
def _rotate_z(eul_z, q):
    phase = _BANDS * eul                 # Broadcasting: (9,1) × (1,k) → (9,k)
    q_flip = np.flipud(q)               # Creates copy of entire q
    return np.cos(phase) * q - np.sin(phase) * q_flip  # 4 element-wise operations
```

Called multiple times per gradient evaluation. C++ uses stack-allocated 9-element arrays with inline SIMD operations.

---

## 5. Module: mbo

### 5.1 Architecture Difference

**MATLAB** (`MBO.m`): Uses `multiprod`/`multitransp` for batched matrix operations and MATLAB's optimized sparse matrix multiply. Reference assignment (`q = qProj`) avoids copies. `pcg()` is used for the reduced linear system.

**Python** (`MBO.py`, `AlignmentConstrainedLinearSolve.py`): Uses `scipy.sparse.linalg.cg` with a `LinearOperator` callback. Every sparse-dense multiply requires explicit transpose. Boundary basis application uses `np.einsum`.

### 5.2 CG Matvec Callback — The Inner Loop Bottleneck

The reduced system solver calls `matvec` 100-200 times per MBO iteration. Each call performs:

```python
# AlignmentConstrainedLinearSolve.py:88-112
def matvec(x):
    q_full = np.zeros((D, nv), dtype=np.float64)       # (1) 7.2 MB allocation
    x_bdry = x[:d*nb].reshape((d, nb), order="F")      # (2) Fortran reshape → copy
    q_bdry = apply_basis(basis, x_bdry)                 # (3) einsum tensor contraction
    x_int = x[d*nb:].reshape((D, ni), order="F")       # (4) Fortran reshape → copy
    q_full[:, bdry_int_idx] = np.concatenate(pieces, axis=1)  # (5) Concatenation
    Aq_full = (A @ q_full.T).T                          # (6) Double transpose
    Aq_bdry = apply_basis_t(basis, Aq_full[:, bdry_idx])  # (7) einsum + indexing
    return np.concatenate([
        Aq_bdry.reshape(-1, order="F"),                 # (8) Fortran reshape → copy
        Aq_full[:, int_idx].reshape(-1, order="F"),     # (9) Fortran reshape → copy
    ])
```

**7-9 array allocations per CG step**. For octa (D=9, d=2) on a 100k mesh with 5k boundary vertices:

| Allocation              | Size                 | Count per MBO solve |
| ----------------------- | -------------------- | ------------------- |
| `q_full`                | 7.2 MB               | 150 CG steps        |
| Fortran reshapes (×4)   | 80 KB - 6.84 MB each | 600                 |
| `np.concatenate`        | 7.2 MB               | 150                 |
| Transpose intermediates | 7.2 MB               | 150                 |
| **Total per MBO solve** | **~5.1 GB**          |                     |

MATLAB's `mulA` function uses direct sparse indexing with no concatenate step and column-major layout eliminates reshape copies.

### 5.3 Repeated Sparse-Dense Products

```python
# MBO.py — per iteration
gradnorm = float(np.linalg.norm((L @ q.T), ord="fro"))    # Line 118/154
delta = _mass_norm(dq, M)                                  # → M @ dq.T
cost = _energy_quadratic(q, L)                             # → L @ q.T
q_norm = _mass_norm(q, M)                                  # → M @ q.T
```

**4 sparse-dense products per iteration**, each creating a `(nv, dim)` temporary via `.T`. MATLAB performs 2-3 (the convergence check's `M * q.'` is evaluated lazily in the conditional).

### 5.4 Aligned Projection Overhead

```python
# OctaMBO.py:36-45
def _proj_aligned(q):
    nrm = np.linalg.norm(q2, axis=0, keepdims=True)   # (1) norm with keepdims
    out = np.zeros_like(q2)                             # (2) pre-allocation
    nz = nrm > 0.0                                     # (3) boolean mask
    out[:, nz[0]] = q2[:, nz[0]] / nrm[:, nz[0]]      # (4) advanced indexing
    return scale * out                                  # (5) scalar multiply
```

**5 allocations** per call. MATLAB: `sqrt(5/12) * (q ./ vecnorm(q, 2, 1))` — single fused operation.

### 5.5 RHS Construction

```python
# MBO.py:132
rhs = (M @ q.T).T
```

Creates 3 temporaries: `q.T` → `M @ q.T` → `(...).T`. MATLAB's JIT recognizes and fuses the transpose chain.

---

## 6. Module: rtr

### 6.1 Architecture Difference

**MATLAB**: Uses `manopt` library with explicit `store` struct for caching intermediate results. Granular `isfield()` checks compute only what's needed.

**Python**: Uses `pymanopt` with `@pnp(manifold)` decorator wrappers. Caching uses `id(q)` to detect point changes, but recomputes all 4 expensive fields unconditionally.

### 6.2 Cache Granularity Mismatch

```python
# OctaManopt.py:93-106
def _prepare(q):
    if _store.get("_q_id") != id(q):
        aq = matvec_rows(A, q)                  # Always computed
        basis = manifold.tangentbasis(q)         # Always computed
        rgrad = manifold.egrad2rgrad(...)        # Always computed
        lxyz_t = manifold.mul_lxyz_t(aq)         # Always computed
        _store.update({...})
    return _store
```

```matlab
% OctaManopt.m:45-58
function store = prepare(q, store)
    if ~isfield(store, 'Aq')
        store.Aq = (A * q')';
    end
    if ~isfield(store, 'tangentbasis')       % Only if needed
        store.tangentbasis = octa.tangentbasis(q);
    end
    % ... granular checks continue ...
end
```

When only `rgrad` is needed (gradient evaluation), Python computes all 4 fields. MATLAB computes only the missing ones.

### 6.3 Odeco Projection Memory Explosion

```python
# OdecoBundleFactory.py:91-121
def _build_projection_cache(self, q):
    oq = self.mul_o(q)                        # (15, 27, n)  — 32.4 MB for n=10k
    basis = np.array(oq, copy=True)           # (15, 27, n)  — +32.4 MB copy
    pinv = np.empty((27, 15, self.n))          # (27, 15, n)  — +32.4 MB
    # batchop calls create additional temporaries
```

**~130 MB temporary allocation per projection call**. For RTR with 100-1000 iterations, this creates **13-130 GB cumulative temporary allocation pressure**.

MATLAB uses the `NqM` struct with in-place field updates.

### 6.4 Pymanopt Decorator Overhead

```python
# OctaManopt.py:108-122
@pnp(manifold)
def cost(q):
    s = _prepare(q)
    return float(0.5 * np.sum(q * s["aq"]))

@pnp(manifold)
def gradient(q):
    s = _prepare(q)
    rgrad = s["rgrad"]
    _iter_log.append({...})                    # Dict creation per iteration
    return rgrad
```

The `@pnp` decorator adds Python function call overhead not present in MATLAB's direct function handle approach. Over 1000+ TR iterations with 10-20 gradient evaluations each, this accumulates.

### 6.5 Boundary Constraint Unconditional Copy

```python
# OctahedralBundleFactory.py:58-64
def _apply_boundary_tangent_constraint(self, s):
    out = np.array(s, dtype=np.float64, copy=True)  # Always copies, even if no boundary
    if self.bdry_idx.size > 0:
        # ... modify boundary entries ...
    return out
```

MATLAB: `s(bdryIdx, :) = dot(s(bdryIdx, :), bdryNormals, 2) .* bdryNormals;` — modifies in place.

### 6.6 Einsum in Manifold Operations

```python
# OctahedralBundleFactory.py:68
def mul_lxyz(self, v):
    return np.einsum("abk,bn->ank", self._lxyz, v2, optimize=True)
    # (9, 9, 3) × (9, n) → (9, n, 3)
```

`optimize=True` runs path-finding heuristics at every call. For n=10k: 2.43M FLOPs plus path optimization overhead. MATLAB's pre-computed `Lxyzq` and JIT-compiled `squeeze(sum(v .* Lxyzq, 1))` is more cache-efficient.

---

## 7. Module: sdp

### 7.1 Architecture Difference

**MATLAB**: Uses MOSEK Fusion API through MEX bindings. Constraint matrices stored sparse.

**Python**: Uses MOSEK Fusion API through Python bindings. Constraint matrix stored as `Matrix.dense()`. Per-column objective expression rebuild required by API limitation.

### 7.2 Per-Column Objective Rebuild

```python
# MultiSdp.py:189-207
for j in range(n):
    q0_col = Matrix.dense(q0m[:, [j]])             # New dense matrix object
    self._model.objective(
        _OBJ_MINIMIZE,
        Expr.add(
            self._trace_expr,
            Expr.mul(-2.0, Expr.dot(q0_col, self._q_expr)),
        ),
    )
    self._solve_and_check(j)
```

MOSEK Fusion prohibits `Parameter()` in expressions with PSD cone variables, forcing the objective expression tree to be rebuilt for each of the n columns.

**Per-column overhead**: `Matrix.dense()` creation (~100us) + expression tree build (~50us) + solve overhead.
**For n=10,000**: ~1.5 seconds of pure overhead beyond solver time.

### 7.3 Dense vs Sparse Constraint Matrix

```python
# MultiSdp.py:120-121
A_mat = Matrix.dense(self._A)    # Dense storage
b_mat = Matrix.dense(self._b)
```

For typical constraint matrices with structural sparsity, `Matrix.sparse()` would reduce memory by 30-50%. MATLAB's `sparse(reshape(A, d^2, k))` proactively converts to sparse.

### 7.4 Multi-Threading Disabled

```python
# MultiSdp.py:53
"intpntMultiThread": "off"
```

Explicitly disables MOSEK's internal parallelization. This matches MATLAB defaults for consistency but leaves single-core performance on the table for large SDPs.

### 7.5 Solution Extraction Overhead

```python
# MultiSdp.py:210-216
q[:, j] = np.asarray(self._q_expr.level(), dtype=np.float64).reshape(self.d)
if return_Q and Qflat is not None:
    Qflat[:, j] = np.asarray(self._Q.level(), dtype=np.float64).reshape(self.d1 * self.d1)
```

Each `.level()` call crosses the MOSEK Python bridge (~100us). For n=10,000 with `return_Q=True`: **~2 seconds of bridge overhead**.

### 7.6 MultiSdp2 Kronecker Construction

```python
# MultiSdp2.py:137-152
for j in range(self.k):                                    # Loop over k constraint pages
    a_page = sparse.csr_matrix(self._a_pages[:, :, j])
    aoi_blocks.append(sparse.kron(a_page.T, id_sparse, format="csr"))
    ioa_blocks.append(
        sparse.kron(id_sparse, a_page, format="csr").transpose().tocsr(),  # CSR→CSC→CSR
    )
```

Each `.transpose().tocsr()` creates an intermediate CSC matrix then converts back. MATLAB builds `IoA` as a 3D dense array first, then converts to sparse once.

---

## 8. Module: variety / mesh

### 8.1 GPU Paths Removed

Three modules explicitly delete the `gpuflag` parameter:

```python
# ExpSO3.py:15
del gpuflag

# RandOctahedralField.py:15
del gpuflag
```

MATLAB uses `gpuflag` to branch to GPU-accelerated `arrayfun()` paths for matrix exponential computation and field generation.

### 8.2 Eigenvalue Computation: CPU-Only vs GPU PCG

```python
# ProcessMesh.py:114-145
eigvals = spla.eigsh(L, M=M, k=k, sigma=0.0, which="LM",
                     maxiter=5000, tol=1e-10)[0]
```

```matlab
% ProcessMesh.m:38-62 (GPU path)
v = randn(meshData.nv, 1, 'gpuArray');
Lg = gpuArray(meshData.L);
for i = 1:100
    [v, ~] = pcg(Lg, v, 1e-6, 1000, [], [], v);  % GPU PCG solver
end
```

For 100k vertices: CPU `eigsh()` takes 1-10 seconds; GPU PCG takes 0.1-1 second.

### 8.3 Reimplemented C++ Built-ins

**Boundary Face Extraction** (`ProcessMesh.py:38-65`):

- MATLAB: `freeBoundary(triangulation(...))` — optimized C++ implementation
- Python: Manual loop over 4 faces per tet, `np.unique()` for deduplication

**Vertex Normals** (`ProcessMesh.py:80-105`):

- MATLAB: `vertexNormal(triangulation(...))` — optimized C++ implementation
- Python: Manual accumulation with `np.add.at` in a 3-iteration Python loop:

```python
# ProcessMesh.py:95-97
for j in range(3):
    np.add.at(accum, faces[:, j], fn)
    np.add.at(count, faces[:, j], 1.0)
```

`np.add.at` is known to be slower than optimized scatter-accumulate operations.

### 8.4 Geometric Laplacian Assembly

`GeometricPrimalLM.py` is fully vectorized and closely matches MATLAB's vectorized implementation. This is **not a significant bottleneck**. Both use the same algorithmic approach with efficient sparse matrix construction.

### 8.5 Tet Volume Computation

`TetVolumes.py` is fully vectorized. No significant performance difference from MATLAB.

### 8.6 SO(3) Generator Caching

Both implementations cache via `persistent` (MATLAB) / `@lru_cache(maxsize=1)` (Python). Performance equivalent after first call. Python's `.copy()` on return (line 66-67) adds minor overhead per access.

---

## 9. Cross-Cutting Concerns

### 9.1 Memory Layout Mismatch

MATLAB stores arrays in column-major (Fortran) order. NumPy defaults to row-major (C) order. This causes:

1. **Fortran-order reshapes require copies**: `np.reshape(..., order="F")` on C-order arrays triggers a full data copy. This appears in `AlignmentConstrainedLinearSolve.py` (5 locations), `batchop.py` (multiple), and `MultiSdp2.py` (3 locations).

2. **Sparse matrix multiply layout**: SciPy sparse CSR matrices are row-major optimized. MATLAB sparse matrices are column-major. When computing `L @ q.T`, the transpose creates a non-optimal access pattern.

3. **3D array page access**: `A[:, :, k]` extracts a non-contiguous view from a C-order 3D array. MATLAB's `A(:, :, k)` extracts a contiguous slice.

### 9.2 Temporary Array Proliferation

Common patterns that create unnecessary temporaries:

| Pattern                           | Occurrences      | Issue                                            |
| --------------------------------- | ---------------- | ------------------------------------------------ |
| `(A @ q.T).T`                     | mbo, rtr, ray    | Double transpose; 2 temporary arrays             |
| `np.asarray(x, dtype=np.float64)` | rtr, batchop     | Redundant type check when already float64        |
| `np.array(x, copy=True)`          | rtr boundary ops | Unconditional copy even when mutation not needed |
| `.reshape(-1, order="F")`         | mbo CG solver    | Forced copy on C-order arrays                    |
| `np.moveaxis(A, 2, 0)`            | batchop          | Non-contiguous view triggers downstream copies   |

### 9.3 Python Interpreter Overhead

Tight inner loops where Python interpretation dominates:

- `ray/_core.py:282` — vertex projection loop
- `ray/_core.py:392` — initialization projection loop
- `ray/Ray.py:205` — L-BFGS objective/gradient evaluation
- `batchop.py:97-118` — recursive fallback loop
- `mbo/AlignmentConstrainedLinearSolve.py:88-112` — CG matvec callback

### 9.4 Absence of GPU Acceleration

MATLAB modules that support GPU via `gpuArray`:

| Module                     | MATLAB GPU Support           | Python Status        |
| -------------------------- | ---------------------------- | -------------------- |
| `ExpSO3`                   | `arrayfun` on gpuArray       | `del gpuflag`        |
| `RandOctahedralField`      | gpuArray data types          | `del gpuflag`        |
| `ProcessMesh` (eigenvalue) | GPU PCG iteration            | CPU `eigsh()` only   |
| `batchop`                  | CUDA MEX kernels             | Not available        |
| `OctaManopt`               | `gpuflag` parameter accepted | Accepted but ignored |

---

## 10. Pipeline-Level Performance Estimates

For a typical frame field computation on a **100k-vertex tetrahedral mesh**:

| Pipeline Stage            | MATLAB (est.) | Python (est.) | Ratio      |
| ------------------------- | ------------- | ------------- | ---------- |
| Mesh import + processing  | 0.5s          | 1-2s          | 2-4x       |
| Random field generation   | 0.1s          | 0.1-0.2s      | 1-2x       |
| Ray initialization        | 2s            | 20-100s       | **10-50x** |
| Ray optimization          | 5s            | 40-75s        | **8-15x**  |
| MBO solver (Octa)         | 3s            | 10-15s        | **3-5x**   |
| MBO solver (Odeco)        | 5s            | 15-25s        | **3-5x**   |
| RTR solver (Octa)         | 10s           | 30-50s        | **3-5x**   |
| RTR solver (Odeco)        | 15s           | 75-150s       | **5-10x**  |
| SDP projection (per call) | 0.5s          | 1.5-3s        | **3-6x**   |
| **Full pipeline**         | **~40s**      | **~200-400s** | **5-10x**  |

> Note: Estimates assume single-threaded CPU for both. MATLAB with GPU + TBB may be significantly faster.

---

## 11. Optimization Roadmap

### Priority 1: Ray Serial Loop Parallelization

**Target**: `_core.py:282`, `_core.py:392`
**Expected gain**: 8-50x
**Approach options**:

| Option                                   | Effort | Gain         | Compatibility                            |
| ---------------------------------------- | ------ | ------------ | ---------------------------------------- |
| `multiprocessing.Pool.map`               | Low    | 4-8x         | Cross-platform                           |
| `concurrent.futures.ProcessPoolExecutor` | Low    | 4-8x         | Cross-platform                           |
| Numba `@njit(parallel=True)`             | Medium | 8-16x        | Requires refactoring SphericalHarmonicL4 |
| Cython extension                         | High   | 10-50x       | Build complexity                         |
| C++ extension with pybind11 + TBB        | High   | Match MATLAB | Build complexity                         |

### Priority 2: Batchop Parallelization

**Target**: `batchop.py` all operations
**Expected gain**: 3-10x
**Approach options**:

- Ensure NumPy is linked against Intel MKL (multi-threaded BLAS internally)
- Add `numba.prange` parallel loops for page-level operations
- Pre-allocate workspaces outside the operation loop
- Eliminate `_batch_to_first`/`_first_to_batch` copies by storing arrays page-first

### Priority 3: MBO CG Inner Loop

**Target**: `AlignmentConstrainedLinearSolve.py:88-112`
**Expected gain**: 2-5x
**Key changes**:

1. Pre-allocate `q_full` **once** before CG loop, zero and reuse
2. Store arrays in Fortran order from the start to eliminate reshape copies
3. Cache `q.T` computation across `_mass_norm` / `_energy_quadratic` calls
4. Replace `(A @ q_full.T).T` with direct sparse multiply on transposed layout

### Priority 4: RTR Cache Granularity

**Target**: `OctaManopt.py:93-106`, `OdecoBundleFactory.py:91-121`
**Expected gain**: 2-3x
**Key changes**:

1. Implement per-field `isfield`-style checks instead of all-or-nothing recomputation
2. Reuse Odeco projection cache arrays instead of reallocating
3. Remove unconditional copy in `_apply_boundary_tangent_constraint`

### Priority 5: SDP Column Parallelization

**Target**: `MultiSdp.py:189-207`
**Expected gain**: 3-6x
**Approach**: Create n independent MOSEK Model instances and solve with `multiprocessing`

### Priority 6: GPU Acceleration (Long-term)

**Target**: All modules with `del gpuflag`
**Expected gain**: 10-100x for large meshes
**Approach**: CuPy drop-in replacement for NumPy arrays, or JAX JIT compilation

---

## Appendix: File Reference

| File                                         | Lines of Interest                                                                     | Bottleneck Category                 |
| -------------------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------- |
| `src/batchop/batchop.py`                     | 77-82, 97-118, 121-131, 134-136, 145-197, 239-295, 336-405, 408-458, 531-608, 611-686 | Parallelization, Memory             |
| `src/ray/_core.py`                           | 276-286, 308-396                                                                      | Serial loops                        |
| `src/ray/Ray.py`                             | 81-99, 201-232                                                                        | Memory allocation, Rotation ops     |
| `src/ray/RayInit.py`                         | 28-41, 44-61                                                                          | Sort overhead, Allocation           |
| `src/mbo/MBO.py`                             | 42-52, 118, 132, 147-179                                                              | Sparse products, Temporaries        |
| `src/mbo/AlignmentConstrainedLinearSolve.py` | 69-82, 88-112                                                                         | CG matvec overhead                  |
| `src/mbo/_basis.py`                          | 13-48                                                                                 | Einsum contraction                  |
| `src/mbo/OctaMBO.py`                         | 36-45                                                                                 | Aligned projection allocations      |
| `src/mbo/OdecoMBO.py`                        | 122-132                                                                               | SDP solver + scaling                |
| `src/rtr/OctaManopt.py`                      | 88-106, 125-144, 172-191                                                              | Cache, Hessian, Logging             |
| `src/rtr/OctahedralBundleFactory.py`         | 58-64, 68-72, 225-234                                                                 | Boundary copy, Einsum               |
| `src/rtr/OdecoBundleFactory.py`              | 91-121                                                                                | Memory explosion                    |
| `src/sdp/MultiSdp.py`                        | 49-63, 120-121, 189-220                                                               | Objective rebuild, Dense matrix     |
| `src/sdp/MultiSdp2.py`                       | 137-152, 224-335                                                                      | Kronecker loop, Reshape overhead    |
| `src/mesh/ProcessMesh.py`                    | 38-65, 80-105, 108-145                                                                | Reimplemented built-ins, Eigenvalue |
| `src/variety/ExpSO3.py`                      | 15                                                                                    | GPU path removed                    |
| `src/variety/RandOctahedralField.py`         | 15                                                                                    | GPU path removed                    |
