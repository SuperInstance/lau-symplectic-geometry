# lau-symplectic-geometry

A symplectic geometry library bridging contact geometry, Hamiltonian mechanics, and optimal control for robotics trajectory planning — implemented in Rust.

Symplectic manifolds, symplectic matrices, Hamiltonian vector fields, symplectic integrators, Poisson brackets, cotangent bundles, Darboux coordinates, Liouville's theorem, and Poincaré recurrence.

---

## What This Does

This crate provides the geometric foundations for Hamiltonian mechanics and conservative dynamical systems:

- **Symplectic forms** — Non-degenerate closed 2-forms ω on 2n-dimensional manifolds, with subspace classification (isotropic, Lagrangian, symplectic)
- **Symplectic matrices** — The group Sp(2n) of matrices satisfying M^T J M = J, with rotations, shears, inverses
- **Hamiltonian systems** — Energy functions H(q,p) with symplectic gradient vector fields X_H = J∇H
- **Symplectic integrators** — Structure-preserving numerical methods: Symplectic Euler (1st order) and Störmer-Verlet (2nd order, time-reversible)
- **Poisson brackets** — The bracket {f,g} = ω(X_f, X_g) with verification of skew-symmetry, bilinearity, Jacobi identity, and Leibniz rule
- **Cotangent bundles** — T*M as the canonical symplectic manifold, with tautological 1-form, vertical lifts, and projection maps
- **Darboux coordinates** — Local coordinates where ω = Σ dx_i ∧ dp_i (the Darboux theorem: no local invariants)
- **Liouville's theorem** — Phase space volume preservation under Hamiltonian flow
- **Poincaré recurrence** — Almost every point in a bounded Hamiltonian system returns arbitrarily close to its initial state

## Key Idea

Hamiltonian mechanics lives on **symplectic manifolds** — even-dimensional spaces equipped with a non-degenerate closed 2-form ω. This structure is what makes energy conservation work and why certain numerical methods (symplectic integrators) preserve the qualitative behavior of dynamical systems forever.

The canonical example is the cotangent bundle T*M of a configuration manifold M, with its canonical symplectic form ω = -dθ where θ is the tautological 1-form. In local coordinates: ω = Σ dx_i ∧ dp_i.

The fundamental insight: **symplectic geometry is the geometry of conservative dynamics**. Every Hamiltonian system is a symplectic manifold, every symplectic transformation preserves the structure, and symplectic integrators preserve this structure numerically.

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-symplectic-geometry = "0.1.0"
```

Requires Rust 2021 edition or later.

## Quick Start

```rust
use lau_symplectic_geometry::*;

// Create a Hamiltonian system (harmonic oscillator: H = (q² + p²)/2)
let sys = HamiltonianSystem::harmonic_oscillator();
let q = DVector::from_vec(vec![1.0]);
let p = DVector::from_vec(vec![0.0]);
assert_eq!(sys.energy(&q, &p), 0.5);

// Hamiltonian vector field: X_H = (∂H/∂p, -∂H/∂q)
let vf = sys.vector_field(&q, &p);
// At (q=1, p=0): dq/dt = 0, dp/dt = -1

// Integrate with Störmer-Verlet (2nd order, symplectic, time-reversible)
let verlet = StormerVerlet::new(0.01);
let trajectory = verlet.integrate(&q, &p, &sys, 10000);
let energy_drift = sys.energy_drift(&trajectory);
assert!(energy_drift < 0.01); // Bounded energy error!

// Symplectic matrices: rotations and shears
let rot = SymplecticMatrix::plane_rotation(2, 0, 0.7);
let shear = SymplecticMatrix::shear_x(2, 1, 1.5);
let product = rot.multiply(&shear)?;
assert!(product.verify_symplectic());

// Poisson brackets
let pb = PoissonBracket::new(2);
let grad_f = DVector::from_vec(vec![1.0, 0.0, 0.0, 1.0]);
let grad_g = DVector::from_vec(vec![0.0, 1.0, 1.0, 0.0]);
let bracket = pb.bracket(&grad_f, &grad_g);
assert!(pb.verify_skew_symmetry(&grad_f, &grad_g));

// Darboux coordinates
let omega = SymplecticForm::canonical(3);
let darboux = DarbouxCoordinates::from_symplectic_form(&omega)?;
assert!(darboux.verify(&omega));
```

## API Reference

### `SymplecticForm`
| Method | Description |
|--------|-------------|
| `canonical(n)` | The standard form Ω = [0 I; -I 0] on R^{2n} |
| `from_matrix(m)` | Construct from a skew-symmetric non-degenerate matrix |
| `apply(u, v)` | Evaluate ω(u, v) = u^T Ω v |
| `is_closed()` | Check dω = 0 |
| `is_isotropic(basis)` | Check if a subspace is isotropic (ω|_L = 0) |
| `is_lagrangian(basis)` | Check if a subspace is Lagrangian (maximal isotropic) |
| `is_symplectic_subspace(basis)` | Check if ω restricted to subspace is non-degenerate |

### `SymplecticMatrix`
| Method | Description |
|--------|-------------|
| `canonical_j(n)` | The matrix J = [0 I; -I 0] |
| `identity(n)` | Identity in Sp(2n) |
| `from_matrix(m)` | Construct verifying M^T J M = J |
| `plane_rotation(n, i, θ)` | Rotation in the (q_i, p_i) plane |
| `shear_x(n, i, a)` | Shear: (q_i, p_i) → (q_i + a·p_i, p_i) |
| `shear_p(n, i, b)` | Shear: (q_i, p_i) → (q_i, p_i + b·q_i) |
| `multiply(&other)` | Product of two symplectic matrices |
| `inverse()` | M^{-1} = -J M^T J |
| `determinant()` | Always 1 for M ∈ Sp(2n) |

### `HamiltonianSystem`
| Method | Description |
|--------|-------------|
| `new(name, n, H, ∇H)` | Create from energy function and gradient |
| `harmonic_oscillator()` | H = (q² + p²)/2 |
| `pendulum()` | H = p²/2 + (1 - cos(q)) |
| `kepler_2d()` | Kepler two-body problem |
| `energy(q, p)` | Evaluate H(q, p) |
| `vector_field(q, p)` | X_H = J∇H |
| `energy_drift(traj)` | Maximum energy deviation along trajectory |

### `SymplecticEuler` (1st order)
| Method | Description |
|--------|-------------|
| `new(dt)` | Create with time step |
| `step_separable(q, p, ∇V, ∇T)` | Step for separable H = T(p) + V(q) |
| `step(q, p, system)` | Step for general Hamiltonian |
| `integrate(q0, p0, sys, n)` | Run n steps |

### `StormerVerlet` (2nd order)
| Method | Description |
|--------|-------------|
| `new(dt)` | Create with time step |
| `step_separable(q, p, ∇V, ∇T)` | Leapfrog step for separable H |
| `step(q, p, system)` | Step for general Hamiltonian |
| `integrate(q0, p0, sys, n)` | Run n steps |
| `verify_time_reversibility(q, p, sys)` | Check time-reversal symmetry |

### `PoissonBracket`
| Method | Description |
|--------|-------------|
| `new(n)` | Poisson bracket on R^{2n} |
| `bracket(∇f, ∇g)` | Compute {f, g} |
| `verify_skew_symmetry(...)` | {f,g} = -{g,f} |
| `verify_bilinearity(...)` | Linearity in both arguments |
| `verify_jacobi(...)` | {f,{g,h}} + cyclic = 0 |
| `verify_leibniz(...)` | {f, gh} = g{f,h} + h{f,g} |

### `CotangentBundle`
| Method | Description |
|--------|-------------|
| `new(n)` | Cotangent bundle of R^n |
| `euclidean(n)` | With Euclidean metric |
| `canonical_symplectic_form()` | The canonical ω on T*M |
| `tautological_form(p, v)` | The Liouville 1-form θ |
| `free_hamiltonian()` | Free particle H = p^T g^{-1} p / 2 |
| `vertical_lift(α)` | Lift covector to vertical vector |
| `projection(qp)` | Canonical projection π: T*M → M |
| `is_vertical(v)` | Check if tangent vector is vertical |

### `DarbouxCoordinates`
| Method | Description |
|--------|-------------|
| `standard(n)` | Identity transformation (already Darboux) |
| `from_symplectic_form(ω)` | Find Darboux coordinates for arbitrary ω |
| `to_darboux(point)` | Transform to Darboux coordinates |
| `from_darboux(point)` | Transform back to original coordinates |
| `verify(ω)` | Check S^T Ω S = J |

### `LiouvilleTheorem`
| Method | Description |
|--------|-------------|
| `verify_volume_preservation(J)` | Check det(J) = 1 |
| `phase_space_volume(ω)` | Compute Ω^n volume form |
| `parallelepiped_volume(vectors)` | Volume of a parallelepiped |
| `verify_along_flow(Js)` | Check volume preservation along flow |

### `PoincareRecurrence`
| Method | Description |
|--------|-------------|
| `new(tolerance)` | Create with return tolerance |
| `find_first_return(traj)` | First index returning near start |
| `find_all_returns(traj)` | All recurrence indices |
| `estimate_recurrence_time(traj, dt)` | Average recurrence interval |
| `poincare_section(traj, dt, idx, val)` | Compute Poincaré section crossings |

## How It Works

The crate builds up from abstract structure to numerical methods:

1. **Symplectic forms** (`symplectic_form.rs`): A 2-form ω is represented as a 2n×2n skew-symmetric matrix. The canonical form has the block structure `[0 I; -I 0]`. Non-degeneracy is checked via determinant, and subspaces are classified as isotropic (ω vanishes), Lagrangian (maximal isotropic, dim n), or symplectic (ω|_S is non-degenerate).

2. **Symplectic matrices** (`symplectic_matrix.rs`): Matrices in Sp(2n) satisfy M^T J M = J. These form a group under multiplication. The inverse is `M^{-1} = -J M^T J`, and the determinant is always 1. Generators include plane rotations and shears.

3. **Hamiltonian systems** (`hamiltonian.rs`): A Hamiltonian H : R^{2n} → R generates the vector field X_H = J∇H, which gives Hamilton's equations: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q. Pre-built systems include the harmonic oscillator, pendulum, and Kepler problem.

4. **Integrators** (`integrators.rs`): 
   - **Symplectic Euler** updates momentum first, then position using the new momentum. It's first-order but exactly preserves a modified Hamiltonian.
   - **Störmer-Verlet** (leapfrog) does a half momentum step, full position step, half momentum step. It's second-order, symplectic, time-reversible, and has excellent long-term energy conservation.

5. **Poisson brackets** (`poisson.rs`): The bracket {f,g} = Σ(∂f/∂q_i · ∂g/∂p_i - ∂f/∂p_i · ∂g/∂q_i) satisfies skew-symmetry, bilinearity, the Jacobi identity, and the Leibniz rule. Jacobi identity is verified numerically using provided Hessian matrices.

6. **Cotangent bundles** (`cotangent.rs`): T*M with its tautological 1-form θ (the Liouville form) and canonical symplectic form ω = -dθ. Supports vertical lifts, projections, and the free Hamiltonian.

7. **Darboux coordinates** (`darboux.rs`): The Darboux theorem states that every symplectic form locally looks like the canonical one. The implementation finds a symplectic basis using a Gram-Schmidt-like procedure, constructing a transformation S such that S^T Ω S = J.

8. **Liouville's theorem** (`liouville.rs`): Hamiltonian flows preserve phase space volume (det of the flow Jacobian = 1). The module verifies this numerically and computes symplectic capacities.

9. **Poincaré recurrence** (`poincare.rs`): In bounded phase space regions, trajectories return arbitrarily close to their starting points infinitely often. The module detects returns, estimates recurrence times, and computes Poincaré sections.

## The Math

**Symplectic geometry** studies symplectic manifolds: pairs (M, ω) where M is a 2n-dimensional manifold and ω is a closed, non-degenerate 2-form.

Key structures:

| Concept | Formula | Meaning |
|---------|---------|---------|
| Symplectic form | dω = 0, ω^n ≠ 0 | Closed and non-degenerate |
| Hamilton's equations | q̇ = ∂H/∂p, ṗ = -∂H/∂q | Time evolution |
| Poisson bracket | {f,g} = ω(X_f, X_g) | Algebra of observables |
| Canonical form | ω = Σ dx_i ∧ dp_i | Darboux coordinates |
| Liouville | det(Φ_t) = 1 | Volume preservation |
| Symplectic matrix | M^T J M = J | Linear symplectomorphism |
| Lagrangian subspace | ω|_L = 0, dim L = n | Maximal isotropic |

The **symplectic gradient** X_H = J∇H is orthogonal to the usual gradient: it flows along level sets of H rather than climbing them. This is why energy is conserved.

**Symplectic integrators** preserve the symplectic form exactly (up to floating point), which means they conserve a modified Hamiltonian H̃ ≈ H + O(dt^p). The energy error oscillates but never grows secularly — even over millions of time steps.

**Darboux's theorem** says symplectic manifolds have no local invariants (unlike Riemannian geometry with curvature). All symplectic manifolds of the same dimension look the same locally.

## Tests

The crate contains **80 unit tests** covering:
- Symplectic form construction, skew-symmetry, non-degeneracy
- Subspace classification (isotropic, Lagrangian, symplectic)
- Symplectic matrix operations (product, inverse, transpose, generators)
- Hamiltonian energy evaluation and vector fields
- Symplectic Euler bounded energy drift
- Störmer-Verlet energy conservation, time-reversibility, period approximation
- Poisson bracket properties (skew-symmetry, bilinearity, Jacobi, Leibniz)
- Cotangent bundle operations (tautological form, vertical lift, projection)
- Darboux coordinate transformations
- Liouville volume preservation
- Poincaré recurrence detection and estimation

Run with:

```bash
cargo test
```

## License

MIT
