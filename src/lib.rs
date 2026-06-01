//! # lau-symplectic-geometry
//!
//! Symplectic geometry as the bridge between contact geometry, optimal control,
//! and Hamiltonian mechanics for robotics trajectory planning.
//!
//! This crate provides:
//! - **Symplectic manifolds**: 2n-dimensional manifolds with a non-degenerate closed 2-form ω
//! - **Symplectic matrices**: Sp(2n) group verification (M^T J M = J)
//! - **Hamiltonian vector fields**: X_H = J∇H (symplectic gradient)
//! - **Hamiltonian flows**: Symplectic Euler and Störmer-Verlet integrators
//! - **Poisson brackets**: {f,g} = ω(X_f, X_g) with Jacobi identity
//! - **Cotangent bundles**: T*M as canonical symplectic manifold
//! - **Darboux theorem**: Local coordinates where ω = Σ dx_i ∧ dp_i
//! - **Liouville's theorem**: Phase space volume preservation
//! - **Poincaré recurrence theorem**

pub mod symplectic_form;
pub mod symplectic_matrix;
pub mod hamiltonian;
pub mod poisson;
pub mod cotangent;
pub mod darboux;
pub mod liouville;
pub mod poincare;
pub mod integrators;

pub use symplectic_form::SymplecticForm;
pub use symplectic_matrix::SymplecticMatrix;
pub use hamiltonian::HamiltonianSystem;
pub use poisson::PoissonBracket;
pub use cotangent::CotangentBundle;
pub use darboux::DarbouxCoordinates;
pub use liouville::LiouvilleTheorem;
pub use poincare::PoincareRecurrence;
pub use integrators::{SymplecticEuler, StormerVerlet};
