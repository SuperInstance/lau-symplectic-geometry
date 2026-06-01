//! Cotangent bundles T*M as canonical symplectic manifolds.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::hamiltonian::HamiltonianSystem;
use crate::symplectic_form::SymplecticForm;

/// A cotangent bundle T*M, the canonical example of a symplectic manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CotangentBundle {
    /// Dimension of the base manifold M
    pub base_dim: usize,
    /// Optional metric tensor on the base manifold (n × n)
    pub metric: Option<DMatrix<f64>>,
    /// Name of the configuration manifold
    pub name: String,
}

impl CotangentBundle {
    /// Create a cotangent bundle for an n-dimensional base manifold.
    pub fn new(n: usize) -> Self {
        Self {
            base_dim: n,
            metric: None,
            name: format!("T*R^{}", n),
        }
    }

    /// Create the cotangent bundle of R^n with Euclidean metric.
    pub fn euclidean(n: usize) -> Self {
        Self {
            base_dim: n,
            metric: Some(DMatrix::identity(n, n)),
            name: format!("T*R^{}", n),
        }
    }

    /// Get the total dimension of the phase space (2n).
    pub fn total_dim(&self) -> usize {
        2 * self.base_dim
    }

    /// Get the canonical symplectic form on T*M.
    pub fn canonical_symplectic_form(&self) -> SymplecticForm {
        SymplecticForm::canonical(self.base_dim)
    }

    /// Evaluate the tautological (Liouville) 1-form θ at a point.
    pub fn tautological_form(&self, p: &DVector<f64>, v: &DVector<f64>) -> f64 {
        let n = self.base_dim;
        let v_q = v.rows(0, n);
        p.dot(&v_q)
    }

    /// Compute the canonical symplectic 2-form from the tautological 1-form.
    pub fn symplectic_from_tautological(&self) -> SymplecticForm {
        self.canonical_symplectic_form()
    }

    /// Create a Hamiltonian system for free particle on M.
    pub fn free_hamiltonian(&self) -> HamiltonianSystem {
        let n = self.base_dim;
        let metric_inv = self.metric.clone()
            .unwrap_or_else(|| DMatrix::identity(n, n))
            .try_inverse()
            .unwrap_or_else(|| DMatrix::identity(n, n));
        let metric_inv2 = metric_inv.clone();

        HamiltonianSystem::new(
            &format!("Free particle on {}", self.name),
            n,
            move |q: &DVector<f64>, p: &DVector<f64>| {
                let _ = q;
                0.5 * (metric_inv.transpose() * p).dot(p)
            },
            move |q: &DVector<f64>, p: &DVector<f64>| {
                let _ = q;
                let grad_p = &metric_inv2 * p;
                let mut grad = DVector::zeros(2 * n);
                for i in 0..n {
                    grad[n + i] = grad_p[i];
                }
                grad
            },
        )
    }

    /// Vertical lift of a covector.
    pub fn vertical_lift(&self, alpha: &DVector<f64>) -> DVector<f64> {
        let n = self.base_dim;
        let mut result = DVector::zeros(2 * n);
        for i in 0..n {
            result[n + i] = alpha[i];
        }
        result
    }

    /// Canonical projection π: T*M → M.
    pub fn projection(&self, qp: &DVector<f64>) -> DVector<f64> {
        qp.rows(0, self.base_dim).into_owned()
    }

    /// Check if a vector is vertical.
    pub fn is_vertical(&self, v: &DVector<f64>) -> bool {
        v.rows(0, self.base_dim).iter().all(|&x| x.abs() < 1e-10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_cotangent_bundle_creation() {
        let tb = CotangentBundle::new(3);
        assert_eq!(tb.base_dim, 3);
        assert_eq!(tb.total_dim(), 6);
    }

    #[test]
    fn test_canonical_symplectic_form() {
        let tb = CotangentBundle::new(2);
        let omega = tb.canonical_symplectic_form();
        assert_eq!(omega.dim(), 4);
        assert_eq!(omega.half_dim(), 2);
    }

    #[test]
    fn test_tautological_form() {
        let tb = CotangentBundle::new(2);
        let p = DVector::from_vec(vec![1.0, 2.0]);
        let v = DVector::from_vec(vec![3.0, 4.0, 5.0, 6.0]);
        assert_abs_diff_eq!(tb.tautological_form(&p, &v), 11.0);
    }

    #[test]
    fn test_free_hamiltonian_energy() {
        let tb = CotangentBundle::euclidean(2);
        let sys = tb.free_hamiltonian();
        let q = DVector::from_vec(vec![0.0, 0.0]);
        let p = DVector::from_vec(vec![1.0, 0.0]);
        assert_abs_diff_eq!(sys.energy(&q, &p), 0.5);
    }

    #[test]
    fn test_vertical_lift() {
        let tb = CotangentBundle::new(2);
        let alpha = DVector::from_vec(vec![1.0, 2.0]);
        let lifted = tb.vertical_lift(&alpha);
        assert_abs_diff_eq!(lifted[0], 0.0);
        assert_abs_diff_eq!(lifted[1], 0.0);
        assert_abs_diff_eq!(lifted[2], 1.0);
        assert_abs_diff_eq!(lifted[3], 2.0);
    }

    #[test]
    fn test_projection() {
        let tb = CotangentBundle::new(2);
        let qp = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let q = tb.projection(&qp);
        assert_abs_diff_eq!(q[0], 1.0);
        assert_abs_diff_eq!(q[1], 2.0);
    }

    #[test]
    fn test_is_vertical() {
        let tb = CotangentBundle::new(2);
        let v_vert = DVector::from_vec(vec![0.0, 0.0, 1.0, 2.0]);
        let v_horiz = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        assert!(tb.is_vertical(&v_vert));
        assert!(!tb.is_vertical(&v_horiz));
    }

    #[test]
    fn test_symplectic_from_tautological() {
        let tb = CotangentBundle::new(3);
        let omega = tb.symplectic_from_tautological();
        assert_eq!(omega.dim(), 6);
        assert!(omega.is_closed());
    }
}
