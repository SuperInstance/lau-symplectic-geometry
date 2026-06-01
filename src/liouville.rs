//! Liouville's theorem: phase space volume is preserved under Hamiltonian flow.

use nalgebra::{DMatrix, DVector};

use crate::symplectic_form::SymplecticForm;

/// Liouville's theorem: the phase space volume form is preserved by Hamiltonian flows.
pub struct LiouvilleTheorem;

fn det(m: &DMatrix<f64>) -> f64 {
    m.clone().lu().determinant()
}

impl LiouvilleTheorem {
    /// Verify that a transformation preserves phase space volume (det = 1).
    pub fn verify_volume_preservation(jacobian: &DMatrix<f64>) -> bool {
        (det(jacobian) - 1.0).abs() < 1e-6
    }

    /// Compute the phase space volume form Ω^n.
    pub fn phase_space_volume(omega: &SymplecticForm) -> f64 {
        let d = det(omega.matrix());
        let n = omega.half_dim();
        let n_factorial = (1..=n).product::<usize>() as f64;
        d / n_factorial
    }

    /// Compute the volume of a parallelepiped in phase space.
    pub fn parallelepiped_volume(vectors: &[DVector<f64>]) -> f64 {
        if vectors.is_empty() {
            return 0.0;
        }
        let dim = vectors[0].nrows();
        let k = vectors.len();
        if k != dim {
            let mut gram = DMatrix::zeros(k, k);
            for i in 0..k {
                for j in 0..k {
                    gram[(i, j)] = vectors[i].dot(&vectors[j]);
                }
            }
            return det(&gram).sqrt();
        }
        let mut matrix = DMatrix::zeros(dim, k);
        for (i, v) in vectors.iter().enumerate() {
            matrix.set_column(i, v);
        }
        det(&matrix).abs()
    }

    /// Verify Liouville's theorem numerically along a Hamiltonian flow.
    pub fn verify_along_flow(flow_jacobians: &[DMatrix<f64>]) -> Vec<(usize, f64, bool)> {
        flow_jacobians
            .iter()
            .enumerate()
            .map(|(i, j)| {
                let d = det(j);
                let preserved = (d - 1.0).abs() < 1e-4;
                (i, d, preserved)
            })
            .collect()
    }

    /// Compute the symplectic capacity (lower bound) of an ellipsoid.
    pub fn symplectic_capacity_lower_bound(ellipsoid_matrix: &DMatrix<f64>) -> f64 {
        let eigenvalues = ellipsoid_matrix.symmetric_eigenvalues();
        let min_eigenvalue = eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);
        if min_eigenvalue > 0.0 {
            std::f64::consts::PI / min_eigenvalue.sqrt()
        } else {
            f64::INFINITY
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SymplecticMatrix;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_identity_preserves_volume() {
        let id = DMatrix::identity(4, 4);
        assert!(LiouvilleTheorem::verify_volume_preservation(&id));
    }

    #[test]
    fn test_rotation_preserves_volume() {
        let theta: f64 = 0.5;
        let mut rot = DMatrix::identity(4, 4);
        rot[(0, 0)] = theta.cos();
        rot[(0, 2)] = -theta.sin();
        rot[(2, 0)] = theta.sin();
        rot[(2, 2)] = theta.cos();
        assert!(LiouvilleTheorem::verify_volume_preservation(&rot));
    }

    #[test]
    fn test_scaling_does_not_preserve_volume() {
        let mut scale = DMatrix::identity(4, 4);
        scale[(0, 0)] = 2.0;
        assert!(!LiouvilleTheorem::verify_volume_preservation(&scale));
    }

    #[test]
    fn test_phase_space_volume_canonical() {
        let omega = SymplecticForm::canonical(2);
        let vol = LiouvilleTheorem::phase_space_volume(&omega);
        assert_abs_diff_eq!(vol, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_parallelepiped_unit_cube() {
        let v1 = DVector::from_vec(vec![1.0, 0.0]);
        let v2 = DVector::from_vec(vec![0.0, 1.0]);
        let vol = LiouvilleTheorem::parallelepiped_volume(&[v1, v2]);
        assert_abs_diff_eq!(vol, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_parallelepiped_scaled() {
        let v1 = DVector::from_vec(vec![2.0, 0.0]);
        let v2 = DVector::from_vec(vec![0.0, 3.0]);
        let vol = LiouvilleTheorem::parallelepiped_volume(&[v1, v2]);
        assert_abs_diff_eq!(vol, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_verify_along_flow_symplectic() {
        let flow_jacobians = vec![
            DMatrix::identity(4, 4),
            DMatrix::identity(4, 4),
            DMatrix::identity(4, 4),
        ];
        let results = LiouvilleTheorem::verify_along_flow(&flow_jacobians);
        for (_, det, preserved) in &results {
            assert!(preserved);
            assert_abs_diff_eq!(*det, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_symplectic_capacity_unit_ball() {
        let a = DMatrix::identity(4, 4);
        let cap = LiouvilleTheorem::symplectic_capacity_lower_bound(&a);
        assert_abs_diff_eq!(cap, std::f64::consts::PI, epsilon = 1e-10);
    }

    #[test]
    fn test_symplectic_matrix_det_is_one() {
        let m = SymplecticMatrix::plane_rotation(2, 0, 1.0);
        assert_abs_diff_eq!(m.determinant(), 1.0, epsilon = 1e-10);
        assert!(LiouvilleTheorem::verify_volume_preservation(m.matrix()));
    }
}
