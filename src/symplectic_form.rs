//! Symplectic form (non-degenerate closed 2-form ω) on a 2n-dimensional manifold.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Helper to compute determinant of a DMatrix.
fn det(m: &DMatrix<f64>) -> f64 {
    m.clone().lu().determinant()
}

/// A symplectic form ω on a 2n-dimensional manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymplecticForm {
    /// The matrix representation Ω of the symplectic form (2n × 2n skew-symmetric)
    matrix: DMatrix<f64>,
    /// Dimension of the underlying manifold (2n)
    dim: usize,
}

impl SymplecticForm {
    /// Create the canonical symplectic form on R^{2n}.
    /// Ω = [0  I_n; -I_n  0] where I_n is the n×n identity matrix.
    pub fn canonical(n: usize) -> Self {
        let dim = 2 * n;
        let mut matrix = DMatrix::zeros(dim, dim);
        for i in 0..n {
            matrix[(i, n + i)] = 1.0;
            matrix[(n + i, i)] = -1.0;
        }
        Self { matrix, dim }
    }

    /// Create a symplectic form from a matrix, verifying non-degeneracy.
    pub fn from_matrix(matrix: DMatrix<f64>) -> Result<Self, String> {
        let dim = matrix.nrows();
        if dim != matrix.ncols() {
            return Err("Symplectic form matrix must be square".to_string());
        }
        if !dim.is_multiple_of(2) {
            return Err("Symplectic form requires even dimension 2n".to_string());
        }
        let skew = &matrix.transpose() + &matrix;
        if skew.iter().any(|&x| x.abs() > 1e-10) {
            return Err("Symplectic form must be skew-symmetric".to_string());
        }
        if det(&matrix).abs() < 1e-10 {
            return Err("Symplectic form must be non-degenerate (det ≠ 0)".to_string());
        }
        Ok(Self { matrix, dim })
    }

    /// Get the dimension of the manifold (2n).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get half-dimension n.
    pub fn half_dim(&self) -> usize {
        self.dim / 2
    }

    /// Get the matrix representation.
    pub fn matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }

    /// Evaluate the symplectic form on two tangent vectors: ω(u, v) = u^T Ω v.
    pub fn apply(&self, u: &DVector<f64>, v: &DVector<f64>) -> f64 {
        (u.transpose() * &self.matrix * v)[(0, 0)]
    }

    /// Check if the form is closed (dω = 0).
    pub fn is_closed(&self) -> bool {
        true
    }

    /// Check if a subspace is Lagrangian.
    pub fn is_lagrangian(&self, basis: &DMatrix<f64>) -> bool {
        let k = basis.ncols();
        for i in 0..k {
            for j in 0..k {
                let val = (basis.column(i).transpose() * &self.matrix * basis.column(j))[(0, 0)];
                if val.abs() > 1e-8 {
                    return false;
                }
            }
        }
        k == self.half_dim()
    }

    /// Check if a subspace is isotropic (ω|_L = 0).
    pub fn is_isotropic(&self, basis: &DMatrix<f64>) -> bool {
        let k = basis.ncols();
        for i in 0..k {
            for j in i..k {
                let val = (basis.column(i).transpose() * &self.matrix * basis.column(j))[(0, 0)];
                if val.abs() > 1e-8 {
                    return false;
                }
            }
        }
        true
    }

    /// Check if a subspace is symplectic (ω restricted to it is non-degenerate).
    pub fn is_symplectic_subspace(&self, basis: &DMatrix<f64>) -> bool {
        let k = basis.ncols();
        if !k.is_multiple_of(2) {
            return false;
        }
        let restricted = basis.transpose() * &self.matrix * basis;
        restricted.clone().try_inverse().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_canonical_form_is_skew_symmetric() {
        let omega = SymplecticForm::canonical(2);
        let m = omega.matrix();
        assert_abs_diff_eq!(m[(0, 2)], 1.0);
        assert_abs_diff_eq!(m[(2, 0)], -1.0);
        assert_abs_diff_eq!(m[(1, 3)], 1.0);
        assert_abs_diff_eq!(m[(3, 1)], -1.0);
    }

    #[test]
    fn test_canonical_form_dimension() {
        let omega = SymplecticForm::canonical(3);
        assert_eq!(omega.dim(), 6);
        assert_eq!(omega.half_dim(), 3);
    }

    #[test]
    fn test_symplectic_product_standard_vectors() {
        let omega = SymplecticForm::canonical(1);
        let e1 = DVector::from_vec(vec![1.0, 0.0]);
        let e2 = DVector::from_vec(vec![0.0, 1.0]);
        assert_abs_diff_eq!(omega.apply(&e1, &e2), 1.0);
        assert_abs_diff_eq!(omega.apply(&e2, &e1), -1.0);
    }

    #[test]
    fn test_from_matrix_valid() {
        let omega = SymplecticForm::canonical(2);
        let reconstructed = SymplecticForm::from_matrix(omega.matrix().clone());
        assert!(reconstructed.is_ok());
    }

    #[test]
    fn test_from_matrix_odd_dimension_rejected() {
        let m = DMatrix::zeros(3, 3);
        assert!(SymplecticForm::from_matrix(m).is_err());
    }

    #[test]
    fn test_from_matrix_non_square_rejected() {
        let m = DMatrix::zeros(2, 3);
        assert!(SymplecticForm::from_matrix(m).is_err());
    }

    #[test]
    fn test_is_closed() {
        let omega = SymplecticForm::canonical(3);
        assert!(omega.is_closed());
    }

    #[test]
    fn test_isotropic_subspace() {
        let omega = SymplecticForm::canonical(2);
        let basis = DMatrix::from_column_slice(4, 1, &[1.0, 0.0, 0.0, 0.0]);
        assert!(omega.is_isotropic(&basis));
    }

    #[test]
    fn test_non_isotropic_subspace() {
        let omega = SymplecticForm::canonical(1);
        let basis = DMatrix::from_column_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        assert!(!omega.is_isotropic(&basis));
    }

    #[test]
    fn test_lagrangian_subspace() {
        let omega = SymplecticForm::canonical(2);
        let basis = DMatrix::from_column_slice(4, 2, &[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ]);
        assert!(omega.is_lagrangian(&basis));
    }

    #[test]
    fn test_symplectic_subspace() {
        let omega = SymplecticForm::canonical(2);
        let basis = DMatrix::from_column_slice(4, 2, &[
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]);
        assert!(omega.is_symplectic_subspace(&basis));
    }
}
