//! Symplectic matrices Sp(2n): matrices M satisfying M^T J M = J.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Helper to compute determinant of a DMatrix.
fn det(m: &DMatrix<f64>) -> f64 {
    m.clone().lu().determinant()
}

/// A symplectic matrix M ∈ Sp(2n), satisfying M^T Ω M = Ω.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymplecticMatrix {
    matrix: DMatrix<f64>,
    n: usize,
}

impl SymplecticMatrix {
    /// Create the canonical symplectic matrix J = [0 I_n; -I_n 0].
    pub fn canonical_j(n: usize) -> Self {
        let dim = 2 * n;
        let mut matrix = DMatrix::zeros(dim, dim);
        for i in 0..n {
            matrix[(i, n + i)] = 1.0;
            matrix[(n + i, i)] = -1.0;
        }
        Self { matrix, n }
    }

    /// Create an identity symplectic matrix.
    pub fn identity(n: usize) -> Self {
        Self {
            matrix: DMatrix::identity(2 * n, 2 * n),
            n,
        }
    }

    /// Construct from a matrix, verifying the symplectic condition M^T J M = J.
    pub fn from_matrix(matrix: DMatrix<f64>) -> Result<Self, String> {
        let dim = matrix.nrows();
        if dim != matrix.ncols() {
            return Err("Matrix must be square".to_string());
        }
        if !dim.is_multiple_of(2) {
            return Err("Symplectic matrix must have even dimension 2n".to_string());
        }
        let n = dim / 2;
        let j = Self::canonical_j(n);
        let product = matrix.transpose() * j.matrix() * &matrix;
        let diff = &product - j.matrix();
        let max_err = diff.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        if max_err > 1e-8 {
            return Err(format!("Matrix does not satisfy M^T J M = J (max error: {})", max_err));
        }
        Ok(Self { matrix, n })
    }

    /// Get the underlying matrix.
    pub fn matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }

    /// Get the half-dimension n.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the full dimension 2n.
    pub fn dim(&self) -> usize {
        2 * self.n
    }

    /// Verify that this matrix satisfies the symplectic condition.
    pub fn verify_symplectic(&self) -> bool {
        let j = Self::canonical_j(self.n);
        let product = self.matrix.transpose() * j.matrix() * &self.matrix;
        let diff = &product - j.matrix();
        diff.iter().all(|&x| x.abs() < 1e-8)
    }

    /// Product of two symplectic matrices.
    pub fn multiply(&self, other: &SymplecticMatrix) -> Result<SymplecticMatrix, String> {
        if self.n != other.n {
            return Err("Dimensions must match".to_string());
        }
        SymplecticMatrix::from_matrix(&self.matrix * &other.matrix)
    }

    /// Inverse: M^{-1} = -J M^T J
    pub fn inverse(&self) -> SymplecticMatrix {
        let j = Self::canonical_j(self.n);
        let inv = -j.matrix() * self.matrix.transpose() * j.matrix();
        SymplecticMatrix { matrix: inv, n: self.n }
    }

    /// Transpose.
    pub fn transpose(&self) -> SymplecticMatrix {
        SymplecticMatrix { matrix: self.matrix.transpose(), n: self.n }
    }

    /// Determinant. For M ∈ Sp(2n), det(M) = 1.
    pub fn determinant(&self) -> f64 {
        det(&self.matrix)
    }

    /// Rotation in the (x_i, p_i) plane.
    pub fn plane_rotation(n: usize, i: usize, theta: f64) -> Self {
        let dim = 2 * n;
        let mut m = DMatrix::identity(dim, dim);
        m[(i, i)] = theta.cos();
        m[(i, n + i)] = -theta.sin();
        m[(n + i, i)] = theta.sin();
        m[(n + i, n + i)] = theta.cos();
        Self { matrix: m, n }
    }

    /// Shear: (x_i, p_i) -> (x_i + a*p_i, p_i).
    pub fn shear_x(n: usize, i: usize, a: f64) -> Self {
        let dim = 2 * n;
        let mut m = DMatrix::identity(dim, dim);
        m[(i, n + i)] = a;
        Self { matrix: m, n }
    }

    /// Shear: (x_i, p_i) -> (x_i, p_i + b*x_i).
    pub fn shear_p(n: usize, i: usize, b: f64) -> Self {
        let dim = 2 * n;
        let mut m = DMatrix::identity(dim, dim);
        m[(n + i, i)] = b;
        Self { matrix: m, n }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_identity_is_symplectic() {
        let m = SymplecticMatrix::identity(2);
        assert!(m.verify_symplectic());
    }

    #[test]
    fn test_canonical_j_is_symplectic() {
        let j = SymplecticMatrix::canonical_j(2);
        assert!(j.verify_symplectic());
    }

    #[test]
    fn test_det_is_one() {
        let m = SymplecticMatrix::identity(3);
        assert_abs_diff_eq!(m.determinant(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_is_symplectic() {
        let m = SymplecticMatrix::plane_rotation(2, 0, 0.7);
        assert!(m.verify_symplectic());
        assert_abs_diff_eq!(m.determinant(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_shear_x_is_symplectic() {
        let m = SymplecticMatrix::shear_x(2, 0, 3.5);
        assert!(m.verify_symplectic());
    }

    #[test]
    fn test_shear_p_is_symplectic() {
        let m = SymplecticMatrix::shear_p(2, 1, -2.0);
        assert!(m.verify_symplectic());
    }

    #[test]
    fn test_product_is_symplectic() {
        let m1 = SymplecticMatrix::plane_rotation(2, 0, 0.5);
        let m2 = SymplecticMatrix::shear_x(2, 1, 1.5);
        let prod = m1.multiply(&m2).unwrap();
        assert!(prod.verify_symplectic());
    }

    #[test]
    fn test_inverse_is_symplectic() {
        let m = SymplecticMatrix::plane_rotation(3, 1, 1.2);
        let inv = m.inverse();
        assert!(inv.verify_symplectic());
    }

    #[test]
    fn test_inverse_times_identity() {
        let m = SymplecticMatrix::plane_rotation(2, 0, 0.8);
        let inv = m.inverse();
        let product = m.matrix() * inv.matrix();
        let identity = DMatrix::identity(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                assert_abs_diff_eq!(product[(i, j)], identity[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_transpose_is_symplectic() {
        let m = SymplecticMatrix::plane_rotation(2, 0, 0.3);
        let t = m.transpose();
        assert!(t.verify_symplectic());
    }

    #[test]
    fn test_from_matrix_rejects_non_symplectic() {
        let m = DMatrix::from_row_slice(4, 4, &[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 2.0, 0.0,
            0.0, 0.0, 0.0, 2.0,
        ]);
        assert!(SymplecticMatrix::from_matrix(m).is_err());
    }

    #[test]
    fn test_j_squared_is_minus_identity() {
        let j = SymplecticMatrix::canonical_j(2);
        let j2 = j.matrix() * j.matrix();
        let neg_id = -DMatrix::identity(4, 4);
        for i in 0..4 {
            for j_idx in 0..4 {
                assert_abs_diff_eq!(j2[(i, j_idx)], neg_id[(i, j_idx)], epsilon = 1e-10);
            }
        }
    }
}
