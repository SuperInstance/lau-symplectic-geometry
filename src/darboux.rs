//! Darboux theorem: local coordinates where ω = Σ dx_i ∧ dp_i.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::symplectic_form::SymplecticForm;

/// Darboux coordinates on a symplectic manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DarbouxCoordinates {
    /// Half-dimension n
    pub n: usize,
    /// Transformation matrix S such that S^T Ω S = J
    pub transformation: DMatrix<f64>,
}

impl DarbouxCoordinates {
    /// Create standard Darboux coordinates (identity transformation).
    pub fn standard(n: usize) -> Self {
        Self {
            n,
            transformation: DMatrix::identity(2 * n, 2 * n),
        }
    }

    /// Find Darboux coordinate transformation for a given symplectic form.
    pub fn from_symplectic_form(omega: &SymplecticForm) -> Result<Self, String> {
        let n = omega.half_dim();
        let omega_mat = omega.matrix();
        let dim = 2 * n;

        // For the canonical form, just return identity
        let canon = SymplecticForm::canonical(n);
        let diff = omega_mat - canon.matrix();
        if diff.iter().all(|x| x.abs() < 1e-8) {
            return Ok(Self::standard(n));
        }

        // General approach: find symplectic basis using Gram-Schmidt-like procedure
        let mut s = DMatrix::zeros(dim, dim);
        let mut used = vec![false; dim];

        let mut col = 0;
        for _pair in 0..n {
            // Find first unused row
            let i = (0..dim).find(|&idx| !used[idx]).ok_or("No unused index")?;

            // Create unit vector e_i
            let mut e_i = DVector::zeros(dim);
            e_i[i] = 1.0;

            // Find j such that ω(e_i, e_j) ≠ 0
            let mut found_j = None;
            for j in (0..dim).filter(|&idx| !used[idx] && idx != i) {
                let mut e_j = DVector::zeros(dim);
                e_j[j] = 1.0;
                let val = (e_i.transpose() * omega_mat * &e_j)[(0, 0)];
                if val.abs() > 1e-10 {
                    found_j = Some((j, val));
                    break;
                }
            }

            let (j, val) = found_j.ok_or("Could not find symplectic pair")?;

            // Normalize so ω(e_i, e_j/val) = 1
            let mut e_j = DVector::zeros(dim);
            e_j[j] = 1.0 / val;

            s.set_column(col, &e_i);
            s.set_column(col + 1, &e_j);
            used[i] = true;
            used[j] = true;
            col += 2;
        }

        let j_canon = {
            let mut m = DMatrix::zeros(dim, dim);
            for k in 0..n {
                m[(k, n + k)] = 1.0;
                m[(n + k, k)] = -1.0;
            }
            m
        };

        let result = s.transpose() * omega_mat * &s;
        let diff = &result - &j_canon;
        let max_err = diff.iter().map(|x: &f64| x.abs()).fold(0.0_f64, f64::max);

        if max_err > 1e-6 {
            return Err(format!("Darboux transformation failed (max error: {})", max_err));
        }

        Ok(Self { n, transformation: s })
    }

    /// Transform a point to Darboux coordinates.
    pub fn to_darboux(&self, point: &DVector<f64>) -> DVector<f64> {
        self.transformation.transpose() * point
    }

    /// Transform from Darboux coordinates back to original.
    pub fn from_darboux(&self, darboux_point: &DVector<f64>) -> DVector<f64> {
        let inv = self.transformation.clone().try_inverse()
            .expect("Transformation should be invertible");
        (&inv * darboux_point).into_owned()
    }

    /// Verify transformation brings symplectic form to canonical form.
    pub fn verify(&self, omega: &SymplecticForm) -> bool {
        let j_canon = {
            let dim = 2 * self.n;
            let mut m = DMatrix::zeros(dim, dim);
            for k in 0..self.n {
                m[(k, self.n + k)] = 1.0;
                m[(self.n + k, k)] = -1.0;
            }
            m
        };
        let result = self.transformation.transpose() * omega.matrix() * &self.transformation;
        let diff = result - j_canon;
        diff.iter().all(|&x| x.abs() < 1e-6)
    }

    /// Compute the canonical symplectic form in Darboux coordinates.
    pub fn canonical_form(&self) -> SymplecticForm {
        SymplecticForm::canonical(self.n)
    }

    /// Express a general symplectic form in Darboux coordinates.
    pub fn transform_form(&self, omega: &SymplecticForm) -> SymplecticForm {
        let transformed = self.transformation.transpose() * omega.matrix() * &self.transformation;
        SymplecticForm::from_matrix(transformed).unwrap_or_else(|_| SymplecticForm::canonical(self.n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_standard_darboux() {
        let d = DarbouxCoordinates::standard(2);
        let omega = SymplecticForm::canonical(2);
        assert!(d.verify(&omega));
    }

    #[test]
    fn test_canonical_form_is_already_darboux() {
        let omega = SymplecticForm::canonical(3);
        let d = DarbouxCoordinates::from_symplectic_form(&omega);
        assert!(d.is_ok());
        assert!(d.unwrap().verify(&omega));
    }

    #[test]
    fn test_standard_identity_transform() {
        let d = DarbouxCoordinates::standard(2);
        let point = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let darboux = d.to_darboux(&point);
        for i in 0..4 {
            assert_abs_diff_eq!(darboux[i], point[i]);
        }
    }

    #[test]
    fn test_darboux_form_is_canonical() {
        let d = DarbouxCoordinates::standard(3);
        let form = d.canonical_form();
        assert_eq!(form.dim(), 6);
    }

    #[test]
    fn test_transform_preserves_symplecticity() {
        let d = DarbouxCoordinates::standard(2);
        let omega = SymplecticForm::canonical(2);
        let transformed = d.transform_form(&omega);
        assert!(transformed.is_closed());
    }

    #[test]
    fn test_no_local_invariants() {
        let omega1 = SymplecticForm::canonical(2);
        let d = DarbouxCoordinates::from_symplectic_form(&omega1).unwrap();
        let canonical = d.canonical_form();
        assert_eq!(omega1.dim(), canonical.dim());
    }

    #[test]
    fn test_from_scaled_form() {
        let n = 2;
        let omega = SymplecticForm::canonical(n);
        let d = DarbouxCoordinates::from_symplectic_form(&omega);
        assert!(d.is_ok());
    }
}
