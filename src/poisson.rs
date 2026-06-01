//! Poisson brackets: {f,g} = ω(X_f, X_g), Jacobi identity.

use nalgebra::DVector;

/// Poisson bracket structure on a 2n-dimensional phase space.
///
/// The Poisson bracket of two functions f, g is defined as:
///   {f, g} = Σ_i (∂f/∂q_i ∂g/∂p_i - ∂f/∂p_i ∂g/∂q_i)
///
/// This satisfies:
/// - Skew-symmetry: {f, g} = -{g, f}
/// - Bilinearity: {af + bg, h} = a{f,h} + b{g,h}
/// - Jacobi identity: {f, {g, h}} + {g, {h, f}} + {h, {f, g}} = 0
/// - Leibniz rule: {f, gh} = g{f,h} + h{f,g}
pub struct PoissonBracket {
    /// Half-dimension n
    pub n: usize,
}

impl PoissonBracket {
    /// Create a Poisson bracket on R^{2n}.
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    /// Compute the Poisson bracket {f, g} at point (q, p).
    ///
    /// Given gradients ∇f = [∂f/∂q; ∂f/∂p] and ∇g = [∂g/∂q; ∂g/∂p],
    /// {f,g} = (∂f/∂q)^T (∂g/∂p) - (∂f/∂p)^T (∂g/∂q)
    pub fn bracket(&self, grad_f: &DVector<f64>, grad_g: &DVector<f64>) -> f64 {
        let mut result = 0.0;
        for i in 0..self.n {
            result += grad_f[i] * grad_g[self.n + i];
            result -= grad_f[self.n + i] * grad_g[i];
        }
        result
    }

    /// Verify skew-symmetry: {f, g} = -{g, f}.
    pub fn verify_skew_symmetry(
        &self,
        grad_f: &DVector<f64>,
        grad_g: &DVector<f64>,
    ) -> bool {
        let fg = self.bracket(grad_f, grad_g);
        let gf = self.bracket(grad_g, grad_f);
        (fg + gf).abs() < 1e-10
    }

    /// Verify bilinearity: {af + bg, h} = a{f,h} + b{g,h}.
    pub fn verify_bilinearity(
        &self,
        a: f64,
        b: f64,
        grad_f: &DVector<f64>,
        grad_g: &DVector<f64>,
        grad_h: &DVector<f64>,
    ) -> bool {
        let combined = a * grad_f + b * grad_g;
        let lhs = self.bracket(&combined, grad_h);
        let rhs = a * self.bracket(grad_f, grad_h) + b * self.bracket(grad_g, grad_h);
        (lhs - rhs).abs() < 1e-8
    }

    /// Verify the Jacobi identity for three gradient vectors.
    ///
    /// This checks {f,{g,h}} + {g,{h,f}} + {h,{f,g}} = 0.
    /// Since we can't take second derivatives easily, we use a numerical approach
    /// with Hessians provided explicitly.
    pub fn verify_jacobi(
        &self,
        grad_f: &DVector<f64>,
        grad_g: &DVector<f64>,
        grad_h: &DVector<f64>,
        hess_f: &nalgebra::DMatrix<f64>,
        hess_g: &nalgebra::DMatrix<f64>,
        hess_h: &nalgebra::DMatrix<f64>,
    ) -> bool {
        // {f, {g, h}} requires the gradient of {g, h} w.r.t. (q, p)
        // {g, h} = sum_i (∂g/∂q_i ∂h/∂p_i - ∂g/∂p_i ∂h/∂q_i)
        // ∂{g,h}/∂q_k = sum_i (∂²g/∂q_k∂q_i ∂h/∂p_i + ∂g/∂q_i ∂²h/∂q_k∂p_i
        //                - ∂²g/∂q_k∂p_i ∂h/∂q_i - ∂g/∂p_i ∂²h/∂q_k∂q_i)

        let grad_gh = self._jacobi_grad(grad_g, grad_h, hess_g, hess_h);
        let grad_hf = self._jacobi_grad(grad_h, grad_f, hess_h, hess_f);
        let grad_fg = self._jacobi_grad(grad_f, grad_g, hess_f, hess_g);

        let term1 = self.bracket(grad_f, &grad_gh);
        let term2 = self.bracket(grad_g, &grad_hf);
        let term3 = self.bracket(grad_h, &grad_fg);

        (term1 + term2 + term3).abs() < 1e-6
    }

    /// Helper: compute gradient of {f, g} for Jacobi identity verification.
    fn _jacobi_grad(
        &self,
        grad_f: &DVector<f64>,
        grad_g: &DVector<f64>,
        hess_f: &nalgebra::DMatrix<f64>,
        hess_g: &nalgebra::DMatrix<f64>,
    ) -> DVector<f64> {
        let dim = 2 * self.n;
        let mut result = DVector::zeros(dim);
        for k in 0..dim {
            let mut val = 0.0;
            for i in 0..self.n {
                // From ∂f/∂q_i * ∂g/∂p_i:
                val += hess_f[(k, i)] * grad_g[self.n + i];
                val += grad_f[i] * hess_g[(k, self.n + i)];
                // From -∂f/∂p_i * ∂g/∂q_i:
                val -= hess_f[(k, self.n + i)] * grad_g[i];
                val -= grad_f[self.n + i] * hess_g[(k, i)];
            }
            result[k] = val;
        }
        result
    }

    /// Compute canonical coordinate brackets.
    /// {q_i, q_j} = 0, {p_i, p_j} = 0, {q_i, p_j} = δ_{ij}
    pub fn canonical_brackets(&self) -> CanonicalBrackets {
        CanonicalBrackets { n: self.n }
    }

    /// Verify Leibniz rule: {f, gh} = g{f,h} + h{f,g}.
    pub fn verify_leibniz(
        &self,
        grad_f: &DVector<f64>,
        grad_g: &DVector<f64>,
        grad_h: &DVector<f64>,
        g_val: f64,
        h_val: f64,
        grad_gh: &DVector<f64>,
    ) -> bool {
        let lhs = self.bracket(grad_f, grad_gh);
        let rhs = g_val * self.bracket(grad_f, grad_h) + h_val * self.bracket(grad_f, grad_g);
        (lhs - rhs).abs() < 1e-8
    }
}

/// Canonical Poisson bracket relations.
pub struct CanonicalBrackets {
    #[allow(dead_code)]
    n: usize,
}

impl CanonicalBrackets {
    /// {q_i, q_j} = 0
    pub fn qq(&self, _i: usize, _j: usize) -> f64 {
        0.0
    }

    /// {p_i, p_j} = 0
    pub fn pp(&self, _i: usize, _j: usize) -> f64 {
        0.0
    }

    /// {q_i, p_j} = δ_{ij}
    pub fn qp(&self, i: usize, j: usize) -> f64 {
        if i == j { 1.0 } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_canonical_qq_bracket() {
        let pb = PoissonBracket::new(2);
        let cb = pb.canonical_brackets();
        assert_abs_diff_eq!(cb.qq(0, 1), 0.0);
        assert_abs_diff_eq!(cb.qq(0, 0), 0.0);
    }

    #[test]
    fn test_canonical_pp_bracket() {
        let pb = PoissonBracket::new(2);
        let cb = pb.canonical_brackets();
        assert_abs_diff_eq!(cb.pp(0, 1), 0.0);
    }

    #[test]
    fn test_canonical_qp_bracket() {
        let pb = PoissonBracket::new(2);
        let cb = pb.canonical_brackets();
        assert_abs_diff_eq!(cb.qp(0, 0), 1.0);
        assert_abs_diff_eq!(cb.qp(0, 1), 0.0);
        assert_abs_diff_eq!(cb.qp(1, 1), 1.0);
    }

    #[test]
    fn test_bracket_computation() {
        let pb = PoissonBracket::new(1);
        // f = q, g = p => {f,g} = ∂f/∂q * ∂g/∂p - ∂f/∂p * ∂g/∂q = 1*1 - 0*0 = 1
        let grad_f = DVector::from_vec(vec![1.0, 0.0]); // ∇f = (1, 0)
        let grad_g = DVector::from_vec(vec![0.0, 1.0]); // ∇g = (0, 1)
        assert_abs_diff_eq!(pb.bracket(&grad_f, &grad_g), 1.0);
    }

    #[test]
    fn test_skew_symmetry() {
        let pb = PoissonBracket::new(2);
        let grad_f = DVector::from_vec(vec![1.0, 2.0, 0.5, -1.0]);
        let grad_g = DVector::from_vec(vec![0.5, -1.0, 2.0, 3.0]);
        assert!(pb.verify_skew_symmetry(&grad_f, &grad_g));
    }

    #[test]
    fn test_bilinearity() {
        let pb = PoissonBracket::new(2);
        let grad_f = DVector::from_vec(vec![1.0, 0.0, 0.0, 1.0]);
        let grad_g = DVector::from_vec(vec![0.0, 1.0, 1.0, 0.0]);
        let grad_h = DVector::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        assert!(pb.verify_bilinearity(2.0, 3.0, &grad_f, &grad_g, &grad_h));
    }

    #[test]
    fn test_jacobi_identity_quadratic() {
        let pb = PoissonBracket::new(1);
        // f = q^2 + p^2, g = q*p, h = q - p
        let grad_f = DVector::from_vec(vec![2.0, 2.0]);
        let grad_g = DVector::from_vec(vec![1.0, 1.0]);
        let grad_h = DVector::from_vec(vec![1.0, -1.0]);
        // Hessians
        let hess_f = nalgebra::DMatrix::from_row_slice(2, 2, &[
            2.0, 0.0,
            0.0, 2.0,
        ]);
        let hess_g = nalgebra::DMatrix::from_row_slice(2, 2, &[
            0.0, 1.0,
            1.0, 0.0,
        ]);
        let hess_h = nalgebra::DMatrix::zeros(2, 2);
        assert!(pb.verify_jacobi(&grad_f, &grad_g, &grad_h, &hess_f, &hess_g, &hess_h));
    }

    #[test]
    fn test_leibniz_rule() {
        let pb = PoissonBracket::new(1);
        let grad_f = DVector::from_vec(vec![1.0, 0.0]);
        let grad_g = DVector::from_vec(vec![0.0, 1.0]);
        let grad_h = DVector::from_vec(vec![1.0, 1.0]);
        // gh has gradient = g * ∇h + h * ∇g
        let g_val = 1.0;
        let h_val = 1.0;
        let grad_gh = &grad_g + &grad_h;
        assert!(pb.verify_leibniz(&grad_f, &grad_g, &grad_h, g_val, h_val, &grad_gh));
    }
}
