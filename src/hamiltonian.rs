//! Hamiltonian vector fields and systems: X_H = J∇H (symplectic gradient).

use nalgebra::DVector;

/// A Hamiltonian system defined by a Hamiltonian function H: R^{2n} -> R.
pub struct HamiltonianSystem {
    /// Name/description
    pub name: String,
    /// Half-dimension n
    pub n: usize,
    /// Hamiltonian H(q, p) -> f64
    #[allow(clippy::type_complexity)]
    pub hamiltonian: Box<dyn Fn(&DVector<f64>, &DVector<f64>) -> f64>,
    /// Gradient ∇H as 2n vector: [∂H/∂q; ∂H/∂p]
    #[allow(clippy::type_complexity)]
    pub grad_hamiltonian: Box<dyn Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>>,
}

impl HamiltonianSystem {
    /// Create a new Hamiltonian system.
    pub fn new<H, G>(name: &str, n: usize, h: H, grad_h: G) -> Self
    where
        H: Fn(&DVector<f64>, &DVector<f64>) -> f64 + 'static,
        G: Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64> + 'static,
    {
        Self {
            name: name.to_string(),
            n,
            hamiltonian: Box::new(h),
            grad_hamiltonian: Box::new(grad_h),
        }
    }

    /// Get the full dimension 2n.
    pub fn dim(&self) -> usize {
        2 * self.n
    }

    /// Evaluate the Hamiltonian at state (q, p).
    pub fn energy(&self, q: &DVector<f64>, p: &DVector<f64>) -> f64 {
        (self.hamiltonian)(q, p)
    }

    /// Compute the Hamiltonian vector field X_H = J∇H at state (q, p).
    pub fn vector_field(&self, q: &DVector<f64>, p: &DVector<f64>) -> DVector<f64> {
        let grad = (self.grad_hamiltonian)(q, p);
        let mut result = DVector::zeros(2 * self.n);
        for i in 0..self.n {
            result[i] = grad[self.n + i];           // dq/dt = ∂H/∂p
            result[self.n + i] = -grad[i];          // dp/dt = -∂H/∂q
        }
        result
    }

    /// Create the canonical simple harmonic oscillator H = (p^2 + q^2)/2.
    pub fn harmonic_oscillator() -> Self {
        Self::new(
            "Harmonic Oscillator",
            1,
            |q, p| (q[0].powi(2) + p[0].powi(2)) / 2.0,
            |q, p| DVector::from_vec(vec![q[0], p[0]]),
        )
    }

    /// Create the mathematical pendulum: H = p^2/2 + (1 - cos(q)).
    pub fn pendulum() -> Self {
        Self::new(
            "Pendulum",
            1,
            |q, p| p[0].powi(2) / 2.0 + 1.0 - q[0].cos(),
            |q, p| DVector::from_vec(vec![q[0].sin(), p[0]]),
        )
    }

    /// Create the Kepler problem in 2D (4D phase space).
    pub fn kepler_2d() -> Self {
        Self::new(
            "Kepler 2D",
            2,
            |q, p| {
                let r = (q[0].powi(2) + q[1].powi(2)).sqrt();
                (p[0].powi(2) + p[1].powi(2)) / 2.0 - 1.0 / r
            },
            |q, p| {
                let r = (q[0].powi(2) + q[1].powi(2)).sqrt();
                let r3 = r.powi(3);
                DVector::from_vec(vec![
                    q[0] / r3,
                    q[1] / r3,
                    p[0],
                    p[1],
                ])
            },
        )
    }

    /// Symplectic gradient (same as vector field).
    pub fn symplectic_gradient(&self, q: &DVector<f64>, p: &DVector<f64>) -> DVector<f64> {
        self.vector_field(q, p)
    }

    /// Numerically verify energy conservation along a trajectory.
    pub fn energy_drift(&self, trajectory: &[(DVector<f64>, DVector<f64>)]) -> f64 {
        if trajectory.is_empty() {
            return 0.0;
        }
        let e0 = self.energy(&trajectory[0].0, &trajectory[0].1);
        let mut max_drift: f64 = 0.0;
        for (q, p) in trajectory {
            let e = self.energy(q, p);
            max_drift = max_drift.max((e - e0).abs());
        }
        max_drift
    }
}

// Manual Clone impl since Box<dyn Fn> doesn't impl Clone
impl Clone for HamiltonianSystem {
    fn clone(&self) -> Self {
        // We can't clone the closures, so we re-create from known systems
        // This is a limitation — users should reconstruct as needed
        panic!("HamiltonianSystem cannot be cloned — reconstruct from constructor");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_harmonic_oscillator_energy() {
        let sys = HamiltonianSystem::harmonic_oscillator();
        let q = DVector::from_vec(vec![1.0]);
        let p = DVector::from_vec(vec![0.0]);
        assert_abs_diff_eq!(sys.energy(&q, &p), 0.5);
    }

    #[test]
    fn test_harmonic_oscillator_vector_field() {
        let sys = HamiltonianSystem::harmonic_oscillator();
        let q = DVector::from_vec(vec![1.0]);
        let p = DVector::from_vec(vec![0.0]);
        let vf = sys.vector_field(&q, &p);
        assert_abs_diff_eq!(vf[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(vf[1], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_harmonic_oscillator_at_origin() {
        let sys = HamiltonianSystem::harmonic_oscillator();
        let q = DVector::from_vec(vec![0.0]);
        let p = DVector::from_vec(vec![0.0]);
        let vf = sys.vector_field(&q, &p);
        assert_abs_diff_eq!(vf[0], 0.0);
        assert_abs_diff_eq!(vf[1], 0.0);
    }

    #[test]
    fn test_pendulum_equilibrium() {
        let sys = HamiltonianSystem::pendulum();
        let q = DVector::from_vec(vec![0.0]);
        let p = DVector::from_vec(vec![0.0]);
        assert_abs_diff_eq!(sys.energy(&q, &p), 0.0);
        let vf = sys.vector_field(&q, &p);
        assert_abs_diff_eq!(vf[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(vf[1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pendulum_vector_field() {
        let sys = HamiltonianSystem::pendulum();
        let q = DVector::from_vec(vec![std::f64::consts::FRAC_PI_2]);
        let p = DVector::from_vec(vec![0.0]);
        let vf = sys.vector_field(&q, &p);
        assert_abs_diff_eq!(vf[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(vf[1], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kepler_2d_circular_orbit() {
        let sys = HamiltonianSystem::kepler_2d();
        let q = DVector::from_vec(vec![1.0, 0.0]);
        let p = DVector::from_vec(vec![0.0, 1.0]);
        let vf = sys.vector_field(&q, &p);
        assert_abs_diff_eq!(vf[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(vf[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(vf[2], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(vf[3], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_energy_drift_empty_trajectory() {
        let sys = HamiltonianSystem::harmonic_oscillator();
        assert_abs_diff_eq!(sys.energy_drift(&[]), 0.0);
    }

    #[test]
    fn test_energy_drift_constant() {
        let sys = HamiltonianSystem::harmonic_oscillator();
        let traj = vec![
            (DVector::from_vec(vec![1.0]), DVector::from_vec(vec![0.0])),
            (DVector::from_vec(vec![0.0]), DVector::from_vec(vec![1.0])),
            (DVector::from_vec(vec![-1.0]), DVector::from_vec(vec![0.0])),
        ];
        assert_abs_diff_eq!(sys.energy_drift(&traj), 0.0, epsilon = 1e-10);
    }
}
