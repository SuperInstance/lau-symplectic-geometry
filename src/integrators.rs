//! Symplectic integrators: Symplectic Euler and Störmer-Verlet.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

use crate::hamiltonian::HamiltonianSystem;

/// Symplectic Euler integrator (first-order symplectic method).
///
/// The symplectic Euler method preserves the symplectic structure exactly
/// (but not the energy exactly — it has bounded energy error).
///
/// For H(q,p) with equations dq/dt = ∂H/∂p, dp/dt = -∂H/∂q:
///   p_{n+1} = p_n - dt * ∂H/∂q(q_n, p_{n+1})  (implicit in p)
///   q_{n+1} = q_n + dt * ∂H/∂p(q_n, p_{n+1})  (explicit)
///
/// Or the variant:
///   q_{n+1} = q_n + dt * ∂H/∂p(q_{n+1}, p_n)
///   p_{n+1} = p_n - dt * ∂H/∂q(q_{n+1}, p_n)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymplecticEuler {
    /// Time step
    pub dt: f64,
}

impl SymplecticEuler {
    /// Create a new Symplectic Euler integrator with given time step.
    pub fn new(dt: f64) -> Self {
        Self { dt }
    }

    /// Perform one step of symplectic Euler (position variant).
    /// Uses the separable structure: H(q,p) = T(p) + V(q).
    ///
    /// Step 1: p_{n+1} = p_n - dt * ∇V(q_n)
    /// Step 2: q_{n+1} = q_n + dt * ∇T(p_{n+1})
    pub fn step_separable(
        &self,
        q: &DVector<f64>,
        p: &DVector<f64>,
        grad_v: fn(&DVector<f64>) -> DVector<f64>,
        grad_t: fn(&DVector<f64>) -> DVector<f64>,
    ) -> (DVector<f64>, DVector<f64>) {
        let p_new = p - self.dt * grad_v(q);
        let q_new = q + self.dt * grad_t(&p_new);
        (q_new, p_new)
    }

    /// Perform one step using the general Hamiltonian system.
    /// For the general case, we use a simple splitting approach.
    pub fn step(
        &self,
        q: &DVector<f64>,
        p: &DVector<f64>,
        system: &HamiltonianSystem,
    ) -> (DVector<f64>, DVector<f64>) {
        let vf = system.vector_field(q, p);
        let n = system.n;
        let mut q_new = q.clone();
        let mut p_new = p.clone();
        // Symplectic Euler: update p first, then q with new p
        for i in 0..n {
            p_new[i] += self.dt * vf[n + i];     // dp/dt
        }
        let vf_new = system.vector_field(q, &p_new);
        for i in 0..n {
            q_new[i] += self.dt * vf_new[i]; // dq/dt
        }
        (q_new, p_new)
    }

    /// Integrate for multiple steps.
    pub fn integrate(
        &self,
        q0: &DVector<f64>,
        p0: &DVector<f64>,
        system: &HamiltonianSystem,
        n_steps: usize,
    ) -> Vec<(DVector<f64>, DVector<f64>)> {
        let mut trajectory = Vec::with_capacity(n_steps + 1);
        trajectory.push((q0.clone(), p0.clone()));
        let (mut q, mut p) = (q0.clone(), p0.clone());
        for _ in 0..n_steps {
            let (q_new, p_new) = self.step(&q, &p, system);
            q = q_new;
            p = p_new;
            trajectory.push((q.clone(), p.clone()));
        }
        trajectory
    }
}

/// Störmer-Verlet (leapfrog) integrator (second-order symplectic method).
///
/// This is the gold standard for Hamiltonian systems. It's:
/// - Second-order accurate
/// - Symplectic (preserves the symplectic form)
/// - Time-reversible
/// - Has excellent long-term energy conservation
///
/// For separable H(q,p) = T(p) + V(q):
///   p_{n+1/2} = p_n - (dt/2) * ∇V(q_n)
///   q_{n+1}   = q_n + dt * ∇T(p_{n+1/2})
///   p_{n+1}   = p_{n+1/2} - (dt/2) * ∇V(q_{n+1})
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StormerVerlet {
    /// Time step
    pub dt: f64,
}

impl StormerVerlet {
    /// Create a new Störmer-Verlet integrator with given time step.
    pub fn new(dt: f64) -> Self {
        Self { dt }
    }

    /// Perform one step of Störmer-Verlet for separable Hamiltonians.
    pub fn step_separable(
        &self,
        q: &DVector<f64>,
        p: &DVector<f64>,
        grad_v: fn(&DVector<f64>) -> DVector<f64>,
        grad_t: fn(&DVector<f64>) -> DVector<f64>,
    ) -> (DVector<f64>, DVector<f64>) {
        let half_dt = 0.5 * self.dt;
        // Half-step momentum
        let p_half = p - half_dt * grad_v(q);
        // Full-step position
        let q_new = q + self.dt * grad_t(&p_half);
        // Half-step momentum
        let p_new = p_half - half_dt * grad_v(&q_new);
        (q_new, p_new)
    }

    /// Perform one step using a general Hamiltonian system.
    pub fn step(
        &self,
        q: &DVector<f64>,
        p: &DVector<f64>,
        system: &HamiltonianSystem,
    ) -> (DVector<f64>, DVector<f64>) {
        let half_dt = 0.5 * self.dt;
        let n = system.n;

        // Half-step momentum update
        let vf = system.vector_field(q, p);
        let mut p_half = p.clone();
        for i in 0..n {
            p_half[i] += half_dt * vf[n + i]; // dp/dt
        }

        // Full-step position update
        let vf_half = system.vector_field(q, &p_half);
        let mut q_new = q.clone();
        for i in 0..n {
            q_new[i] += self.dt * vf_half[i]; // dq/dt
        }

        // Half-step momentum update
        let vf_new = system.vector_field(&q_new, &p_half);
        let mut p_new = p_half;
        for i in 0..n {
            p_new[i] += half_dt * vf_new[n + i];
        }

        (q_new, p_new)
    }

    /// Integrate for multiple steps.
    pub fn integrate(
        &self,
        q0: &DVector<f64>,
        p0: &DVector<f64>,
        system: &HamiltonianSystem,
        n_steps: usize,
    ) -> Vec<(DVector<f64>, DVector<f64>)> {
        let mut trajectory = Vec::with_capacity(n_steps + 1);
        trajectory.push((q0.clone(), p0.clone()));
        let (mut q, mut p) = (q0.clone(), p0.clone());
        for _ in 0..n_steps {
            let (q_new, p_new) = self.step(&q, &p, system);
            q = q_new;
            p = p_new;
            trajectory.push((q.clone(), p.clone()));
        }
        trajectory
    }

    /// Verify time-reversibility: stepping forward then backward returns to start.
    pub fn verify_time_reversibility(
        &self,
        q: &DVector<f64>,
        p: &DVector<f64>,
        system: &HamiltonianSystem,
    ) -> bool {
        let (q_fwd, p_fwd) = self.step(q, p, system);
        // Reverse momentum
        let p_rev = -&p_fwd;
        // Create reverse integrator
        let reverse = StormerVerlet { dt: -self.dt };
        let (q_back, p_back) = reverse.step(&q_fwd, &p_rev, system);
        // Check we're back
        let p_back_unrev = -p_back;
        let q_err = (q - &q_back).norm();
        let p_err = (p - &p_back_unrev).norm();
        q_err < 1e-8 && p_err < 1e-8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ho_grad_v(q: &DVector<f64>) -> DVector<f64> {
        q.clone() // V = q^2/2, ∇V = q
    }

    fn ho_grad_t(p: &DVector<f64>) -> DVector<f64> {
        p.clone() // T = p^2/2, ∇T = p
    }

    #[test]
    fn test_symplectic_euler_harmonic_oscillator() {
        let integrator = SymplecticEuler::new(0.01);
        let _sys = HamiltonianSystem::harmonic_oscillator();
        let q0 = DVector::from_vec(vec![1.0]);
        let p0 = DVector::from_vec(vec![0.0]);

        let (q1, p1) = integrator.step_separable(&q0, &p0, ho_grad_v, ho_grad_t);
        // p should decrease (dp = -dt * q = -0.01)
        assert!(p1[0] < 0.0);
        // q should stay near 1 (small step)
        assert!(q1[0] > 0.9);
    }

    #[test]
    fn test_symplectic_euler_energy_bounded() {
        let integrator = SymplecticEuler::new(0.01);
        let sys = HamiltonianSystem::harmonic_oscillator();
        let q0 = DVector::from_vec(vec![1.0]);
        let p0 = DVector::from_vec(vec![0.0]);

        let trajectory = integrator.integrate(&q0, &p0, &sys, 1000);
        let drift = sys.energy_drift(&trajectory);
        // Symplectic methods have bounded energy error
        assert!(drift < 0.1, "Energy drift {} should be bounded", drift);
    }

    #[test]
    fn test_stormer_verlet_harmonic_oscillator() {
        let integrator = StormerVerlet::new(0.01);
        let q0 = DVector::from_vec(vec![1.0]);
        let p0 = DVector::from_vec(vec![0.0]);

        let (q1, p1) = integrator.step_separable(&q0, &p0, ho_grad_v, ho_grad_t);
        assert!(p1[0] < 0.0);
        assert!(q1[0] > 0.9);
    }

    #[test]
    fn test_stormer_verlet_energy_conservation() {
        let integrator = StormerVerlet::new(0.01);
        let sys = HamiltonianSystem::harmonic_oscillator();
        let q0 = DVector::from_vec(vec![1.0]);
        let p0 = DVector::from_vec(vec![0.0]);

        let trajectory = integrator.integrate(&q0, &p0, &sys, 10000);
        let drift = sys.energy_drift(&trajectory);
        // Störmer-Verlet should have very small energy drift
        assert!(
            drift < 0.01,
            "Energy drift {} should be very small for Verlet",
            drift
        );
    }

    #[test]
    fn test_stormer_verlet_time_reversibility() {
        let integrator = StormerVerlet::new(0.1);
        let q = DVector::from_vec(vec![1.0]);
        let p = DVector::from_vec(vec![0.5]);

        // Forward step
        let (q_fwd, p_fwd) = integrator.step_separable(&q, &p, ho_grad_v, ho_grad_t);
        // Time reversibility: step forward from (q_fwd, -p_fwd) should return to (q, -p)
        let p_rev = -p_fwd;
        let (q_back, p_back) = integrator.step_separable(&q_fwd, &p_rev, ho_grad_v, ho_grad_t);
        let p_back_unrev = -p_back;

        let q_err = (q - &q_back).norm();
        let p_err = (p - &p_back_unrev).norm();
        assert!(q_err < 1e-10, "q error: {}", q_err);
        assert!(p_err < 1e-10, "p error: {}", p_err);
    }

    #[test]
    fn test_stormer_verlet_better_than_euler() {
        let euler = SymplecticEuler::new(0.01);
        let verlet = StormerVerlet::new(0.01);
        let sys = HamiltonianSystem::harmonic_oscillator();
        let q0 = DVector::from_vec(vec![1.0]);
        let p0 = DVector::from_vec(vec![0.0]);

        let euler_traj = euler.integrate(&q0, &p0, &sys, 10000);
        let verlet_traj = verlet.integrate(&q0, &p0, &sys, 10000);

        let euler_drift = sys.energy_drift(&euler_traj);
        let verlet_drift = sys.energy_drift(&verlet_traj);

        assert!(
            verlet_drift <= euler_drift + 1e-10,
            "Verlet (drift={}) should be at least as good as Euler (drift={})",
            verlet_drift,
            euler_drift
        );
    }

    #[test]
    fn test_verlet_trajectory_length() {
        let integrator = StormerVerlet::new(0.01);
        let sys = HamiltonianSystem::harmonic_oscillator();
        let q0 = DVector::from_vec(vec![1.0]);
        let p0 = DVector::from_vec(vec![0.0]);

        let traj = integrator.integrate(&q0, &p0, &sys, 100);
        assert_eq!(traj.len(), 101); // 100 steps + initial
    }

    #[test]
    fn test_euler_trajectory_length() {
        let integrator = SymplecticEuler::new(0.01);
        let sys = HamiltonianSystem::harmonic_oscillator();
        let q0 = DVector::from_vec(vec![1.0]);
        let p0 = DVector::from_vec(vec![0.0]);

        let traj = integrator.integrate(&q0, &p0, &sys, 100);
        assert_eq!(traj.len(), 101);
    }

    #[test]
    fn test_verlet_period_approximation() {
        let integrator = StormerVerlet::new(0.001);
        let sys = HamiltonianSystem::harmonic_oscillator();
        let q0 = DVector::from_vec(vec![1.0]);
        let p0 = DVector::from_vec(vec![0.0]);

        let traj = integrator.integrate(&q0, &p0, &sys, 10000);
        // Find first return to initial state (period ≈ 2π)
        // Skip early steps to avoid matching the initial transient
        let mut first_return = None;
        for (i, (q, _p)) in traj.iter().enumerate().skip(2000) {
            if (q[0] - 1.0).abs() < 0.001 {
                first_return = Some(i);
                break;
            }
        }
        assert!(first_return.is_some());
        let period = first_return.unwrap() as f64 * 0.001;
        assert!(
            (period - 2.0 * std::f64::consts::PI).abs() < 0.1,
            "Period {} should be close to 2π",
            period
        );
    }
}
