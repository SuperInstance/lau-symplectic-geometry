//! Poincaré recurrence theorem: in a finite measure-preserving system,
//! almost every point returns arbitrarily close to its initial position.

use nalgebra::DVector;

/// Poincaré recurrence theorem implementation.
///
/// Statement: If Φ_t is a Hamiltonian flow on a bounded region R of phase space,
/// then for almost every point x ∈ R, there exists a sequence t_n → ∞ such that
/// Φ_{t_n}(x) → x.
///
/// Equivalently: the system returns arbitrarily close to its initial state
/// infinitely often.
pub struct PoincareRecurrence {
    /// Tolerance for "close to initial state"
    pub tolerance: f64,
}

impl PoincareRecurrence {
    /// Create a new Poincaré recurrence checker with given tolerance.
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// Check if a trajectory returns close to its initial point.
    /// Returns the first index where |x_i - x_0| < tolerance.
    pub fn find_first_return(&self, trajectory: &[DVector<f64>]) -> Option<usize> {
        if trajectory.len() < 2 {
            return None;
        }
        let initial = &trajectory[0];
        for (i, point) in trajectory.iter().enumerate().skip(1) {
            if (point - initial).norm() < self.tolerance {
                return Some(i);
            }
        }
        None
    }

    /// Find all recurrence times (indices where trajectory returns close to start).
    pub fn find_all_returns(&self, trajectory: &[DVector<f64>]) -> Vec<usize> {
        if trajectory.len() < 2 {
            return vec![];
        }
        let initial = &trajectory[0];
        trajectory
            .iter()
            .enumerate()
            .skip(1)
            .filter_map(|(i, point)| {
                if (point - initial).norm() < self.tolerance {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Estimate the Poincaré recurrence time from a trajectory.
    /// Returns the average interval between returns.
    pub fn estimate_recurrence_time(&self, trajectory: &[DVector<f64>], dt: f64) -> Option<f64> {
        let returns = self.find_all_returns(trajectory);
        if returns.len() < 2 {
            return None;
        }
        let total_intervals = returns.len() - 1;
        let total_time = (returns[returns.len() - 1] - returns[0]) as f64 * dt;
        Some(total_time / total_intervals as f64)
    }

    /// Compute the minimum distance from the initial point over the trajectory.
    pub fn min_return_distance(&self, trajectory: &[DVector<f64>]) -> f64 {
        if trajectory.len() < 2 {
            return f64::INFINITY;
        }
        let initial = &trajectory[0];
        trajectory
            .iter()
            .skip(1)
            .map(|p| (p - initial).norm())
            .fold(f64::INFINITY, f64::min)
    }

    /// Check Poincaré recurrence for a 2D phase space trajectory.
    /// Simulate a simple harmonic oscillator and verify recurrence.
    pub fn verify_harmonic_oscillator_recurrence(
        n_steps: usize,
        dt: f64,
        tolerance: f64,
    ) -> RecurrenceResult {
        let mut trajectory = Vec::with_capacity(n_steps);
        let q0 = 1.0;
        let _p0 = 0.0;
        let omega = 1.0; // frequency

        for i in 0..n_steps {
            let t = i as f64 * dt;
            let q = q0 * (omega * t).cos();
            let p = -q0 * (omega * t).sin();
            trajectory.push(DVector::from_vec(vec![q, p]));
        }

        let checker = Self::new(tolerance);
        let first_return = checker.find_first_return(&trajectory);
        let all_returns = checker.find_all_returns(&trajectory);
        let min_dist = checker.min_return_distance(&trajectory);

        RecurrenceResult {
            first_return,
            num_returns: all_returns.len(),
            min_distance: min_dist,
            n_steps,
            tolerance,
        }
    }

    /// Compute the Poincaré section at a given q value.
    /// Records (p, t) whenever the trajectory crosses q = q_section.
    pub fn poincare_section(
        trajectory: &[DVector<f64>],
        dt: f64,
        q_index: usize,
        q_value: f64,
    ) -> Vec<(f64, f64)> {
        let mut section = Vec::new();
        for i in 1..trajectory.len() {
            let q_prev = trajectory[i - 1][q_index];
            let q_curr = trajectory[i][q_index];
            // Detect crossing: q_prev < q_value <= q_curr or q_prev > q_value >= q_curr
            if (q_prev - q_value) * (q_curr - q_value) < 0.0 {
                // Linear interpolation
                let frac = (q_value - q_prev) / (q_curr - q_prev);
                let t = (i - 1) as f64 * dt + frac * dt;
                let p = trajectory[i - 1][trajectory[0].nrows() / 2 + q_index]
                    * (1.0 - frac)
                    + trajectory[i][trajectory[0].nrows() / 2 + q_index] * frac;
                section.push((p, t));
            }
        }
        section
    }
}

/// Result of a Poincaré recurrence analysis.
#[derive(Debug, Clone)]
pub struct RecurrenceResult {
    /// First return index (None if no return found)
    pub first_return: Option<usize>,
    /// Total number of returns
    pub num_returns: usize,
    /// Minimum distance from initial point
    pub min_distance: f64,
    /// Number of steps in trajectory
    pub n_steps: usize,
    /// Tolerance used
    pub tolerance: f64,
}

impl std::fmt::Display for RecurrenceResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Poincaré recurrence: first_return={:?}, num_returns={}, min_dist={:.6}, tolerance={:.6}",
            self.first_return, self.num_returns, self.min_distance, self.tolerance
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_harmonic_oscillator_recurrence() {
        // Period of harmonic oscillator with ω=1 is 2π ≈ 6.28
        let result = PoincareRecurrence::verify_harmonic_oscillator_recurrence(
            10000, 0.01, 0.005,
        );
        assert!(result.first_return.is_some());
        // First return should be near step 628 (2π / 0.01)
        if let Some(idx) = result.first_return {
            assert!((idx as f64 - 628.0).abs() < 10.0);
        }
    }

    #[test]
    fn test_find_first_return_empty() {
        let checker = PoincareRecurrence::new(0.1);
        let traj: Vec<DVector<f64>> = vec![DVector::from_vec(vec![1.0, 0.0])];
        assert!(checker.find_first_return(&traj).is_none());
    }

    #[test]
    fn test_find_first_return_single_point() {
        let checker = PoincareRecurrence::new(0.1);
        let traj = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![0.05, 0.0]),
        ];
        assert_eq!(checker.find_first_return(&traj), Some(1));
    }

    #[test]
    fn test_find_all_returns() {
        let checker = PoincareRecurrence::new(0.1);
        let traj = vec![
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![2.0, 0.0]),
            DVector::from_vec(vec![0.99, 0.0]),
            DVector::from_vec(vec![3.0, 0.0]),
            DVector::from_vec(vec![1.01, 0.0]),
        ];
        let returns = checker.find_all_returns(&traj);
        assert_eq!(returns, vec![2, 4]);
    }

    #[test]
    fn test_min_return_distance() {
        let checker = PoincareRecurrence::new(0.1);
        let traj = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![5.0, 0.0]),
            DVector::from_vec(vec![0.1, 0.0]),
        ];
        assert_abs_diff_eq!(checker.min_return_distance(&traj), 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_recurrence_result_display() {
        let result = RecurrenceResult {
            first_return: Some(42),
            num_returns: 3,
            min_distance: 0.001,
            n_steps: 100,
            tolerance: 0.01,
        };
        let s = format!("{}", result);
        assert!(s.contains("42"));
        assert!(s.contains("3"));
    }

    #[test]
    fn test_poincare_section() {
        // Simple trajectory crossing q=0 twice
        let traj = vec![
            DVector::from_vec(vec![-1.0, 0.5]),  // q=-1
            DVector::from_vec(vec![1.0, 0.5]),   // q=1 (crossing!)
            DVector::from_vec(vec![0.5, 0.5]),   // q=0.5
            DVector::from_vec(vec![-0.5, 0.5]),  // q=-0.5 (crossing!)
        ];
        let section = PoincareRecurrence::poincare_section(&traj, 0.1, 0, 0.0);
        assert_eq!(section.len(), 2);
    }

    #[test]
    fn test_estimate_recurrence_time() {
        let checker = PoincareRecurrence::new(0.1);
        let traj = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.05, 0.0]),
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.05, 0.0]),
        ];
        let recurrence_time = checker.estimate_recurrence_time(&traj, 0.1);
        assert!(recurrence_time.is_some());
        // Returns at indices 2, 4 → time = (4-2)*0.1 / 1 = 0.2
        assert_abs_diff_eq!(recurrence_time.unwrap(), 0.2, epsilon = 1e-10);
    }
}
