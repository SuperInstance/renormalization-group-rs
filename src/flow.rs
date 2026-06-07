//! RG flow equations — tracking how coupling constants evolve under scale changes.

/// An RG flow trajectory: a sequence of coupling constants at different scales.
#[derive(Debug, Clone)]
pub struct FlowTrajectory {
    pub points: Vec<Vec<f64>>,
    pub scales: Vec<f64>,
}

impl FlowTrajectory {
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            scales: Vec::new(),
        }
    }

    pub fn add_point(&mut self, coupling: Vec<f64>, scale: f64) {
        self.points.push(coupling);
        self.scales.push(scale);
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Get the beta function (rate of change) between consecutive points.
    pub fn beta_function(&self) -> Vec<Vec<f64>> {
        self.points
            .windows(2)
            .map(|w| {
                let ds = if w.len() > 1 {
                    let s0 = self.scales[self.points.iter().position(|p| *p == w[0]).unwrap_or(0)];
                    let s1 = self.scales[self.points.iter().position(|p| *p == w[1]).unwrap_or(1)];
                    if (s1 - s0).abs() < 1e-15 {
                        1.0
                    } else {
                        s1 - s0
                    }
                } else {
                    1.0
                };
                w[1].iter().zip(w[0].iter()).map(|(a, b)| (a - b) / ds).collect()
            })
            .collect()
    }

    /// Check if the flow has converged (last few points are close).
    pub fn has_converged(&self, tolerance: f64) -> bool {
        if self.points.len() < 3 {
            return false;
        }
        let n = self.points.len();
        for i in (n - 3)..n.saturating_sub(1) {
            let dist: f64 = self.points[i]
                .iter()
                .zip(self.points[i + 1].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            if dist > tolerance {
                return false;
            }
        }
        true
    }
}

impl Default for FlowTrajectory {
    fn default() -> Self {
        Self::new()
    }
}

type BetaFn = Box<dyn Fn(&[f64]) -> Vec<f64>>;

/// Beta function: defines how coupling constants change with scale.
/// β(g) = dg / d(ln Λ)
pub struct BetaFunction {
    /// The beta function as a closure.
    pub f: BetaFn,
}

impl BetaFunction {
    pub fn new(f: impl Fn(&[f64]) -> Vec<f64> + 'static) -> Self {
        Self { f: Box::new(f) }
    }

    /// Evaluate the beta function at given couplings.
    pub fn evaluate(&self, couplings: &[f64]) -> Vec<f64> {
        (self.f)(couplings)
    }

    /// Integrate the flow using Euler method.
    pub fn flow_euler(
        &self,
        initial: &[f64],
        scale_steps: usize,
        step_size: f64,
    ) -> FlowTrajectory {
        let mut traj = FlowTrajectory::new();
        let mut current = initial.to_vec();
        let mut scale = 0.0;
        traj.add_point(current.clone(), scale);

        for _ in 0..scale_steps {
            let beta = self.evaluate(&current);
            current = current.iter().zip(beta.iter()).map(|(g, b)| g + step_size * b).collect();
            scale += step_size;
            traj.add_point(current.clone(), scale);
        }

        traj
    }

    /// Integrate the flow using RK4.
    pub fn flow_rk4(
        &self,
        initial: &[f64],
        scale_steps: usize,
        step_size: f64,
    ) -> FlowTrajectory {
        let mut traj = FlowTrajectory::new();
        let mut current = initial.to_vec();
        let mut scale = 0.0;
        traj.add_point(current.clone(), scale);

        for _ in 0..scale_steps {
            let k1 = self.evaluate(&current);
            let k2_input: Vec<f64> = current.iter().zip(k1.iter()).map(|(g, k)| g + 0.5 * step_size * k).collect();
            let k2 = self.evaluate(&k2_input);
            let k3_input: Vec<f64> = current.iter().zip(k2.iter()).map(|(g, k)| g + 0.5 * step_size * k).collect();
            let k3 = self.evaluate(&k3_input);
            let k4_input: Vec<f64> = current.iter().zip(k3.iter()).map(|(g, k)| g + step_size * k).collect();
            let k4 = self.evaluate(&k4_input);

            for i in 0..current.len() {
                current[i] += step_size / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            }
            scale += step_size;
            traj.add_point(current.clone(), scale);
        }

        traj
    }
}

/// Linear beta function: β(g) = -ε g + a g² (typical phi-4 theory).
pub fn phi4_beta(epsilon: f64, a: f64) -> BetaFunction {
    BetaFunction::new(move |g: &[f64]| {
        g.iter().map(|&gi| -epsilon * gi + a * gi * gi).collect()
    })
}

/// Gaussian beta function: β(g) = -ε g (trivial fixed point at g=0).
pub fn gaussian_beta(epsilon: f64) -> BetaFunction {
    BetaFunction::new(move |g: &[f64]| g.iter().map(|&gi| -epsilon * gi).collect())
}

/// Check if a flow is asymptotically free (couplings → 0 at high energy).
pub fn is_asymptotically_free(traj: &FlowTrajectory) -> bool {
    if traj.points.len() < 2 {
        return false;
    }
    let first: f64 = traj.points.first().unwrap().iter().map(|x| x * x).sum();
    let last: f64 = traj.points.last().unwrap().iter().map(|x| x * x).sum();
    last < first * 0.01
}

/// Compute the critical exponent from the eigenvalue at a fixed point.
pub fn critical_exponent(eigenvalue: f64) -> f64 {
    if eigenvalue.abs() < 1e-15 {
        return 0.0;
    }
    eigenvalue.ln() / (2.0_f64).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_trajectory_add() {
        let mut traj = FlowTrajectory::new();
        traj.add_point(vec![1.0], 0.0);
        traj.add_point(vec![2.0], 1.0);
        assert_eq!(traj.len(), 2);
        assert!(!traj.is_empty());
    }

    #[test]
    fn test_flow_trajectory_beta() {
        let mut traj = FlowTrajectory::new();
        traj.add_point(vec![1.0], 0.0);
        traj.add_point(vec![2.0], 1.0);
        let beta = traj.beta_function();
        assert_eq!(beta.len(), 1);
        assert!((beta[0][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_convergence() {
        let mut traj = FlowTrajectory::new();
        traj.add_point(vec![1.0], 0.0);
        traj.add_point(vec![1.001], 1.0);
        traj.add_point(vec![1.0001], 2.0);
        assert!(traj.has_converged(0.01));
    }

    #[test]
    fn test_no_convergence() {
        let mut traj = FlowTrajectory::new();
        traj.add_point(vec![1.0], 0.0);
        traj.add_point(vec![5.0], 1.0);
        traj.add_point(vec![10.0], 2.0);
        assert!(!traj.has_converged(0.1));
    }

    #[test]
    fn test_euler_flow_gaussian() {
        let beta = gaussian_beta(1.0);
        let traj = beta.flow_euler(&[1.0], 10, 0.1);
        assert_eq!(traj.len(), 11);
        // Should decay toward zero
        assert!(traj.points.last().unwrap()[0] < 0.5);
    }

    #[test]
    fn test_rk4_flow_gaussian() {
        let beta = gaussian_beta(1.0);
        let traj = beta.flow_rk4(&[1.0], 10, 0.1);
        assert_eq!(traj.len(), 11);
        assert!(traj.points.last().unwrap()[0] < 0.5);
    }

    #[test]
    fn test_phi4_beta() {
        let beta = phi4_beta(1.0, 1.0);
        let result = beta.evaluate(&[0.5]);
        // β(0.5) = -1*0.5 + 1*0.25 = -0.25
        assert!((result[0] - (-0.25)).abs() < 1e-10);
    }

    #[test]
    fn test_asymptotically_free() {
        let mut traj = FlowTrajectory::new();
        traj.add_point(vec![1.0], 0.0);
        traj.add_point(vec![0.01], 1.0);
        traj.add_point(vec![0.001], 2.0);
        assert!(is_asymptotically_free(&traj));
    }

    #[test]
    fn test_not_asymptotically_free() {
        let mut traj = FlowTrajectory::new();
        traj.add_point(vec![0.1], 0.0);
        traj.add_point(vec![1.0], 1.0);
        traj.add_point(vec![5.0], 2.0);
        assert!(!is_asymptotically_free(&traj));
    }

    #[test]
    fn test_critical_exponent() {
        // eigenvalue = 2 => exponent = ln(2)/ln(2) = 1.0
        assert!((critical_exponent(2.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_critical_exponent_zero() {
        assert_eq!(critical_exponent(0.0), 0.0);
    }

    #[test]
    fn test_empty_trajectory() {
        let traj = FlowTrajectory::new();
        assert!(traj.is_empty());
        assert!(!traj.has_converged(0.1));
    }
}
