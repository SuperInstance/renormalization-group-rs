//! RG fixed points and their stability analysis.

/// A fixed point of the renormalization group transformation.
#[derive(Debug, Clone)]
pub struct FixedPoint {
    pub coordinates: Vec<f64>,
    pub stability: StabilityType,
    pub label: String,
}

/// Stability classification of a fixed point.
#[derive(Debug, Clone, PartialEq)]
pub enum StabilityType {
    /// All relevant directions flow away (UV stable).
    Stable,
    /// At least one relevant direction (IR attractive).
    Unstable,
    /// Marginal directions exist.
    Marginal,
}

impl FixedPoint {
    pub fn new(coordinates: Vec<f64>, stability: StabilityType, label: &str) -> Self {
        Self {
            coordinates,
            stability,
            label: label.to_string(),
        }
    }

    pub fn dimension(&self) -> usize {
        self.coordinates.len()
    }

    /// Distance to another point.
    pub fn distance_to(&self, other: &[f64]) -> f64 {
        self.coordinates
            .iter()
            .zip(other.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }
}

/// Linearized RG transformation around a fixed point.
/// The Jacobian matrix determines stability.
#[derive(Debug, Clone)]
pub struct LinearizedRG {
    pub fixed_point: Vec<f64>,
    pub jacobian: Vec<Vec<f64>>,
}

impl LinearizedRG {
    pub fn new(fixed_point: Vec<f64>, jacobian: Vec<Vec<f64>>) -> Self {
        Self {
            fixed_point,
            jacobian,
        }
    }

    /// Compute eigenvalues of the Jacobian using power iteration.
    pub fn eigenvalues(&self, iterations: usize) -> Vec<f64> {
        let n = self.jacobian.len();
        if n == 0 {
            return vec![];
        }

        let mut eigenvalues = Vec::new();
        let mut deflated = self.jacobian.clone();

        for _ in 0..n {
            let ev = power_iteration_eigenvalue(&deflated, iterations);
            eigenvalues.push(ev);

            // Deflate
            let vec = power_iteration_vector(&deflated, iterations);
            deflate_matrix(&mut deflated, ev, &vec);
        }

        eigenvalues
    }

    /// Classify stability based on eigenvalue magnitudes.
    pub fn classify_stability(&self, iterations: usize) -> StabilityType {
        let eigenvalues = self.eigenvalues(iterations);
        let mut has_marginal = false;
        for ev in &eigenvalues {
            let mag = ev.abs();
            if mag > 1.0 + 1e-6 {
                return StabilityType::Unstable;
            }
            if (mag - 1.0).abs() < 1e-6 {
                has_marginal = true;
            }
        }
        if has_marginal {
            StabilityType::Marginal
        } else {
            StabilityType::Stable
        }
    }

    /// Count relevant directions (|eigenvalue| > 1).
    pub fn relevant_directions(&self, iterations: usize) -> usize {
        self.eigenvalues(iterations)
            .iter()
            .filter(|ev| ev.abs() > 1.0 + 1e-6)
            .count()
    }

    /// Count irrelevant directions (|eigenvalue| < 1).
    pub fn irrelevant_directions(&self, iterations: usize) -> usize {
        self.eigenvalues(iterations)
            .iter()
            .filter(|ev| ev.abs() < 1.0 - 1e-6)
            .count()
    }
}

fn power_iteration_eigenvalue(mat: &[Vec<f64>], iterations: usize) -> f64 {
    let n = mat.len();
    let mut v = vec![1.0; n];
    for _ in 0..iterations {
        let mut new_v = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                new_v[i] += mat[i][j] * v[j];
            }
        }
        let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for x in new_v.iter_mut() {
                *x /= norm;
            }
        }
        v = new_v;
    }
    // Rayleigh quotient
    let mut mv = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            mv[i] += mat[i][j] * v[j];
        }
    }
    v.iter().zip(mv.iter()).map(|(a, b)| a * b).sum()
}

fn power_iteration_vector(mat: &[Vec<f64>], iterations: usize) -> Vec<f64> {
    let n = mat.len();
    let mut v = vec![1.0; n];
    for _ in 0..iterations {
        let mut new_v = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                new_v[i] += mat[i][j] * v[j];
            }
        }
        let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for x in new_v.iter_mut() {
                *x /= norm;
            }
        }
        v = new_v;
    }
    v
}

fn deflate_matrix(mat: &mut [Vec<f64>], eigenvalue: f64, eigenvector: &[f64]) {
    let n = mat.len();
    let norm_sq: f64 = eigenvector.iter().map(|x| x * x).sum();
    if norm_sq < 1e-15 {
        return;
    }
    for i in 0..n {
        for j in 0..n {
            mat[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j] / norm_sq;
        }
    }
}

/// Find a fixed point by iterating the RG transformation.
pub fn find_fixed_point(
    transform: &dyn Fn(&[f64]) -> Vec<f64>,
    initial: &[f64],
    max_iterations: usize,
    tolerance: f64,
) -> Option<FixedPoint> {
    let mut current = initial.to_vec();
    for _ in 0..max_iterations {
        let next = transform(&current);
        let dist: f64 = current
            .iter()
            .zip(next.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        if dist < tolerance {
            return Some(FixedPoint::new(next, StabilityType::Stable, "found"));
        }
        current = next;
    }
    None
}

/// Trivial (Gaussian) fixed point — all zeros.
pub fn gaussian_fixed_point(dim: usize) -> FixedPoint {
    FixedPoint::new(vec![0.0; dim], StabilityType::Stable, "gaussian")
}

/// Wilson-Fisher fixed point proxy (non-trivial fixed point in 4-ε dimensions).
pub fn wilson_fisher_proxy(epsilon: f64) -> FixedPoint {
    // Approximate WF coordinates for φ⁴ theory
    let coords = vec![epsilon / 6.0, -(epsilon * epsilon) / 54.0];
    FixedPoint::new(coords, StabilityType::Unstable, "wilson-fisher")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_point_distance() {
        let fp = FixedPoint::new(vec![1.0, 0.0], StabilityType::Stable, "test");
        assert!((fp.distance_to(&[4.0, 4.0]) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_point_dimension() {
        let fp = FixedPoint::new(vec![1.0, 2.0, 3.0], StabilityType::Stable, "test");
        assert_eq!(fp.dimension(), 3);
    }

    #[test]
    fn test_linearized_rg_identity_is_stable() {
        let rg = LinearizedRG::new(vec![0.0], vec![vec![0.5]]);
        assert_eq!(rg.classify_stability(20), StabilityType::Stable);
    }

    #[test]
    fn test_linearized_rg_unstable() {
        let rg = LinearizedRG::new(vec![0.0], vec![vec![2.0]]);
        assert_eq!(rg.classify_stability(20), StabilityType::Unstable);
    }

    #[test]
    fn test_relevant_directions() {
        let rg = LinearizedRG::new(
            vec![0.0, 0.0],
            vec![vec![2.0, 0.0], vec![0.0, 0.5]],
        );
        assert_eq!(rg.relevant_directions(20), 1);
        assert_eq!(rg.irrelevant_directions(20), 1);
    }

    #[test]
    fn test_find_fixed_point_identity() {
        let transform = |v: &[f64]| v.to_vec();
        let result = find_fixed_point(&transform, &[0.5, 0.3], 100, 1e-10);
        assert!(result.is_some());
        let fp = result.unwrap();
        assert!((fp.coordinates[0] - 0.5).abs() < 1e-8);
    }

    #[test]
    fn test_find_fixed_point_contraction() {
        let transform = |v: &[f64]| vec![v[0] * 0.5];
        let result = find_fixed_point(&transform, &[1.0], 100, 1e-8);
        assert!(result.is_some());
        assert!(result.unwrap().coordinates[0].abs() < 1e-6);
    }

    #[test]
    fn test_gaussian_fixed_point() {
        let fp = gaussian_fixed_point(3);
        assert_eq!(fp.coordinates, vec![0.0, 0.0, 0.0]);
        assert_eq!(fp.stability, StabilityType::Stable);
    }

    #[test]
    fn test_wilson_fisher_proxy() {
        let fp = wilson_fisher_proxy(1.0);
        assert_eq!(fp.label, "wilson-fisher");
        assert!(!fp.coordinates.is_empty());
    }

    #[test]
    fn test_2x2_eigenvalues() {
        let rg = LinearizedRG::new(
            vec![0.0, 0.0],
            vec![vec![3.0, 1.0], vec![0.0, 2.0]],
        );
        let eigenvalues = rg.eigenvalues(30);
        assert_eq!(eigenvalues.len(), 2);
        // Dominant eigenvalue should be close to 3
        assert!((eigenvalues[0] - 3.0).abs() < 0.5);
    }
}
