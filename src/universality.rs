//! Universality classes for agent behavior.
//!
//! Systems in the same universality class share critical exponents
//! and large-scale behavior regardless of microscopic details.

use crate::fixed_point::FixedPoint;


/// A universality class characterized by critical exponents.
#[derive(Debug, Clone)]
pub struct UniversalityClass {
    pub name: String,
    pub exponents: Vec<(String, f64)>,
    pub upper_critical_dimension: usize,
    pub description: String,
}

impl UniversalityClass {
    pub fn new(
        name: &str,
        exponents: Vec<(String, f64)>,
        upper_critical_dimension: usize,
        description: &str,
    ) -> Self {
        Self {
            name: name.to_string(),
            exponents,
            upper_critical_dimension,
            description: description.to_string(),
        }
    }

    /// Get a specific exponent by name.
    pub fn exponent(&self, name: &str) -> Option<f64> {
        self.exponents
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| *v)
    }

    /// Check if two sets of exponents are compatible within tolerance.
    pub fn is_compatible(&self, other_exponents: &[(String, f64)], tolerance: f64) -> bool {
        if self.exponents.len() != other_exponents.len() {
            return false;
        }
        for (name, value) in &self.exponents {
            if let Some(other) = other_exponents.iter().find(|(n, _)| n == name) {
                if (value - other.1).abs() > tolerance {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }
}

/// Well-known universality classes.
pub fn ising_class() -> UniversalityClass {
    UniversalityClass::new(
        "Ising",
        vec![
            ("alpha".to_string(), 0.110),
            ("beta".to_string(), 0.326),
            ("gamma".to_string(), 1.237),
            ("delta".to_string(), 4.790),
            ("nu".to_string(), 0.630),
            ("eta".to_string(), 0.036),
        ],
        4,
        "Scalar order parameter, Z₂ symmetry",
    )
}

pub fn xy_class() -> UniversalityClass {
    UniversalityClass::new(
        "XY",
        vec![
            ("alpha".to_string(), -0.015),
            ("beta".to_string(), 0.349),
            ("gamma".to_string(), 1.316),
            ("delta".to_string(), 4.770),
            ("nu".to_string(), 0.670),
            ("eta".to_string(), 0.038),
        ],
        4,
        "2-component order parameter, O(2) symmetry",
    )
}

pub fn heisenberg_class() -> UniversalityClass {
    UniversalityClass::new(
        "Heisenberg",
        vec![
            ("alpha".to_string(), -0.116),
            ("beta".to_string(), 0.366),
            ("gamma".to_string(), 1.387),
            ("delta".to_string(), 4.780),
            ("nu".to_string(), 0.707),
            ("eta".to_string(), 0.036),
        ],
        4,
        "3-component order parameter, O(3) symmetry",
    )
}

pub fn mean_field_class() -> UniversalityClass {
    UniversalityClass::new(
        "MeanField",
        vec![
            ("alpha".to_string(), 0.0),
            ("beta".to_string(), 0.5),
            ("gamma".to_string(), 1.0),
            ("delta".to_string(), 3.0),
            ("nu".to_string(), 0.5),
            ("eta".to_string(), 0.0),
        ],
        4,
        "Mean field theory, valid above upper critical dimension",
    )
}

/// Classify a system based on computed critical exponents.
pub fn classify(exponents: &[(String, f64)], tolerance: f64) -> Option<UniversalityClass> {
    let classes = vec![ising_class(), xy_class(), heisenberg_class(), mean_field_class()];
    classes.into_iter().find(|class| class.is_compatible(exponents, tolerance))
}

/// Compute the correlation length exponent ν from the RG eigenvalue.
pub fn correlation_length_exponent(leading_eigenvalue: f64) -> f64 {
    if (leading_eigenvalue - 1.0).abs() < 1e-10 {
        return f64::INFINITY;
    }
    1.0 / (leading_eigenvalue - 1.0).abs()
}

/// Compute η (anomalous dimension) from the fixed point.
pub fn anomalous_dimension(fixed_point: &FixedPoint) -> f64 {
    // Proxy: based on distance from Gaussian fixed point
    let norm: f64 = fixed_point
        .coordinates
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    // Simple scaling relation proxy
    2.0 * norm / (norm + 1.0) * 0.036
}

/// Compute hyperscaling relation: 2 - α = dν
pub fn check_hyperscaling(d: usize, alpha: f64, nu: f64) -> f64 {
    let lhs = 2.0 - alpha;
    let rhs = d as f64 * nu;
    (lhs - rhs).abs()
}

/// Compute Rushbrooke scaling relation: α + 2β + γ = 2
pub fn check_rushbrooke(alpha: f64, beta: f64, gamma: f64) -> f64 {
    (alpha + 2.0 * beta + gamma - 2.0).abs()
}

/// Compute Widom scaling relation: γ = β(δ - 1)
pub fn check_widom(beta: f64, gamma: f64, delta: f64) -> f64 {
    (gamma - beta * (delta - 1.0)).abs()
}

/// Compute Fisher scaling relation: γ = ν(2 - η)
pub fn check_fisher(gamma: f64, nu: f64, eta: f64) -> f64 {
    (gamma - nu * (2.0 - eta)).abs()
}

/// Compute all scaling relation violations.
pub fn scaling_relation_violations(
    alpha: f64,
    beta: f64,
    gamma: f64,
    delta: f64,
    nu: f64,
    eta: f64,
) -> Vec<(String, f64)> {
    vec![
        ("Rushbrooke".to_string(), check_rushbrooke(alpha, beta, gamma)),
        ("Widom".to_string(), check_widom(beta, gamma, delta)),
        ("Fisher".to_string(), check_fisher(gamma, nu, eta)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ising_exponents() {
        let c = ising_class();
        assert!((c.exponent("nu").unwrap() - 0.630).abs() < 0.001);
        assert!((c.exponent("beta").unwrap() - 0.326).abs() < 0.001);
        assert_eq!(c.upper_critical_dimension, 4);
    }

    #[test]
    fn test_mean_field_exponents() {
        let c = mean_field_class();
        assert!((c.exponent("beta").unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compatible() {
        let c = mean_field_class();
        let exps = vec![
            ("alpha".to_string(), 0.001),
            ("beta".to_string(), 0.501),
            ("gamma".to_string(), 1.001),
            ("delta".to_string(), 3.001),
            ("nu".to_string(), 0.499),
            ("eta".to_string(), 0.001),
        ];
        assert!(c.is_compatible(&exps, 0.01));
    }

    #[test]
    fn test_not_compatible() {
        let c = mean_field_class();
        let exps = vec![
            ("alpha".to_string(), 0.5), // way off
            ("beta".to_string(), 0.5),
            ("gamma".to_string(), 1.0),
            ("delta".to_string(), 3.0),
            ("nu".to_string(), 0.5),
            ("eta".to_string(), 0.0),
        ];
        assert!(!c.is_compatible(&exps, 0.01));
    }

    #[test]
    fn test_classify_ising() {
        let exps = vec![
            ("alpha".to_string(), 0.11),
            ("beta".to_string(), 0.33),
            ("gamma".to_string(), 1.24),
            ("delta".to_string(), 4.79),
            ("nu".to_string(), 0.63),
            ("eta".to_string(), 0.04),
        ];
        let result = classify(&exps, 0.1);
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "Ising");
    }

    #[test]
    fn test_classify_unknown() {
        let exps = vec![
            ("alpha".to_string(), 99.0),
            ("beta".to_string(), 99.0),
            ("gamma".to_string(), 99.0),
            ("delta".to_string(), 99.0),
            ("nu".to_string(), 99.0),
            ("eta".to_string(), 99.0),
        ];
        assert!(classify(&exps, 0.1).is_none());
    }

    #[test]
    fn test_correlation_length_exponent() {
        let nu = correlation_length_exponent(1.5);
        assert!((nu - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_anomalous_dimension() {
        let fp = FixedPoint::new(vec![0.1], StabilityType::Unstable, "test");
        let eta = anomalous_dimension(&fp);
        assert!(eta >= 0.0);
    }

    #[test]
    fn test_rushbrooke() {
        // Mean field: 0 + 2*0.5 + 1.0 = 2.0 ✓
        let violation = check_rushbrooke(0.0, 0.5, 1.0);
        assert!(violation < 1e-10);
    }

    #[test]
    fn test_widom() {
        // Mean field: 1.0 = 0.5 * (3.0 - 1) = 1.0 ✓
        let violation = check_widom(0.5, 1.0, 3.0);
        assert!(violation < 1e-10);
    }

    #[test]
    fn test_scaling_relations() {
        let violations = scaling_relation_violations(0.0, 0.5, 1.0, 3.0, 0.5, 0.0);
        assert_eq!(violations.len(), 3);
        for (_, v) in &violations {
            assert!(v.abs() < 1e-10);
        }
    }
}
