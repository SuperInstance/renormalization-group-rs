# renormalization-group-rs

Renormalization group for multi-scale analysis of agent systems — coarse-grain, flow, find fixed points, classify universality.

When you zoom out on a system of agents, which properties survive and which
wash away? This crate implements the renormalization group (RG) framework to
answer that question: partition agents into blocks, aggregate their states,
track how coupling constants evolve under scale transformations, and classify
the resulting behavior into universality classes.

## Why Care?

You have 1000 agents and you want to understand their collective behavior. You
can simulate all 1000 — or you can group them into 500 super-agents, then 250,
then 125, and see what's stable across scales. The properties that survive
coarse-graining are the ones that matter at the macro level.

```
Level 0: 8 agents    [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
Level 1: 4 blocks    [2.0, 6.0, 10.0, 14.0]
Level 2: 2 blocks    [4.0, 12.0]
Level 3: 1 block     [8.0]
```

Each level loses information (variance drops). The *scaling ratios* between
levels tell you how fast information is lost — and whether there's a scale
where the system becomes "simple" (a fixed point).

## Quick Start

```toml
# Cargo.toml
[dependencies]
renormalization-group-rs = "0.1.0"
```

```rust
use renormalization_group_rs::coarse_grain::{
    coarse_grain, multi_level_coarse_grain, CoarseGrainConfig, AggregationMethod,
};
use renormalization_group_rs::fixed_point::{FixedPoint, StabilityType};
use renormalization_group_rs::flow::BetaFunction;
use renormalization_group_rs::universality::{ising_class, classify};

// 1. Coarse-grain 8 agents into blocks of 2
let state = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0];
let config = CoarseGrainConfig {
    block_size: 2,
    method: AggregationMethod::Mean,
};
let coarse = coarse_grain(&state, &config);
println!("{:?}", coarse);
// => [2.0, 6.0, 10.0, 14.0]

// 2. Multi-level: zoom out repeatedly
let levels = multi_level_coarse_grain(&state, &config, 3);
for (i, level) in levels.iter().enumerate() {
    println!("Level {}: {:?}", i, level);
}
// => Level 0: [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
// => Level 1: [2.0, 6.0, 10.0, 14.0]
// => Level 2: [4.0, 12.0]
// => Level 3: [8.0]

// 3. Integrate RG flow with a beta function
let beta = BetaFunction::new(|g: &[f64]| vec![-g[0]]); // dg/ds = -g → flows to 0
let traj = beta.flow_euler(&[1.0], 50, 0.1);
println!("Final coupling: {:.6}", traj.points.last().unwrap()[0]);
// => Final coupling: ~0.0059 (decaying toward Gaussian fixed point)

// 4. Classify measured exponents against known universality classes
let measured = vec![
    ("alpha".into(), 0.11), ("beta".into(), 0.33),
    ("gamma".into(), 1.24), ("delta".into(), 4.8),
    ("nu".into(), 0.63), ("eta".into(), 0.04),
];
if let Some(class) = classify(&measured, 0.1) {
    println!("System is in the {} universality class", class.name);
    // => System is in the Ising universality class
}
```

## Core Concepts Through Code

### Coarse Graining

Group agents into blocks and aggregate their values:

```rust
use renormalization_group_rs::coarse_grain::{
    coarse_grain, partition_into_blocks, aggregate,
    CoarseGrainConfig, AggregationMethod, Block,
};

let state = vec![1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0];

// Different aggregation methods
let mean_config = CoarseGrainConfig { block_size: 3, method: AggregationMethod::Mean };
let max_config  = CoarseGrainConfig { block_size: 3, method: AggregationMethod::Max };
let med_config  = CoarseGrainConfig { block_size: 3, method: AggregationMethod::Median };

println!("Mean: {:?}", coarse_grain(&state, &mean_config));
// => Mean: [3.0, 5.6667, 4.0]

println!("Max:  {:?}", coarse_grain(&state, &max_config));
// => Max:  [5.0, 8.0, 4.0]

println!("Median: {:?}", coarse_grain(&state, &med_config));
// => Median: [3.0, 7.0, 4.0]

// Weighted mean: give center agents more importance
let weights = vec![0.5, 1.0, 0.5];
let wconfig = CoarseGrainConfig {
    block_size: 3,
    method: AggregationMethod::WeightedMean(weights),
};
println!("Weighted: {:?}", coarse_grain(&state, &wconfig));
// => Weighted: [3.0, 5.6667, 4.0]

// Inspect the block structure
let blocks = partition_into_blocks(7, 3);
for b in &blocks {
    println!("Super-agent {} ← agents {:?}", b.super_id, b.agent_ids);
}
// => Super-agent 0 ← agents [0, 1, 2]
// => Super-agent 1 ← agents [3, 4, 5]
// => Super-agent 2 ← agents [6]
```

### Information Loss

Each coarse-graining step loses information. Measure how much:

```rust
use renormalization_group_rs::coarse_grain::{
    coarse_grain, multi_level_coarse_grain, scaling_ratios,
    information_loss, CoarseGrainConfig, AggregationMethod,
};

let state = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0];
let config = CoarseGrainConfig::default(); // block_size=2, Mean

let levels = multi_level_coarse_grain(&state, &config, 3);

// Information loss at each level
for i in 1..levels.len() {
    let loss = information_loss(&levels[i-1], &levels[i]);
    println!("Level {}→{}: {:.4} loss", i-1, i, loss);
}
// => Level 0→1: positive (variance reduced)
// => Level 1→2: more loss
// => Level 2→3: even more

// Scaling ratios: how fast does variance shrink?
let ratios = scaling_ratios(&levels);
println!("Scaling ratios: {:?}", ratios);
// Each ratio < 1.0 means variance is decreasing under coarse-graining
```

### Fixed Points

Find where the RG transformation stabilizes:

```rust
use renormalization_group_rs::fixed_point::{
    FixedPoint, StabilityType, LinearizedRG,
    find_fixed_point, gaussian_fixed_point, wilson_fisher_proxy,
};

// The Gaussian fixed point: all couplings are zero
let gfp = gaussian_fixed_point(3);
println!("Gaussian FP: {:?}", gfp.coordinates);
// => [0.0, 0.0, 0.0]
println!("Stability: {:?}", gfp.stability);
// => Stable

// Wilson-Fisher fixed point (non-trivial, for φ⁴ theory in 4-ε dimensions)
let wf = wilson_fisher_proxy(1.0); // ε = 1.0
println!("WF FP: {:?}", wf.coordinates);
// => [0.1667, -0.0185]
println!("Stability: {:?}", wf.stability);
// => Unstable

// Find a fixed point by iterating a contraction
let transform = |v: &[f64]| vec![v[0] * 0.5]; // shrinks toward 0
let fp = find_fixed_point(&transform, &[1.0], 100, 1e-8);
assert!(fp.is_some());
assert!(fp.unwrap().coordinates[0].abs() < 1e-6);
// The contraction map has a fixed point at the origin
```

### Stability Analysis via Eigenvalues

The Jacobian at a fixed point determines stability:

```rust
use renormalization_group_rs::fixed_point::{LinearizedRG, StabilityType};

// A 2D system with one relevant and one irrelevant direction
let rg = LinearizedRG::new(
    vec![0.0, 0.0],
    vec![
        vec![2.0, 0.0],  // eigenvalue = 2.0 (relevant: |λ| > 1)
        vec![0.0, 0.5],  // eigenvalue = 0.5 (irrelevant: |λ| < 1)
    ],
);

println!("Stability: {:?}", rg.classify_stability(20));
// => Unstable (has relevant direction)

println!("Relevant directions:   {}", rg.relevant_directions(20));
// => 1

println!("Irrelevant directions: {}", rg.irrelevant_directions(20));
// => 1

let eigenvalues = rg.eigenvalues(30);
println!("Eigenvalues: {:?}", eigenvalues);
// => [~2.0, ~0.5]
```

### RG Flow Integration

Track coupling constants as the scale changes:

```rust
use renormalization_group_rs::flow::{
    BetaFunction, FlowTrajectory,
    phi4_beta, gaussian_beta, is_asymptotically_free, critical_exponent,
};

// φ⁴ theory: β(g) = -ε·g + a·g²
let beta = phi4_beta(1.0, 1.0);
let traj = beta.flow_rk4(&[0.5], 200, 0.01);

println!("Initial coupling: {:.4}", traj.points[0][0]);
println!("Final coupling:   {:.4}", traj.points.last().unwrap()[0]);
// => Flows toward a fixed point

// Check convergence
if traj.has_converged(1e-4) {
    println!("Flow converged to a fixed point");
}

// Euler integration for comparison
let euler_traj = beta.flow_euler(&[0.5], 200, 0.01);
// RK4 is more accurate but Euler is simpler

// Gaussian flow: β(g) = -ε·g → always flows to zero
let gauss_beta = gaussian_beta(1.0);
let gauss_traj = gauss_beta.flow_rk4(&[1.0], 100, 0.1);
println!("Asymptotically free: {}", is_asymptotically_free(&gauss_traj));
// => true (couplings → 0 at high energy)

// Critical exponent from eigenvalue
let nu = critical_exponent(2.0);
println!("Critical exponent: {:.4}", nu);
// => 1.0 (= ln(2)/ln(2))
```

### Building Custom Beta Functions

```rust
use renormalization_group_rs::flow::{BetaFunction, FlowTrajectory};

// Two-coupling system: [g1, g2]
let two_coupling = BetaFunction::new(|g: &[f64]| {
    vec![
        -g[0] + g[0] * g[1],        // β₁ = -g₁ + g₁g₂
        -2.0 * g[1] + g[0] * g[0],  // β₂ = -2g₂ + g₁²
    ]
});

let traj = two_coupling.flow_rk4(&[0.5, 0.1], 500, 0.01);
println!("Trajectory length: {} points", traj.len());

// Extract beta function numerically from trajectory
let numerical_beta = traj.beta_function();
for (i, b) in numerical_beta.iter().enumerate().take(3) {
    println!("Step {}: β = [{:.4}, {:.4}]", i, b[0], b[1]);
}
```

### Universality Classes

Systems with the same critical exponents behave identically at large scales:

```rust
use renormalization_group_rs::universality::{
    ising_class, xy_class, heisenberg_class, mean_field_class,
    classify, correlation_length_exponent,
    check_rushbrooke, check_widom, check_fisher,
    scaling_relation_violations,
};

// Built-in universality classes with known critical exponents
let ising = ising_class();
println!("Ising ν = {}", ising.exponent("nu").unwrap());
// => 0.630

let mf = mean_field_class();
println!("MF β = {}", mf.exponent("beta").unwrap());
// => 0.5

// Classify your measured exponents
let my_exponents = vec![
    ("alpha".into(), 0.0),   ("beta".into(), 0.5),
    ("gamma".into(), 1.0),   ("delta".into(), 3.0),
    ("nu".into(), 0.5),      ("eta".into(), 0.0),
];
match classify(&my_exponents, 0.01) {
    Some(c) => println!("Matched: {} — {}", c.name, c.description),
    None    => println!("No known class matches"),
}
// => Matched: MeanField — Mean field theory, valid above upper critical dimension

// Correlation length exponent from leading eigenvalue
let nu = correlation_length_exponent(1.5);
println!("ν = {:.4}", nu);
// => 2.0 (= 1/|1.5 - 1|)

// Verify scaling relations (should be ≈ 0 if satisfied)
let rushbrooke = check_rushbrooke(0.0, 0.5, 1.0);
let widom = check_widom(0.5, 1.0, 3.0);
let fisher = check_fisher(1.0, 0.5, 0.0);
println!("Rushbrooke violation: {:.6} (α + 2β + γ = 2)", rushbrooke);
// => 0.000000
println!("Widom violation:      {:.6} (γ = β(δ-1))", widom);
// => 0.000000
println!("Fisher violation:     {:.6} (γ = ν(2-η))", fisher);
// => 0.000000

// Check all scaling relations at once
let violations = scaling_relation_violations(0.0, 0.5, 1.0, 3.0, 0.5, 0.0);
for (name, v) in &violations {
    println!("  {}: {:.6}", name, v);
}
// => Rushbrooke: 0.000000
// => Widom:      0.000000
// => Fisher:     0.000000
```

## API Reference

### `coarse_grain` Module

```rust
// Core types
pub struct Block {
    pub agent_ids: Vec<usize>,
    pub super_id: usize,
}

pub struct CoarseGrainConfig {
    pub block_size: usize,
    pub method: AggregationMethod,
}

pub enum AggregationMethod {
    Mean,
    Max,
    Min,
    Median,
    WeightedMean(Vec<f64>),
}

// Functions
pub fn aggregate(values: &[f64], method: &AggregationMethod) -> f64
pub fn partition_into_blocks(n_agents: usize, block_size: usize) -> Vec<Block>
pub fn coarse_grain(state: &[f64], config: &CoarseGrainConfig) -> Vec<f64>
pub fn multi_level_coarse_grain(state: &[f64], config: &CoarseGrainConfig, levels: usize) -> Vec<Vec<f64>>
pub fn information_loss(fine: &[f64], coarse: &[f64]) -> f64
pub fn scaling_ratios(levels: &[Vec<f64>]) -> Vec<f64>
```

### `fixed_point` Module

```rust
pub struct FixedPoint {
    pub coordinates: Vec<f64>,
    pub stability: StabilityType,
    pub label: String,
}

pub enum StabilityType { Stable, Unstable, Marginal }

pub struct LinearizedRG {
    pub fixed_point: Vec<f64>,
    pub jacobian: Vec<Vec<f64>>,
}

// Functions
pub fn find_fixed_point(
    transform: &dyn Fn(&[f64]) -> Vec<f64>,
    initial: &[f64],
    max_iterations: usize,
    tolerance: f64,
) -> Option<FixedPoint>
pub fn gaussian_fixed_point(dim: usize) -> FixedPoint
pub fn wilson_fisher_proxy(epsilon: f64) -> FixedPoint

// Methods on LinearizedRG
impl LinearizedRG {
    pub fn eigenvalues(&self, iterations: usize) -> Vec<f64>
    pub fn classify_stability(&self, iterations: usize) -> StabilityType
    pub fn relevant_directions(&self, iterations: usize) -> usize
    pub fn irrelevant_directions(&self, iterations: usize) -> usize
}
```

### `flow` Module

```rust
pub struct FlowTrajectory {
    pub points: Vec<Vec<f64>>,
    pub scales: Vec<f64>,
}

pub struct BetaFunction { /* opaque */ }

// BetaFunction methods
impl BetaFunction {
    pub fn new(f: impl Fn(&[f64]) -> Vec<f64> + 'static) -> Self
    pub fn evaluate(&self, couplings: &[f64]) -> Vec<f64>
    pub fn flow_euler(&self, initial: &[f64], steps: usize, step_size: f64) -> FlowTrajectory
    pub fn flow_rk4(&self, initial: &[f64], steps: usize, step_size: f64) -> FlowTrajectory
}

// FlowTrajectory methods
impl FlowTrajectory {
    pub fn new() -> Self
    pub fn add_point(&mut self, coupling: Vec<f64>, scale: f64)
    pub fn beta_function(&self) -> Vec<Vec<f64>>
    pub fn has_converged(&self, tolerance: f64) -> bool
}

// Preset beta functions
pub fn phi4_beta(epsilon: f64, a: f64) -> BetaFunction
pub fn gaussian_beta(epsilon: f64) -> BetaFunction
pub fn is_asymptotically_free(traj: &FlowTrajectory) -> bool
pub fn critical_exponent(eigenvalue: f64) -> f64
```

### `universality` Module

```rust
pub struct UniversalityClass {
    pub name: String,
    pub exponents: Vec<(String, f64)>,
    pub upper_critical_dimension: usize,
    pub description: String,
}

// Built-in classes
pub fn ising_class() -> UniversalityClass
pub fn xy_class() -> UniversalityClass
pub fn heisenberg_class() -> UniversalityClass
pub fn mean_field_class() -> UniversalityClass

// Classification
pub fn classify(exponents: &[(String, f64)], tolerance: f64) -> Option<UniversalityClass>
pub fn correlation_length_exponent(leading_eigenvalue: f64) -> f64
pub fn anomalous_dimension(fixed_point: &FixedPoint) -> f64

// Scaling relation checks
pub fn check_hyperscaling(d: usize, alpha: f64, nu: f64) -> f64
pub fn check_rushbrooke(alpha: f64, beta: f64, gamma: f64) -> f64
pub fn check_widom(beta: f64, gamma: f64, delta: f64) -> f64
pub fn check_fisher(gamma: f64, nu: f64, eta: f64) -> f64
pub fn scaling_relation_violations(alpha: f64, beta: f64, gamma: f64, delta: f64, nu: f64, eta: f64) -> Vec<(String, f64)>
```

## Advanced Examples

### Multi-Scale Fleet Analysis Pipeline

```rust
use renormalization_group_rs::coarse_grain::{
    multi_level_coarse_grain, scaling_ratios, information_loss,
    CoarseGrainConfig, AggregationMethod,
};
use renormalization_group_rs::fixed_point::find_fixed_point;
use renormalization_group_rs::flow::BetaFunction;

// Step 1: Generate agent coupling data
fn simulate_agents(n: usize) -> Vec<f64> {
    (0..n).map(|i| {
        let x = i as f64 / n as f64;
        (2.0 * std::f64::consts::PI * 3.0 * x).sin() * (1.0 - x)
    }).collect()
}

let state = simulate_agents(64);

// Step 2: Multi-level coarse-graining
let config = CoarseGrainConfig {
    block_size: 2,
    method: AggregationMethod::Mean,
};
let levels = multi_level_coarse_grain(&state, &config, 6);

println!("Scale hierarchy:");
for (i, level) in levels.iter().enumerate() {
    println!("  Level {} ({} agents): mean={:.4}, var={:.4}",
        i, level.len(),
        level.iter().sum::<f64>() / level.len() as f64,
        variance(level));
}

// Step 3: Check scaling ratios (is there a scale-invariant regime?)
let ratios = scaling_ratios(&levels);
println!("Scaling ratios: {:?}", ratios);
// Ratios near 1.0 indicate scale invariance → universality!

// Step 4: Find the RG fixed point
let coarse_grain_transform = |v: &[f64]| {
    let mut result = Vec::new();
    for chunk in v.chunks(2) {
        result.push(chunk.iter().sum::<f64>() / chunk.len() as f64);
    }
    result
};

let fp = find_fixed_point(&coarse_grain_transform, &state, 20, 0.01);
if let Some(fp) = fp {
    println!("Fixed point found: {:?}", fp.coordinates);
} else {
    println!("No fixed point within iteration limit");
}

fn variance(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
}
```

### Checking Scaling Relations for Measured Exponents

```rust
use renormalization_group_rs::universality::{
    scaling_relation_violations, classify, ising_class,
};

// Suppose you measured these critical exponents from simulation
let measured = vec![
    ("alpha".to_string(), 0.09),
    ("beta".to_string(),  0.34),
    ("gamma".to_string(), 1.20),
    ("delta".to_string(), 4.60),
    ("nu".to_string(),    0.64),
    ("eta".to_string(),   0.04),
];

// Check scaling relations
let violations = scaling_relation_violations(0.09, 0.34, 1.20, 4.60, 0.64, 0.04);
println!("Scaling relation violations:");
for (name, v) in &violations {
    println!("  {}: {:.4} {}", name, v, if *v < 0.1 { "✓" } else { "✗" });
}
// Small violations mean your exponents are self-consistent

// Try to classify
match classify(&measured, 0.1) {
    Some(c) => println!("→ {} universality class", c.name),
    None => println!("→ Unknown universality class (new physics?)"),
}
```

## Conservation Law Connections

The renormalization group reveals which quantities survive coarse-graining:
those are the *relevant* operators. Conservation laws correspond to relevant
directions in coupling space — if total entropy is conserved at the microscopic
level, it had better be conserved at the macroscopic level too.

The coarse-graining operations in this crate preserve certain invariants:
- **Sum preservation** under `Mean` aggregation when block sizes are equal
- **Variance scaling** follows predictable ratios between levels
- **Fixed points** represent self-similar states where the system looks the
  same at every scale

### Relation to Other SuperInstance Crates

- **`entropy-conservation-rs`** — Use `is_conserved()` to verify entropy
  is preserved through each coarse-graining step
- **`sheaf-coherence-rs`** — Coarse-graining is a special case of sheaf
  restriction; global sections at one scale must project consistently to
  coarser scales
- **`constraint-dynamics-rs`** — Conservation laws are constraints; the
  relevant/irrelevant decomposition tells you which constraints survive
  coarse-graining

## Performance

- Coarse graining: O(n) for n agents per level
- Multi-level: O(n + n/2 + n/4 + ...) = O(n) total
- Flow integration (Euler/RK4): O(steps × dim) per step
- Power iteration eigenvalues: O(iterations × n²) per eigenvalue
- Suitable for ~10⁴ agents and ~20 coupling constants at interactive speed

## License

MIT
