# INTEGRATION.md — renormalization-group-rs × conservation-law-rs × entropy-conservation-rs

**Renormalization group** coarse-grains agent systems across scales,
finding fixed points and critical exponents. It connects to symplectic
integration for scale evolution and entropy tracking for information loss.

## Synergy Map

```
conservation-law-rs         renormalization-group-rs        entropy-conservation-rs
┌──────────────────┐        ┌──────────────────────┐       ┌─────────────────────┐
│ SymplecticIntegr  │        │ Block                │       │ decompose           │
│ AgentState        │◄──────►│ CoarseGrainConfig    │◄─────►│ is_conserved        │
│ total_energy      │        │ coarse_grain         │       │ conservation_violat│
│ verify_noether    │        │ multi_level_coarse_gr│       │ HodgeComponents     │
└──────────────────┘        │ FixedPoint           │       └─────────────────────┘
                            │ LinearizedRG         │
                            │ BetaFunction         │
                            │ flow_rk4             │
                            │ UniversalityClass    │
                            └──────────────────────┘
```

## Key Insight

When you coarse-grain a fleet of agents, you lose fine-grained information.
The renormalization group quantifies that loss via `information_loss` and
finds the emergent fixed points where the system becomes scale-invariant.
Conservation-law verifies that energy is preserved across scales, and
entropy-conservation tracks whether the lost information is truly destroyed
or merely hidden in cyclic correlations.

## Example 1: Scale Evolution with Symplectic Integration

Evolve an agent fleet through scale space using RK4 flow, then verify
energy conservation at each scale.

```rust
use conservation_law::lagrangian::{AgentState, MechanicalLagrangian, total_energy};
use renormalization_group::flow::{BetaFunction, FlowTrajectory};
use renormalization_group::coarse_grain::{coarse_grain, CoarseGrainConfig, AggregationMethod};

fn scale_evolution(initial_state: &[f64]) {
    // Define a beta function for agent coupling strengths
    let beta = BetaFunction::new(|couplings: &[f64]| {
        couplings.iter().map(|&g| -0.1 * g * g * g).collect()
    });

    // Integrate RG flow from UV to IR
    let traj = beta.flow_rk4(initial_state, 100, 0.01);

    // Verify energy-like conservation at each scale
    let lagrangian = MechanicalLagrangian {
        mass: 1.0,
        potential_fn: |q: &[f64; 1]| 0.5 * q[0] * q[0],
    };

    for (i, point) in traj.points.iter().enumerate() {
        if point.is_empty() {
            continue;
        }
        let state = AgentState::new([point[0]], [0.0]);
        let e = total_energy(&lagrangian, &state);
        println!("scale {}: coupling = {:.4}, energy = {:.4}", i, point[0], e);
    }
}
```

## Example 2: Track Information Loss During Coarse-Graining

Coarse-grain a fleet and decompose the residual into Hodge components.

```rust
use renormalization_group::coarse_grain::{coarse_grain, CoarseGrainConfig, AggregationMethod};
use entropy_conservation::hodge_decomposition::{decompose, HodgeComponents};

fn analyze_information_loss(fine_state: &[f64]) -> HodgeComponents {
    let config = CoarseGrainConfig {
        block_size: 2,
        method: AggregationMethod::Mean,
    };

    // Coarse-grain by averaging pairs
    let coarse = coarse_grain(fine_state, &config);
    println!("Fine: {:?} -> Coarse: {:?}", fine_state, coarse);

    // Build a 2×N matrix: [fine_state; coarse_state_padded]
    let mut matrix = vec![vec![0.0; fine_state.len()]; 2];
    for (i, &v) in fine_state.iter().enumerate() {
        matrix[0][i] = v;
        matrix[1][i] = if i < coarse.len() { coarse[i] } else { 0.0 };
    }

    // Decompose the scale-transition matrix
    let hodge = decompose(&matrix);
    println!("Gradient (conservative rescaling): {:.4}", hodge.gradient_energy());
    println!("Curl (cyclic mixing): {:.4}", hodge.curl_energy());
    println!("Harmonic (information destruction): {:.4}", hodge.harmonic_energy());
    hodge
}
```

## Example 3: Fixed-Point Stability via Spectral Analysis

Linearize around a fixed point and use power-iteration eigenvalues to
classify stability.

```rust
use renormalization_group::fixed_point::{LinearizedRG, find_fixed_point, StabilityType};
use renormalization_group::flow::BetaFunction;

fn classify_agent_fixed_point(dim: usize) -> StabilityType {
    let transform = |state: &[f64]| {
        state.iter().map(|&x| x * 0.95).collect::<Vec<f64>>()
    };

    let fp = find_fixed_point(&transform, &vec![1.0; dim], 1000, 1e-8)
        .expect("fixed point should exist");

    let jacobian = vec![
        vec![0.95, 0.0],
        vec![0.0, 0.95],
    ];
    let linearized = LinearizedRG::new(fp.coordinates.clone(), jacobian);
    let stability = linearized.classify_stability(100);

    println!("Fixed point {:?} is {:?}", fp.coordinates, stability);
    println!("Relevant directions: {}", linearized.relevant_directions(100));
    println!("Irrelevant directions: {}", linearized.irrelevant_directions(100));
    stability
}
```

## Cargo.toml Wiring

```toml
[dependencies]
renormalization-group = { git = "https://github.com/SuperInstance/renormalization-group-rs" }
conservation-law = { git = "https://github.com/SuperInstance/conservation-law-rs" }
entropy-conservation = { git = "https://github.com/SuperInstance/entropy-conservation-rs" }
```

## Design Patterns

### Pattern: Scale-Invariant Fleet Descriptors

Build a descriptor that is stable under coarse-graining by composing RG
blocks with conservation checks:

```rust
use renormalization_group::coarse_grain::{multi_level_coarse_grain, CoarseGrainConfig, AggregationMethod};
use renormalization_group::universality::classify;

fn scale_invariant_descriptor(state: &[f64]) -> String {
    let config = CoarseGrainConfig {
        block_size: 2,
        method: AggregationMethod::Mean,
    };
    let levels = multi_level_coarse_grain(state, &config, 3);
    let ratios = renormalization_group::coarse_grain::scaling_ratios(&levels);

    if let Some(cls) = classify(&[
        ("nu".to_string(), ratios[0]),
    ], 0.1) {
        cls.name.clone()
    } else {
        "unknown".to_string()
    }
}
```

This pattern is useful when you need fleet summaries that remain stable
as the fleet grows or shrinks.
