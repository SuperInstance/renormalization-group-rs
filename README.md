# renormalization-group-rs

**Renormalization group for multi-scale analysis of agent systems.**

This crate implements the renormalization group (RG) framework in Rust: coarse-grain collections of agents into super-agents, track how coupling constants flow under scale transformations, identify fixed points and their stability, and classify system behavior into universality classes. With 46 tests spanning aggregation, flow convergence, eigenvalue computation, and class matching, it provides the mathematical backbone for understanding emergent behavior across scales.

## Why This Matters

The renormalization group is physics' most powerful tool for understanding how microscopic rules produce macroscopic behavior. In AGI, the same principle applies: individual agent interactions at one scale produce collective intelligence (or dysfunction) at another. RG flow analysis reveals whether a multi-agent system will converge to a stable collective, oscillate, or exhibit critical behavior. Universality classes tell you when *different* systems will behave the same way at large scales — the mathematical basis for transfer learning between architectures. If you're building hierarchical agent systems, RG is the theory of what survives at each level of abstraction.

## Quick Start

```toml
# Cargo.toml
[dependencies]
renormalization-group-rs = "0.1.0"
```

```rust
use renormalization_group_rs::coarse_grain::{coarse_grain, AggregationMethod, CoarseGrainConfig};
use renormalization_group_rs::flow::{FlowTrajectory, integrate_flow};
use renormalization_group_rs::fixed_point::FixedPoint;
use renormalization_group_rs::universality::{ising_class, UniversalityClass};

// Coarse-grain 100 agents into 50 super-agents using mean aggregation
let agents: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
let config = CoarseGrainConfig {
    block_size: 2,
    method: AggregationMethod::Mean,
};
let (super_agents, blocks) = coarse_grain(&agents, &config);
println!("{} agents → {} super-agents", agents.len(), super_agents.len());

// Integrate an RG flow using a beta function
let beta = |g: &[f64]| vec![-g[0] * g[0]]; // g² flow → Gaussian fixed point
let mut traj = FlowTrajectory::new();
let result = integrate_flow(&mut traj, vec![0.5], &beta, 0.01, 100);
println!("Converged: {}", traj.has_converged(1e-6));

// Check compatibility with the Ising universality class
let ising = ising_class();
let measured = vec![
    ("alpha".into(), 0.12), ("beta".into(), 0.33),
    ("gamma".into(), 1.24), ("delta".into(), 4.8),
    ("nu".into(), 0.63), ("eta".into(), 0.04),
];
println!("Ising-compatible: {}", ising.is_compatible(&measured, 0.1));
```

## Architecture

| Module | Purpose |
|---|---|
| `coarse_grain` | Block partitioning, aggregation (mean/max/min/median/weighted), scale reduction |
| `fixed_point` | Fixed point detection, stability analysis, Jacobian eigenvalues |
| `flow` | RG flow trajectories, beta functions, Euler integration, convergence detection |
| `universality` | Universality class definition, critical exponents, compatibility testing |

## API Tour

### Coarse Graining (`coarse_grain`)

- **`Block { agent_ids, super_id }`** — Maps a group of agents to one super-agent
- **`CoarseGrainConfig { block_size, method }`** — Configuration for partitioning
- **`AggregationMethod`** — Enum: `Mean`, `Max`, `Min`, `Median`, `WeightedMean(weights)`
- **`coarse_grain(values, config) → (Vec<f64>, Vec<Block>)`** — Partition and aggregate
- **`aggregate(values, method) → f64`** — Apply a single aggregation

### Fixed Points (`fixed_point`)

- **`FixedPoint { coordinates, stability, label }`** — A point in coupling space
  - `.distance_to(&other)` — Euclidean distance
  - `.dimension()` — Dimensionality of coupling space
- **`StabilityType`** — `Stable`, `Unstable`, `Marginal`
- **`LinearizedRG { fixed_point, jacobian }`** — Linearized transformation
  - `.eigenvalues(iterations)` — Power iteration eigenvalues
  - `.relevant_directions(threshold)` — Eigendirections with |λ| > 1

### Flow Equations (`flow`)

- **`FlowTrajectory { points, scales }`** — Sequence of coupling constants at different scales
  - `.add_point(coupling, scale)` — Record a step
  - `.beta_function()` — Numerical beta function between steps
  - `.has_converged(tolerance)` — Convergence check on last 3 points
- **`integrate_flow(traj, initial, beta_fn, step_size, steps) → FlowResult`** — Euler integration
- **`find_fixed_point(traj, tolerance) → Option<Vec<f64>>`** — Extract fixed point from trajectory

### Universality (`universality`)

- **`UniversalityClass { name, exponents, upper_critical_dimension, description }`**
  - `.exponent("nu")` — Look up a specific critical exponent
  - `.is_compatible(measured, tolerance)` — Compare against measured exponents
- **`ising_class()`** — 3D Ising model exponents
- **`xy_class()`** — 3D XY model exponents
- **`percolation_class()`** — Percolation critical exponents

## Performance

- Coarse graining is O(n) for n agents
- Flow integration is O(steps × dim²) per step for dim coupling constants
- Power iteration eigenvalues: O(iterations × n²) per eigenvalue with deflation
- Suitable for systems with up to ~10⁴ agents and ~20 coupling constants at interactive speed

## Ecosystem

Part of the **SuperInstance** family:

- [`sheaf-coherence-rs`](https://github.com/SuperInstance/sheaf-coherence-rs) — Local-to-global consistency
- [`optimal-transport-rs`](https://github.com/SuperInstance/optimal-transport-rs) — Wasserstein geometry for distributions
- [`spectral-prosody-rs`](https://github.com/SuperInstance/spectral-prosody-rs) — Spectral features in agent communication
- [`constraint-dynamics-rs`](https://github.com/SuperInstance/constraint-dynamics-rs) — Constraint satisfaction dynamics
- [`agent-homeostasis-rs`](https://github.com/SuperInstance/agent-homeostasis-rs) — Homeostatic regulation

## Ideas for Improvement

- **Monte Carlo RG** — Stochastic block-spin transformations
- **Functional RG** — Exact renormalization group equations (Wetterich equation)
- **Momentum-shell integration** — Wilsonian RG with Fourier modes
- **Automatic exponent extraction** — Fit critical exponents from simulation data
- **Multi-parameter flows** — 2D and 3D flow diagrams with cross-sections
- **GPU-accelerated coarse graining** — For 10⁶+ agent systems

## License

MIT
