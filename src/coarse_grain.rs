//! Coarse-graining transformations for multi-scale analysis.

/// A coarse-graining block that groups agents into super-agents.
#[derive(Debug, Clone)]
pub struct Block {
    pub agent_ids: Vec<usize>,
    pub super_id: usize,
}

impl Block {
    pub fn new(super_id: usize, agent_ids: Vec<usize>) -> Self {
        Self { super_id, agent_ids }
    }

    pub fn size(&self) -> usize {
        self.agent_ids.len()
    }
}

/// Configuration for coarse-graining.
#[derive(Debug, Clone)]
pub struct CoarseGrainConfig {
    pub block_size: usize,
    pub method: AggregationMethod,
}

/// How to aggregate values within a block.
#[derive(Debug, Clone, PartialEq)]
pub enum AggregationMethod {
    Mean,
    Max,
    Min,
    Median,
    WeightedMean(Vec<f64>),
}

impl Default for CoarseGrainConfig {
    fn default() -> Self {
        Self {
            block_size: 2,
            method: AggregationMethod::Mean,
        }
    }
}

/// Apply aggregation to a slice of values.
pub fn aggregate(values: &[f64], method: &AggregationMethod) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    match method {
        AggregationMethod::Mean => values.iter().sum::<f64>() / values.len() as f64,
        AggregationMethod::Max => values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        AggregationMethod::Min => values.iter().cloned().fold(f64::INFINITY, f64::min),
        AggregationMethod::Median => {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = sorted.len() / 2;
            if sorted.len().is_multiple_of(2) {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            }
        }
        AggregationMethod::WeightedMean(weights) => {
            let total_w: f64 = weights.iter().take(values.len()).sum();
            if total_w.abs() < 1e-15 {
                return 0.0;
            }
            values
                .iter()
                .zip(weights.iter())
                .map(|(v, w)| v * w)
                .sum::<f64>()
                / total_w
        }
    }
}

/// Partition agents into blocks.
pub fn partition_into_blocks(n_agents: usize, block_size: usize) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut super_id = 0;
    let mut i = 0;
    while i < n_agents {
        let end = (i + block_size).min(n_agents);
        let ids: Vec<usize> = (i..end).collect();
        blocks.push(Block::new(super_id, ids));
        super_id += 1;
        i = end;
    }
    blocks
}

/// Coarse-grain a state vector by partitioning into blocks and aggregating.
pub fn coarse_grain(state: &[f64], config: &CoarseGrainConfig) -> Vec<f64> {
    let blocks = partition_into_blocks(state.len(), config.block_size);
    blocks
        .iter()
        .map(|block| {
            let values: Vec<f64> = block.agent_ids.iter().map(|&i| state[i]).collect();
            aggregate(&values, &config.method)
        })
        .collect()
}

/// Multi-level coarse-graining: apply repeatedly.
pub fn multi_level_coarse_grain(
    state: &[f64],
    config: &CoarseGrainConfig,
    levels: usize,
) -> Vec<Vec<f64>> {
    let mut result = Vec::with_capacity(levels + 1);
    result.push(state.to_vec());
    let mut current = state.to_vec();
    for _ in 0..levels {
        if current.len() < config.block_size {
            break;
        }
        current = coarse_grain(&current, config);
        result.push(current.clone());
    }
    result
}

/// Compute the information loss between two scales (KL-like divergence proxy).
pub fn information_loss(fine: &[f64], coarse: &[f64]) -> f64 {
    if coarse.is_empty() {
        return 0.0;
    }
    // Simple proxy: variance reduction ratio
    let fine_var = variance(fine);
    let coarse_var = variance(coarse);
    if fine_var < 1e-15 {
        0.0
    } else {
        1.0 - coarse_var / fine_var
    }
}

fn variance(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    data.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / data.len() as f64
}

/// Compute the scaling ratio between consecutive levels.
pub fn scaling_ratios(levels: &[Vec<f64>]) -> Vec<f64> {
    levels
        .windows(2)
        .map(|w| {
            let coarse_var = variance(&w[1]);
            let fine_var = variance(&w[0]);
            if fine_var < 1e-15 {
                1.0
            } else {
                coarse_var / fine_var
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregate_mean() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        assert!((aggregate(&v, &AggregationMethod::Mean) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_aggregate_max() {
        let v = vec![1.0, 5.0, 3.0];
        assert!((aggregate(&v, &AggregationMethod::Max) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_aggregate_min() {
        let v = vec![1.0, 5.0, 3.0];
        assert!((aggregate(&v, &AggregationMethod::Min) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_aggregate_median_odd() {
        let v = vec![3.0, 1.0, 2.0];
        assert!((aggregate(&v, &AggregationMethod::Median) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_aggregate_median_even() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        assert!((aggregate(&v, &AggregationMethod::Median) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_aggregate_weighted_mean() {
        let v = vec![1.0, 2.0, 3.0];
        let w = vec![1.0, 2.0, 1.0];
        assert!((aggregate(&v, &AggregationMethod::WeightedMean(w)) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_aggregate_empty() {
        assert_eq!(aggregate(&[], &AggregationMethod::Mean), 0.0);
    }

    #[test]
    fn test_partition_into_blocks() {
        let blocks = partition_into_blocks(7, 3);
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].agent_ids, vec![0, 1, 2]);
        assert_eq!(blocks[1].agent_ids, vec![3, 4, 5]);
        assert_eq!(blocks[2].agent_ids, vec![6]);
    }

    #[test]
    fn test_coarse_grain_mean() {
        let state = vec![1.0, 3.0, 5.0, 7.0];
        let config = CoarseGrainConfig::default();
        let cg = coarse_grain(&state, &config);
        assert_eq!(cg.len(), 2);
        assert!((cg[0] - 2.0).abs() < 1e-10);
        assert!((cg[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_multi_level() {
        let state = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0];
        let config = CoarseGrainConfig::default();
        let levels = multi_level_coarse_grain(&state, &config, 3);
        assert_eq!(levels.len(), 4); // original + 3 levels
        assert_eq!(levels[0].len(), 8);
        assert_eq!(levels[1].len(), 4);
        assert_eq!(levels[2].len(), 2);
        assert_eq!(levels[3].len(), 1);
    }

    #[test]
    fn test_information_loss() {
        let fine = vec![1.0, 3.0, 5.0, 7.0];
        let coarse = vec![2.0, 6.0];
        let loss = information_loss(&fine, &coarse);
        // Coarse has same mean, lower variance => positive loss
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_scaling_ratios() {
        let levels = vec![
            vec![1.0, 3.0, 5.0, 7.0],
            vec![2.0, 6.0],
        ];
        let ratios = scaling_ratios(&levels);
        assert_eq!(ratios.len(), 1);
        assert!(ratios[0] > 0.0);
    }

    #[test]
    fn test_block_size() {
        let b = Block::new(0, vec![1, 2, 3]);
        assert_eq!(b.size(), 3);
    }
}
