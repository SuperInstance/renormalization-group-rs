//! # renormalization-group-rs
//!
//! Renormalization group for multi-scale analysis of agent systems.
//!
//! Provides coarse-graining transformations, fixed point analysis,
//! RG flow equations, and universality class detection.

#![allow(clippy::needless_range_loop)]

pub mod coarse_grain;
pub mod fixed_point;
pub mod flow;
pub mod universality;
