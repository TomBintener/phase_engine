//! A macro-based engine for generating multiple path sum backends.
//!
//! This module defines the `generate_phase_engine!` macro, which creates
//! self-contained, type-specific modules for path sum evaluation. It then
//! uses this macro to instantiate a fast 64-bit engine and a wide 128-bit
//! engine, each optimized for a specific memory layout and variable capacity.

use smallvec::SmallVec;
use std::cmp::Ordering;
use std::f64::consts::TAU;
use std::hash::{Hash, Hasher};

#[allow(dead_code)]
const EPSILON: f64 = 1e-10;

// Include the files that define the logic-generating macros.
// This must be done at the top level of the module.
include!("canonical_phase_poly.rs");
include!("continuous_poly.rs");
include!("evaluator.rs");
include!("reduction.rs");
include!("tests.rs");
include!("pmh.rs");
include!("steiner.rs");

macro_rules! generate_phase_engine {
    (
        $module_name:ident,
        $primitive:ty,
        $phase_mask:expr,
        $phase_shift:expr,
        $poly_capacity:expr,
        $canon_capacity:expr
    ) => {
        pub mod $module_name {
            use super::*;

            // Call the macros to generate the logic for this specific engine.
            define_canonical_phase_poly_logic!($primitive, $phase_mask, $phase_shift, $poly_capacity, $canon_capacity);
            define_continuous_poly_logic!($primitive, $poly_capacity);
            define_evaluator_logic!($primitive);
            define_reduction_logic!($primitive, $poly_capacity);
            define_pmh_logic!($primitive);
            define_steiner_logic!($primitive);

            #[cfg(test)]
            pub mod tests {
                use super::*;
                define_tests_logic!($primitive, $phase_shift);
            }
        }
    };
}

// The Fast-Path 64-bit Engine
generate_phase_engine!(
    engine_64,
    u64,
    0xE000_0000_0000_0000,
    61,
    8,
    16
);

// The Wide-Path 128-bit Engine
generate_phase_engine!(
    engine_128,
    u128,
    0xE000_0000_0000_0000_0000_0000_0000_0000,
    125,
    4,
    8
);
mod debug_test;
