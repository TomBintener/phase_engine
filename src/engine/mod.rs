//! A macro-based engine for generating multiple path sum backends.
//!
//! This module defines the `generate_phase_engine!` macro, which creates
//! self-contained, type-specific modules for path sum evaluation. It then
//! uses this macro to instantiate a fast 64-bit engine and a wide 128-bit
//! engine, each optimized for a specific memory layout and variable capacity.

use smallvec::SmallVec;
use std::cmp::Ordering;
use std::f64::consts::TAU;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

/// Unique tokens for H/SX capacity overflow. Never zero; never reused.
static NEXT_OVERFLOW_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) fn next_overflow_id() -> u64 {
    NEXT_OVERFLOW_ID.fetch_add(1, AtomicOrdering::Relaxed)
}

#[allow(dead_code)]
const EPSILON: f64 = 1e-10;

/// Continuous phases live on an integer lattice with this many ticks per 2π
/// (one tick ≈ 1.17e-8 rad, the resolution of the previous 1e-8 `f64` snap).
/// A power of two divisible by 8 makes π/4 multiples, negation and modular
/// sums exact, so the discrete/continuous split and interned `Eq` never depend
/// on the order in which angles were accumulated.
pub const TICKS_PER_TURN: u64 = 1 << 29;
pub const TICKS_PER_PI_4: u64 = TICKS_PER_TURN / 8;
pub const TICKS_PER_PI_2: u64 = TICKS_PER_TURN / 4;

/// Round an angle in radians onto the tick lattice (mod 2π).
#[inline(always)]
pub fn angle_to_ticks(theta: f64) -> u32 {
    let t = (theta / TAU * TICKS_PER_TURN as f64).round();
    if !t.is_finite() {
        return 0;
    }
    (t as i64).rem_euclid(TICKS_PER_TURN as i64) as u32
}

/// Representative angle (radians, in `[0, 2π)`) of a tick count.
#[inline(always)]
pub fn ticks_to_angle(ticks: u32) -> f64 {
    ticks as f64 * (TAU / TICKS_PER_TURN as f64)
}

#[inline(always)]
pub fn add_ticks(a: u32, b: u32) -> u32 {
    ((a as u64 + b as u64) % TICKS_PER_TURN) as u32
}

#[inline(always)]
pub fn negate_ticks(a: u32) -> u32 {
    ((TICKS_PER_TURN - (a as u64 % TICKS_PER_TURN)) % TICKS_PER_TURN) as u32
}

// Include the files that define the logic-generating macros.
// This must be done at the top level of the module.
include!("canonical_phase_poly.rs");
include!("continuous_poly.rs");
include!("evaluator.rs");
include!("reduction.rs");
include!("tests.rs");
include!("reference.rs");
include!("soundness_tests.rs");
include!("pmh.rs");
include!("steiner.rs");
include!("gray.rs");

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
            define_gray_logic!($primitive);

            #[cfg(test)]
            pub mod tests {
                use super::*;
                define_tests_logic!($primitive, $phase_shift);
                define_reference_logic!($primitive);
                define_soundness_tests_logic!($primitive);
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

#[cfg(test)]
mod pmh_tests;
