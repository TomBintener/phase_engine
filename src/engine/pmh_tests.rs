//! Matrix-level Patel–Markov–Hayes tests. Soundness is "apply CX list to I
//! equals the source matrix", never AST-string equality with GE or Qiskit.

use super::engine_128;
use super::engine_64;

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

/// Dense invertible matrix: random unit-diagonal L and U, then `M = L U` over F2.
fn dense_gl_64(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = seed;
    let mut rand_bit = || lcg(&mut rng) & 1 == 1;
    let mut l = vec![0u64; n];
    let mut u = vec![0u64; n];
    for i in 0..n {
        l[i] |= 1u64 << i;
        u[i] |= 1u64 << i;
        for j in 0..i {
            if rand_bit() {
                l[i] |= 1u64 << j;
            }
        }
        for j in (i + 1)..n {
            if rand_bit() {
                u[i] |= 1u64 << j;
            }
        }
    }
    let mut m = vec![0u64; n];
    for i in 0..n {
        for j in 0..n {
            if ((l[i] >> j) & 1) == 1 {
                m[i] ^= u[j];
            }
        }
    }
    m
}

fn assert_implements_64(matrix: &[u64], n: usize, cnots: &[(usize, usize)]) {
    let got = engine_64::apply_cnots_to_identity(cnots, n);
    assert_eq!(got, matrix, "CX list does not implement the source matrix");
}

fn assert_implements_128(matrix: &[u128], n: usize, cnots: &[(usize, usize)]) {
    let got = engine_128::apply_cnots_to_identity(cnots, n);
    assert_eq!(got, matrix, "CX list does not implement the source matrix");
}

#[test]
fn identity_is_empty() {
    let n = 4;
    let m: Vec<u64> = (0..n).map(|i| 1u64 << i).collect();
    let cnots = engine_64::synthesize_cnot_matrix(m, n).unwrap();
    assert!(cnots.is_empty(), "identity should emit no CXs: {cnots:?}");
}

#[test]
fn swap_matrix_round_trips() {
    // Swap on qubits 0,1: rows [x1, x0, x2].
    let n = 3;
    let m = vec![1u64 << 1, 1u64 << 0, 1u64 << 2];
    let cnots = engine_64::synthesize_cnot_matrix(m.clone(), n).unwrap();
    assert_implements_64(&m, n, &cnots);
}

#[test]
fn three_cycle_and_cx_ladder_round_trip() {
    let n = 3;
    // Start I, apply CX 0,1; 1,2; 2,0.
    let mut m: Vec<u64> = (0..n).map(|i| 1u64 << i).collect();
    m[1] ^= m[0];
    m[2] ^= m[1];
    m[0] ^= m[2];
    let cnots = engine_64::synthesize_cnot_matrix(m.clone(), n).unwrap();
    assert_implements_64(&m, n, &cnots);

    let mut ladder: Vec<u64> = (0..n).map(|i| 1u64 << i).collect();
    ladder[1] ^= ladder[0];
    ladder[2] ^= ladder[1];
    let cnots = engine_64::synthesize_cnot_matrix(ladder.clone(), n).unwrap();
    assert_implements_64(&ladder, n, &cnots);
}

#[test]
fn random_gl_n2_to_16_and_wide_64() {
    let mut ns: Vec<usize> = (2..=16).collect();
    ns.extend([20, 32, 40]);
    for n in ns {
        for k in 0..8 {
            let m = dense_gl_64(n, 0xC0FFEE + (n as u64) * 1000 + k);
            let cnots = engine_64::synthesize_cnot_matrix(m.clone(), n)
                .unwrap_or_else(|e| panic!("n={n} k={k}: {e}"));
            assert_implements_64(&m, n, &cnots);
        }
    }
}

#[test]
fn section_sizes_2_3_4_all_implement() {
    let n = 8;
    let m = dense_gl_64(n, 42);
    for sec in [2usize, 3, 4] {
        let cnots = engine_64::synthesize_cnot_matrix_pmh(m.clone(), n, Some(sec))
            .unwrap_or_else(|e| panic!("m={sec}: {e}"));
        assert_implements_64(&m, n, &cnots);
    }
}

#[test]
fn dense_suite_mean_cx_below_ge() {
    // Suite-level, not per-matrix. Dense random GL(n) at n where GE is O(n^2).
    for n in [8usize, 12, 16] {
        let mut pmh_total = 0usize;
        let mut ge_total = 0usize;
        let samples = 64;
        for k in 0..samples {
            let m = dense_gl_64(n, 0xA11CE + (n as u64) * 10_000 + k);
            let pmh = engine_64::synthesize_cnot_matrix_pmh(m.clone(), n, None)
                .unwrap_or_else(|e| panic!("pmh n={n} k={k}: {e}"));
            let ge = engine_64::synthesize_cnot_matrix_ge(m.clone(), n).unwrap();
            assert_implements_64(&m, n, &pmh);
            assert_implements_64(&m, n, &ge);
            pmh_total += pmh.len();
            ge_total += ge.len();
        }
        assert!(
            pmh_total < ge_total,
            "n={n}: mean PMH CX {} not below GE {}",
            pmh_total as f64 / samples as f64,
            ge_total as f64 / samples as f64
        );
    }
}

#[test]
fn wide_n70_no_u64_truncation() {
    let n = 70;
    // Bits 0, 65, 69 must survive: a u64 cast would drop 65 and 69.
    let mut m: Vec<u128> = (0..n).map(|i| 1u128 << i).collect();
    m[65] ^= m[0];
    m[69] ^= m[65];
    m[0] ^= m[69];
    let cnots =
        engine_128::synthesize_cnot_matrix(m.clone(), n).unwrap_or_else(|e| panic!("n=70: {e}"));
    assert_implements_128(&m, n, &cnots);
    assert!(
        cnots.iter().any(|&(c, t)| c >= 64 || t >= 64),
        "expected at least one CX on a wire >= 64: {cnots:?}"
    );
}
