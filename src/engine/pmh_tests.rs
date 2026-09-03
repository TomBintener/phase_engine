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

fn dense_gl_128(n: usize, seed: u64) -> Vec<u128> {
    let mut rng = seed;
    let mut rand_bit = || lcg(&mut rng) & 1 == 1;
    let mut l = vec![0u128; n];
    let mut u = vec![0u128; n];
    for i in 0..n {
        l[i] |= 1u128 << i;
        u[i] |= 1u128 << i;
        for j in 0..i {
            if rand_bit() {
                l[i] |= 1u128 << j;
            }
        }
        for j in (i + 1)..n {
            if rand_bit() {
                u[i] |= 1u128 << j;
            }
        }
    }
    let mut m = vec![0u128; n];
    for i in 0..n {
        for j in 0..n {
            if ((l[i] >> j) & 1) == 1 {
                m[i] ^= u[j];
            }
        }
    }
    m
}

/// Sparse invertible matrix: identity plus a random CX product (window-like).
fn sparse_cx_product_64(n: usize, n_ops: usize, seed: u64) -> Vec<u64> {
    let mut rng = seed;
    let mut m: Vec<u64> = (0..n).map(|i| 1u64 << i).collect();
    for _ in 0..n_ops {
        let c = (lcg(&mut rng) as usize) % n;
        let t = (lcg(&mut rng) as usize) % n;
        if c != t {
            m[t] ^= m[c];
        }
    }
    m
}

#[test]
fn extended_dense_sparse_and_wide_suite() {
    // Heavier pre-merge pass: more n, more samples, 128-bit, sparse products.
    // Production entry (PMH + GE fallback) must implement every matrix.
    // Pure PMH must not fall back on invertible dense GL.
    for &(n, samples) in &[
        (8usize, 256),
        (12, 256),
        (16, 256),
        (20, 128),
        (24, 128),
        (32, 64),
    ] {
        let mut pmh_total = 0usize;
        let mut ge_total = 0usize;
        let mut max_pmh = 0usize;
        let mut max_ge = 0usize;
        let first = dense_gl_64(n, 0xE7E7 + (n as u64) * 100_003);
        let mut saw_distinct = false;
        for k in 0..samples {
            let m = dense_gl_64(n, 0xE7E7 + (n as u64) * 100_003 + k);
            if k > 0 && m != first {
                saw_distinct = true;
            }
            let pmh = engine_64::synthesize_cnot_matrix_pmh(m.clone(), n, None)
                .unwrap_or_else(|e| panic!("pmh failed on dense GL n={n} k={k}: {e}"));
            assert_implements_64(&m, n, &pmh);
            pmh_total += pmh.len();
            max_pmh = max_pmh.max(pmh.len());

            let ge = engine_64::synthesize_cnot_matrix_ge(m.clone(), n).unwrap();
            assert_implements_64(&m, n, &ge);
            ge_total += ge.len();
            max_ge = max_ge.max(ge.len());

            let prod = engine_64::synthesize_cnot_matrix(m.clone(), n).unwrap();
            assert_implements_64(&m, n, &prod);
        }
        let pmh_mean = pmh_total as f64 / samples as f64;
        let ge_mean = ge_total as f64 / samples as f64;
        eprintln!(
            "extended dense n={n} samples={samples}: mean PMH {pmh_mean:.2} vs GE {ge_mean:.2} \
             (max {max_pmh} vs {max_ge})"
        );
        if !saw_distinct {
            panic!("n={n}: all dense samples were identical");
        }
        if n >= 12 {
            assert!(
                pmh_total < ge_total,
                "n={n}: mean PMH {pmh_mean} not below GE {ge_mean}"
            );
        }
    }

    // Sparse CX products (typical window residuals): implement, no PMH Err.
    for n in [4usize, 8, 12, 16, 24] {
        for k in 0..64 {
            let m = sparse_cx_product_64(n, n * 3, 0x5A5A + (n as u64) * 17 + k);
            let pmh = engine_64::synthesize_cnot_matrix_pmh(m.clone(), n, None)
                .unwrap_or_else(|e| panic!("sparse n={n} k={k}: {e}"));
            assert_implements_64(&m, n, &pmh);
            let prod = engine_64::synthesize_cnot_matrix(m.clone(), n).unwrap();
            assert_implements_64(&m, n, &prod);
        }
    }

    // 128-bit dense + high-wire: implement at n=16, 40, 70.
    for &(n, samples) in &[(16usize, 32), (40, 16), (70, 8)] {
        for k in 0..samples {
            let m = dense_gl_128(n, 0x7070 + (n as u64) * 1009 + k);
            let cnots = engine_128::synthesize_cnot_matrix_pmh(m.clone(), n, None)
                .unwrap_or_else(|e| panic!("128 n={n} k={k}: {e}"));
            assert_implements_128(&m, n, &cnots);
            let prod = engine_128::synthesize_cnot_matrix(m.clone(), n).unwrap();
            assert_implements_128(&m, n, &prod);
        }
    }
}

fn identity_64(n: usize) -> Vec<u64> {
    (0..n).map(|i| 1u64 << i).collect()
}

fn mul_f2_64(a: &[u64], b: &[u64], n: usize) -> Vec<u64> {
    let mut out = vec![0u64; n];
    for i in 0..n {
        for j in 0..n {
            if ((a[i] >> j) & 1) == 1 {
                out[i] ^= b[j];
            }
        }
    }
    out
}

fn invert_cnots(cnots: &[(usize, usize)]) -> Vec<(usize, usize)> {
    cnots.iter().rev().copied().collect()
}

fn permutation_gl_64(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = seed;
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (lcg(&mut rng) as usize) % (i + 1);
        perm.swap(i, j);
    }
    perm.into_iter().map(|src| 1u64 << src).collect()
}

fn involution_perm_64(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = seed;
    let mut leftover: Vec<usize> = (0..n).collect();
    let mut dest = vec![0usize; n];
    while leftover.len() >= 2 {
        let i = (lcg(&mut rng) as usize) % leftover.len();
        let a = leftover.swap_remove(i);
        if (lcg(&mut rng) & 1) == 0 {
            dest[a] = a;
            continue;
        }
        let j = (lcg(&mut rng) as usize) % leftover.len();
        let b = leftover.swap_remove(j);
        dest[a] = b;
        dest[b] = a;
    }
    for a in leftover {
        dest[a] = a;
    }
    dest.into_iter().map(|src| 1u64 << src).collect()
}

fn block_diag_64(left: &[u64], n_left: usize, right: &[u64], n_right: usize) -> Vec<u64> {
    let n = n_left + n_right;
    let mut m = vec![0u64; n];
    m[..n_left].copy_from_slice(&left[..n_left]);
    for i in 0..n_right {
        m[n_left + i] = right[i] << n_left;
    }
    m
}

#[test]
fn default_section_size_matches_spec() {
    assert_eq!(engine_64::default_pmh_section_size(0), 1);
    assert_eq!(engine_64::default_pmh_section_size(1), 1);
    assert_eq!(engine_64::default_pmh_section_size(2), 2);
    assert_eq!(engine_64::default_pmh_section_size(3), 2);
    assert_eq!(engine_64::default_pmh_section_size(4), 2);
    assert_eq!(engine_64::default_pmh_section_size(8), 3);
    assert_eq!(engine_64::default_pmh_section_size(16), 4);
    assert_eq!(engine_64::default_pmh_section_size(32), 4);
    assert_eq!(engine_64::default_pmh_section_size(7), 2);
    assert_eq!(
        engine_64::default_pmh_section_size(16),
        engine_128::default_pmh_section_size(16)
    );
}

#[test]
fn pre_merge_inverse_section_and_ensemble_suite() {
    // Second pre-merge pass: inverse tapes, section-size sweep, ensembles
    // that are not LU-dense, composition, and wider 128-bit widths.

    for &(n, samples) in &[(8usize, 96), (12, 64), (16, 64), (24, 32)] {
        let mut default_total = 0usize;
        let mut ge_total = 0usize;
        let sections = [2usize, 3, 4, 5];
        let mut section_totals = [0usize; 4];
        let mut section_ok = [0usize; 4];
        let id = identity_64(n);

        for k in 0..samples {
            let m = dense_gl_64(n, 0xBEE5 + (n as u64) * 90_011 + k);
            let pmh = engine_64::synthesize_cnot_matrix_pmh(m.clone(), n, None)
                .unwrap_or_else(|e| panic!("default pmh n={n} k={k}: {e}"));
            assert_implements_64(&m, n, &pmh);
            default_total += pmh.len();

            let inv = invert_cnots(&pmh);
            let recovered = engine_64::apply_cnots_to_identity(&inv, n);
            let product = mul_f2_64(&m, &recovered, n);
            assert_eq!(product, id, "n={n} k={k}: M * M^{{-1}} != I");
            let undone = {
                let mut work = m.clone();
                for &(c, t) in &inv {
                    work[t] ^= work[c];
                }
                work
            };
            assert_eq!(undone, id, "n={n} k={k}: inverse tape did not undo M");

            let ge = engine_64::synthesize_cnot_matrix_ge(m.clone(), n).unwrap();
            assert_implements_64(&m, n, &ge);
            ge_total += ge.len();

            for (si, &sec) in sections.iter().enumerate() {
                if sec > n {
                    continue;
                }
                let tape = engine_64::synthesize_cnot_matrix_pmh(m.clone(), n, Some(sec))
                    .unwrap_or_else(|e| panic!("section {sec} n={n} k={k}: {e}"));
                assert_implements_64(&m, n, &tape);
                section_totals[si] += tape.len();
                section_ok[si] += 1;
            }
        }

        let default_mean = default_total as f64 / samples as f64;
        let ge_mean = ge_total as f64 / samples as f64;
        eprintln!(
            "pre-merge inverse/section n={n} samples={samples}: default PMH {default_mean:.2} vs GE {ge_mean:.2}"
        );
        for (si, &sec) in sections.iter().enumerate() {
            if section_ok[si] == 0 {
                continue;
            }
            eprintln!(
                "  section m={sec}: mean CX {:.2} ({}/{} ok)",
                section_totals[si] as f64 / section_ok[si] as f64,
                section_ok[si],
                samples
            );
        }
        if n >= 12 {
            assert!(
                default_total < ge_total,
                "n={n}: default PMH {default_mean} not below GE {ge_mean}"
            );
        }
    }

    // Permutations, involutions, and block-diagonal GL: implement + invert.
    for n in [6usize, 8, 12, 16, 24] {
        for k in 0..48 {
            for (label, m) in [
                ("perm", permutation_gl_64(n, 0xA0A0 + (n as u64) * 31 + k)),
                (
                    "involution",
                    involution_perm_64(n, 0x0B0B + (n as u64) * 37 + k),
                ),
            ] {
                let pmh = engine_64::synthesize_cnot_matrix_pmh(m.clone(), n, None)
                    .unwrap_or_else(|e| panic!("{label} n={n} k={k}: {e}"));
                assert_implements_64(&m, n, &pmh);
                let inv = invert_cnots(&pmh);
                assert_eq!(
                    mul_f2_64(&m, &engine_64::apply_cnots_to_identity(&inv, n), n),
                    identity_64(n),
                    "{label} n={n} k={k}: inverse failed"
                );
            }
        }
        if n % 2 == 0 {
            let half = n / 2;
            for k in 0..24 {
                let left = dense_gl_64(half, 0x1111 + (n as u64) * 13 + k);
                let right = dense_gl_64(half, 0x2222 + (n as u64) * 17 + k);
                let m = block_diag_64(&left, half, &right, half);
                let pmh = engine_64::synthesize_cnot_matrix_pmh(m.clone(), n, None)
                    .unwrap_or_else(|e| panic!("block-diag n={n} k={k}: {e}"));
                assert_implements_64(&m, n, &pmh);
            }
        }
    }

    // Composition: synth(A) then synth(B) implements B A, same as synth(B A).
    for n in [8usize, 12, 16] {
        for k in 0..32 {
            let a = dense_gl_64(n, 0xCA11 + (n as u64) * 101 + k);
            let b = dense_gl_64(n, 0xCA12 + (n as u64) * 103 + k);
            let ba = mul_f2_64(&b, &a, n);
            let ca = engine_64::synthesize_cnot_matrix_pmh(a.clone(), n, None).unwrap();
            let cb = engine_64::synthesize_cnot_matrix_pmh(b.clone(), n, None).unwrap();
            let mut composed = ca;
            composed.extend(cb);
            assert_implements_64(&ba, n, &composed);
            let direct = engine_64::synthesize_cnot_matrix_pmh(ba.clone(), n, None)
                .unwrap_or_else(|e| panic!("compose n={n} k={k}: {e}"));
            assert_implements_64(&ba, n, &direct);
        }
    }

    // Wider 128-bit: n=80 / 96 / 120, plus inverse on n=40.
    for &(n, samples) in &[(40usize, 12), (80, 6), (96, 4), (120, 3)] {
        for k in 0..samples {
            let m = dense_gl_128(n, 0x8080 + (n as u64) * 401 + k);
            let cnots = engine_128::synthesize_cnot_matrix_pmh(m.clone(), n, None)
                .unwrap_or_else(|e| panic!("wide128 n={n} k={k}: {e}"));
            assert_implements_128(&m, n, &cnots);
            let inv = invert_cnots(&cnots);
            let mut work = m.clone();
            for &(c, t) in &inv {
                work[t] ^= work[c];
            }
            let id: Vec<u128> = (0..n).map(|i| 1u128 << i).collect();
            assert_eq!(work, id, "wide128 n={n} k={k}: inverse tape did not undo M");
            let prod = engine_128::synthesize_cnot_matrix(m.clone(), n).unwrap();
            assert_implements_128(&m, n, &prod);
        }
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
