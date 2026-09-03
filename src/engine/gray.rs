macro_rules! define_gray_logic {
    ($primitive:ty) => {
        /// Converts a multilinear (monomial-basis) phase polynomial into the
        /// parity basis by Mobius inversion.
        ///
        /// The canonical phase poly stores `phase(x) = sum c_m * prod_{i in m} x_i`
        /// (verified empirically: T applied to x0^x1 stores three monomials,
        /// not one parity). A CX+RZ region's phase function always admits a
        /// parity decomposition `sum theta_p * XOR_{i in p} x_i`; using the
        /// expansion `XOR_{i in p} x_i = sum_{0 != S subset p} (-2)^{|S|-1} prod_S x_i`,
        /// the heaviest surviving monomial `m` can only come from parity
        /// `p = m`, so we peel parities from the top down. Any consistent
        /// angle lift produces an exact decomposition by construction (the
        /// subtraction removes exactly what the emitted parity contributes).
        ///
        /// Returns `None` when a monomial is too wide to expand (popcount >
        /// 16) - callers treat that as "do not synthesize".
        pub fn phase_monomials_to_parities(
            monomials: &[($primitive, f64)],
        ) -> Option<Vec<($primitive, f64)>> {
            use std::collections::HashMap;
            let mut coeff: HashMap<$primitive, f64> = HashMap::new();
            for &(m, c) in monomials {
                if m != 0 {
                    *coeff.entry(m).or_insert(0.0) += c;
                }
            }
            let mut parities: Vec<($primitive, f64)> = Vec::new();
            const EPS: f64 = 1e-9;
            loop {
                // Heaviest surviving monomial (ties broken by mask value for
                // determinism).
                let top = coeff
                    .iter()
                    .filter(|&(_, &c)| c.abs() > EPS)
                    .max_by_key(|&(&m, _)| (m.count_ones(), m));
                let (&m, &c) = match top {
                    Some(t) => t,
                    None => break,
                };
                let k = m.count_ones();
                if k == 1 {
                    parities.push((m, c));
                    coeff.remove(&m);
                    continue;
                }
                if k > 16 {
                    return None;
                }
                let theta = c / (-2.0f64).powi(k as i32 - 1);
                parities.push((m, theta));
                // Subtract theta * expansion(parity m) over all nonempty
                // subsets of m.
                let mut s = m;
                loop {
                    let sk = s.count_ones();
                    let contrib = theta * (-2.0f64).powi(sk as i32 - 1);
                    let e = coeff.entry(s).or_insert(0.0);
                    *e -= contrib;
                    if e.abs() < EPS {
                        coeff.remove(&s);
                    }
                    if s == 0 || (s - 1) & m == 0 {
                        break;
                    }
                    s = (s - 1) & m;
                }
            }
            Some(parities)
        }

        fn gray_merge_terms(parities: &[($primitive, f64)]) -> Vec<($primitive, f64)> {
            use std::collections::HashMap;
            let mut merged: HashMap<$primitive, f64> = HashMap::new();
            for &(mask, angle) in parities {
                if mask == 0 {
                    continue;
                }
                *merged.entry(mask).or_insert(0.0) += angle;
            }
            let tau = std::f64::consts::TAU;
            let mut out: Vec<($primitive, f64)> = merged
                .into_iter()
                .filter_map(|(m, a)| {
                    let mut a = a % tau;
                    if a < 0.0 {
                        a += tau;
                    }
                    if a.abs() < 1e-12 || (tau - a).abs() < 1e-12 {
                        None
                    } else {
                        Some((m, a))
                    }
                })
                .collect();
            out.sort_by_key(|&(m, _)| m);
            out
        }

        /// Solve `XOR_{i in combo} rows[i] == m` over GF(2). Combo is a
        /// bitmask over wire indices.
        fn gray_solve(rows: &[$primitive], m: $primitive, n: usize) -> Option<u128> {
            let mut work: Vec<($primitive, u128)> = rows
                .iter()
                .enumerate()
                .map(|(i, &r)| (r, 1u128 << i))
                .collect();
            let mut rhs = m;
            let mut rhs_combo: u128 = 0;
            let mut used = vec![false; n];
            for bit in 0..n {
                let bmask = (1 as $primitive) << bit;
                let mut pivot = None;
                for (i, &(r, _)) in work.iter().enumerate() {
                    if !used[i] && (r & bmask) != 0 {
                        pivot = Some(i);
                        break;
                    }
                }
                let p = match pivot {
                    Some(p) => p,
                    None => continue,
                };
                used[p] = true;
                let (pr, pc) = work[p];
                for i in 0..n {
                    if i != p && !used[i] && (work[i].0 & bmask) != 0 {
                        work[i].0 ^= pr;
                        work[i].1 ^= pc;
                    }
                }
                if (rhs & bmask) != 0 {
                    rhs ^= pr;
                    rhs_combo ^= pc;
                }
            }
            if rhs == 0 {
                Some(rhs_combo)
            } else {
                None
            }
        }

        fn gray_residual_pmh(
            rows: &[$primitive],
            target: &[$primitive],
            n: usize,
            instructions: &mut Vec<String>,
            total_cnots: &mut i64,
        ) -> Option<()> {
            let mut residual: Vec<$primitive> = Vec::with_capacity(n);
            for q in 0..n {
                let combo = gray_solve(rows, target[q], n)?;
                let mut row: $primitive = 0;
                for i in 0..n {
                    if (combo >> i) & 1 == 1 {
                        row |= (1 as $primitive) << i;
                    }
                }
                residual.push(row);
            }
            match synthesize_cnot_matrix(residual, n) {
                Ok(cnots) => {
                    for (c, t) in cnots {
                        instructions.push(format!("cx {},{}", c, t));
                        *total_cnots += 1;
                    }
                    Some(())
                }
                Err(_) => None,
            }
        }

        /// Place an RZ on every remaining dual parity of Hamming weight 1
        /// (Lemma 4.1: that live wire currently holds χ_y) and drop it.
        fn gray_place_ready(s: &mut Vec<($primitive, f64)>, instructions: &mut Vec<String>) {
            s.retain(|&(y, angle)| {
                if y == 0 {
                    false
                } else if y.count_ones() == 1 {
                    let q = y.trailing_zeros() as usize;
                    instructions.push(format!("rz {},{}", q, angle.to_bits()));
                    false
                } else {
                    true
                }
            });
        }

        /// Dual update for CX(c, t): y_c ^= y_t on every stacked parity.
        fn gray_dual_cx(s: &mut [($primitive, f64)], c: usize, t: usize) {
            let tmask = (1 as $primitive) << t;
            let cmask = (1 as $primitive) << c;
            for (y, _) in s.iter_mut() {
                if (*y & tmask) != 0 {
                    *y ^= cmask;
                }
            }
        }

        /// Previous nearest-neighbour assemble tour (test oracle only).
        pub fn synthesize_gray_network_nn(
            parities: &[($primitive, f64)],
            target: &[$primitive],
            n: usize,
        ) -> Option<(Vec<String>, i64)> {
            let mut instructions: Vec<String> = Vec::new();
            let mut total_cnots: i64 = 0;
            let mut terms = gray_merge_terms(parities);
            terms.sort_by_key(|&(m, _)| (m.count_ones(), m ^ (m >> 1)));
            if terms.len() > 2 && terms.len() <= 512 {
                let mut chained: Vec<($primitive, f64)> = Vec::with_capacity(terms.len());
                let mut rest = terms;
                chained.push(rest.remove(0));
                while !rest.is_empty() {
                    let last = chained.last().unwrap().0;
                    let (best_idx, _) = rest
                        .iter()
                        .enumerate()
                        .map(|(i, &(m, _))| (i, (m ^ last).count_ones()))
                        .min_by_key(|&(_, d)| d)
                        .unwrap();
                    chained.push(rest.remove(best_idx));
                }
                terms = chained;
            }

            let mut rows: Vec<$primitive> = (0..n).map(|i| (1 as $primitive) << i).collect();
            for &(mask, angle) in &terms {
                let combo = gray_solve(&rows, mask, n)?;
                if combo == 0 {
                    return None;
                }
                let members: Vec<usize> = (0..n).filter(|&i| (combo >> i) & 1 == 1).collect();
                let &tq = members
                    .iter()
                    .max_by_key(|&&i| (rows[i] & mask).count_ones())
                    .unwrap();
                for &s in &members {
                    if s != tq {
                        instructions.push(format!("cx {},{}", s, tq));
                        rows[tq] ^= rows[s];
                        total_cnots += 1;
                    }
                }
                debug_assert_eq!(rows[tq], mask);
                instructions.push(format!("rz {},{}", tq, angle.to_bits()));
            }
            gray_residual_pmh(&rows, target, n, &mut instructions, &mut total_cnots)?;
            Some((instructions, total_cnots))
        }

        /// Amy–Azimzadeh–Mosca GraySynth (arXiv:1712.01859 Algorithm 1) plus
        /// a Patel–Markov–Hayes residual to the target linear map.
        ///
        /// Stack of `(S, I, i)` frames. Each `S` carries dual parities with
        /// their RZ angles. A bit `j ∈ I` that maximises the larger cofactor
        /// is expanded; `S0` keeps the existing target and `S1` takes `j` as
        /// the target on the first 1-branch. Common 1-bits are CNOTed onto
        /// the designated target *without uncomputing* (`y_c ^= y_t` on every
        /// stacked dual). An RZ is emitted when a dual has weight 1 (Lemma
        /// 4.1: that wire currently holds χ_y). Empty `S` continues (the
        /// paper's `return` would abort remaining frames). Empty `I` still
        /// runs the common-1 loop so a singleton is reduced to weight 1.
        ///
        /// Unlike Steiner this shares a Gray-code tour across terms instead
        /// of a fresh CNOT tree per parity.
        pub fn synthesize_gray_network(
            parities: &[($primitive, f64)],
            target: &[$primitive],
            n: usize,
        ) -> Option<(Vec<String>, i64)> {
            let mut instructions: Vec<String> = Vec::new();
            let mut total_cnots: i64 = 0;
            let mut terms = gray_merge_terms(parities);
            let mut rows: Vec<$primitive> = (0..n).map(|i| (1 as $primitive) << i).collect();

            gray_place_ready(&mut terms, &mut instructions);

            struct Frame {
                s: Vec<($primitive, f64)>,
                unused: Vec<usize>,
                target: Option<usize>,
            }
            let mut stack: Vec<Frame> = Vec::new();
            if !terms.is_empty() {
                stack.push(Frame {
                    s: terms,
                    unused: (0..n).collect(),
                    target: None,
                });
            }

            while let Some(mut frame) = stack.pop() {
                gray_place_ready(&mut frame.s, &mut instructions);
                if frame.s.is_empty() {
                    continue;
                }
                if let Some(tgt) = frame.target {
                    loop {
                        let mut found = None;
                        for j in 0..n {
                            if j == tgt {
                                continue;
                            }
                            let all_j = frame.s.iter().all(|&(y, _)| (y >> j) & 1 == 1);
                            let some_tgt = frame.s.iter().any(|&(y, _)| (y >> tgt) & 1 == 1);
                            if all_j && some_tgt {
                                found = Some(j);
                                break;
                            }
                        }
                        let j = match found {
                            Some(j) => j,
                            None => break,
                        };
                        instructions.push(format!("cx {},{}", j, tgt));
                        rows[tgt] ^= rows[j];
                        total_cnots += 1;
                        gray_dual_cx(&mut frame.s, j, tgt);
                        for f in stack.iter_mut() {
                            gray_dual_cx(&mut f.s, j, tgt);
                        }
                        gray_place_ready(&mut frame.s, &mut instructions);
                        for f in stack.iter_mut() {
                            gray_place_ready(&mut f.s, &mut instructions);
                        }
                        if frame.s.is_empty() {
                            break;
                        }
                    }
                }
                if frame.s.is_empty() {
                    continue;
                }
                // Paper typesets `if S=∅ or I=∅ then return`; returning
                // would drop the rest of the stack. Continue after common-1.
                if frame.unused.is_empty() {
                    if !frame.s.is_empty() {
                        return None;
                    }
                    continue;
                }
                let mut best_j = frame.unused[0];
                let mut best_score = 0usize;
                for &j in &frame.unused {
                    let ones = frame.s.iter().filter(|&&(y, _)| (y >> j) & 1 == 1).count();
                    let zeros = frame.s.len() - ones;
                    let score = ones.max(zeros);
                    if score > best_score || (score == best_score && j < best_j) {
                        best_score = score;
                        best_j = j;
                    }
                }
                let (mut s0, mut s1) = (Vec::new(), Vec::new());
                for &(y, angle) in &frame.s {
                    if (y >> best_j) & 1 == 1 {
                        s1.push((y, angle));
                    } else {
                        s0.push((y, angle));
                    }
                }
                let unused: Vec<usize> = frame
                    .unused
                    .iter()
                    .copied()
                    .filter(|&u| u != best_j)
                    .collect();
                let t1 = match frame.target {
                    None => Some(best_j),
                    Some(t) => Some(t),
                };
                // Push S1 then S0 so S0 is popped first (paper stack order).
                if !s1.is_empty() {
                    stack.push(Frame {
                        s: s1,
                        unused: unused.clone(),
                        target: t1,
                    });
                }
                if !s0.is_empty() {
                    stack.push(Frame {
                        s: s0,
                        unused,
                        target: frame.target,
                    });
                }
            }

            gray_residual_pmh(&rows, target, n, &mut instructions, &mut total_cnots)?;
            Some((instructions, total_cnots))
        }
    };
}
