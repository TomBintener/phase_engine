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

        /// Gray-ordered parity-network synthesis for CX+RZ phase circuits on
        /// all-to-all connectivity.
        ///
        /// Given the phase parities `(mask, angle)` of a Hadamard-free region
        /// and its target linear reversible function (`target[i]` = XOR mask
        /// of input variables on output wire `i`), emits a circuit computing
        /// every parity exactly once (each rotation applied where the wire
        /// already holds the parity) followed by a Gauss-Jordan fix-up to the
        /// target linear function.
        ///
        /// Unlike the Steiner synthesizer this shares parity prefixes across
        /// terms: consecutive (Gray-ordered) parities are reached by XOR-ing
        /// a few live wires instead of building and unbuilding a fresh CNOT
        /// tree per term.
        ///
        /// Returns `None` only on internal inconsistency (non-invertible wire
        /// basis, unsolvable parity), which cannot happen for well-formed
        /// linear states; callers still guard with a cost check and the
        /// engine's `state_union` equality proof.
        pub fn synthesize_gray_network(
            parities: &[($primitive, f64)],
            target: &[$primitive],
            n: usize,
        ) -> Option<(Vec<String>, i64)> {
            let mut instructions: Vec<String> = Vec::new();
            let mut total_cnots: i64 = 0;

            // Merge duplicate parity masks; drop zero-angle terms.
            use std::collections::HashMap;
            let mut merged: HashMap<$primitive, f64> = HashMap::new();
            for &(mask, angle) in parities {
                if mask == 0 {
                    continue;
                }
                *merged.entry(mask).or_insert(0.0) += angle;
            }
            let tau = std::f64::consts::TAU;
            let mut terms: Vec<($primitive, f64)> = merged
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

            // Gray-flavored ordering: group by popcount, then by the
            // binary-reflected key so neighbors share long prefixes; refine
            // small lists with greedy nearest-neighbor chaining on Hamming
            // distance.
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

            // Wire state: rows[q] = parity of input variables currently held
            // by wire q. Starts as the identity and always stays an
            // invertible GF(2) basis (we only ever XOR distinct rows).
            let mut rows: Vec<$primitive> = (0..n).map(|i| (1 as $primitive) << i).collect();

            // Solve `sum_{i in combo} rows[i] == m` for the current basis.
            let solve = |rows: &[$primitive], m: $primitive| -> Option<u128> {
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
                    // Find a pivot row with this bit.
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
            };

            for &(mask, angle) in &terms {
                let combo = solve(&rows, mask)?;
                if combo == 0 {
                    return None;
                }
                let members: Vec<usize> =
                    (0..n).filter(|&i| (combo >> i) & 1 == 1).collect();
                // Target choice: the member whose row has the largest overlap
                // with the parity (fewest live bits destroyed elsewhere is a
                // wash on all-to-all; overlap keeps rows close to parities
                // that tend to repeat).
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

            // Residual linear fix-up: express the target rows in the current
            // wire basis and synthesize that matrix with Gauss-Jordan.
            let mut residual: Vec<$primitive> = Vec::with_capacity(n);
            for q in 0..n {
                let combo = solve(&rows, target[q])?;
                // combo bits index wires; the residual matrix row for output
                // q is the XOR of those wires.
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
                        total_cnots += 1;
                    }
                }
                Err(_) => return None,
            }

            Some((instructions, total_cnots))
        }
    };
}
