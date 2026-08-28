// Note: This file is included at the top level of `engine/mod.rs`
// and defines a macro that is called by the main engine-generating macro.

macro_rules! define_reduction_logic {
    (
        $primitive:ty,
        $poly_capacity:expr
    ) => {
        impl EvaluatedPathSum {
            /// Returns true if the variable appears in `out_state` or in a continuous
            /// phase parity. Such a variable must never be integrated out, since the
            /// summation identities used by `reduce()` only hold for variables that
            /// occur exclusively in the discrete phase polynomial.
            #[inline(always)]
            fn is_var_live(
                out_state: &[BooleanPoly],
                continuous: &ContinuousPhasePoly,
                v_mask: $primitive,
            ) -> bool {
                out_state.iter().any(|p| (p.variable_mask & v_mask) != 0)
                    || continuous.parities.iter().any(|p| (p.variable_mask & v_mask) != 0)
            }

            pub fn reduce(&mut self) {
                let mut overall_changed = true;
                while overall_changed {
                    overall_changed = false;
                    let mut dead_vars = 0 as $primitive;
                    let original_num_path_vars = self.num_path_vars;
                    let path_var_mask = if original_num_path_vars == 0 {
                        0
                    } else {
                        (<$primitive>::MAX.checked_shl(self.num_qubits as u32).unwrap_or(0))
                            & (<$primitive>::MAX.checked_shr( (<$primitive>::BITS - (self.num_qubits + original_num_path_vars)) as u32).unwrap_or(0))
                    };

                    let mut global_out_mask = self.out_state.iter().fold(0, |acc, p| acc | p.variable_mask);
                    for parity in &self.continuous_poly.parities {
                        global_out_mask |= parity.variable_mask;
                    }

                    let mut continuous_needs_compact = false;
                    
                    // Signature-based variable merging (replaces the O(m^2)
                    // pairwise rescans). For every path variable, one pass over
                    // `out_state` and the continuous parities collects (a) the
                    // ordered list of structures in which it occurs linearly
                    // and (b) whether it occurs nonlinearly anywhere.
                    // Eligibility never reads `phase_poly`. Two variables are
                    // mergeable iff neither is nonlinear and their linear
                    // occurrence lists are identical and non-empty.
                    //
                    // A committed merge only deletes the *lower* variable's
                    // linear occurrences (parities stay pairwise distinct, so
                    // no parity collapse/cancellation can occur), leaving every
                    // other variable's signature intact; groups computed once
                    // per pass therefore stay valid. Within a group
                    // {g1 < g2 < ... < gk} only the chained adjacent pairs
                    // (g1,g2), (g2,g3), ..., (g(k-1),gk) are replayed — after
                    // (g1,g2) commits, g1 occurs nowhere, so the sequential
                    // algorithm would skip every later g1 pairing. Committing
                    // all pairs in ascending (v1, v2) lexicographic order, with
                    // the exact per-pair phase rewrite between commits, matches
                    // the sequential restart-per-merge order: each sequential
                    // restart picks the smallest eligible pair, and the chain
                    // link (g_i, g_{i+1}) always precedes (g_i, g_j) for
                    // j > i+1.
                    let mut basis_changed = false;
                    if original_num_path_vars >= 2 {
                        let lo = self.num_qubits as usize;
                        let m = original_num_path_vars as usize;
                        let mut nonlinear = vec![false; m];
                        let mut lin_occ: Vec<Vec<u32>> = vec![Vec::new(); m];

                        fn scan_structure(
                            terms: &[$primitive],
                            path_var_mask: $primitive,
                            lo: usize,
                            structure_id: u32,
                            nonlinear: &mut [bool],
                            lin_occ: &mut [Vec<u32>],
                        ) {
                            for &t in terms {
                                let mut pv = t & path_var_mask;
                                if pv == 0 { continue; }
                                // Linear occurrence: the term is exactly one
                                // path-variable bit. Anything else touching a
                                // path variable is a nonlinear occurrence.
                                let linear = pv == t && (pv & (pv - 1)) == 0;
                                while pv != 0 {
                                    let idx = pv.trailing_zeros() as usize - lo;
                                    pv &= pv - 1;
                                    if linear {
                                        lin_occ[idx].push(structure_id);
                                    } else {
                                        nonlinear[idx] = true;
                                    }
                                }
                            }
                        }

                        let mut structure_id = 0u32;
                        for poly in &self.out_state {
                            scan_structure(&poly.terms, path_var_mask, lo, structure_id, &mut nonlinear, &mut lin_occ);
                            structure_id += 1;
                        }
                        for parity in &self.continuous_poly.parities {
                            scan_structure(&parity.terms, path_var_mask, lo, structure_id, &mut nonlinear, &mut lin_occ);
                            structure_id += 1;
                        }

                        // Group eligible variables by signature. Variables with
                        // an all-empty signature appear nowhere linearly and are
                        // skipped entirely (the sequential code never merges
                        // them: `appears_anywhere == false`).
                        let mut eligible: Vec<usize> = (0..m)
                            .filter(|&i| !nonlinear[i] && !lin_occ[i].is_empty())
                            .collect();
                        eligible.sort_by(|&a, &b| lin_occ[a].cmp(&lin_occ[b]).then(a.cmp(&b)));

                        let mut pairs: Vec<(usize, usize)> = Vec::new();
                        let mut g = 0;
                        while g < eligible.len() {
                            let mut h = g + 1;
                            while h < eligible.len() && lin_occ[eligible[h]] == lin_occ[eligible[g]] {
                                pairs.push((eligible[h - 1], eligible[h]));
                                h += 1;
                            }
                            g = h;
                        }
                        // Ascending (v1, v2) commit order across groups equals
                        // the sequential commit order.
                        pairs.sort_unstable();

                        for &(i1, i2) in &pairs {
                            let v1_mask = 1 as $primitive << (lo + i1);
                            let v2_mask = 1 as $primitive << (lo + i2);
                            basis_changed = true;

                            for poly in &mut self.out_state {
                                poly.terms.retain(|t| *t != v1_mask);
                                poly.variable_mask = poly.terms.iter().fold(0, |acc, &x| acc | x);
                            }
                            for parity in &mut self.continuous_poly.parities {
                                parity.terms.retain(|t| *t != v1_mask);
                                parity.variable_mask = parity.terms.iter().fold(0, |acc, &x| acc | x);
                            }

                            let mut new_phase_terms = Vec::with_capacity(self.phase_poly.terms.len() * 3);
                            for term in self.phase_poly.terms.iter() {
                                let mono = term.monomial();
                                let phase = term.phase();
                                if (mono & v2_mask) != 0 {
                                    let base = mono & !v2_mask;
                                    if (base & v1_mask) != 0 {
                                        let real_base = base & !v1_mask;
                                        new_phase_terms.push(PackedPhaseTerm::create(v1_mask | real_base, phase));
                                        new_phase_terms.push(PackedPhaseTerm::create(mono, (8 - phase) % 8));
                                    } else {
                                        new_phase_terms.push(PackedPhaseTerm::create(mono, phase));
                                        new_phase_terms.push(PackedPhaseTerm::create(v1_mask | base, phase));
                                        let minus_2c = (8 - (2 * phase) % 8) % 8;
                                        new_phase_terms.push(PackedPhaseTerm::create(v1_mask | mono, minus_2c));
                                    }
                                } else {
                                    new_phase_terms.push(*term);
                                }
                            }
                            self.phase_poly.terms.clear();
                            self.phase_poly.merge_unsorted_batch(new_phase_terms);
                        }
                    }
                    
                    if basis_changed {
                        global_out_mask = self.out_state.iter().fold(0, |acc, p| acc | p.variable_mask);
                        for parity in &self.continuous_poly.parities {
                            global_out_mask |= parity.variable_mask;
                        }
                        self.continuous_poly.compact();
                        overall_changed = true;
                        continue;
                    }

                    // Per-sweep pivot index: one O(T*k) pass over `phase_poly`
                    // collects, for every path variable, the pivot partner
                    // mask, base phase, and validity flag — replacing the
                    // per-candidate O(T) rescans. Staleness discipline: the
                    // index is a pre-filter/cache only. It is rebuilt after
                    // every committed pivot (the `u := E` substitution
                    // rewrites `phase_poly` and can change other variables'
                    // pivot data), and the exact `is_var_live` re-checks plus
                    // the post-substitution `global_out_mask` refresh below
                    // stay in place unchanged.
                    fn build_pivot_index(
                        phase_terms: &[PackedPhaseTerm],
                        path_var_mask: $primitive,
                        partner: &mut [$primitive],
                        base: &mut [u8],
                        invalid: &mut [bool],
                    ) {
                        partner.fill(0);
                        base.fill(0);
                        invalid.fill(false);
                        for term in phase_terms {
                            let mono = term.monomial();
                            let phase = term.phase();
                            let mut pv = mono & path_var_mask;
                            while pv != 0 {
                                let bit_idx = pv.trailing_zeros() as usize;
                                pv &= pv - 1;
                                let v_mask = 1 as $primitive << bit_idx;
                                let remaining = mono & !v_mask;
                                if remaining == 0 {
                                    if phase == 4 {
                                        partner[bit_idx] ^= 1 as $primitive << (<$primitive>::BITS - 1);
                                        base[bit_idx] = 4;
                                    } else if phase == 2 || phase == 6 {
                                        base[bit_idx] = phase;
                                    } else {
                                        invalid[bit_idx] = true;
                                    }
                                } else if remaining.count_ones() == 1 {
                                    if phase == 4 {
                                        partner[bit_idx] ^= remaining;
                                    } else {
                                        invalid[bit_idx] = true;
                                    }
                                } else {
                                    invalid[bit_idx] = true;
                                }
                            }
                        }
                    }

                    let mut piv_partner = [0 as $primitive; <$primitive>::BITS as usize];
                    let mut piv_base = [0u8; <$primitive>::BITS as usize];
                    let mut piv_invalid = [false; <$primitive>::BITS as usize];
                    build_pivot_index(&self.phase_poly.terms, path_var_mask, &mut piv_partner, &mut piv_base, &mut piv_invalid);

                    let mut changed = true;
                    while changed {
                        changed = false;
                        for v_idx in self.num_qubits..self.num_qubits + original_num_path_vars {
                            let v_mask = 1 as $primitive << v_idx;
                            if (dead_vars & v_mask) != 0 { continue; }
                            if (global_out_mask & v_mask) != 0 { continue; }

                            if piv_invalid[v_idx as usize] { continue; }
                            let p_mask = piv_partner[v_idx as usize];
                            let base_phase = piv_base[v_idx as usize];

                            if base_phase == 2 || base_phase == 6 {
                                // Re-verify liveness against the live structures; `global_out_mask`
                                // is only a fast pre-filter and can lag behind substitutions
                                // performed earlier in this pass.
                                if Self::is_var_live(&self.out_state, &self.continuous_poly, v_mask) { continue; }

                                changed = true;
                                let e_poly = BooleanPoly::from_mask(p_mask);

                                let mut next_gen_terms = Vec::with_capacity(self.phase_poly.terms.len());
                                for term in self.phase_poly.terms.iter() {
                                    if (term.monomial() & v_mask) == 0 {
                                        next_gen_terms.push(*term);
                                    }
                                }
                                self.phase_poly.terms.clear();
                                self.phase_poly.merge_unsorted_batch(next_gen_terms);

                                let phase_to_add = if base_phase == 2 { 6 } else { 2 };
                                self.push_phase_expansion(&e_poly.terms, phase_to_add);
                                dead_vars |= v_mask;
                                // The elimination rewrote phase_poly: rebuild the
                                // pivot index before judging further candidates.
                                build_pivot_index(&self.phase_poly.terms, path_var_mask, &mut piv_partner, &mut piv_base, &mut piv_invalid);
                                continue;
                            }

                            let valid_pivots = p_mask & path_var_mask & !dead_vars;
                            if valid_pivots == 0 { continue; }

                            // Re-verify liveness against the live structures; `global_out_mask`
                            // is only a fast pre-filter and can lag behind substitutions
                            // performed earlier in this pass.
                            if Self::is_var_live(&self.out_state, &self.continuous_poly, v_mask) { continue; }

                            changed = true;
                            let pivot_index = valid_pivots.trailing_zeros();
                            let u_mask = 1 as $primitive << pivot_index;
                            let e_mask = p_mask ^ u_mask;
                            let e_poly = BooleanPoly::from_mask(e_mask);

                            for poly in &mut self.out_state {
                                if (poly.variable_mask & u_mask) == 0 { continue; }
                                let mut distributed_terms = SmallVec::<[$primitive; $poly_capacity]>::new();
                                poly.terms.retain(|t| {
                                    if (*t & u_mask) != 0 {
                                        let base = *t & !u_mask;
                                        for &e_term in &e_poly.terms {
                                            let new_t = if e_term == 0 { base } else { base | e_term };
                                            distributed_terms.push(new_t);
                                        }
                                        false
                                    } else {
                                        true
                                    }
                                });
                                if !distributed_terms.is_empty() {
                                    let addition = BooleanPoly::from_terms(distributed_terms);
                                    poly.add_assign(&addition);
                                }
                            }

                            self.continuous_poly.substitute(u_mask, &e_poly);
                            continuous_needs_compact = true;

                            // Substituting `u := E` can make E's variables live in `out_state`
                            // and `continuous_poly`. Refresh the liveness mask immediately so
                            // subsequent candidates in this pass are judged against the
                            // current state rather than a stale snapshot.
                            global_out_mask = self.out_state.iter().fold(0, |acc, p| acc | p.variable_mask);
                            for parity in &self.continuous_poly.parities {
                                global_out_mask |= parity.variable_mask;
                            }

                            let mut next_gen_terms = Vec::with_capacity(self.phase_poly.terms.len());
                            for term in self.phase_poly.terms.iter() {
                                let mono = term.monomial();
                                if (mono & v_mask) != 0 { continue; }
                                if (mono & u_mask) != 0 {
                                    // Exact substitution u := E in a mod-8 phase term requires
                                    // the full multilinear XOR expansion. Distributing linearly
                                    // over E's terms is only valid for pi (phase 4) coefficients
                                    // and silently corrupted S/T-phase terms before.
                                    let base = mono & !u_mask;
                                    Self::push_xor_phase_expansion(&mut next_gen_terms, base, &e_poly.terms, term.phase());
                                } else {
                                    next_gen_terms.push(*term);
                                }
                            }
                            self.phase_poly.terms.clear();
                            self.phase_poly.merge_unsorted_batch(next_gen_terms);
                            dead_vars |= v_mask | u_mask;
                            // The substitution rewrote phase_poly: rebuild the
                            // pivot index before judging further candidates.
                            build_pivot_index(&self.phase_poly.terms, path_var_mask, &mut piv_partner, &mut piv_base, &mut piv_invalid);
                        }
                    }

                    let surviving_mask = path_var_mask & !dead_vars;
                    if surviving_mask.count_ones() < self.num_path_vars as u32 {
                        let mut remapping = [0u32; <$primitive>::BITS as usize];
                        for i in 0..self.num_qubits { remapping[i as usize] = i; }
                        let mut current_new_idx = 0;
                        for i in self.num_qubits..<$primitive>::BITS {
                            if (surviving_mask & (1 as $primitive << i)) != 0 {
                                remapping[i as usize] = self.num_qubits + current_new_idx;
                                current_new_idx += 1;
                            }
                        }

                        let remap_mono = |mono: $primitive| -> $primitive {
                            let mut new_mono = 0 as $primitive;
                            let mut temp = mono;
                            while temp != 0 {
                                let bit_idx = temp.trailing_zeros() as usize;
                                new_mono |= 1 as $primitive << remapping[bit_idx];
                                temp &= temp - 1;
                            }
                            new_mono
                        };

                        for poly in &mut self.out_state {
                            let new_terms = poly.terms.iter().map(|t| remap_mono(*t)).collect();
                            *poly = BooleanPoly::from_terms(new_terms);
                        }

                        for parity in &mut self.continuous_poly.parities {
                            for t in &mut parity.terms { *t = remap_mono(*t); }
                            parity.terms.sort_unstable();
                            parity.variable_mask = parity.terms.iter().fold(0, |acc, &x| acc | x);
                        }
                        continuous_needs_compact = true;

                        let mut new_phase_terms = Vec::with_capacity(self.phase_poly.terms.len());
                        for term in &self.phase_poly.terms {
                            new_phase_terms.push(PackedPhaseTerm::create(remap_mono(term.monomial()), term.phase()));
                        }
                        self.phase_poly.terms.clear();
                        self.phase_poly.merge_unsorted_batch(new_phase_terms);
                        self.num_path_vars = surviving_mask.count_ones();
                    }

                    if continuous_needs_compact {
                        self.continuous_poly.compact();
                        if self.promote_cliffords() {
                            overall_changed = true;
                        }
                    } else {
                        // Even if no substitution happened, we should check for promotions just in case!
                        if self.promote_cliffords() {
                            overall_changed = true;
                        }
                    }
                }
            }
        }
    }
}
