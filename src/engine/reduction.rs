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
                    || continuous.parities.iter().any(|p| (*p & v_mask) != 0)
            }

            /// True iff `mask` is empty or a single bit. Short-circuits on
            /// zero so debug builds do not overflow on `0 - 1`. Release
            /// wraparound of `0 - 1` is `MAX`, and `0 & MAX == 0`, so this
            /// matches the intended power-of-two-or-zero skip.
            #[inline(always)]
            fn is_zero_or_pow2(mask: $primitive) -> bool {
                mask == 0 || mask & (mask - 1) == 0
            }

            /// Substitute `v := v XOR e` everywhere (an invertible affine change
            /// of one path variable, so the denoted operator is unchanged).
            /// `e` must not contain `v`.
            fn gauge_substitute(&mut self, v_mask: $primitive, e_poly: &BooleanPoly) {
                if e_poly.terms.is_empty() { return; }
                for poly in &mut self.out_state {
                    if (poly.variable_mask & v_mask) != 0 && poly.terms.contains(&v_mask) {
                        poly.add_assign(e_poly);
                    }
                }
                for (parity, ticks) in self
                    .continuous_poly
                    .parities
                    .iter_mut()
                    .zip(self.continuous_poly.phases.iter_mut())
                {
                    if (*parity & v_mask) == 0 {
                        continue;
                    }
                    for &t in &e_poly.terms {
                        if t == 0 {
                            *ticks = negate_ticks(*ticks);
                        } else if t.count_ones() == 1 {
                            *parity ^= t;
                        }
                    }
                }
                if (self.phase_poly.terms.iter().fold(0, |acc, t| acc | t.monomial()) & v_mask) == 0 {
                    return;
                }
                // c * v * M  ->  c * (v XOR e_1 XOR ... XOR e_k) * M, exact in Z_8.
                let mut sub_terms: SmallVec<[$primitive; $poly_capacity]> = SmallVec::new();
                sub_terms.push(v_mask);
                sub_terms.extend(e_poly.terms.iter().copied());
                let mut new_terms = Vec::with_capacity(self.phase_poly.terms.len() * 2);
                for term in self.phase_poly.terms.iter() {
                    let mono = term.monomial();
                    if (mono & v_mask) != 0 {
                        Self::push_xor_phase_expansion(&mut new_terms, mono & !v_mask, &sub_terms, term.phase());
                    } else {
                        new_terms.push(*term);
                    }
                }
                self.phase_poly.terms.clear();
                self.phase_poly.merge_unsorted_batch(new_terms);
            }

            /// Canonical form of the path-variable gauge.
            ///
            /// Path variables are only defined up to invertible affine
            /// substitutions and relabeling: `H0 H1` vs `H1 H0` allocate them in
            /// opposite order, `SX` vs `H S H` and nam `h` vs ibm `RZ SX RZ`
            /// leave different affine residues on the wire, and an internal
            /// variable `v` can be traded for `1 XOR v` (negating its rotation).
            /// This pass picks one representative:
            ///
            /// 1. Row echelon: scanning `out_state` rows in order, the first
            ///    not-yet-pivoted path variable of a row becomes that row's
            ///    pivot and the row is reduced to exactly that variable
            ///    (`v := v XOR rest`). Rows without a fresh variable stay as
            ///    combinations of earlier pivots and inputs.
            /// 2. Parity pivots: the same for continuous parities, visited in
            ///    a label-independent order (angle, then the parity's input /
            ///    row-pivot content), so every internal variable carrying a
            ///    rotation is a bare parity.
            /// 3. Constant gauge: for each remaining variable decide between
            ///    `v` and `1 XOR v` from its bare-parity remainder (smaller
            ///    angle wins) and, on a tie, from the discrete coefficients of
            ///    the terms containing it.
            /// 4. Relabel: row pivots in row order, then the remaining
            ///    variables by a label-independent signature (multiset hash of
            ///    the phase terms and parities they occur in, seen through
            ///    input and row-pivot bits), ties keeping the current order.
            ///
            /// Every step is a bijective change of summation variables, so the
            /// denoted operator is unchanged; the pass is idempotent and a
            /// no-op (one early return) when there are no path variables.
            pub fn canonicalize_gauge(&mut self) -> bool {
                if self.is_overflowed() { return false; }
                let m = self.num_path_vars as usize;
                if m == 0 { return false; }
                let lo = self.num_qubits as usize;
                let path_mask: $primitive = (<$primitive>::MAX.checked_shl(self.num_qubits).unwrap_or(0))
                    & (<$primitive>::MAX.checked_shr((<$primitive>::BITS - (self.num_qubits + self.num_path_vars)) as u32).unwrap_or(0));
                let qubit_mask: $primitive = if lo >= <$primitive>::BITS as usize { <$primitive>::MAX } else { (1 as $primitive << lo) - 1 };
                // Gate application keeps rows and parities affine in the path
                // variables; bail on hand-built states that are not.
                let nonlinear = |t: &$primitive| (*t & path_mask) != 0 && !Self::is_zero_or_pow2(*t);
                if self.out_state.iter().any(|p| p.terms.iter().any(nonlinear)) {
                    return false;
                }

                let mut changed = false;
                let mut pivots: $primitive = 0;
                let mut order: Vec<$primitive> = Vec::with_capacity(m);

                // Pivot choice for a structure with several fresh variables:
                // the one with the fewest phase-term occurrences (cheapest
                // substitution), ties to the lowest bit.
                let pick = |fresh: $primitive, phase_poly: &CanonicalPhasePoly| -> $primitive {
                    if Self::is_zero_or_pow2(fresh) { return fresh; }
                    let mut counts = [0u32; <$primitive>::BITS as usize];
                    for t in phase_poly.terms.iter() {
                        let mut pv = t.monomial() & fresh;
                        while pv != 0 {
                            counts[pv.trailing_zeros() as usize] += 1;
                            pv &= pv - 1;
                        }
                    }
                    let mut best = 0 as $primitive;
                    let mut best_c = u32::MAX;
                    let mut f = fresh;
                    while f != 0 {
                        let idx = f.trailing_zeros() as usize;
                        if counts[idx] < best_c { best_c = counts[idx]; best = 1 as $primitive << idx; }
                        f &= f - 1;
                    }
                    best
                };

                // 1. Row echelon.
                for q in 0..self.out_state.len() {
                    let fresh = self.out_state[q].variable_mask & path_mask & !pivots;
                    if fresh == 0 { continue; }
                    let v = pick(fresh, &self.phase_poly);
                    pivots |= v;
                    order.push(v);
                    if self.out_state[q].terms.len() > 1 {
                        let residue: SmallVec<[$primitive; $poly_capacity]> =
                            self.out_state[q].terms.iter().copied().filter(|&t| t != v).collect();
                        changed = true;
                        self.gauge_substitute(v, &BooleanPoly::from_terms(residue));
                    }
                }
                let row_pivots = pivots;
                if changed {
                    self.continuous_poly.compact();
                    self.promote_cliffords();
                }

                // Canonical view of a mask: inputs unchanged, row pivots at their
                // canonical positions, everything else dropped.
                let mut canon_map = [0u32; <$primitive>::BITS as usize];
                for (k, &v) in order.iter().enumerate() {
                    canon_map[v.trailing_zeros() as usize] = (lo + k) as u32;
                }
                let canon_anchor = |mono: $primitive| -> $primitive {
                    let mut out = mono & qubit_mask;
                    let mut pv = mono & row_pivots;
                    while pv != 0 {
                        let idx = pv.trailing_zeros() as usize;
                        out |= 1 as $primitive << canon_map[idx];
                        pv &= pv - 1;
                    }
                    out
                };

                // 2. Parity pivots, smallest (angle, canonical content) first.
                //    The angle enters as `min(r, π/2 - r)`: step 3 may still
                //    negate remainders, and the pivot choice must not depend on
                //    that, or the pass would not be idempotent.
                loop {
                    let mut best: Option<(u64, $primitive, u32, usize)> = None;
                    for (i, &parity) in self.continuous_poly.parities.iter().enumerate() {
                        let fresh = parity & path_mask & !pivots;
                        if fresh == 0 { continue; }
                        let r = self.continuous_poly.phases[i] as u64;
                        let key = (r.min(TICKS_PER_PI_2 - r), canon_anchor(parity), fresh.count_ones(), i);
                        if best.map_or(true, |b| key < b) { best = Some(key); }
                    }
                    let Some((_, _, _, i)) = best else { break };
                    let fresh = self.continuous_poly.parities[i] & path_mask & !pivots;
                    let v = pick(fresh, &self.phase_poly);
                    pivots |= v;
                    if self.continuous_poly.parities[i] != v {
                        let residue = BooleanPoly::from_mask(self.continuous_poly.parities[i] ^ v);
                        changed = true;
                        self.gauge_substitute(v, &residue);
                        // Keep parities canonical (constant fold, sort) and the
                        // split canonical before choosing the next pivot.
                        self.continuous_poly.compact();
                        self.promote_cliffords();
                    }
                }

                let rest_mask = path_mask & !row_pivots;
                if rest_mask == 0 {
                    return self.relabel_gauge(&order, lo) || changed;
                }

                #[inline(always)]
                fn mix(mut x: u64) -> u64 {
                    x ^= x >> 30; x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    x ^= x >> 27; x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
                    x ^ (x >> 31)
                }
                #[inline(always)]
                fn fold_bits(x: $primitive) -> u64 {
                    (x as u64) ^ (x.checked_shr(64).unwrap_or(0) as u64).wrapping_mul(0x2545_f491_4f6c_dd1d)
                }
                // Multiset hash per remaining variable over the terms and
                // parities it occurs in, seen through canonical anchors only.
                let signatures = |this: &Self| -> [u64; <$primitive>::BITS as usize] {
                    let mut sig = [0u64; <$primitive>::BITS as usize];
                    for t in this.phase_poly.terms.iter() {
                        let mono = t.monomial();
                        let mut pv = mono & rest_mask;
                        if pv == 0 { continue; }
                        let h = mix(mix(fold_bits(canon_anchor(mono)) ^ 0x9e37_79b9_7f4a_7c15)
                            ^ ((t.phase() as u64) << 56)
                            ^ ((pv.count_ones() as u64) << 48));
                        while pv != 0 {
                            let idx = pv.trailing_zeros() as usize;
                            sig[idx] = sig[idx].wrapping_add(h);
                            pv &= pv - 1;
                        }
                    }
                    for (parity, &ticks) in this.continuous_poly.parities.iter().zip(this.continuous_poly.phases.iter()) {
                        let mut pv = *parity & rest_mask;
                        if pv == 0 { continue; }
                        let h = mix(mix(fold_bits(canon_anchor(*parity)) ^ 0x7f4a_7c15_9e37_79b9)
                            ^ ((ticks as u64) << 32)
                            ^ (pv.count_ones() as u64));
                        while pv != 0 {
                            let idx = pv.trailing_zeros() as usize;
                            sig[idx] = sig[idx].wrapping_add(h);
                            pv &= pv - 1;
                        }
                    }
                    sig
                };
                let sorted_rest = |sig: &[u64; <$primitive>::BITS as usize]| -> Vec<$primitive> {
                    let mut rest: Vec<(u64, usize)> = Vec::with_capacity(rest_mask.count_ones() as usize);
                    let mut f = rest_mask;
                    while f != 0 {
                        let idx = f.trailing_zeros() as usize;
                        rest.push((sig[idx], idx));
                        f &= f - 1;
                    }
                    rest.sort_unstable();
                    rest.into_iter().map(|(_, idx)| 1 as $primitive << idx).collect()
                };

                // 3. Constant gauge.
                //
                // Per-variable parity statistics in one pass: remainder of the
                // bare parity (if any), number of parities containing the
                // variable, and whether any parity holds two non-row variables
                // (rare; forces the general path below).
                let mut bare = [u64::MAX; <$primitive>::BITS as usize];
                let mut pcount = [0u8; <$primitive>::BITS as usize];
                let mut parity_adj: $primitive = 0;
                for (parity, &ticks) in self.continuous_poly.parities.iter().zip(self.continuous_poly.phases.iter()) {
                    let pv = *parity & rest_mask;
                    if pv == 0 { continue; }
                    if !Self::is_zero_or_pow2(pv) { parity_adj |= pv; }
                    if Self::is_zero_or_pow2(*parity) { bare[pv.trailing_zeros() as usize] = ticks as u64; }
                    let mut f = pv;
                    while f != 0 {
                        pcount[f.trailing_zeros() as usize] += 1;
                        f &= f - 1;
                    }
                }
                // (a) Variables whose bare parity remainder is not exactly π/4
                //     are decided on their own: flipping maps `r` to `π/2 - r`,
                //     so keep the smaller remainder. This is invariant under
                //     every other variable's gauge.
                let mut tied: $primitive = 0;
                let mut f = rest_mask;
                while f != 0 {
                    let idx = f.trailing_zeros() as usize;
                    let v = 1 as $primitive << idx;
                    f &= f - 1;
                    match bare[idx] {
                        r if r != u64::MAX && r != TICKS_PER_PI_4 => {
                            if r > TICKS_PER_PI_4 {
                                changed = true;
                                self.flip_constant_gauge(v);
                            }
                        }
                        _ => tied |= v,
                    }
                }
                // (b) T-type / parity-free variables. Flipping one shifts the
                //     coefficients of its neighbours, so connected components
                //     (edges = shared terms or parities) are decided jointly by
                //     exhaustive search over flip patterns, keeping the
                //     lexicographically smallest component key. Isolated
                //     variables use the closed-form rule; components larger
                //     than the cap fall back to it greedily.
                if tied != 0 {
                    let mut adj = [0 as $primitive; <$primitive>::BITS as usize];
                    for t in self.phase_poly.terms.iter() {
                        let tv = t.monomial() & tied;
                        if Self::is_zero_or_pow2(tv) { continue; }
                        let mut f = tv;
                        while f != 0 {
                            adj[f.trailing_zeros() as usize] |= tv;
                            f &= f - 1;
                        }
                    }
                    for parity in self.continuous_poly.parities.iter() {
                        let tv = *parity & tied;
                        if Self::is_zero_or_pow2(tv) { continue; }
                        let mut f = tv;
                        while f != 0 {
                            adj[f.trailing_zeros() as usize] |= tv;
                            f &= f - 1;
                        }
                    }
                    // Isolated variables that never share a parity with another
                    // non-row variable: one pass decides them all. At the first
                    // coordinate (monomial order) where `c != F(c)` keep the
                    // smaller value; `F(c) = 6·P - c` on the linear term and
                    // `-c` elsewhere.
                    let mut simple: $primitive = 0;
                    let mut f = tied;
                    while f != 0 {
                        let idx = f.trailing_zeros() as usize;
                        let v = 1 as $primitive << idx;
                        f &= f - 1;
                        if adj[idx] & !v == 0 && (parity_adj & v) == 0 { simple |= v; }
                    }
                    let mut undecided = simple;
                    let mut flips: $primitive = 0;
                    for t in self.phase_poly.terms.iter() {
                        if undecided == 0 { break; }
                        let mono = t.monomial();
                        let mut pv = mono & undecided;
                        while pv != 0 {
                            let idx = pv.trailing_zeros() as usize;
                            let v = 1 as $primitive << idx;
                            pv &= pv - 1;
                            let c = t.phase();
                            if mono != v {
                                // Linear coordinate absent: c = 0, F = 6·P.
                                let f_lin = (6 * pcount[idx] as u32) % 8;
                                if f_lin != 0 { undecided &= !v; continue; } // 0 < F: keep
                            }
                            let fc = if mono == v { (6 * pcount[idx] as u32 + 8 - c as u32) % 8 } else { (8 - c as u32) % 8 };
                            if fc != c as u32 {
                                if fc < c as u32 { flips |= v; }
                                undecided &= !v;
                            }
                        }
                    }
                    // Variables with no terms at all: linear coordinate c = 0.
                    // F = 6·P >= 0, so they keep their gauge.
                    let mut f = flips;
                    while f != 0 {
                        let v = 1 as $primitive << f.trailing_zeros();
                        f &= f - 1;
                        changed = true;
                        self.flip_constant_gauge(v);
                    }

                    const JOINT_CAP: u32 = 5;
                    let mut remaining = tied & !simple;
                    while remaining != 0 {
                        // Connected component of `remaining` containing its lowest var.
                        let mut comp = 1 as $primitive << remaining.trailing_zeros();
                        loop {
                            let mut grown = comp;
                            let mut f = comp;
                            while f != 0 {
                                grown |= adj[f.trailing_zeros() as usize] & remaining;
                                f &= f - 1;
                            }
                            if grown == comp { break; }
                            comp = grown;
                        }
                        remaining &= !comp;
                        let k = comp.count_ones();
                        let vars: Vec<$primitive> = {
                            let mut v = Vec::with_capacity(k as usize);
                            let mut c = comp;
                            while c != 0 { v.push(1 as $primitive << c.trailing_zeros()); c &= c - 1; }
                            v
                        };
                        if k == 1 || k > JOINT_CAP {
                            let sig = signatures(self);
                            for v in sorted_rest(&sig).into_iter().filter(|v| (*v & comp) != 0) {
                                if self.constant_gauge_wants_flip(v) {
                                    changed = true;
                                    self.flip_constant_gauge(v);
                                }
                            }
                            continue;
                        }
                        // Gray-code walk over the 2^k patterns: one flip per step.
                        let mut best_key = self.component_key(comp, rest_mask, &canon_anchor, &signatures);
                        let mut best_pattern: u32 = 0;
                        let mut pattern: u32 = 0;
                        for step in 1u32..(1u32 << k) {
                            let bit = step.trailing_zeros();
                            pattern ^= 1 << bit;
                            self.flip_constant_gauge(vars[bit as usize]);
                            let key = self.component_key(comp, rest_mask, &canon_anchor, &signatures);
                            if key < best_key {
                                best_key = key;
                                best_pattern = pattern;
                            }
                        }
                        // Move from the last pattern to the best one.
                        let diff = pattern ^ best_pattern;
                        for bit in 0..k {
                            if (diff >> bit) & 1 == 1 {
                                self.flip_constant_gauge(vars[bit as usize]);
                            }
                        }
                        if best_pattern != 0 { changed = true; }
                    }
                }

                // 4. Relabel.
                let sig = signatures(self);
                order.extend(sorted_rest(&sig));
                self.relabel_gauge(&order, lo) || changed
            }

            /// Cheap follow-up for `apply_x` / `apply_cx` on an otherwise
            /// canonical state: those only edit `out_state`, so the gauge can
            /// only have drifted if a row that carries a path variable is no
            /// longer a bare pivot (or a pivot now appears in an earlier row).
            /// Runs the full pass only in that case.
            pub fn canonicalize_gauge_after_row_op(&mut self) -> bool {
                if self.is_overflowed() { return false; }
                if self.num_path_vars == 0 { return false; }
                let path_mask: $primitive = (<$primitive>::MAX.checked_shl(self.num_qubits).unwrap_or(0))
                    & (<$primitive>::MAX.checked_shr((<$primitive>::BITS - (self.num_qubits + self.num_path_vars)) as u32).unwrap_or(0));
                let lo = self.num_qubits as usize;
                let mut pivots: $primitive = 0;
                let mut next_pivot = 1 as $primitive << lo;
                let mut dirty = false;
                for poly in &self.out_state {
                    let fresh = poly.variable_mask & path_mask & !pivots;
                    if fresh == 0 { continue; }
                    // A canonical row introduces exactly one fresh variable, alone,
                    // and canonical labels put row pivots first in row order.
                    if poly.terms.len() != 1 || fresh != next_pivot {
                        dirty = true;
                        break;
                    }
                    pivots |= fresh;
                    next_pivot <<= 1;
                }
                if !dirty { return false; }
                self.canonicalize_gauge()
            }

            /// `v := 1 XOR v` for a non-row variable, then restore canonical
            /// parities and the canonical discrete/continuous split. Involutive.
            fn flip_constant_gauge(&mut self, v: $primitive) {
                self.gauge_substitute(v, &BooleanPoly::from_terms(smallvec::smallvec![0]));
                self.continuous_poly.compact();
                self.promote_cliffords();
            }

            /// Label-independent rendering of everything that touches the
            /// variables in `comp`: terms and parities with component variables
            /// relabeled by signature rank and other non-row variables erased to
            /// a count. Used to pick the canonical flip pattern of a component.
            fn component_key(
                &self,
                comp: $primitive,
                rest_mask: $primitive,
                canon_anchor: &dyn Fn($primitive) -> $primitive,
                signatures: &dyn Fn(&Self) -> [u64; <$primitive>::BITS as usize],
            ) -> (Vec<($primitive, u8, u32)>, Vec<($primitive, u32, u32)>) {
                let sig = signatures(self);
                let mut ranked: Vec<(u64, usize)> = Vec::new();
                let mut c = comp;
                while c != 0 {
                    let idx = c.trailing_zeros() as usize;
                    ranked.push((sig[idx], idx));
                    c &= c - 1;
                }
                ranked.sort_unstable();
                let mut rank = [0u32; <$primitive>::BITS as usize];
                for (k, &(_, idx)) in ranked.iter().enumerate() {
                    rank[idx] = k as u32;
                }
                let lo = self.num_qubits as usize;
                let view = |mono: $primitive| -> ($primitive, u32) {
                    let mut out = canon_anchor(mono);
                    let mut cv = mono & comp;
                    while cv != 0 {
                        let idx = cv.trailing_zeros() as usize;
                        out |= 1 as $primitive << (lo + rank[idx] as usize);
                        cv &= cv - 1;
                    }
                    (out, (mono & rest_mask & !comp).count_ones())
                };
                let mut terms: Vec<($primitive, u8, u32)> = self
                    .phase_poly
                    .terms
                    .iter()
                    .filter(|t| (t.monomial() & comp) != 0)
                    .map(|t| { let (m, others) = view(t.monomial()); (m, t.phase(), others) })
                    .collect();
                terms.sort_unstable();
                let mut parities: Vec<($primitive, u32, u32)> = self
                    .continuous_poly
                    .parities
                    .iter()
                    .zip(self.continuous_poly.phases.iter())
                    .filter(|(p, _)| (*p & comp) != 0)
                    .map(|(p, &ticks)| { let (m, others) = view(*p); (m, ticks, others) })
                    .collect();
                parities.sort_unstable();
                (terms, parities)
            }

            /// Decide between `v` and `1 XOR v` for a variable that is not a row
            /// pivot. Flipping negates every phase on `v`: a parity remainder `r`
            /// becomes `π/2 - r`, and the discrete coefficient vector `c_M` of
            /// the terms `c_M·v·M` maps to `F(c)_M = -c_M + δ_M`, where `δ_∅ = 6·P`
            /// (`P` = parities containing `v`, each re-splits with quotient 6)
            /// and `δ_{w} = 4·P_vw` (`P_vw` = parities containing both `v` and
            /// `w`); higher `δ` vanish. `F` is an involution, so "prefer the
            /// smaller of `c_M` and `F(c)_M` at the first coordinate where they
            /// differ" is a consistent choice for an isolated variable. Used
            /// only as the greedy fallback for oversized tied components.
            fn constant_gauge_wants_flip(&self, v: $primitive) -> bool {
                // Parity statistics for `v`.
                let mut bare_remainder: Option<u64> = None;
                let mut parity_count: u64 = 0;
                let mut shared: Vec<($primitive, u64)> = Vec::new(); // (other var bit, #shared parities)
                for (parity, &ticks) in self.continuous_poly.parities.iter().zip(self.continuous_poly.phases.iter()) {
                    if (*parity & v) == 0 { continue; }
                    parity_count += 1;
                    if Self::is_zero_or_pow2(*parity) {
                        bare_remainder = Some(ticks as u64);
                    }
                    let mut others = *parity & !v;
                    while others != 0 {
                        let w = 1 as $primitive << others.trailing_zeros();
                        others &= others - 1;
                        match shared.iter_mut().find(|(b, _)| *b == w) {
                            Some(e) => e.1 += 1,
                            None => shared.push((w, 1)),
                        }
                    }
                }
                if let Some(r) = bare_remainder {
                    if r != TICKS_PER_PI_4 {
                        return r > TICKS_PER_PI_4;
                    }
                }

                // Coordinates: (canonical key, c, delta).
                let mut coords: Vec<($primitive, u8, u8)> = Vec::new();
                let mut linear: u8 = 0;
                for t in self.phase_poly.terms.iter() {
                    let mono = t.monomial();
                    if (mono & v) == 0 { continue; }
                    let m = mono & !v;
                    if m == 0 {
                        linear = t.phase();
                        continue;
                    }
                    let delta = if Self::is_zero_or_pow2(m) {
                        shared.iter().find(|(b, _)| *b == m).map_or(0, |(_, k)| ((4 * k) % 8) as u8)
                    } else {
                        0
                    };
                    coords.push((m, t.phase(), delta));
                }
                for &(w, k) in &shared {
                    if !self.phase_poly.terms.iter().any(|t| t.monomial() == (v | w)) {
                        coords.push((w, 0, ((4 * k) % 8) as u8));
                    }
                }
                coords.push((0, linear, ((6 * parity_count) % 8) as u8));
                coords.sort_unstable();
                for (_, c, delta) in coords {
                    let flipped = (delta + 8 - c) % 8;
                    if flipped != c {
                        return flipped < c;
                    }
                }
                false
            }

            /// Relabel path variables so that `order[k]` becomes bit `lo + k`.
            fn relabel_gauge(&mut self, order: &[$primitive], lo: usize) -> bool {
                let identity = order.iter().enumerate().all(|(k, &v)| v == (1 as $primitive << (lo + k)));
                if identity { return false; }
                let mut remapping = [0u32; <$primitive>::BITS as usize];
                for i in 0..lo { remapping[i] = i as u32; }
                for (k, &v) in order.iter().enumerate() {
                    remapping[v.trailing_zeros() as usize] = (lo + k) as u32;
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
                    *parity = remap_mono(*parity);
                }
                self.continuous_poly.compact();
                let mut new_phase_terms = Vec::with_capacity(self.phase_poly.terms.len());
                for term in &self.phase_poly.terms {
                    new_phase_terms.push(PackedPhaseTerm::create(remap_mono(term.monomial()), term.phase()));
                }
                self.phase_poly.terms.clear();
                self.phase_poly.merge_unsorted_batch(new_phase_terms);
                true
            }

            pub fn reduce(&mut self) {
                if self.is_overflowed() { return; }
                let mut overall_changed = true;
                while overall_changed {
                    overall_changed = false;
                    // Canonical gauge first: it can free variables that the
                    // elimination rules below then integrate out.
                    self.canonicalize_gauge();
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
                        global_out_mask |= *parity;
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
                                let linear = pv == t && Self::is_zero_or_pow2(pv);
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
                        for &parity in &self.continuous_poly.parities {
                            let mut terms = SmallVec::<[$primitive; $poly_capacity]>::new();
                            let mut bits = parity;
                            while bits != 0 {
                                let bit = 1 as $primitive << bits.trailing_zeros();
                                terms.push(bit);
                                bits &= bits - 1;
                            }
                            scan_structure(&terms, path_var_mask, lo, structure_id, &mut nonlinear, &mut lin_occ);
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
                                *parity &= !v1_mask;
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
                            global_out_mask |= *parity;
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
                                global_out_mask |= *parity;
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
                            *parity = remap_mono(*parity);
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
                    // Pivot substitutions `u := E` rewrite rows that held `u`,
                    // which can leave affine residues: run one more iteration so
                    // the gauge is canonical again.
                    if dead_vars != 0 {
                        overall_changed = true;
                    }
                }
            }
        }
    }
}
