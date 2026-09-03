// Note: This file is included at the top level of `engine/mod.rs`
// and defines a macro that is called by the main engine-generating macro.

macro_rules! define_continuous_poly_logic {
    (
        $primitive:ty,
        $poly_capacity:expr
    ) => {
        /// Non-Clifford phase content in the parity basis.
        ///
        /// `phases[i]` is the angle on `parities[i]` in lattice ticks
        /// (`TICKS_PER_TURN` per 2π, see `engine/mod.rs`), always in
        /// `1..TICKS_PER_TURN`. All arithmetic is exact modular integer
        /// arithmetic; the only rounding happens once when an `f64` angle
        /// enters through `apply_phase`. Parities are sorted by terms and never
        /// contain the constant monomial (folded into a sign flip, see
        /// `canonicalize_continuous_parity`).
        #[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
        pub struct ContinuousPhasePoly {
            pub parities: Vec<BooleanPoly>,
            pub phases: Vec<u32>,
        }

        /// Fold a constant-1 ANF term (monomial 0) into a global phase and drop it.
        /// `e^{iθ(1⊕p)} = e^{iθ} e^{-iθ p}`; interned Eq already ignores global phase.
        fn canonicalize_continuous_parity(
            parity: BooleanPoly,
            ticks: u32,
        ) -> Option<(BooleanPoly, u32)> {
            if parity.terms.is_empty() {
                return None;
            }
            let has_const = parity.terms.first().copied() == Some(0);
            let signed = if has_const { negate_ticks(ticks) } else { add_ticks(ticks, 0) };
            if signed == 0 {
                return None;
            }
            if !has_const {
                return Some((parity, signed));
            }
            let rest: SmallVec<[$primitive; $poly_capacity]> =
                parity.terms.iter().copied().filter(|&t| t != 0).collect();
            if rest.is_empty() {
                return None;
            }
            Some((BooleanPoly::from_terms(rest), signed))
        }

        impl ContinuousPhasePoly {
            pub fn new() -> Self { Self::default() }

            /// Add `theta` radians on `parity`; the angle is rounded onto the
            /// tick lattice exactly once, here.
            pub fn apply_phase(&mut self, parity: BooleanPoly, theta: f64) {
                self.apply_ticks(parity, angle_to_ticks(theta));
            }

            pub fn apply_ticks(&mut self, parity: BooleanPoly, ticks: u32) {
                let Some((parity, ticks)) = canonicalize_continuous_parity(parity, ticks) else {
                    return;
                };

                match self.parities.binary_search_by(|p| p.terms.cmp(&parity.terms)) {
                    Ok(idx) => {
                        let sum = add_ticks(self.phases[idx], ticks);
                        if sum == 0 {
                            self.parities.remove(idx);
                            self.phases.remove(idx);
                        } else {
                            self.phases[idx] = sum;
                        }
                    }
                    Err(idx) => {
                        self.parities.insert(idx, parity);
                        self.phases.insert(idx, ticks);
                    }
                }
            }

            pub fn compact(&mut self) {
                if self.parities.is_empty() { return; }
                let combined: Vec<_> = self.parities.drain(..).zip(self.phases.drain(..)).collect();
                let mut folded = Vec::with_capacity(combined.len());
                for (parity, ticks) in combined {
                    if let Some(pair) = canonicalize_continuous_parity(parity, ticks) {
                        folded.push(pair);
                    }
                }
                folded.sort_unstable_by(|a, b| a.0.terms.cmp(&b.0.terms));

                let mut i = 0;
                while i < folded.len() {
                    let mut j = i + 1;
                    let mut accumulated = folded[i].1;
                    while j < folded.len() && folded[j].0.terms == folded[i].0.terms {
                        accumulated = add_ticks(accumulated, folded[j].1);
                        j += 1;
                    }
                    if accumulated != 0 {
                        self.parities.push(folded[i].0.clone());
                        self.phases.push(accumulated);
                    }
                    i = j;
                }
            }

            pub fn substitute(&mut self, u_mask: $primitive, e_poly: &BooleanPoly) {
                for parity in self.parities.iter_mut() {
                    if (parity.variable_mask & u_mask) == 0 { continue; }
                    let mut b_poly = BooleanPoly::from_terms(Default::default());
                    parity.terms.retain(|term| {
                        if (*term & u_mask) != 0 {
                            b_poly.terms.push(*term & !u_mask);
                            false
                        } else {
                            true
                        }
                    });
                    b_poly.terms.sort_unstable();
                    b_poly.variable_mask = b_poly.terms.iter().fold(0, |acc, &x| acc | x);
                    if !b_poly.terms.is_empty() {
                        let mut eb_poly = BooleanPoly::from_terms(Default::default());
                        for e_term in &e_poly.terms {
                            let mut shifted_b = b_poly.clone();
                            if *e_term != 0 {
                                for b in &mut shifted_b.terms { *b |= *e_term; }
                                shifted_b.terms.sort_unstable();
                            }
                            eb_poly.add_assign(&shifted_b);
                        }
                        parity.add_assign(&eb_poly);
                    }
                }
            }

            /// Split every parity's angle `θ` into an even lattice quotient
            /// `k·π/4` with `k ∈ {0, 2, 4, 6}` (returned for promotion into the
            /// discrete polynomial) and the remainder `θ mod π/2 ∈ (0, π/2)`
            /// (kept here).
            ///
            /// The split is a function of the parity's *total* angle only, so
            /// `RZ(θ)·T`, `T·RZ(θ)` and `RZ(θ + π/4)` on one parity intern
            /// equal; the previous rule (promote only exact lattice totals)
            /// made the discrete/continuous split depend on gate order. Odd
            /// quotients stay continuous because their cubic monomial expansion
            /// is the dominant cost of T-heavy states, while `reduce()` only
            /// needs π (pivot) and ±π/2 (self-loop) terms in the discrete
            /// polynomial. Known residual: odd-lattice layers that are
            /// identities (T on all 15 parities of 4 qubits) no longer cancel.
            pub fn extract_cliffords(&mut self) -> Vec<(Vec<$primitive>, u8)> {
                let mut extracted = Vec::new();
                let mut write_idx = 0;

                for read_idx in 0..self.phases.len() {
                    let ticks = self.phases[read_idx] as u64;
                    let units = (2 * (ticks / TICKS_PER_PI_2)) as u8;
                    let remainder = (ticks % TICKS_PER_PI_2) as u32;
                    if units != 0 {
                        extracted.push((self.parities[read_idx].terms.to_vec(), units));
                    }
                    if remainder != 0 {
                        self.phases[write_idx] = remainder;
                        if read_idx != write_idx {
                            self.parities[write_idx] = self.parities[read_idx].clone();
                        }
                        write_idx += 1;
                    }
                }

                self.phases.truncate(write_idx);
                self.parities.truncate(write_idx);

                extracted
            }
        }
    }
}
