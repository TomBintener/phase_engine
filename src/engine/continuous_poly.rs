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
        /// enters through `apply_phase`.
        ///
        /// Each parity is a linear XOR of variables, stored as one bitmask
        /// (bit `k` set iff variable `k` appears). The constant monomial is
        /// never stored: `e^{iθ(1⊕p)} = e^{iθ} e^{-iθ p}` folds into a sign
        /// flip, and interned Eq already ignores global phase. Production
        /// gates keep parities linear; product terms are dropped with a
        /// debug assertion (gauge already bails on those hand-built states).
        #[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
        pub struct ContinuousPhasePoly {
            pub parities: Vec<$primitive>,
            pub phases: Vec<u32>,
        }

        /// Convert an affine/linear `BooleanPoly` into `(mask, flip)`.
        /// `flip` is true when the constant-1 term was present (phase negates).
        /// Product monomials are not representable as a mask; they are ignored
        /// after a debug assertion.
        fn linear_mask_and_flip(parity: &BooleanPoly) -> ( $primitive, bool ) {
            let mut mask = 0 as $primitive;
            let mut flip = false;
            for &t in &parity.terms {
                if t == 0 {
                    flip = !flip;
                    continue;
                }
                debug_assert_eq!(
                    t.count_ones(),
                    1,
                    "continuous parity must be a linear XOR of variables"
                );
                if t.count_ones() == 1 {
                    mask ^= t;
                }
            }
            (mask, flip)
        }

        /// Fold a constant-1 ANF term into a global phase and drop it.
        fn canonicalize_continuous_mask(
            mask: $primitive,
            ticks: u32,
            flip: bool,
        ) -> Option<($primitive, u32)> {
            let signed = if flip { negate_ticks(ticks) } else { add_ticks(ticks, 0) };
            if signed == 0 || mask == 0 {
                return None;
            }
            Some((mask, signed))
        }

        impl ContinuousPhasePoly {
            pub fn new() -> Self { Self::default() }

            /// Add `theta` radians on `parity`; the angle is rounded onto the
            /// tick lattice exactly once, here.
            pub fn apply_phase(&mut self, parity: BooleanPoly, theta: f64) {
                self.apply_ticks(parity, angle_to_ticks(theta));
            }

            pub fn apply_ticks(&mut self, parity: BooleanPoly, ticks: u32) {
                let (mask, flip) = linear_mask_and_flip(&parity);
                let Some((mask, ticks)) = canonicalize_continuous_mask(mask, ticks, flip) else {
                    return;
                };

                match self.parities.binary_search(&mask) {
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
                        self.parities.insert(idx, mask);
                        self.phases.insert(idx, ticks);
                    }
                }
            }

            pub fn compact(&mut self) {
                if self.parities.is_empty() { return; }
                let combined: Vec<_> = self.parities.drain(..).zip(self.phases.drain(..)).collect();
                let mut folded = Vec::with_capacity(combined.len());
                for (mask, ticks) in combined {
                    if let Some(pair) = canonicalize_continuous_mask(mask, ticks, false) {
                        folded.push(pair);
                    }
                }
                folded.sort_unstable_by_key(|(mask, _)| *mask);

                let mut i = 0;
                while i < folded.len() {
                    let mut j = i + 1;
                    let mut accumulated = folded[i].1;
                    while j < folded.len() && folded[j].0 == folded[i].0 {
                        accumulated = add_ticks(accumulated, folded[j].1);
                        j += 1;
                    }
                    if accumulated != 0 {
                        self.parities.push(folded[i].0);
                        self.phases.push(accumulated);
                    }
                    i = j;
                }
            }

            pub fn substitute(&mut self, u_mask: $primitive, e_poly: &BooleanPoly) {
                let (e_mask, e_flip) = linear_mask_and_flip(e_poly);
                for (parity, ticks) in self.parities.iter_mut().zip(self.phases.iter_mut()) {
                    if (*parity & u_mask) == 0 { continue; }
                    *parity ^= u_mask;
                    *parity ^= e_mask;
                    if e_flip {
                        *ticks = negate_ticks(*ticks);
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
                        let mut terms = Vec::new();
                        let mut mask = self.parities[read_idx];
                        while mask != 0 {
                            let bit = mask.trailing_zeros();
                            terms.push(1 as $primitive << bit);
                            mask &= mask - 1;
                        }
                        extracted.push((terms, units));
                    }
                    if remainder != 0 {
                        self.phases[write_idx] = remainder;
                        self.parities[write_idx] = self.parities[read_idx];
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
