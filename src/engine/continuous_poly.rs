// Note: This file is included at the top level of `engine/mod.rs`
// and defines a macro that is called by the main engine-generating macro.

macro_rules! define_continuous_poly_logic {
    (
        $primitive:ty,
        $poly_capacity:expr
    ) => {
        const SNAP_PRECISION: f64 = 100_000_000.0;

        #[inline(always)]
        fn snap_phase(val: f64) -> f64 {
            (val * SNAP_PRECISION).round() / SNAP_PRECISION
        }

        #[derive(Debug, Clone, Default)]
        pub struct ContinuousPhasePoly {
            pub parities: Vec<BooleanPoly>,
            pub phases: Vec<f64>,
        }

        /// Fold a constant-1 ANF term (monomial 0) into a global phase and drop it.
        /// `e^{iθ(1⊕p)} = e^{iθ} e^{-iθ p}`; interned Eq already ignores global phase.
        fn canonicalize_continuous_parity(
            parity: BooleanPoly,
            theta: f64,
        ) -> Option<(BooleanPoly, f64)> {
            let mut normalized = snap_phase(theta.rem_euclid(TAU));
            if normalized == 0.0 || normalized == snap_phase(TAU) {
                return None;
            }
            if parity.terms.is_empty() {
                return None;
            }
            if parity.terms.first().copied() != Some(0) {
                return Some((parity, normalized));
            }
            normalized = snap_phase((-normalized).rem_euclid(TAU));
            if normalized == 0.0 || normalized == snap_phase(TAU) {
                return None;
            }
            let rest: SmallVec<[$primitive; $poly_capacity]> =
                parity.terms.iter().copied().filter(|&t| t != 0).collect();
            if rest.is_empty() {
                return None;
            }
            Some((BooleanPoly::from_terms(rest), normalized))
        }

        impl ContinuousPhasePoly {
            pub fn new() -> Self { Self::default() }

            pub fn apply_phase(&mut self, parity: BooleanPoly, theta: f64) {
                let Some((parity, normalized_theta)) =
                    canonicalize_continuous_parity(parity, theta)
                else {
                    return;
                };

                match self.parities.binary_search_by(|p| p.terms.cmp(&parity.terms)) {
                    Ok(idx) => {
                        let new_phase = snap_phase((self.phases[idx] + normalized_theta).rem_euclid(TAU));
                        if new_phase == 0.0 || new_phase == snap_phase(TAU) {
                            self.parities.remove(idx);
                            self.phases.remove(idx);
                        } else {
                            self.phases[idx] = new_phase;
                        }
                    }
                    Err(idx) => {
                        self.parities.insert(idx, parity);
                        self.phases.insert(idx, normalized_theta);
                    }
                }
            }

            pub fn compact(&mut self) {
                if self.parities.is_empty() { return; }
                let combined: Vec<_> = self.parities.drain(..).zip(self.phases.drain(..)).collect();
                let mut folded = Vec::with_capacity(combined.len());
                for (parity, theta) in combined {
                    if let Some(pair) = canonicalize_continuous_parity(parity, theta) {
                        folded.push(pair);
                    }
                }
                folded.sort_unstable_by(|a, b| a.0.terms.cmp(&b.0.terms));

                let mut i = 0;
                while i < folded.len() {
                    let mut j = i + 1;
                    let mut accumulated_phase = folded[i].1;
                    while j < folded.len() && folded[j].0.terms == folded[i].0.terms {
                        accumulated_phase += folded[j].1;
                        j += 1;
                    }
                    let norm = snap_phase(accumulated_phase.rem_euclid(TAU));
                    if norm != 0.0 && norm != snap_phase(TAU) {
                        self.parities.push(folded[i].0.clone());
                        self.phases.push(norm);
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

            pub fn extract_cliffords(&mut self) -> Vec<(Vec<$primitive>, u8)> {
                let mut extracted = Vec::new();
                let mut write_idx = 0;
                
                for read_idx in 0..self.phases.len() {
                    let phase = self.phases[read_idx];
                    let mut units = (phase / std::f64::consts::FRAC_PI_4).round() as i64;
                    if units == 8 { units = 0; }
                    
                    let mut remainder = phase;
                    if units != 0 {
                        let potential_remainder = snap_phase(phase - (units as f64 * std::f64::consts::FRAC_PI_4));
                        if potential_remainder.rem_euclid(TAU) == 0.0 || potential_remainder.rem_euclid(TAU) == snap_phase(TAU) {
                            remainder = 0.0;
                            extracted.push((self.parities[read_idx].terms.to_vec(), units.rem_euclid(8) as u8));
                        } else {
                            units = 0;
                        }
                    }
                    
                    let norm = snap_phase(remainder.rem_euclid(TAU));
                    if norm != 0.0 && norm != snap_phase(TAU) {
                        self.phases[write_idx] = norm;
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

        impl PartialEq for ContinuousPhasePoly {
            fn eq(&self, other: &Self) -> bool {
                if self.parities != other.parities || self.phases.len() != other.phases.len() { return false; }
                self.phases.iter().zip(other.phases.iter()).all(|(&a, &b)| {
                    (a * SNAP_PRECISION).round() as i64 == (b * SNAP_PRECISION).round() as i64
                })
            }
        }
        impl Eq for ContinuousPhasePoly {}
        
        impl Hash for ContinuousPhasePoly {
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.parities.hash(state);
                for &phase in &self.phases { 
                    ((phase * SNAP_PRECISION).round() as i64).hash(state); 
                }
            }
        }
    }
}
