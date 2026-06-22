// Note: This file is included at the top level of `engine/mod.rs`
// and defines a macro that is called by the main engine-generating macro.

macro_rules! define_continuous_poly_logic {
    (
        $primitive:ty,
        $poly_capacity:expr
    ) => {
        #[derive(Debug, Clone, Default)]
        pub struct ContinuousPhasePoly {
            pub parities: Vec<BooleanPoly>,
            pub phases: Vec<f64>,
        }

        fn quantize_phase(phase: f64) -> i64 {
            let norm_phase = phase.rem_euclid(TAU);
            if (TAU - norm_phase) <= EPSILON { return 0; }
            (norm_phase / EPSILON).round() as i64
        }

        impl ContinuousPhasePoly {
            pub fn new() -> Self { Self::default() }

            pub fn apply_phase(&mut self, parity: BooleanPoly, theta: f64) {
                let normalized_theta = theta.rem_euclid(TAU);
                if normalized_theta <= EPSILON || (TAU - normalized_theta) <= EPSILON { return; }
                match self.parities.binary_search_by(|p| p.terms.cmp(&parity.terms)) {
                    Ok(idx) => {
                        let new_phase = (self.phases[idx] + normalized_theta).rem_euclid(TAU);
                        if new_phase <= EPSILON || (TAU - new_phase) <= EPSILON {
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
                let mut combined: Vec<_> = self.parities.drain(..).zip(self.phases.drain(..)).collect();
                combined.sort_unstable_by(|a, b| a.0.terms.cmp(&b.0.terms));
                let mut i = 0;
                while i < combined.len() {
                    let mut j = i + 1;
                    let mut accumulated_phase = combined[i].1;
                    while j < combined.len() && combined[j].0.terms == combined[i].0.terms {
                        accumulated_phase += combined[j].1;
                        j += 1;
                    }
                    let norm = accumulated_phase.rem_euclid(TAU);
                    if norm > EPSILON && (TAU - norm) > EPSILON {
                        self.parities.push(combined[i].0.clone());
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
        }

        impl PartialEq for ContinuousPhasePoly {
            fn eq(&self, other: &Self) -> bool {
                if self.parities != other.parities || self.phases.len() != other.phases.len() { return false; }
                self.phases.iter().zip(other.phases.iter()).all(|(&a, &b)| quantize_phase(a) == quantize_phase(b))
            }
        }
        impl Eq for ContinuousPhasePoly {}
        impl Hash for ContinuousPhasePoly {
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.parities.hash(state);
                for &phase in &self.phases { quantize_phase(phase).hash(state); }
            }
        }
    }
}
