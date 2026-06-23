// Note: This file is included at the top level of `engine/mod.rs`
// and defines a macro that is called by the main engine-generating macro.

macro_rules! define_reduction_logic {
    (
        $primitive:ty,
        $poly_capacity:expr
    ) => {
        impl EvaluatedPathSum {
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
                    let mut changed = true;
                    while changed {
                        changed = false;
                        for v_idx in self.num_qubits..self.num_qubits + original_num_path_vars {
                            let v_mask = 1 as $primitive << v_idx;
                            if (dead_vars & v_mask) != 0 { continue; }
                            if (global_out_mask & v_mask) != 0 { continue; }

                            let mut p_mask = 0 as $primitive;
                            let mut is_linear_and_phase_pi = true;
                            for term in &self.phase_poly.terms {
                                let mono = term.monomial();
                                if (mono & v_mask) != 0 {
                                    if term.phase() != 4 { is_linear_and_phase_pi = false; break; }
                                    let remaining = mono & !v_mask;
                                    if remaining == 0 {
                                        p_mask ^= 1 as $primitive << (<$primitive>::BITS - 1);
                                    } else if remaining.count_ones() == 1 {
                                        p_mask ^= remaining;
                                    } else {
                                        is_linear_and_phase_pi = false;
                                        break;
                                    }
                                }
                            }

                            if !is_linear_and_phase_pi { continue; }
                            let valid_pivots = p_mask & path_var_mask & !dead_vars;
                            if valid_pivots == 0 { continue; }

                            changed = true;
                            let pivot_index = valid_pivots.trailing_zeros();
                            let u_mask = 1 as $primitive << pivot_index;
                            let e_mask = p_mask ^ u_mask;
                            let e_poly = BooleanPoly::from_mask(e_mask);

                            let mut out_state_changed = false;
                            for poly in &mut self.out_state {
                                if (poly.variable_mask & u_mask) == 0 { continue; }
                                out_state_changed = true;
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

                            if out_state_changed {
                                global_out_mask = self.out_state.iter().fold(0, |acc, p| acc | p.variable_mask);
                                for parity in &self.continuous_poly.parities {
                                    global_out_mask |= parity.variable_mask;
                                }
                            }

                            self.continuous_poly.substitute(u_mask, &e_poly);
                            continuous_needs_compact = true;

                            let mut next_gen_terms = Vec::new();
                            for term in self.phase_poly.terms.iter() {
                                let mono = term.monomial();
                                if (mono & v_mask) != 0 { continue; }
                                if (mono & u_mask) != 0 {
                                    let base = mono & !u_mask;
                                    for &e_term in &e_poly.terms {
                                        let new_mono = if e_term == 0 { base } else { base | e_term };
                                        next_gen_terms.push(PackedPhaseTerm::create(new_mono, term.phase()));
                                    }
                                } else {
                                    next_gen_terms.push(*term);
                                }
                            }
                            self.phase_poly.terms.clear();
                            self.phase_poly.merge_unsorted_batch(next_gen_terms);
                            dead_vars |= v_mask | u_mask;
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
