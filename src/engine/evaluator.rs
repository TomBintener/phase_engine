// Note: This file is included at the top level of `engine/mod.rs`
// and defines a macro that is called by the main engine-generating macro.

macro_rules! define_evaluator_logic {
    (
        $primitive:ty
    ) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct EvaluatedPathSum {
            pub num_qubits: u32,
            pub num_path_vars: u32,
            pub out_state: Vec<BooleanPoly>,
            pub phase_poly: CanonicalPhasePoly,
            pub continuous_poly: ContinuousPhasePoly,
        }

        impl EvaluatedPathSum {
            /// Identity *operator* \(I_n\): `out_state[i] = x_i`, empty phases.
            /// Ket constructors (`zero_state` / `basis_state`) are intentionally
            /// absent — AGES equality is operator equality from this baseline.
            pub fn new_id(num_qubits: u32) -> Self {
                let mut out_state = Vec::with_capacity(num_qubits as usize);
                for i in 0..num_qubits {
                    out_state.push(BooleanPoly::from_terms(smallvec::smallvec![1 as $primitive << i]));
                }
                Self {
                    num_qubits,
                    num_path_vars: 0,
                    out_state,
                    phase_poly: CanonicalPhasePoly { terms: smallvec::smallvec![] },
                    continuous_poly: ContinuousPhasePoly::new(),
                }
            }

            pub fn apply_x(&mut self, q: usize) {
                let constant_one = BooleanPoly::from_terms(smallvec::smallvec![0]);
                self.out_state[q].add_assign(&constant_one);
            }

            pub fn apply_cx(&mut self, qc: usize, qt: usize) {
                assert!(qc != qt, "CX control and target must be distinct");
                let (ctrl_poly, tgt_poly) = if qc < qt {
                    let (left, right) = self.out_state.split_at_mut(qt);
                    (&left[qc], &mut right[0])
                } else {
                    let (left, right) = self.out_state.split_at_mut(qc);
                    (&right[0], &mut left[qt])
                };
                tgt_poly.add_assign(ctrl_poly);
            }

            /// Appends the exact phase-polynomial expansion of
            /// `phase_units * (t_1 XOR ... XOR t_n) * base` onto `batch`, where each
            /// `t_i` and `base` are monomials (variable bitmasks, `0` = constant 1).
            ///
            /// Uses the multilinear identity over {0,1}:
            ///   t_1 XOR ... XOR t_n = sum over non-empty subsets S of (-2)^(|S|-1) * prod_S t_i
            /// Subsets with |S| >= 4 carry a factor (-2)^3 = -8 == 0 (mod 8), so truncating
            /// at triples is exact for EVERY phase coefficient. Linear distribution (singles
            /// only) is exact only for phase 4 (pi), which the coefficient guards recover.
            fn push_xor_phase_expansion(
                batch: &mut Vec<PackedPhaseTerm>,
                base: $primitive,
                terms: &[$primitive],
                phase_units: u8,
            ) {
                let phase = (phase_units & 0b111) as i16;
                if phase == 0 { return; }
                let pair_phase = (-2 * phase).rem_euclid(8) as u8;
                let triple_phase = (4 * phase).rem_euclid(8) as u8;
                let n = terms.len();
                for i in 0..n {
                    batch.push(PackedPhaseTerm::create(base | terms[i], phase as u8));
                    if pair_phase == 0 && triple_phase == 0 { continue; }
                    for j in (i + 1)..n {
                        if pair_phase != 0 {
                            batch.push(PackedPhaseTerm::create(base | terms[i] | terms[j], pair_phase));
                        }
                        if triple_phase != 0 {
                            for k in (j + 1)..n {
                                batch.push(PackedPhaseTerm::create(base | terms[i] | terms[j] | terms[k], triple_phase));
                            }
                        }
                    }
                }
            }

            #[inline(always)]
            fn push_phase_expansion(&mut self, terms: &[$primitive], phase_units: u8) {
                let n = terms.len();
                if n == 0 || (phase_units & 0b111) == 0 { return; }
                let capacity = if phase_units % 2 != 0 { // T, Tdg
                    let pairs = (n * n.saturating_sub(1)) / 2;
                    let triplets = (n * n.saturating_sub(1) * n.saturating_sub(2)) / 6;
                    n + pairs + triplets
                } else if phase_units == 2 || phase_units == 6 { // S, Sdg
                    n + (n * n.saturating_sub(1)) / 2
                } else { // Z
                    n
                };
                let mut batch = Vec::with_capacity(capacity);
                Self::push_xor_phase_expansion(&mut batch, 0, terms, phase_units);
                self.phase_poly.merge_unsorted_batch(batch);
            }

            pub fn apply_z(&mut self, q: usize) {
                let terms = self.out_state[q].terms.clone();
                self.push_phase_expansion(&terms, 4);
            }

            pub fn apply_s(&mut self, q: usize) {
                let terms = self.out_state[q].terms.clone();
                self.push_phase_expansion(&terms, 2);
            }

            pub fn apply_sdg(&mut self, q: usize) {
                let terms = self.out_state[q].terms.clone();
                self.push_phase_expansion(&terms, 6);
            }

            pub fn apply_t(&mut self, q: usize) {
                let terms = self.out_state[q].terms.clone();
                self.push_phase_expansion(&terms, 1);
            }

            pub fn apply_tdg(&mut self, q: usize) {
                let terms = self.out_state[q].terms.clone();
                self.push_phase_expansion(&terms, 7);
            }

            pub fn apply_sx(&mut self, q: usize) {
                let var_index = self.num_qubits + self.num_path_vars;
                assert!(var_index < (<$primitive>::BITS - 3), "Exceeded variable limit");
                let v_mask = 1 as $primitive << var_index;
                self.num_path_vars += 1;
                
                // XOR the path variable into the target qubit
                let v_poly = BooleanPoly::from_terms(smallvec::smallvec![v_mask]);
                self.out_state[q].add_assign(&v_poly);
                
                // Add S^dag(v) to the phase polynomial (6 units of pi/4)
                let mut batch = Vec::with_capacity(1);
                batch.push(PackedPhaseTerm::create(v_mask, 6)); 
                self.phase_poly.merge_unsorted_batch(batch);
            }

            pub fn apply_h(&mut self, q: usize) {
                let var_index = self.num_qubits + self.num_path_vars;
                assert!(var_index < (<$primitive>::BITS - 3), "Exceeded variable limit");
                let v_mask = 1 as $primitive << var_index;
                self.num_path_vars += 1;
                let old_state = std::mem::replace(
                    &mut self.out_state[q],
                    BooleanPoly::from_terms(smallvec::smallvec![v_mask]),
                );
                let mut batch = Vec::with_capacity(old_state.terms.len());
                for t in old_state.terms {
                    batch.push(PackedPhaseTerm::create(t | v_mask, 4));
                }
                self.phase_poly.merge_unsorted_batch(batch);
            }

            pub fn promote_cliffords(&mut self) -> bool {
                let cliffords = self.continuous_poly.extract_cliffords();
                if cliffords.is_empty() { return false; }
                
                for (monomials, phase_units) in cliffords {
                    self.push_phase_expansion(&monomials, phase_units);
                }
                true
            }

            pub fn apply_rz(&mut self, q: usize, theta: f64) {
                let current_parity = self.out_state[q].clone();
                self.continuous_poly.apply_phase(current_parity, theta);
                self.promote_cliffords();
            }
        }
    }
}
