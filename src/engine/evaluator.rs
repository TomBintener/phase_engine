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

            pub fn apply_z(&mut self, q: usize) {
                let terms = &self.out_state[q].terms;
                let mut batch = Vec::with_capacity(terms.len());
                for &t in terms { batch.push(PackedPhaseTerm::create(t, 4)); }
                self.phase_poly.merge_unsorted_batch(batch);
            }

            pub fn apply_s(&mut self, q: usize) {
                let terms = &self.out_state[q].terms;
                let n = terms.len();
                let capacity = n + (n * n.saturating_sub(1)) / 2;
                let mut batch = Vec::with_capacity(capacity);
                for i in 0..n {
                    batch.push(PackedPhaseTerm::create(terms[i], 2));
                    for j in (i + 1)..n {
                        batch.push(PackedPhaseTerm::create(terms[i] | terms[j], 4));
                    }
                }
                self.phase_poly.merge_unsorted_batch(batch);
            }

            pub fn apply_sdg(&mut self, q: usize) {
                let terms = &self.out_state[q].terms;
                let n = terms.len();
                let capacity = n + (n * n.saturating_sub(1)) / 2;
                let mut batch = Vec::with_capacity(capacity);
                for i in 0..n {
                    batch.push(PackedPhaseTerm::create(terms[i], 6));
                    for j in (i + 1)..n {
                        batch.push(PackedPhaseTerm::create(terms[i] | terms[j], 4));
                    }
                }
                self.phase_poly.merge_unsorted_batch(batch);
            }

            pub fn apply_t(&mut self, q: usize) {
                let terms = &self.out_state[q].terms;
                let n = terms.len();
                let pairs = (n * n.saturating_sub(1)) / 2;
                let triplets = (n * n.saturating_sub(1) * n.saturating_sub(2)) / 6;
                let capacity = n + pairs + triplets;
                let mut batch = Vec::with_capacity(capacity);
                for i in 0..n {
                    batch.push(PackedPhaseTerm::create(terms[i], 1));
                    for j in (i + 1)..n {
                        batch.push(PackedPhaseTerm::create(terms[i] | terms[j], 6));
                        for k in (j + 1)..n {
                            batch.push(PackedPhaseTerm::create(terms[i] | terms[j] | terms[k], 4));
                        }
                    }
                }
                self.phase_poly.merge_unsorted_batch(batch);
            }

            pub fn apply_tdg(&mut self, q: usize) {
                let terms = &self.out_state[q].terms;
                let n = terms.len();
                let pairs = (n * n.saturating_sub(1)) / 2;
                let triplets = (n * n.saturating_sub(1) * n.saturating_sub(2)) / 6;
                let capacity = n + pairs + triplets;
                let mut batch = Vec::with_capacity(capacity);
                for i in 0..n {
                    batch.push(PackedPhaseTerm::create(terms[i], 7));
                    for j in (i + 1)..n {
                        batch.push(PackedPhaseTerm::create(terms[i] | terms[j], 2));
                        for k in (j + 1)..n {
                            batch.push(PackedPhaseTerm::create(terms[i] | terms[j] | terms[k], 4));
                        }
                    }
                }
                self.phase_poly.merge_unsorted_batch(batch);
            }

            pub fn apply_sx(&mut self, q: usize) {
                self.apply_h(q);
                self.apply_s(q);
                self.apply_h(q);
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

            pub fn apply_rz(&mut self, q: usize, theta: f64) {
                let current_parity = self.out_state[q].clone();
                self.continuous_poly.apply_phase(current_parity, theta);
            }
        }
    }
}
