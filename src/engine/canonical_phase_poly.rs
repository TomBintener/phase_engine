// Note: This file is included at the top level of `engine/mod.rs`
// and defines a macro that is called by the main engine-generating macro.

macro_rules! define_canonical_phase_poly_logic {
    (
        $primitive:ty,
        $phase_mask:expr,
        $phase_shift:expr,
        $poly_capacity:expr,
        $canon_capacity:expr
    ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct PackedPhaseTerm(pub $primitive);

        impl PackedPhaseTerm {
            pub const PHASE_MASK: $primitive = $phase_mask;
            pub const MONOMIAL_MASK: $primitive = !Self::PHASE_MASK;

            #[inline(always)]
            pub fn monomial(&self) -> $primitive { self.0 & Self::MONOMIAL_MASK }

            #[inline(always)]
            pub fn phase(&self) -> u8 { (self.0 >> $phase_shift) as u8 }

            #[inline(always)]
            pub fn create(monomial: $primitive, phase: u8) -> Self {
                let safe_monomial = monomial & Self::MONOMIAL_MASK;
                let safe_phase = (phase & 0b111) as $primitive;
                Self(safe_monomial | (safe_phase << $phase_shift))
            }
        }

        impl PartialOrd for PackedPhaseTerm {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
        }

        impl Ord for PackedPhaseTerm {
            #[inline(always)]
            fn cmp(&self, other: &Self) -> Ordering {
                // Rotate the phase (top bits) to the bottom.
                // This shifts the monomial to the top, allowing a
                // single, branchless primitive integer comparison.
                let shift = (<$primitive>::BITS - $phase_shift) as u32;
                let a = self.0.rotate_left(shift);
                let b = other.0.rotate_left(shift);
                a.cmp(&b)
            }
        }

        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct CanonicalPhasePoly {
            pub terms: SmallVec<[PackedPhaseTerm; $canon_capacity]>,
        }

        impl CanonicalPhasePoly {
            pub fn add_assign(&mut self, other: &Self) {
                // 1. Bulk copy directly via slice (guarantees ptr::copy_nonoverlapping)
                self.terms.extend_from_slice(&other.terms);

                // 2. Sort (Apply the rotate_left trick to Ord for PackedPhaseTerm for max speed)
                self.terms.sort_unstable();

                // 3. In-place merge and modulo 8 phase cancellation
                if self.terms.is_empty() { return; }

                let mut write_idx = 0;
                let mut read_idx = 0;
                let len = self.terms.len();

                while read_idx < len {
                    let current_mono = self.terms[read_idx].monomial();
                    let mut current_phase = self.terms[read_idx].phase();
                    read_idx += 1;

                    // Consume all identical monomials
                    while read_idx < len && self.terms[read_idx].monomial() == current_mono {
                        current_phase = (current_phase + self.terms[read_idx].phase()) % 8;
                        read_idx += 1;
                    }

                    // Only keep the term if the phases haven't entirely canceled out
                    if current_phase != 0 {
                        self.terms[write_idx] = PackedPhaseTerm::create(current_mono, current_phase);
                        write_idx += 1;
                    }
                }

                // 4. Truncate leaves excess heap capacity intact for the next loop iteration
                self.terms.truncate(write_idx);
            } // <-- Re-added missing closing brace here

            pub fn merge_unsorted_batch(&mut self, mut batch: Vec<PackedPhaseTerm>) {
                if batch.is_empty() { return; }
                batch.sort_unstable();
                let mut compacted = SmallVec::<[PackedPhaseTerm; $canon_capacity]>::new();
                if !batch.is_empty() {
                    let mut current_mono = batch[0].monomial();
                    let mut current_phase = batch[0].phase();
                    for term in batch.into_iter().skip(1) {
                        if term.monomial() == current_mono {
                            current_phase = (current_phase + term.phase()) % 8;
                        } else {
                            if current_phase != 0 {
                                compacted.push(PackedPhaseTerm::create(current_mono, current_phase));
                            }
                            current_mono = term.monomial();
                            current_phase = term.phase();
                        }
                    }
                    if current_phase != 0 {
                        compacted.push(PackedPhaseTerm::create(current_mono, current_phase));
                    }
                }
                let batch_poly = CanonicalPhasePoly { terms: compacted };
                self.add_assign(&batch_poly);
            }
        }

        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct BooleanPoly {
            pub terms: SmallVec<[$primitive; $poly_capacity]>,
            pub variable_mask: $primitive,
        }

        impl BooleanPoly {
            pub fn from_terms(mut terms: SmallVec<[$primitive; $poly_capacity]>) -> Self {
                terms.sort_unstable();
                terms.dedup();
                let variable_mask = terms.iter().fold(0, |acc, &x| acc | x);
                Self { terms, variable_mask }
            }

            pub fn add_assign(&mut self, other: &Self) {
                // 1. Bulk copy
                self.terms.extend_from_slice(&other.terms);

                // 2. Sort (this will natively use ipnsort since the elements are raw primitives)
                self.terms.sort_unstable();

                // 3. In-place XOR cancellation
                if self.terms.is_empty() { return; }

                let mut write_idx = 0;
                let mut read_idx = 0;
                let len = self.terms.len();

                while read_idx < len {
                    let current_mono = self.terms[read_idx];
                    let mut count = 1;
                    read_idx += 1;

                    // Consume all identical monomials
                    while read_idx < len && self.terms[read_idx] == current_mono {
                        count += 1;
                        read_idx += 1;
                    }

                    // Only keep if the count is odd (XOR logic for boolean polynomials)
                    if count % 2 != 0 {
                        self.terms[write_idx] = current_mono;
                        write_idx += 1;
                    }
                }

                self.terms.truncate(write_idx);

                // 4. Update the variable mask in a single pass at the end
                self.variable_mask = self.terms.iter().fold(0, |acc, &x| acc | x);
            }

            pub(crate) fn from_mask(mask: $primitive) -> Self {
                let mut terms = SmallVec::new();
                if (mask & (1 as $primitive << (<$primitive>::BITS - 1))) != 0 {
                    terms.push(0);
                }
                let mut var_mask = mask & !(1 as $primitive << (<$primitive>::BITS - 1));
                while var_mask != 0 {
                    let bit = var_mask.trailing_zeros();
                    terms.push(1 as $primitive << bit);
                    var_mask &= var_mask - 1;
                }
                BooleanPoly::from_terms(terms)
            }
        }
    }
}