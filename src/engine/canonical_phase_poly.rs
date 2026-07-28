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
                if other.terms.is_empty() { return; }
                let old_len = self.terms.len();

                // 1. Bulk copy directly via slice (guarantees ptr::copy_nonoverlapping).
                // If the two sorted runs are already in order this is final;
                // otherwise the buffer is rewritten by the back-to-front merge.
                self.terms.extend_from_slice(&other.terms);

                // 2. Linear back-to-front merge of the two sorted runs, O(T+B)
                // instead of the previous O((T+B) log(T+B)) re-sort. Run B is
                // read from `other` (separate memory), so writes can never
                // clobber unread elements: while B is non-empty the write
                // index stays strictly above the unread tail of run A.
                if old_len > 0 && self.terms[old_len - 1] > other.terms[0] {
                    let mut w = self.terms.len();
                    let mut a = old_len;
                    let mut b = other.terms.len();
                    while a > 0 && b > 0 {
                        w -= 1;
                        if self.terms[a - 1] > other.terms[b - 1] {
                            self.terms[w] = self.terms[a - 1];
                            a -= 1;
                        } else {
                            self.terms[w] = other.terms[b - 1];
                            b -= 1;
                        }
                    }
                    while b > 0 {
                        w -= 1;
                        b -= 1;
                        self.terms[w] = other.terms[b];
                    }
                    // Remaining elements of run A are already in place.
                }

                // 3. In-place compaction and modulo 8 phase cancellation across
                // every equal-monomial run. This must count across whole runs:
                // `other` is sorted but may contain equal adjacent monomials
                // (e.g. `ContinuousPhasePoly::substitute` produces them), so
                // duplicates are not necessarily cross-run only.
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
                    // AND drop global phases (current_mono == 0) so the engine natively recognizes quantum equivalences.
                    if current_phase != 0 && current_mono != 0 {
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
                            if current_phase != 0 && current_mono != 0 {
                                compacted.push(PackedPhaseTerm::create(current_mono, current_phase));
                            }
                            current_mono = term.monomial();
                            current_phase = term.phase();
                        }
                    }
                    if current_phase != 0 && current_mono != 0 {
                        compacted.push(PackedPhaseTerm::create(current_mono, current_phase));
                    }
                }
                // Single sort: the batch is sorted+compacted once above, and
                // `add_assign` now merges linearly instead of re-sorting.
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
                // GF(2) parity compaction: duplicate monomials cancel in pairs
                // (XOR semantics), so even-length runs vanish entirely and
                // odd-length runs keep exactly one copy. A plain dedup() would
                // keep one copy of *every* run, which is unsound.
                let len = terms.len();
                let mut write_idx = 0;
                let mut read_idx = 0;
                while read_idx < len {
                    let current = terms[read_idx];
                    let mut count = 1usize;
                    read_idx += 1;
                    while read_idx < len && terms[read_idx] == current {
                        count += 1;
                        read_idx += 1;
                    }
                    if count % 2 != 0 {
                        terms[write_idx] = current;
                        write_idx += 1;
                    }
                }
                terms.truncate(write_idx);
                let variable_mask = terms.iter().fold(0, |acc, &x| acc | x);
                Self { terms, variable_mask }
            }

            pub fn add_assign(&mut self, other: &Self) {
                if other.terms.is_empty() { return; }
                let old_len = self.terms.len();

                // 1. Bulk copy; final if the runs are already in order,
                // otherwise rewritten by the back-to-front merge below.
                self.terms.extend_from_slice(&other.terms);

                // 2. Linear back-to-front merge of the two sorted runs
                // (run B is read from `other`'s separate memory, so writes
                // never clobber unread elements).
                if old_len > 0 && self.terms[old_len - 1] > other.terms[0] {
                    let mut w = self.terms.len();
                    let mut a = old_len;
                    let mut b = other.terms.len();
                    while a > 0 && b > 0 {
                        w -= 1;
                        if self.terms[a - 1] > other.terms[b - 1] {
                            self.terms[w] = self.terms[a - 1];
                            a -= 1;
                        } else {
                            self.terms[w] = other.terms[b - 1];
                            b -= 1;
                        }
                    }
                    while b > 0 {
                        w -= 1;
                        b -= 1;
                        self.terms[w] = other.terms[b];
                    }
                }

                // 3. In-place XOR cancellation across every equal-monomial run.
                // Parity must be counted across whole runs: `other` can carry
                // within-run duplicates (ContinuousPhasePoly::substitute ORs a
                // single-bit e-term into monomials, which can collide them).
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