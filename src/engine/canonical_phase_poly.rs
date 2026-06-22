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
                self.monomial().cmp(&other.monomial())
                    .then_with(|| self.phase().cmp(&other.phase()))
            }
        }

        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct CanonicalPhasePoly {
            pub terms: SmallVec<[PackedPhaseTerm; $canon_capacity]>,
        }

        impl CanonicalPhasePoly {
            pub fn add_assign(&mut self, other: &Self) {
                let mut result = SmallVec::with_capacity(self.terms.len() + other.terms.len());
                let mut i = 0;
                let mut j = 0;

                while i < self.terms.len() && j < other.terms.len() {
                    let a = self.terms[i];
                    let b = other.terms[j];

                    match a.monomial().cmp(&b.monomial()) {
                        Ordering::Less => { result.push(a); i += 1; }
                        Ordering::Greater => { result.push(b); j += 1; }
                        Ordering::Equal => {
                            let new_phase = (a.phase() + b.phase()) % 8;
                            if new_phase != 0 {
                                result.push(PackedPhaseTerm::create(a.monomial(), new_phase));
                            }
                            i += 1;
                            j += 1;
                        }
                    }
                }
                result.extend_from_slice(&self.terms[i..]);
                result.extend_from_slice(&other.terms[j..]);
                self.terms = result;
            }

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
                let mut result = SmallVec::with_capacity(self.terms.len() + other.terms.len());
                let mut i = 0;
                let mut j = 0;
                while i < self.terms.len() && j < other.terms.len() {
                    match self.terms[i].cmp(&other.terms[j]) {
                        Ordering::Less => { result.push(self.terms[i]); i += 1; }
                        Ordering::Greater => { result.push(other.terms[j]); j += 1; }
                        Ordering::Equal => { i += 1; j += 1; }
                    }
                }
                result.extend_from_slice(&self.terms[i..]);
                result.extend_from_slice(&other.terms[j..]);
                self.terms = result;
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
