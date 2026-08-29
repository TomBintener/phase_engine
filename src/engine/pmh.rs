macro_rules! define_pmh_logic {
    ($primitive:ty) => {
        /// Default Patel–Markov–Hayes section width:
        /// `min(4, max(2, floor(log2 n)))`, `2` for `n<4`, capped at `8` and `n`.
        pub fn default_pmh_section_size(n: usize) -> usize {
            if n == 0 {
                return 1;
            }
            if n < 4 {
                return n.min(2).max(1);
            }
            let log = (usize::BITS - 1 - n.leading_zeros()) as usize;
            4.min(2.max(log)).min(n).min(8)
        }

        fn pmh_row_mask(n: usize) -> $primitive {
            if n >= <$primitive>::BITS as usize {
                !0
            } else {
                ((1 as $primitive) << n) - 1
            }
        }

        fn pmh_subrow_pattern(row: $primitive, start: usize, width: usize) -> $primitive {
            let mask = if width >= <$primitive>::BITS as usize {
                !0
            } else {
                ((1 as $primitive) << width) - 1
            };
            (row >> start) & mask
        }

        fn pmh_transpose(matrix: &[$primitive], n: usize) -> Vec<$primitive> {
            let mut out = vec![0 as $primitive; n];
            for i in 0..n {
                let row = matrix[i];
                for j in 0..n {
                    if ((row >> j) & 1) == 1 {
                        out[j] |= 1 as $primitive << i;
                    }
                }
            }
            out
        }

        fn pmh_cancel_adjacent(cnots: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
            let mut out = Vec::with_capacity(cnots.len());
            for g in cnots {
                if out.last() == Some(&g) {
                    out.pop();
                } else {
                    out.push(g);
                }
            }
            out
        }

        /// Apply a CX list to \(I_n\) (row \(t \oplus=\) row \(c\)).
        pub fn apply_cnots_to_identity(
            cnots: &[(usize, usize)],
            n: usize,
        ) -> Vec<$primitive> {
            let mut m: Vec<$primitive> = (0..n).map(|i| 1 as $primitive << i).collect();
            for &(c, t) in cnots {
                m[t] ^= m[c];
            }
            m
        }

        /// Paper Lwr_CNOT_Synth (quant-ph/0302002 Alg. 1), without Qiskit's
        /// back-reduce tweak. Returns `None` if a diagonal 1 cannot be placed.
        fn lwr_cnot_synth(
            matrix: &mut [$primitive],
            n: usize,
            section_size: usize,
        ) -> Option<Vec<(usize, usize)>> {
            let mut circuit = Vec::new();
            let n_secs = n.div_ceil(section_size);
            for sec in 0..n_secs {
                let start = sec * section_size;
                let end = (start + section_size).min(n);
                let width = end - start;

                let mut patt = vec![None; 1usize << width];
                for row in start..n {
                    let p = pmh_subrow_pattern(matrix[row], start, width) as usize;
                    if p == 0 {
                        continue;
                    }
                    if let Some(first) = patt[p] {
                        matrix[row] ^= matrix[first];
                        circuit.push((first, row));
                    } else {
                        patt[p] = Some(row);
                    }
                }

                for col in start..end {
                    let mut diag_one = ((matrix[col] >> col) & 1) == 1;
                    for row in (col + 1)..n {
                        if ((matrix[row] >> col) & 1) == 1 {
                            if !diag_one {
                                matrix[col] ^= matrix[row];
                                circuit.push((row, col));
                                diag_one = true;
                            }
                            matrix[row] ^= matrix[col];
                            circuit.push((col, row));
                        }
                    }
                    if !diag_one {
                        return None;
                    }
                }
            }
            Some(circuit)
        }

        fn synthesize_cnot_matrix_pmh_inner(
            matrix: &[$primitive],
            n: usize,
            section_size: usize,
        ) -> Option<Vec<(usize, usize)>> {
            let mask = pmh_row_mask(n);
            let mut work: Vec<$primitive> = matrix.iter().take(n).map(|r| r & mask).collect();
            if work.len() < n {
                return None;
            }

            let mut circuit_l = lwr_cnot_synth(&mut work, n, section_size)?;
            work = pmh_transpose(&work, n);
            let circuit_u = lwr_cnot_synth(&mut work, n, section_size)?;

            // Qiskit/paper stitch: swapped circuit_u, then reversed circuit_l.
            let mut cnots: Vec<(usize, usize)> = circuit_u.into_iter().map(|(c, t)| (t, c)).collect();
            circuit_l.reverse();
            cnots.extend(circuit_l);
            let cnots = pmh_cancel_adjacent(cnots);

            let got = apply_cnots_to_identity(&cnots, n);
            let want: Vec<$primitive> = matrix.iter().take(n).map(|r| r & mask).collect();
            if got != want {
                return None;
            }
            Some(cnots)
        }

        /// Sectioned Patel–Markov–Hayes CNOT synthesis (quant-ph/0302002).
        ///
        /// `section_size = None` uses [`default_pmh_section_size`]. On a failed
        /// invertibility check this returns `Err` so callers can fall back to GE.
        pub fn synthesize_cnot_matrix_pmh(
            matrix: Vec<$primitive>,
            num_qubits: usize,
            section_size: Option<usize>,
        ) -> Result<Vec<(usize, usize)>, String> {
            let n = num_qubits;
            if n == 0 {
                return Ok(Vec::new());
            }
            if matrix.len() < n {
                return Err("matrix shorter than num_qubits".to_string());
            }
            let m = section_size
                .unwrap_or_else(|| default_pmh_section_size(n))
                .clamp(1, n.min(8));
            synthesize_cnot_matrix_pmh_inner(&matrix, n, m)
                .ok_or_else(|| "pmh failed invertibility check".to_string())
        }

        /// One-pass Gauss–Jordan over \(\mathbb{F}_2\) (legacy helper).
        ///
        /// Row swaps are three CXs. Singular columns are skipped, so this
        /// always returns `Ok`. Kept as a fallback and A/B oracle.
        pub fn synthesize_cnot_matrix_ge(
            mut matrix: Vec<$primitive>,
            num_qubits: usize,
        ) -> Result<Vec<(usize, usize)>, String> {
            let mut cnots: Vec<(usize, usize)> = Vec::new();
            let n = num_qubits;

            let push_cnot = |c: usize, r: usize, cnots: &mut Vec<(usize, usize)>| {
                if let Some(&last) = cnots.last() {
                    if last == (c, r) {
                        cnots.pop();
                        return;
                    }
                }
                cnots.push((c, r));
            };

            for c in 0..n {
                let mut pivot = c;
                while pivot < n && (matrix[pivot] & ((1 as $primitive) << c)) == 0 {
                    pivot += 1;
                }

                if pivot == n {
                    pivot = 0;
                    while pivot < c && (matrix[pivot] & ((1 as $primitive) << c)) == 0 {
                        pivot += 1;
                    }
                    if pivot == c {
                        continue;
                    }
                }

                if pivot != c {
                    push_cnot(pivot, c, &mut cnots);
                    push_cnot(c, pivot, &mut cnots);
                    push_cnot(pivot, c, &mut cnots);
                    matrix.swap(pivot, c);
                }

                for r in 0..n {
                    if r != c && (matrix[r] & ((1 as $primitive) << c)) != 0 {
                        matrix[r] ^= matrix[c];
                        push_cnot(c, r, &mut cnots);
                    }
                }
            }

            cnots.reverse();
            Ok(cnots)
        }

        /// Synthesize a CNOT network for an \(n \times n\) matrix over \(\mathbb{F}_2\).
        ///
        /// Production entry: Patel–Markov–Hayes sectioned elimination, falling
        /// back to [`synthesize_cnot_matrix_ge`] if PMH cannot invert the matrix.
        /// Returns `(control, target)` pairs. Python bindings keep this name.
        pub fn synthesize_cnot_matrix(
            matrix: Vec<$primitive>,
            num_qubits: usize,
        ) -> Result<Vec<(usize, usize)>, String> {
            match synthesize_cnot_matrix_pmh(matrix.clone(), num_qubits, None) {
                Ok(cnots) => Ok(cnots),
                Err(_) => synthesize_cnot_matrix_ge(matrix, num_qubits),
            }
        }
    };
}
