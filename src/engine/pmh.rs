macro_rules! define_pmh_logic {
    ($primitive:ty) => {
        /// Synthesizes a CNOT matrix into an optimal sequence of CNOT operations 
        /// using Gauss-Jordan Elimination over GF(2).
        /// Returns a list of (control, target) qubit pairs.
        pub fn synthesize_cnot_matrix(mut matrix: Vec<$primitive>, num_qubits: usize) -> Result<Vec<(usize, usize)>, String> {
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
                    // Try to find a pivot above c
                    pivot = 0;
                    while pivot < c && (matrix[pivot] & ((1 as $primitive) << c)) == 0 {
                        pivot += 1;
                    }
                    if pivot == c {
                        continue; // Singular matrix
                    }
                }

                // Swap pivot row if needed
                if pivot != c {
                    // Swap is 3 CNOTs
                    push_cnot(pivot, c, &mut cnots);
                    push_cnot(c, pivot, &mut cnots);
                    push_cnot(pivot, c, &mut cnots);
                    
                    matrix.swap(pivot, c);
                }

                // Eliminate column c for all OTHER rows
                for r in 0..n {
                    if r != c && (matrix[r] & ((1 as $primitive) << c)) != 0 {
                        matrix[r] ^= matrix[c];
                        push_cnot(c, r, &mut cnots);
                    }
                }
            }

            // The operations we applied transform M to I.
            // To transform I to M, we apply the operations in reverse.
            cnots.reverse();

            Ok(cnots)
        }
    };
}
