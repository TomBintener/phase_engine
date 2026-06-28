macro_rules! define_pmh_logic {
    ($primitive:ty) => {
        /// Synthesizes a CNOT matrix into an optimal sequence of CNOT operations 
        /// using Gauss-Jordan Elimination over GF(2).
        /// Returns a list of (control, target) qubit pairs.
        pub fn synthesize_cnot_matrix(mut matrix: Vec<$primitive>, num_qubits: usize) -> Result<Vec<(usize, usize)>, String> {
            let mut cnots = Vec::new();
            let n = num_qubits;

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
                    cnots.push((pivot, c));
                    cnots.push((c, pivot));
                    cnots.push((pivot, c));
                    
                    matrix.swap(pivot, c);
                }

                // Eliminate column c for all OTHER rows
                for r in 0..n {
                    if r != c && (matrix[r] & ((1 as $primitive) << c)) != 0 {
                        matrix[r] ^= matrix[c];
                        cnots.push((c, r));
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
