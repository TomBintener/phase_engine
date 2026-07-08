macro_rules! define_steiner_logic {
    ($primitive:ty) => {
        use std::collections::{HashSet, VecDeque};
        
        pub struct Topology {
            pub n: usize,
            pub dist: Vec<Vec<usize>>,
            pub next: Vec<Vec<usize>>,
            pub adj: Vec<Vec<usize>>,
        }

        impl Topology {
            pub fn new(n: usize, edge_str: &str) -> Self {
                let edge_str = edge_str.trim_matches('"');
                let mut adj = vec![vec![]; n];
                if !edge_str.is_empty() {
                    for pair in edge_str.split(';') {
                        let parts: Vec<&str> = pair.split('-').collect();
                        if parts.len() == 2 {
                            if let (Ok(u), Ok(v)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                                if u < n && v < n {
                                    adj[u].push(v);
                                    adj[v].push(u);
                                }
                            }
                        }
                    }
                } else {
                    // Default to fully connected if empty
                    for i in 0..n {
                        for j in 0..n {
                            if i != j {
                                adj[i].push(j);
                            }
                        }
                    }
                }

                let mut dist = vec![vec![usize::MAX; n]; n];
                let mut next = vec![vec![usize::MAX; n]; n];

                for i in 0..n {
                    dist[i][i] = 0;
                    next[i][i] = i;
                    for &j in &adj[i] {
                        dist[i][j] = 1;
                        next[i][j] = j;
                    }
                }

                // Floyd-Warshall
                for k in 0..n {
                    for i in 0..n {
                        for j in 0..n {
                            if dist[i][k] != usize::MAX && dist[k][j] != usize::MAX {
                                if dist[i][k] + dist[k][j] < dist[i][j] {
                                    dist[i][j] = dist[i][k] + dist[k][j];
                                    next[i][j] = next[i][k];
                                }
                            }
                        }
                    }
                }

                Topology { n, dist, next, adj }
            }

            pub fn path(&self, u: usize, v: usize) -> Vec<usize> {
                let mut p = Vec::new();
                if self.dist[u][v] == usize::MAX {
                    return p;
                }
                let mut curr = u;
                p.push(curr);
                while curr != v {
                    curr = self.next[curr][v];
                    p.push(curr);
                }
                p
            }

            // Returns undirected edges that form the Steiner tree
            pub fn steiner_tree_edges(&self, terminals: &[usize]) -> Vec<(usize, usize)> {
                if terminals.is_empty() {
                    return vec![];
                }
                if terminals.len() == 1 {
                    return vec![];
                }
                // Prim's algorithm on the complete graph of terminals
                let mut in_tree = vec![false; terminals.len()];
                in_tree[0] = true;
                let mut edges = HashSet::new();

                for _ in 1..terminals.len() {
                    let mut min_dist = usize::MAX;
                    let mut best_u = 0;
                    let mut best_v = 0;
                    let mut best_j = 0;

                    for i in 0..terminals.len() {
                        if in_tree[i] {
                            for j in 0..terminals.len() {
                                if !in_tree[j] {
                                    let d = self.dist[terminals[i]][terminals[j]];
                                    if d < min_dist {
                                        min_dist = d;
                                        best_u = terminals[i];
                                        best_v = terminals[j];
                                        best_j = j;
                                    }
                                }
                            }
                        }
                    }

                    if min_dist == usize::MAX {
                        break;
                    }

                    in_tree[best_j] = true;
                    let p = self.path(best_u, best_v);
                    for k in 0..p.len().saturating_sub(1) {
                        let a = std::cmp::min(p[k], p[k+1]);
                        let b = std::cmp::max(p[k], p[k+1]);
                        edges.insert((a, b));
                    }
                }
                edges.into_iter().collect()
            }
        }
        
        // CNOT ladder helper for remote CNOT without SWAPs. 
        // Corrupts intermediate state but perfectly sets target.
        // We use this for routing PMH unconstrained CNOTs.
        pub fn apply_remote_cnot(c: usize, t: usize, topo: &Topology, cnots: &mut Vec<(usize, usize)>) {
            let p = topo.path(c, t);
            if p.is_empty() { return; }
            let k = p.len() - 1;
            if k == 1 {
                cnots.push((c, t));
                return;
            }
            // Standard SWAP route to preserve intermediate qubits
            for i in 0..k-1 {
                cnots.push((p[i], p[i+1]));
                cnots.push((p[i+1], p[i]));
                cnots.push((p[i], p[i+1]));
            }
            cnots.push((p[k-1], p[k]));
            for i in (0..k-1).rev() {
                cnots.push((p[i], p[i+1]));
                cnots.push((p[i+1], p[i]));
                cnots.push((p[i], p[i+1]));
            }
        }
    };
}
