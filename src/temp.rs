use egglog::EGraph;
use std::time::Instant;

fn profile_egglog_file(path: &str) {
    let program = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path, e));
    
    println!("=== Profiling: {} ===", path);
    println!("Program size: {} chars, {} lines", program.len(), program.lines().count());
    
    // Split the program into individual commands
    // Egglog programs are semicolon-delimited
    let lines: Vec<&str> = program.lines().collect();
    
    // Phase 1: Parse + Type check + Setup
    let t_total = Instant::now();
    let mut egraph = EGraph::default();
    
    let t_parse = Instant::now();
    let result = egraph.parse_and_run_program(Some(path.to_owned()), &program);
    let parse_elapsed = t_parse.elapsed();
    
    match result {
        Ok(outputs) => {
            println!("  Execution succeeded with {} outputs", outputs.len());
            for (i, output) in outputs.iter().take(5).enumerate() {
                println!("    output[{}]: {}", i, output);
            }
        }
        Err(e) => {
            println!("  Execution failed: {}", e);
        }
    }
    
    let total_elapsed = t_total.elapsed();
    
    println!("\n  TIMING:");
    println!("    Total wall time:    {:>10.3}ms", total_elapsed.as_secs_f64() * 1000.0);
    println!("    Parse+Run:          {:>10.3}ms", parse_elapsed.as_secs_f64() * 1000.0);

    // Phase 2: Serialization benchmark
    let t_serialize = Instant::now();
    let _serialized = egraph.serialize(egglog::SerializeConfig::default());
    let serialize_elapsed = t_serialize.elapsed();
    println!("    Serialization:      {:>10.3}ms", serialize_elapsed.as_secs_f64() * 1000.0);
    
    // Print e-graph stats
    println!("\n  E-GRAPH STATS:");
    let report = egraph.get_overall_run_report();
    println!("    Iterations (reported): {}", report.iterations.len());
    let total_matches: usize = report.num_matches_per_rule.values().sum();
    println!("    Total matches (reported):  {}", total_matches);
    
    // Per-rule breakdown
    println!("\n  TOP RULES BY TIME:");
    let mut rule_times: Vec<_> = report.search_and_apply_time_per_rule.iter().collect();
    rule_times.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    for (i, (rule, time)) in rule_times.iter().take(15).enumerate() {
        let matches = report.num_matches_per_rule.get(*rule).copied().unwrap_or(0);
        println!("    {:>2}. {:>10.3}ms | {:>8} matches | {}", 
            i + 1, 
            time.as_secs_f64() * 1000.0, 
            matches,
            // Truncate rule name for readability
            {
                let s = format!("{}", rule);
                if s.len() > 100 { format!("{}...", &s[..100]) } else { s }
            }
        );
    }
    
    println!("\n  TOP RULES BY MATCHES:");
    let mut rule_matches: Vec<_> = report.num_matches_per_rule.iter().collect();
    rule_matches.sort_by(|a, b| b.1.cmp(a.1));
    
    for (i, (rule, matches)) in rule_matches.iter().take(10).enumerate() {
        let time = report.search_and_apply_time_per_rule.get(*rule)
            .map(|t| t.as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        println!("    {:>2}. {:>8} matches | {:>10.3}ms | {}", 
            i + 1, 
            matches,
            time,
            {
                let s = format!("{}", rule);
                if s.len() > 100 { format!("{}...", &s[..100]) } else { s }
            }
        );
    }
    
    println!("\n{}", "=".repeat(80));
}


fn main() {
    // Profile the slow circuit
    profile_egglog_file("tests/egglog_dump_test_continuous_cx.egg");
    
    println!("\n\n");
    
    // Profile the fast circuit for comparison
    profile_egglog_file("tests/egglog_dump_10_phasetrap.egg");
}
