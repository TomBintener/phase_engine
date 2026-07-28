use egglog::EGraph;

fn main() {
    // Mirror egglog-python's default: a single-threaded rayon pool.
    let num_threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();
    eprintln!("rayon threads: {}", rayon::current_num_threads());
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "_repro_program.egg".to_owned());
    // The recorded program contains a ~400-deep nested term; run on a big stack.
    let handle = std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024 * 1024)
        .spawn(move || {
            let program = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("Failed to read {path}: {e}"));
            let mut egraph = EGraph::default();
            match egraph.parse_and_run_program(Some(path), &program) {
                Ok(outputs) => {
                    eprintln!("Execution succeeded with {} outputs", outputs.len());
                    for (i, output) in outputs.iter().enumerate() {
                        println!("=== output[{i}] ===");
                        println!("{output}");
                    }
                }
                Err(e) => {
                    eprintln!("Execution failed: {e}");
                    std::process::exit(1);
                }
            }
        })
        .unwrap();
    handle.join().unwrap();
}
