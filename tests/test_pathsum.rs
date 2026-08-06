use egglog::EGraph;
#[test]
fn test_pathsum_sort() {
    let mut eg = EGraph::default();
    let cmds = eg.parse_program(None, "(function test () PathSum64 :merge (old))").unwrap();
    eg.run_program(cmds).unwrap();
}
