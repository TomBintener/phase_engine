import re

with open("src/pathsum.rs", "r") as f:
    content = f.read()

content = content.replace(
    "add_primitive!(eg, \"rust_synthesize_steiner_64\" = |s: PSum64, count: i64, top: S| -> S { S::new(synthesize_steiner_logic_64(s, count, top.to_string())) });",
    "add_primitive!(eg, \"rust_synthesize_steiner_64\" = |s: PSum64, count: i64, top: S| -> S { S::new(synthesize_steiner_logic_64(s, count, top.to_string())) });\n        add_primitive!(eg, \"rust_debug_pathsum_64\" = |s: PSum64| -> S { S::new(debug_pathsum_logic_64(s)) });"
)

content = content.replace(
    "add_primitive!(eg, \"rust_synthesize_steiner_128\" = |s: PSum128, count: i64, top: S| -> S { S::new(synthesize_steiner_logic_128(s, count, top.to_string())) });",
    "add_primitive!(eg, \"rust_synthesize_steiner_128\" = |s: PSum128, count: i64, top: S| -> S { S::new(synthesize_steiner_logic_128(s, count, top.to_string())) });\n        add_primitive!(eg, \"rust_debug_pathsum_128\" = |s: PSum128| -> S { S::new(debug_pathsum_logic_128(s)) });"
)

content += """
pub fn debug_pathsum_logic_64(state: PSum64) -> String {
    let mut out = String::new();
    out.push_str(&format!("Continuous Parities: {:?}\\n", state.continuous_poly.parities));
    out.push_str(&format!("Continuous Phases: {:?}\\n", state.continuous_poly.phases));
    out.push_str("Phase Poly Terms: ");
    for term in &state.phase_poly.terms {
        let phase_unit = term.0 >> 61;
        let mask = term.0 & 0x1FFFFFFFFFFFFFFF;
        out.push_str(&format!("(mask={}, phase_unit={}) ", mask, phase_unit));
    }
    out
}

pub fn debug_pathsum_logic_128(state: PSum128) -> String {
    let mut out = String::new();
    out.push_str(&format!("Continuous Parities: {:?}\\n", state.continuous_poly.parities));
    out.push_str(&format!("Continuous Phases: {:?}\\n", state.continuous_poly.phases));
    out.push_str("Phase Poly Terms: ");
    for term in &state.phase_poly.terms {
        let phase_unit = term.0 >> 125;
        let mask = term.0 & 0x1FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
        out.push_str(&format!("(mask={}, phase_unit={}) ", mask, phase_unit));
    }
    out
}
"""

with open("src/pathsum.rs", "w") as f:
    f.write(content)

