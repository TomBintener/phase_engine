# PathSum in this crate

This repository (`TomBintener/phase_engine`) is a fork of egglog. The Cargo
package name is still `egglog`. AGES compiles **this** crate for PathSum, not
`egraphs-good/egglog` and not `egglog-pathsum-python`.

The upstream egglog README below this file does **not** document PathSum.
This file is the PathSum contract. Every claim cites a symbol in this tree.

## How AGES installs it

Pip package name is `egglog` (maturin). The wheel is built from
`python-bindings/`, which path-depends the parent crate:

| Claim | Bind |
|---|---|
| pip project name `egglog` | `python-bindings/pyproject.toml` `[project] name` |
| cdylib crate name `egglog` | `python-bindings/Cargo.toml` `[lib] name` |
| PathSum engine is the parent crate | `python-bindings/Cargo.toml` `egglog = { path = "..", default-features = false }` |
| workspace excludes the nested maturin project | root `Cargo.toml` `exclude = ["python-bindings"]` |

AGES pin:

```
egglog @ git+https://github.com/TomBintener/phase_engine.git@main#subdirectory=python-bindings
```

Local: `pip install -e python-bindings`.

## Operator, not a ket

Equality is interned **operator** equality from the identity \(I_n\). There are
no ket constructors.

| Claim | Bind |
|---|---|
| Value type | `src/engine/evaluator.rs` `EvaluatedPathSum` (`out_state`, `phase_poly`, `continuous_poly`, `num_qubits`, `num_path_vars`) |
| Identity \(I_n\): `out_state[i] = x_i`, empty phases | `EvaluatedPathSum::new_id` |
| Ket constructors absent | `new_id` docstring; no `zero_state` / `basis_state` / `comp_mask` / `val_mask` in this tree |
| Two widths | `src/engine/mod.rs` `generate_phase_engine!(engine_64, u64, …)` and `engine_128` (`u128`) |
| Interned egglog sort | `src/pathsum.rs` `PSum64` / `PSum128` = `Boxed<Arc<EvaluatedPathSum>>` |
| Host FFI for \(I_n\) | `id_pathsum_logic_{64,128}` → primitives `rust_id_pathsum_{64,128}` |

`apply_cx` is always linear XOR on `out_state` (`EvaluatedPathSum::apply_cx`).
There is no computational-basis shortcut.

## Eager `reduce()` — and the two exceptions

The module comment at the top of `src/pathsum.rs` used to say every gate
reduces. The registered primitives are the contract:

| Primitive | Wrapper | `reduce()`? |
|---|---|---|
| `rust_apply_{z,s,sdg,t,tdg,h,sx}_*` | `apply_gate_logic_*` | yes, after the apply |
| `rust_apply_rz_*` | `apply_rz_logic_*` | yes, after `apply_rz` |
| `rust_apply_x_*` | `apply_gate_no_reduce_logic_*` | **no** |
| `rust_apply_cx_*` | `apply_cx_logic_*` | **no** |

`x` and `cx` are affine / linear XOR updates and do not allocate path
variables. Registration: `add_primitive!` block in `src/pathsum.rs`
(`rust_apply_x_64` through `rust_apply_cx_64`, and the `_128` copies).

## Panic shields and path-variable budget

Out-of-range qubit indices, or `cx` with `qc == qt`, return the **unmodified**
PathSum (`apply_gate_logic_*`, `apply_cx_logic_*`).

Not shielded: `apply_h` / `apply_sx` `assert!(var_index < BITS-3)` with
`var_index = num_qubits + num_path_vars` (`EvaluatedPathSum::apply_h`,
`apply_sx`). On PathSum64 that is `< 61` (`u64::BITS - 3`). AGES `mode="auto"`
switches to 128-bit at `n > 45` for headroom, not because 45 is a hard cap.

## Continuous snap lattice

`apply_rz` unpacks IEEE-754 bits in `apply_rz_logic_*`, then
`ContinuousPhasePoly::apply_phase` does `rem_euclid(TAU)` and snaps at
`SNAP_PRECISION = 1e8` **before** `promote_cliffords()`. Interned `Eq`/`Hash`
on `ContinuousPhasePoly` compare/hash that same rounded `i64` lattice
(`src/engine/continuous_poly.rs`).

`rust_add_rz_bits_*` / `rust_negate_rz_bits_*` (`rust_add_rz_bits_logic`,
`rust_negate_rz_bits_logic`) apply the same wrap+snap to **bit payloads** for
AGES fusion rules. They are not PathSum state updates.

## Fingerprints

| Claim | Bind |
|---|---|
| Equality oracle dump | `state_fingerprint_logic_{64,128}` = `format!("{:?}", EvaluatedPathSum)` |
| Debug dump omits `out_state` | `debug_pathsum_logic_{64,128}` (phases only). Do not use as `Eq`. |

## In-cycle synthesizers

Registered as `rust_synthesize_{pmh,steiner,gray}_{64,128}`. Backends:
`src/engine/{pmh,steiner,gray}.rs` (instantiated by `generate_phase_engine!`).
AGES `synthesis_rules` call these; merge still requires interned PathSum `Eq`.
