# Python bindings for this PathSum engine

Maturin package name is `egglog` so AGES can `import egglog`.

| Claim | Bind |
|---|---|
| pip name `egglog` | `python-bindings/pyproject.toml` `[project] name = "egglog"` |
| cdylib `egglog` | `python-bindings/Cargo.toml` `[lib] name = "egglog"` |
| Compiles **this** PathSum crate (\(I_n\)-only) | `python-bindings/Cargo.toml` `egglog = { path = "..", default-features = false }` |
| PathSum FFI | parent `src/pathsum.rs` (`rust_id_pathsum_*`, `rust_apply_*`, …) |
| Operator contract | [../PATHSUM.md](../PATHSUM.md) |

AGES installs this directory with:

```
egglog @ git+https://github.com/TomBintener/phase_engine.git@main#subdirectory=python-bindings
```

Local:

```
pip install -e python-bindings
```
