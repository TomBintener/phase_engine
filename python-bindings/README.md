# Python bindings for this PathSum engine

Maturin package name is `egglog` so AGES can `import egglog`. Cargo path-depends
on the parent `phase_engine` crate (I_n-only PathSum).

AGES installs this directory with:

```
egglog @ git+https://github.com/TomBintener/phase_engine.git@main#subdirectory=python-bindings
```

Local:

```
pip install -e python-bindings
```
