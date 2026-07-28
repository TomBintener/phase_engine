use crate::{ast::ResolvedVar, core::ResolvedCall};

pub(crate) type BuildHasher = std::hash::BuildHasherDefault<rustc_hash::FxHasher>;
pub(crate) type HashMap<K, V> = hashbrown::HashMap<K, V, BuildHasher>;
pub(crate) type HashSet<K> = hashbrown::HashSet<K, BuildHasher>;
pub(crate) type HEntry<'a, A, B> = hashbrown::hash_map::Entry<'a, A, B, BuildHasher>;
pub type IndexMap<K, V> = indexmap::IndexMap<K, V, BuildHasher>;
pub type IndexSet<K> = indexmap::IndexSet<K, BuildHasher>;

pub use egglog_ast::generic_ast_helpers::INTERNAL_SYMBOL_PREFIX;

/// Generates fresh symbols for internal use during typechecking and flattening.
/// These are guaranteed not to collide with the
/// user's symbols because they use a reserved prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolGen {
    hint_to_count: HashMap<String, usize>,
    /// All names handed out so far. Because generated names are formed by
    /// concatenating `hint` and a counter, two different hints can produce the
    /// same string (e.g. hint `f2` with the counter left off vs. hint `f` with
    /// counter 2). Tracking every generated name lets `fresh` detect and skip
    /// such collisions, which would otherwise silently alias two distinct
    /// variables (see `fresh_symbols_no_collision_*` tests).
    generated: HashSet<String>,
    reserved_string: String,
    leave_off_zero: bool,
}

impl SymbolGen {
    /// Create a new symbol generator with the given reserved prefix.
    pub fn new(reserved_string: String) -> Self {
        Self {
            hint_to_count: HashMap::default(),
            generated: HashSet::default(),
            reserved_string,
            leave_off_zero: true,
        }
    }

    /// Produce a fresh, never-before-generated name for `hint`.
    ///
    /// Names are `{reserved}{hint}{count}` (with the count left off for the
    /// first use of a hint when `leave_off_zero` is set). Since the count is
    /// appended directly to the hint, hints ending in digits can produce the
    /// same string for different (hint, count) pairs; the `generated` set
    /// catches those cases and the counter is bumped until the name is unique.
    fn fresh_name(&mut self, hint: &str) -> String {
        let entry = self.hint_to_count.entry(hint.to_string()).or_insert(0);
        loop {
            let count = *entry;
            *entry += 1;
            let name = format!(
                "{}{}{}",
                self.reserved_string,
                hint,
                if self.leave_off_zero && count == 0 {
                    String::new()
                } else {
                    count.to_string()
                }
            );
            if self.generated.insert(name.clone()) {
                return name;
            }
        }
    }

    /// By default, the first symbol generated with a given hint
    /// does not have a numeric suffix (e.g., "var" instead of "var0").
    /// This method changes that behavior.
    pub fn include_zero(&mut self, include: bool) {
        self.leave_off_zero = !include;
    }

    /// Check if this symbol generator has been used to generate any symbols.
    pub fn has_been_used(&self) -> bool {
        !self.hint_to_count.is_empty()
    }

    /// Get the reserved prefix used by this symbol generator.
    pub fn reserved_prefix(&self) -> &str {
        &self.reserved_string
    }

    /// Check if the given symbol is reserved (i.e., starts with the reserved prefix).
    pub fn is_reserved(&self, symbol: &str) -> bool {
        !self.reserved_string.is_empty() && symbol.starts_with(&self.reserved_string)
    }
}

/// This trait lets us statically dispatch between `fresh` methods for generic structs.
pub trait FreshGen<Head: ?Sized, Leaf> {
    fn fresh(&mut self, name_hint: &Head) -> Leaf;
}

impl FreshGen<str, String> for SymbolGen {
    fn fresh(&mut self, name_hint: &str) -> String {
        self.fresh_name(name_hint)
    }
}

impl FreshGen<String, String> for SymbolGen {
    fn fresh(&mut self, name_hint: &String) -> String {
        self.fresh(name_hint.as_str())
    }
}

impl FreshGen<ResolvedCall, ResolvedVar> for SymbolGen {
    fn fresh(&mut self, name_hint: &ResolvedCall) -> ResolvedVar {
        let name = self.fresh_name(&format!("{name_hint}"));
        let sort = match name_hint {
            ResolvedCall::Func(f) => f.output.clone(),
            ResolvedCall::Primitive(prim) => prim.output().clone(),
        };
        ResolvedVar {
            name,
            sort,
            // fresh variables are never global references, since globals
            // are desugared away by `remove_globals`
            is_global_ref: false,
        }
    }
}
