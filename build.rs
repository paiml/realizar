// build.rs — Read provable-contracts binding.yaml and set CONTRACT_* env vars
//
// Policy: WarnOnGaps (Phase 4). We emit warnings for partial/not_implemented
// bindings but do NOT fail the build. Phase 5 will switch to AllImplemented.
//
// The env vars follow the pattern:
//   CONTRACT_<CONTRACT_STEM>_<EQUATION>=<status>
//
// Example:
//   CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX=implemented
//   CONTRACT_SILU_KERNEL_V1_SILU=not_implemented
//
// These are consumed at compile time by the #[contract] proc macro to gate
// code generation and emit compile_error! for missing implementations.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

/// Minimal subset of the binding.yaml schema — just enough to parse what we need.
#[derive(Deserialize)]
struct BindingFile {
    #[allow(dead_code)]
    version: String,
    #[allow(dead_code)]
    target_crate: String,
    bindings: Vec<Binding>,
}

#[derive(Deserialize)]
struct Binding {
    contract: String,
    equation: String,
    status: String,
    #[serde(default)]
    notes: Option<String>,
}

/// Convert a contract filename + equation into a canonical env var name.
///
/// "softmax-kernel-v1.yaml" + "softmax" → "CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX"
fn env_var_name(contract: &str, equation: &str) -> String {
    let stem = contract
        .trim_end_matches(".yaml")
        .trim_end_matches(".yml")
        .to_uppercase()
        .replace('-', "_");
    let eq = equation.to_uppercase().replace('-', "_");
    format!("CONTRACT_{stem}_{eq}")
}

// ─────────────────────────────────────────────────────────────────────────────
// PMAT-228: Architecture requirements YAML → Rust codegen
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal schema for architecture-requirements-v1.yaml.
#[derive(Deserialize)]
struct ArchRequirements {
    weight_roles: BTreeMap<String, WeightRoleDef>,
    role_sets: BTreeMap<String, RoleSetDef>,
    constraint_matrix: BTreeMap<String, ConstraintCell>,
}

#[derive(Deserialize)]
struct WeightRoleDef {
    description: String,
    field_name: String,
    #[allow(dead_code)]
    category: String,
}

#[derive(Deserialize)]
struct RoleSetDef {
    #[allow(dead_code)]
    description: String,
    #[allow(dead_code)]
    count: usize,
    roles: Vec<String>,
}

#[derive(Deserialize)]
struct ConstraintCell {
    has_qk_norm: bool,
    has_bias: bool,
    required_sets: Vec<String>,
    total_required: usize,
}

/// Convert YAML role name to Rust enum variant name.
///
/// Special cases preserve backward compatibility with existing codebase:
/// - gate_proj → FfnGate, up_proj → FfnUp, down_proj → FfnDown
/// - Everything else: snake_case → PascalCase (e.g. attn_q_norm → AttnQNorm)
fn role_to_variant(s: &str) -> String {
    match s {
        "gate_proj" => "FfnGate".to_string(),
        "up_proj" => "FfnUp".to_string(),
        "down_proj" => "FfnDown".to_string(),
        _ => s
            .split('_')
            .map(|part| {
                let mut chars = part.chars();
                match chars.next() {
                    Some(c) => {
                        let upper: String = c.to_uppercase().collect();
                        format!("{upper}{}", chars.as_str())
                    },
                    None => String::new(),
                }
            })
            .collect(),
    }
}

/// Generate the arch_requirements.rs content from YAML data.
fn generate_arch_requirements(req: &ArchRequirements) -> String {
    let mut out = String::with_capacity(4096);

    // Header — use // not //! since this is included via include!()
    out.push_str(
        "// Per-architecture required weight roles.\n\
         //\n\
         // AUTO-GENERATED from architecture-requirements-v1.yaml by build.rs — DO NOT EDIT.\n\
         // See: provable-contracts/contracts/architecture-requirements-v1.yaml\n\
         //\n\
         // UCBD §4 / GH-279: Compile-time enforcement that every loader\n\
         // provides all tensors required by the target architecture.\n\
         \n\
         use crate::gguf::ArchConstraints;\n\
         \n",
    );

    // WeightRole enum
    out.push_str(
        "/// Weight roles that may be required for a transformer layer.\n\
         /// Each architecture requires a subset of these.\n\
         #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]\n\
         pub enum WeightRole {\n",
    );
    for (name, def) in &req.weight_roles {
        out.push_str(&format!("    /// {}\n", def.description));
        out.push_str(&format!("    {},\n", role_to_variant(name)));
    }
    out.push_str("}\n\n");

    // field_name() method
    out.push_str(
        "impl WeightRole {\n\
         \x20   /// Returns the field name prefix as it appears in `IndexedLayerWeights`.\n\
         \x20   #[must_use]\n\
         \x20   pub const fn field_name(&self) -> &'static str {\n\
         \x20       match self {\n",
    );
    for (name, def) in &req.weight_roles {
        out.push_str(&format!(
            "            Self::{} => \"{}\",\n",
            role_to_variant(name),
            def.field_name
        ));
    }
    out.push_str("        }\n    }\n}\n\n");

    // Const arrays for each constraint cell
    // Sort cells by (has_qk_norm, has_bias) for deterministic output
    let mut cells: Vec<_> = req.constraint_matrix.iter().collect();
    cells.sort_by_key(|(_, c)| (c.has_qk_norm, c.has_bias));

    for (cell_name, cell) in &cells {
        let const_name = cell_const_name(cell_name);
        let roles = expand_role_sets(&cell.required_sets, &req.role_sets);
        out.push_str(&format!(
            "/// Roles for constraint cell: {} (has_qk_norm={}, has_bias={}).\n",
            cell_name, cell.has_qk_norm, cell.has_bias
        ));
        out.push_str(&format!("const {const_name}: &[WeightRole] = &[\n"));
        for role in &roles {
            out.push_str(&format!("    WeightRole::{},\n", role_to_variant(role)));
        }
        out.push_str("];\n\n");

        // Compile-time assertion on count
        out.push_str(&format!(
            "const _: () = assert!({const_name}.len() == {}, \"YAML declares {} roles\");\n\n",
            cell.total_required, cell.total_required
        ));
    }

    // required_roles() function
    out.push_str(
        "/// Returns the required weight roles for a given architecture.\n\
         ///\n\
         /// Exhaustive match on `(has_qk_norm, has_bias)` — adding a new architecture\n\
         /// combination without updating this function will still match one of the\n\
         /// four arms, but the contract test FALSIFY-ARCH-001 will catch mismatches.\n\
         #[must_use]\n\
         pub fn required_roles(arch: &ArchConstraints) -> &'static [WeightRole] {\n\
         \x20   match (arch.has_qk_norm, arch.has_bias) {\n",
    );
    for (cell_name, cell) in &cells {
        let const_name = cell_const_name(cell_name);
        out.push_str(&format!(
            "        ({}, {}) => {const_name},\n",
            cell.has_qk_norm, cell.has_bias
        ));
    }
    out.push_str("    }\n}\n");

    out
}

/// Convert YAML cell name to a Rust const name.
/// "no_qk_norm_no_bias" → "ROLES_NO_QK_NORM_NO_BIAS"
fn cell_const_name(cell_name: &str) -> String {
    format!("ROLES_{}", cell_name.to_uppercase())
}

/// Expand a list of role set names into a flat list of role names.
fn expand_role_sets(set_names: &[String], sets: &BTreeMap<String, RoleSetDef>) -> Vec<String> {
    let mut roles = Vec::new();
    for set_name in set_names {
        if let Some(set) = sets.get(set_name) {
            roles.extend(set.roles.iter().cloned());
        }
    }
    roles
}

fn main() {
    // Phase 1: Architecture requirements codegen (PMAT-228)
    generate_arch_requirements_file();

    // Phase 2: Contract binding env vars
    // Re-run if binding.yaml changes
    let binding_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("provable-contracts")
        .join("contracts")
        .join("aprender")
        .join("binding.yaml");

    // Always tell Cargo to re-run if the file appears or changes
    println!("cargo:rerun-if-changed={}", binding_path.display());

    if !binding_path.exists() {
        // Graceful fallback: CI/crates.io builds won't have the sibling repo.
        // The #[contract] proc macro should handle missing env vars gracefully.
        println!(
            "cargo:warning=provable-contracts binding.yaml not found at {}; \
             CONTRACT_* env vars will not be set (CI/crates.io build)",
            binding_path.display()
        );
        // Set a sentinel so the proc macro knows we ran but had no file
        println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
        return;
    }

    let yaml_content = match std::fs::read_to_string(&binding_path) {
        Ok(s) => s,
        Err(e) => {
            println!(
                "cargo:warning=Failed to read binding.yaml: {e}; \
                 CONTRACT_* env vars will not be set"
            );
            println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
            return;
        },
    };

    let bindings: BindingFile = match serde_yaml::from_str(&yaml_content) {
        Ok(b) => b,
        Err(e) => {
            println!(
                "cargo:warning=Failed to parse binding.yaml: {e}; \
                 CONTRACT_* env vars will not be set"
            );
            println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
            return;
        },
    };

    // Counters for the summary
    let mut implemented = 0u32;
    let mut partial = 0u32;
    let mut not_implemented = 0u32;

    // De-duplicate: the binding.yaml has multiple entries for the same
    // (contract, equation) pair across different crates. We take the "best"
    // status (implemented > partial > not_implemented).
    let mut seen = std::collections::HashMap::<String, String>::new();

    for binding in &bindings.bindings {
        let var_name = env_var_name(&binding.contract, &binding.equation);
        let status = binding.status.as_str();

        // Keep the best status seen so far for this (contract, equation) pair
        let dominated = match (seen.get(&var_name).map(|s| s.as_str()), status) {
            (None, _) => false,                        // first time
            (Some("implemented"), _) => true,          // already best
            (Some("partial"), "implemented") => false, // upgrade
            (Some("partial"), _) => true,              // keep partial
            (Some("not_implemented"), "not_implemented") => true,
            (Some("not_implemented"), _) => false, // upgrade
            _ => false,
        };

        if dominated {
            continue;
        }

        seen.insert(var_name, status.to_string());
    }

    // Now emit env vars and warnings
    let mut keys: Vec<_> = seen.keys().cloned().collect();
    keys.sort();

    for var_name in &keys {
        let status = &seen[var_name];

        println!("cargo:rustc-env={var_name}={status}");

        match status.as_str() {
            "implemented" => implemented += 1,
            "partial" => {
                partial += 1;
                // Find the original binding for the note
                let note = bindings
                    .bindings
                    .iter()
                    .find(|b| &env_var_name(&b.contract, &b.equation) == var_name)
                    .and_then(|b| b.notes.as_deref())
                    .unwrap_or("");
                println!("cargo:warning=[contract] PARTIAL: {var_name} — {note}");
            },
            "not_implemented" => {
                not_implemented += 1;
                // WarnOnGaps: warn but do NOT fail the build
                let note = bindings
                    .bindings
                    .iter()
                    .find(|b| &env_var_name(&b.contract, &b.equation) == var_name)
                    .and_then(|b| b.notes.as_deref())
                    .unwrap_or("");
                println!("cargo:warning=[contract] GAP: {var_name} — {note}");
            },
            other => {
                println!("cargo:warning=[contract] UNKNOWN STATUS '{other}': {var_name}");
            },
        }
    }

    let total = implemented + partial + not_implemented;
    println!(
        "cargo:warning=[contract] Summary: {implemented}/{total} implemented, \
         {partial} partial, {not_implemented} gaps (WarnOnGaps policy)"
    );

    // Set metadata env vars for the proc macro
    println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=binding.yaml");
    println!(
        "cargo:rustc-env=CONTRACT_BINDING_VERSION={}",
        bindings.version
    );
    println!("cargo:rustc-env=CONTRACT_TOTAL={total}");
    println!("cargo:rustc-env=CONTRACT_IMPLEMENTED={implemented}");
    println!("cargo:rustc-env=CONTRACT_PARTIAL={partial}");
    println!("cargo:rustc-env=CONTRACT_GAPS={not_implemented}");
}

/// PMAT-228: Read architecture-requirements-v1.yaml and generate arch_requirements.rs.
fn generate_arch_requirements_file() {
    let yaml_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("provable-contracts")
        .join("contracts")
        .join("architecture-requirements-v1.yaml");

    println!("cargo:rerun-if-changed={}", yaml_path.display());

    if !yaml_path.exists() {
        // Graceful fallback for CI/crates.io — write a stub generated file
        println!(
            "cargo:warning=[PMAT-228] architecture-requirements-v1.yaml not found; \
             using hand-written arch_requirements.rs fallback"
        );
        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
        let out_path = Path::new(&out_dir).join("arch_requirements_generated.rs");
        std::fs::write(&out_path, include_str!("src/arch_requirements_fallback.rs"))
            .expect("Failed to write fallback arch_requirements_generated.rs");
        return;
    }

    let yaml_content = match std::fs::read_to_string(&yaml_path) {
        Ok(s) => s,
        Err(e) => {
            println!(
                "cargo:warning=[PMAT-228] Failed to read architecture-requirements-v1.yaml: {e}"
            );
            return;
        },
    };

    let req: ArchRequirements = match serde_yaml::from_str(&yaml_content) {
        Ok(r) => r,
        Err(e) => {
            println!(
                "cargo:warning=[PMAT-228] Failed to parse architecture-requirements-v1.yaml: {e}"
            );
            return;
        },
    };

    let generated = generate_arch_requirements(&req);

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = Path::new(&out_dir).join("arch_requirements_generated.rs");
    std::fs::write(&out_path, generated).expect("Failed to write generated arch_requirements.rs");

    println!(
        "cargo:warning=[PMAT-228] Generated arch_requirements from YAML ({} weight roles, {} constraint cells)",
        req.weight_roles.len(),
        req.constraint_matrix.len()
    );
}
