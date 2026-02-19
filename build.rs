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

fn main() {
    // Re-run if binding.yaml changes
    let binding_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("provable-contracts")
        .join("contracts")
        .join("aprender")
        .join("binding.yaml");

    // Always tell Cargo to re-run if the file appears or changes
    println!(
        "cargo:rerun-if-changed={}",
        binding_path.display()
    );

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
        }
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
        }
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
            (None, _) => false,                            // first time
            (Some("implemented"), _) => true,              // already best
            (Some("partial"), "implemented") => false,     // upgrade
            (Some("partial"), _) => true,                  // keep partial
            (Some("not_implemented"), "not_implemented") => true,
            (Some("not_implemented"), _) => false,         // upgrade
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
                println!(
                    "cargo:warning=[contract] PARTIAL: {var_name} — {note}"
                );
            }
            "not_implemented" => {
                not_implemented += 1;
                // WarnOnGaps: warn but do NOT fail the build
                let note = bindings
                    .bindings
                    .iter()
                    .find(|b| &env_var_name(&b.contract, &b.equation) == var_name)
                    .and_then(|b| b.notes.as_deref())
                    .unwrap_or("");
                println!(
                    "cargo:warning=[contract] GAP: {var_name} — {note}"
                );
            }
            other => {
                println!(
                    "cargo:warning=[contract] UNKNOWN STATUS '{other}': {var_name}"
                );
            }
        }
    }

    let total = implemented + partial + not_implemented;
    println!(
        "cargo:warning=[contract] Summary: {implemented}/{total} implemented, \
         {partial} partial, {not_implemented} gaps (WarnOnGaps policy)"
    );

    // Set metadata env vars for the proc macro
    println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=binding.yaml");
    println!("cargo:rustc-env=CONTRACT_BINDING_VERSION={}", bindings.version);
    println!("cargo:rustc-env=CONTRACT_TOTAL={total}");
    println!("cargo:rustc-env=CONTRACT_IMPLEMENTED={implemented}");
    println!("cargo:rustc-env=CONTRACT_PARTIAL={partial}");
    println!("cargo:rustc-env=CONTRACT_GAPS={not_implemented}");
}
