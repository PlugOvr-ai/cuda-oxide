/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::PathBuf;

/// Explicit backend knobs; replaces every `CUDA_OXIDE_*` env read inside the
/// backend. `run_pipeline` (mir-importer) builds one from the environment at
/// its own boundary. The experimental API builds one from typed compile
/// options without reading the environment.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BackendOptions {
    /// Hard target override (`llc -mcpu=`), e.g. `"sm_120"`.
    pub target_arch: Option<String>,
    /// Human-readable name for whatever set `target_arch`, used only to
    /// describe target provenance in diagnostics and errors (e.g.
    /// `"CUDA_OXIDE_TARGET"` for the env-driven rustc pipeline, or a
    /// caller-facing description for the standalone experimental API).
    ///
    /// Keep this in step with `target_arch`: whoever writes one writes the
    /// other, or a target error names a source the caller never used.
    pub target_arch_source: &'static str,
    /// Advisory local-GPU arch; used only when it satisfies detected features.
    pub device_arch_hint: Option<String>,
    /// Skip the `opt -O2` middle-end.
    pub no_opt: bool,
    /// Suppress `llc -fp-contract=fast` (fmul+fadd fusion to fma).
    pub no_fma: bool,
    /// Print progress and tool-selection notes to stderr.
    pub verbose: bool,
    /// Explicit `llc` binary (was `CUDA_OXIDE_LLC`).
    pub llc_override: Option<PathBuf>,
    /// Explicit `opt` binary (was `CUDA_OXIDE_OPT`).
    pub opt_override: Option<PathBuf>,
}

impl Default for BackendOptions {
    fn default() -> Self {
        Self {
            target_arch: None,
            target_arch_source: "CUDA_OXIDE_TARGET",
            device_arch_hint: None,
            no_opt: false,
            no_fma: false,
            verbose: false,
            llc_override: None,
            opt_override: None,
        }
    }
}

impl BackendOptions {
    /// Reads the historical `CUDA_OXIDE_*` variables; called by rustc-pipeline
    /// hosts, never by the backend itself. The only other env access in this
    /// crate is `CUDA_OXIDE_LLVM_LINK` in `llvm_tools::resolve_sibling_tool`
    /// (a per-toolchain tool override, not a compile option).
    pub fn from_env() -> Self {
        // A per-crate target, which is what makes one binary able to carry
        // kernels for two architectures.
        //
        // The device target is resolved once per rustc invocation, and cargo
        // runs one invocation per crate, so a crate is already the unit that
        // gets its own artifact bundle — `load_embedded_module` selects
        // between them by name at run time. What was missing was any way to
        // give those bundles *different* targets: `CUDA_OXIDE_TARGET` is one
        // process-wide value, so asking for `sm_110a` anywhere asked for it
        // everywhere, and the portable baseline stopped loading on the cards
        // it was the baseline for.
        //
        // `CUDA_OXIDE_TARGET_<CRATE_NAME>` (upper-cased, `-` as `_`) overrides
        // it for one crate. Cargo sets `CARGO_CRATE_NAME` per invocation, so
        // this needs no new plumbing and no build-system cooperation:
        //
        //     CUDA_OXIDE_TARGET_OXIDE_BLACKWELL=sm_110a cargo oxide build
        //
        // leaves every other crate on its own detected target.
        let per_crate = std::env::var("CARGO_CRATE_NAME").ok().and_then(|name| {
            let key = format!("CUDA_OXIDE_TARGET_{}", name.to_uppercase().replace('-', "_"));
            std::env::var(&key).ok().map(|value| (key, value))
        });
        let (target_arch_source, target_arch) = match per_crate {
            Some((key, value)) => (
                Box::leak(key.into_boxed_str()) as &'static str,
                Some(value),
            ),
            None => (
                "CUDA_OXIDE_TARGET",
                std::env::var("CUDA_OXIDE_TARGET").ok(),
            ),
        };
        Self {
            target_arch,
            target_arch_source,
            device_arch_hint: std::env::var("CUDA_OXIDE_DEVICE_ARCH").ok(),
            no_opt: std::env::var("CUDA_OXIDE_NO_OPT").is_ok(),
            no_fma: std::env::var("CUDA_OXIDE_NO_FMA").is_ok(),
            verbose: std::env::var("CUDA_OXIDE_VERBOSE").is_ok(),
            llc_override: std::env::var("CUDA_OXIDE_LLC").ok().map(PathBuf::from),
            opt_override: std::env::var("CUDA_OXIDE_OPT").ok().map(PathBuf::from),
        }
    }
}
