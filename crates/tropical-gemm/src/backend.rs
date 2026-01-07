use tropical_gemm_simd::SimdLevel;

/// Available backends for tropical GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Pure Rust portable implementation.
    Portable,
    /// SIMD-accelerated (AVX2, NEON, etc.).
    Simd,
}

impl Backend {
    /// Get the currently active backend based on CPU features.
    pub fn current() -> Self {
        match tropical_gemm_simd::simd_level() {
            SimdLevel::Scalar => Backend::Portable,
            _ => Backend::Simd,
        }
    }

    /// Get a description of the current SIMD capabilities.
    pub fn description() -> String {
        let level = tropical_gemm_simd::simd_level();
        match level {
            SimdLevel::Scalar => "Portable (no SIMD)".to_string(),
            SimdLevel::Sse2 => "x86-64 SSE2 (128-bit)".to_string(),
            SimdLevel::Avx => "x86-64 AVX (256-bit float)".to_string(),
            SimdLevel::Avx2 => "x86-64 AVX2 (256-bit)".to_string(),
            SimdLevel::Avx512 => "x86-64 AVX-512 (512-bit)".to_string(),
            SimdLevel::Neon => "ARM NEON (128-bit)".to_string(),
        }
    }
}

/// Get information about the library configuration.
pub fn version_info() -> String {
    format!(
        "tropical-gemm v{}\nBackend: {}\nSIMD Level: {:?}",
        env!("CARGO_PKG_VERSION"),
        Backend::description(),
        tropical_gemm_simd::simd_level()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_detection() {
        let backend = Backend::current();
        println!("Current backend: {:?}", backend);
        println!("Description: {}", Backend::description());
        println!("Version info:\n{}", version_info());
    }
}
