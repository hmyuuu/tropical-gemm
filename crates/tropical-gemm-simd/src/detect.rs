/// CPU feature detection for runtime SIMD dispatch.

/// Available SIMD instruction sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    /// No SIMD, use scalar code.
    Scalar,
    /// SSE2 (128-bit, available on all x86-64).
    Sse2,
    /// AVX (256-bit float).
    Avx,
    /// AVX2 (256-bit integer + FMA).
    Avx2,
    /// AVX-512 (512-bit).
    Avx512,
    /// ARM NEON (128-bit).
    Neon,
}

impl SimdLevel {
    /// Detect the best available SIMD level at runtime.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return SimdLevel::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return SimdLevel::Avx2;
            }
            if is_x86_feature_detected!("avx") {
                return SimdLevel::Avx;
            }
            // SSE2 is always available on x86-64
            return SimdLevel::Sse2;
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on AArch64
            return SimdLevel::Neon;
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            SimdLevel::Scalar
        }
    }

    /// Get the SIMD width in bytes.
    pub fn width_bytes(&self) -> usize {
        match self {
            SimdLevel::Scalar => 1,
            SimdLevel::Sse2 | SimdLevel::Neon => 16,
            SimdLevel::Avx | SimdLevel::Avx2 => 32,
            SimdLevel::Avx512 => 64,
        }
    }

    /// Get the number of f32 elements that fit in one SIMD register.
    pub fn f32_width(&self) -> usize {
        self.width_bytes() / 4
    }

    /// Get the number of f64 elements that fit in one SIMD register.
    pub fn f64_width(&self) -> usize {
        self.width_bytes() / 8
    }
}

/// Global cached SIMD level.
static SIMD_LEVEL: std::sync::OnceLock<SimdLevel> = std::sync::OnceLock::new();

/// Get the detected SIMD level (cached).
pub fn simd_level() -> SimdLevel {
    *SIMD_LEVEL.get_or_init(SimdLevel::detect)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect() {
        let level = SimdLevel::detect();
        println!("Detected SIMD level: {:?}", level);

        // Should detect at least Scalar
        assert!(level >= SimdLevel::Scalar);

        // On x86-64, should detect at least SSE2
        #[cfg(target_arch = "x86_64")]
        assert!(level >= SimdLevel::Sse2);

        // On AArch64, should detect NEON
        #[cfg(target_arch = "aarch64")]
        assert_eq!(level, SimdLevel::Neon);
    }

    #[test]
    fn test_width() {
        assert_eq!(SimdLevel::Avx2.f32_width(), 8);
        assert_eq!(SimdLevel::Avx2.f64_width(), 4);
        assert_eq!(SimdLevel::Sse2.f32_width(), 4);
    }
}
