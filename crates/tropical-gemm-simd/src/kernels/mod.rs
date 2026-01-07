//! SIMD microkernel implementations.

pub mod avx2;
pub mod neon;
pub mod portable;

pub use avx2::{
    Avx2MaxMulF32Kernel, Avx2MaxPlusF32Kernel, Avx2MaxPlusF64Kernel, Avx2MinPlusF32Kernel,
};
pub use neon::{NeonMaxPlusF32Kernel, NeonMaxPlusF64Kernel, NeonMinPlusF32Kernel};
pub use portable::PortableKernel;
