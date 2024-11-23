use crate::spherical_harmonics::SH_DEGREE_MAX;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Invalid pixel count: {0}. It should not be zero or excessively large.")]
    InvalidPixelCount(usize),

    #[error("Mismatched point count: {0}. It should be {1}.")]
    MismatchedPointCount(usize, String),

    #[error("Mismatched tensor shape: {0:?}. It should be {1:?}.")]
    MismatchedTensorShape(Vec<usize>, Vec<usize>),

    #[error("Unsupported spherical harmonics degree: {0}, It should be less than {SH_DEGREE_MAX}.")]
    UnsupportedSphericalHarmonicsDegree(u32),
}
