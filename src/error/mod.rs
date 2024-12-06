//! Error module.

pub use gausplat_loader::source::polygon;

use crate::spherical_harmonics::SH_DEGREE_MAX;

/// Error variants.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Error from invalid pixel count.
    #[error("Invalid pixel count: {0}. It should not be zero or excessively large.")]
    InvalidPixelCount(usize),
    /// Error from I/O.
    #[error("Io error: {0}")]
    Io(#[from] std::io::Error),
    /// Error from [`gausplat_loader`].
    #[error("Gausplat loader error: {0}")]
    Loader(#[from] gausplat_loader::error::Error),
    /// Error from mismatched polygon header.
    #[error(
        "Mismatched polygon header (3DGS PLY). \
        Please check the file again:\n--------\n{0}--------\n\
        * Note: You can reference the header here: \"\
        https://raw.githubusercontent.com/AsherJingkongChen/\
        gausplat-renderer/main/src/scene/gaussian_3d/header.3dgs.ply\""
    )]
    MismatchedPolygonHeader3DGS(Box<polygon::Header>),
    /// Error from mismatched point count.
    #[error("Mismatched point count: {0}. It should be {1}.")]
    MismatchedPointCount(usize, String),
    /// Error from mismatched tensor shape.
    #[error("Mismatched tensor shape: {0:?}. It should be {1:?}.")]
    MismatchedTensorShape(Vec<usize>, Vec<usize>),
    /// Error from unsupported spherical harmonics degree.
    #[error(
        "Unsupported spherical harmonics degree: {0}. \
        It should be less than {SH_DEGREE_MAX}."
    )]
    UnsupportedSphericalHarmonicsDegree(u32),
}
