use std::{f64::consts::PI, sync::LazyLock};

/// The count of spherical harmonics coefficients
pub const SH_COUNT_MAX: usize = (SH_DEGREE_MAX as usize + 1).pow(2);

/// The maximum degree of spherical harmonics
pub const SH_DEGREE_MAX: u32 = 3;

/// The real coefficients of orthonormalized spherical harmonics from degree 0 to 3
///
/// ## Examples
///
/// ```rust
/// use gausplat_renderer::spherical_harmonics::SH_COEF;
///
/// assert_eq!(SH_COEF.0[0], 0.28209479177387814);
/// assert_eq!(SH_COEF.3[3 + 2], 1.445305721320277);
///
/// assert_eq!(
///     *SH_COEF,
///     (
///         [0.28209479177387814],
///         [
///             -0.4886025119029199,
///             0.4886025119029199,
///             -0.4886025119029199,
///         ],
///         [
///             1.0925484305920792,
///             -1.0925484305920792,
///             0.31539156525252005,
///             -1.0925484305920792,
///             0.5462742152960396,
///         ],
///         [
///             -0.5900435899266435,
///             2.890611442640554,
///             -0.4570457994644658,
///             0.3731763325901154,
///             -0.4570457994644658,
///             1.445305721320277,
///             -0.5900435899266435,
///         ],
///     )
/// );
///
/// ```
#[allow(clippy::type_complexity)]
pub static SH_COEF: LazyLock<([f64; 1], [f64; 3], [f64; 5], [f64; 7])> =
    LazyLock::new(|| {
        (
            [(1.0 / 4.0 / PI).sqrt()],
            [
                -(3.0 / 4.0 / PI).sqrt(),
                (3.0 / 4.0 / PI).sqrt(),
                -(3.0 / 4.0 / PI).sqrt(),
            ],
            [
                (15.0 / 4.0 / PI).sqrt(),
                -(15.0 / 4.0 / PI).sqrt(),
                (5.0 / 16.0 / PI).sqrt(),
                -(15.0 / 4.0 / PI).sqrt(),
                (15.0 / 16.0 / PI).sqrt(),
            ],
            [
                -(35.0 / 32.0 / PI).sqrt(),
                (105.0 / 4.0 / PI).sqrt(),
                -(21.0 / 32.0 / PI).sqrt(),
                (7.0 / 16.0 / PI).sqrt(),
                -(21.0 / 32.0 / PI).sqrt(),
                (105.0 / 16.0 / PI).sqrt(),
                -(35.0 / 32.0 / PI).sqrt(),
            ],
        )
    });
