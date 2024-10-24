use lazy_static::lazy_static;
use std::f64::consts::PI;

/// The count of spherical harmonics coefficients
pub const SH_COUNT_MAX: usize = (SH_DEGREE_MAX as usize + 1).pow(2);

/// The maximum degree of spherical harmonics
pub const SH_DEGREE_MAX: u32 = 3;

lazy_static! {
    /// The real coefficients of orthonormalized spherical harmonics from degree 0 to 3
    ///
    /// ## Type
    ///
    /// ```ignore
    /// ([f64; 1], [f64; 3], [f64; 5], [f64; 7])
    /// ```
    ///
    /// ## Example
    /// ```rust
    /// use gausplat_renderer::spherical_harmonics::SH_C;
    ///
    /// assert_eq!(SH_C.0[0], 0.28209479177387814);
    /// assert_eq!(SH_C.3[3 + 2], 1.445305721320277);
    ///
    /// assert_eq!(
    ///     *SH_C,
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
    pub static ref SH_C: ([f64; 1], [f64; 3], [f64; 5], [f64; 7]) = (
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
    );
}
