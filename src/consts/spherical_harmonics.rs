use lazy_static::lazy_static;
use std::f64::consts::PI;

lazy_static! {
    /// The real coefficients of orthonormalized spherical harmonics from degree 0 to 3
    ///
    /// `[[f64; 1], [f64; 3], [f64; 5], [f64; 7]]`
    ///
    /// ## Example
    /// ```rust
    /// use gausplat_renderer::consts::spherical_harmonics::SH_C;
    ///
    /// let degree = 0;
    /// let order = 0;
    /// assert_eq!(SH_C[degree][degree + order], 0.28209479177387814);
    ///
    /// let degree = 3;
    /// let order = 2;
    /// assert_eq!(SH_C[degree][degree + order], 1.445305721320277);
    ///
    /// ```
    pub static ref SH_C: Vec<Vec<f64>> = vec![
        vec![(1.0 / 4.0 / PI).sqrt()],
        vec![
            -(3.0 / 4.0 / PI).sqrt(),
            (3.0 / 4.0 / PI).sqrt(),
            -(3.0 / 4.0 / PI).sqrt(),
        ],
        vec![
            (15.0 / 4.0 / PI).sqrt(),
            -(15.0 / 4.0 / PI).sqrt(),
            (5.0 / 16.0 / PI).sqrt(),
            -(15.0 / 4.0 / PI).sqrt(),
            (15.0 / 16.0 / PI).sqrt(),
        ],
        vec![
            -(35.0 / 32.0 / PI).sqrt(),
            (105.0 / 4.0 / PI).sqrt(),
            -(21.0 / 32.0 / PI).sqrt(),
            (7.0 / 16.0 / PI).sqrt(),
            -(21.0 / 32.0 / PI).sqrt(),
            (105.0 / 16.0 / PI).sqrt(),
            -(35.0 / 32.0 / PI).sqrt(),
        ],
    ];
}
