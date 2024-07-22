use lazy_static::lazy_static;
use std::f32::consts::PI;

lazy_static! {
    /// The real coefficients of orthonormalized spherical harmonics from degree 0 to 4
    ///
    /// ## Example
    /// ```rust
    /// use gausplat_renderer::function::spherical_harmonics::SH_C;
    /// let degree = 3;
    /// let order = 2;
    /// let sh_c_p3_p2 = SH_C[degree][degree + order];
    /// assert_eq!(sh_c_p3_p2, 1.4453057);
    ///
    /// ```
    pub static ref SH_C: Vec<Vec<f32>> = vec![
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
        vec![
            (315.0 / 16.0 / PI).sqrt(),
            (315.0 / 32.0 / PI).sqrt(),
            (45.0 / 16.0 / PI).sqrt(),
            (45.0 / 32.0 / PI).sqrt(),
            (9.0 / 256.0 / PI).sqrt(),
            (45.0 / 32.0 / PI).sqrt(),
            (45.0 / 64.0 / PI).sqrt(),
            (315.0 / 32.0 / PI).sqrt(),
            (315.0 / 256.0 / PI).sqrt(),
        ]
    ];
}
