use lazy_static::lazy_static;
use std::f32::consts::PI;

lazy_static! {
    /// The real coefficients of orthonormalized spherical harmonics
    static ref SH_C: Vec<Vec<f32>> = vec![
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
