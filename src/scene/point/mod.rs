pub mod points;

pub use points::*;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    /// Normalized
    pub color_rgb: [f32; 3],
    pub position: [f64; 3],
}
