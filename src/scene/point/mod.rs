pub mod points;

pub use gausplat_loader::source::colmap;
pub use points::*;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    /// Normalized
    pub color_rgb: [f32; 3],
    pub position: [f64; 3],
}

impl From<colmap::Point> for Point {
    fn from(point: colmap::Point) -> Self {
        Self {
            color_rgb: point.color_rgb_normalized(),
            position: point.position,
        }
    }
}
