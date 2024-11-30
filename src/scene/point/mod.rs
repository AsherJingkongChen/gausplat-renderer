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

impl Into<colmap::Point> for Point {
    fn into(self) -> colmap::Point {
        colmap::Point {
            color_rgb: [
                self.color_rgb[0].mul_add(255.0, 0.5).clamp(0.0, 255.0) as u8,
                self.color_rgb[1].mul_add(255.0, 0.5).clamp(0.0, 255.0) as u8,
                self.color_rgb[2].mul_add(255.0, 0.5).clamp(0.0, 255.0) as u8,
            ],
            position: self.position,
        }
    }
}
