pub mod points;

pub use gausplat_loader::source::colmap;
pub use points::*;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    /// Normalized RGB color.
    pub color_rgb: [f32; 3],
    pub position: [f64; 3],
}

impl From<colmap::Point> for Point {
    #[inline]
    fn from(point: colmap::Point) -> Self {
        Self {
            color_rgb: point.color_rgb_normalized(),
            position: point.position,
        }
    }
}

impl From<Point> for colmap::Point {
    #[inline]
    fn from(point: Point) -> Self {
        let color_rgb = point.color_rgb;
        let position = point.position;
        Self {
            color_rgb: [
                color_rgb[0].mul_add(255.0, 0.5).clamp(0.0, 255.0) as u8,
                color_rgb[1].mul_add(255.0, 0.5).clamp(0.0, 255.0) as u8,
                color_rgb[2].mul_add(255.0, 0.5).clamp(0.0, 255.0) as u8,
            ],
            position,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn from_and_into_colmap_point() {
        use super::*;

        let point = Point {
            color_rgb: [0.2509804, 0.5019608, 0.7529412],
            position: [1.0, 2.0, 3.0],
        };
        let colmap_point = colmap::Point {
            color_rgb: [64, 128, 192],
            position: [1.0, 2.0, 3.0],
        };

        assert_eq!(Point::from(colmap_point), point);
        assert_eq!(colmap::Point::from(point), colmap_point);
    }
}
