pub mod views;

pub use views::*;

/// A view in 3D space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct View {
    /// The horizontal field of view in radians.
    pub field_of_view_x: f64,
    /// The vertical field of view in radians.
    pub field_of_view_y: f64,
    /// Image height.
    pub image_height: u32,
    /// Image width.
    pub image_width: u32,
    /// View ID.
    pub view_id: u32,
    /// Position in world space.
    pub view_position: [f64; 3],
    /// Affine transformation from world space to view space.
    ///
    /// It is in **column-major order**, i.e., `M[col][row]`.
    ///
    /// # Format
    ///
    /// ```plaintext
    /// [R_v   | T_v]
    /// [...   | ...]
    /// [0 0 0 | 1  ]
    /// ```
    pub view_transform: [[f64; 4]; 4],
}

/// Linear transformations.
impl View {
    /// Returns the affine transformation matrix.
    ///
    /// It is in **column-major order**, i.e., `M[col][row]`.
    #[inline]
    pub const fn transform(
        rotation: &[[f64; 3]; 3],
        translation: &[f64; 3],
    ) -> [[f64; 4]; 4] {
        let r = rotation;
        let t = [translation];
        [
            [r[0][0], r[0][1], r[0][2], 0.0],
            [r[1][0], r[1][1], r[1][2], 0.0],
            [r[2][0], r[2][1], r[2][2], 0.0],
            [t[0][0], t[0][1], t[0][2], 1.0],
        ]
    }
}

/// Dimension operations
impl View {
    /// Returns the aspect ratio (`width / height`).
    #[inline]
    pub const fn aspect_ratio(&self) -> f32 {
        self.image_width as f32 / self.image_height as f32
    }

    /// Resizing the view to the maximum side length of `to`.
    pub fn resize_max(
        &mut self,
        to: u32,
    ) -> &mut Self {
        let ratio = self.aspect_ratio();
        if ratio > 1.0 {
            self.image_width = to;
            self.image_height = (to as f32 / ratio).ceil() as u32;
        } else {
            self.image_width = (to as f32 * ratio).ceil() as u32;
            self.image_height = to;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn transform() {
        use super::*;

        let target = [
            [
                0.9870946659543874,
                0.011754269038001336,
                0.1597058471183149,
                0.0000000000000000,
            ],
            [
                -0.000481623211642526,
                0.9975159094549839,
                -0.07043989227191047,
                0.0000000000000000,
            ],
            [
                -0.1601370927782764,
                0.0694539238889973,
                0.9846482945564589,
                0.0000000000000000,
            ],
            [
                0.129242027423,
                0.0000000000000000,
                -0.3424233862,
                1.0000000000000000,
            ],
        ];
        let output = View::transform(
            &[
                [0.9870946659543874, 0.011754269038001336, 0.1597058471183149],
                [
                    -0.000481623211642526,
                    0.9975159094549839,
                    -0.07043989227191047,
                ],
                [-0.1601370927782764, 0.0694539238889973, 0.9846482945564589],
            ],
            &[0.129242027423, 0.0, -0.3424233862],
        );
        assert_eq!(output, target);
    }

    #[test]
    fn resize_max() {
        use super::*;

        let mut view = View {
            image_width: 1920,
            image_height: 1080,
            ..Default::default()
        };
        view.resize_max(1080);
        assert_eq!(view.image_width, 1080);
        assert_eq!(view.image_height, 608);

        let mut view = View {
            image_width: 720,
            image_height: 1080,
            ..Default::default()
        };
        view.resize_max(1080);
        assert_eq!(view.image_width, 720);
        assert_eq!(view.image_height, 1080);
    }
}
