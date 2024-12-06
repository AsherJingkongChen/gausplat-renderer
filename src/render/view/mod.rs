pub mod views;

pub use views::*;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct View {
    pub field_of_view_x: f64,
    pub field_of_view_y: f64,
    pub image_height: u32,
    pub image_width: u32,
    pub view_id: u32,

    /// The view position in world space.
    pub view_position: [f64; 3],

    /// The affine transformation matrix mapping
    /// from world space to view space **(in column-major order)**.
    ///
    /// ## Details
    ///
    /// ```plaintext
    /// Tr_v = [R_v   | T_v]
    ///        [0 0 0 | 1  ]
    /// ```
    pub view_transform: [[f64; 4]; 4],
}

/// Linear transformations for 3D views.
impl View {
    /// ## Arguments
    ///
    /// * `rotation_to_view` - A 3D rotation matrix mapping
    ///   from world space to view space **(in column-major order)**.
    ///
    /// * `translation_to_view` - A 3D translation vector mapping
    ///   from world space to view space.
    ///
    /// ## Returns
    ///
    /// A 3D view position in world space **(in column-major order)**.
    ///
    /// ## Details
    ///
    /// ```plaintext
    /// * P_w is the view position in world space.
    /// * P_v is the view position in view space, which is the origin.
    /// * R_v is the rotation matrix mapping from world space to view space.
    /// * T_v is the translation vector mapping from world space to view space.
    ///
    /// P_v = 0 = R_v * P_w + T_v
    /// P_w = -R_v^t * T_v
    /// ```
    pub fn position(
        rotation_to_view: &[[f64; 3]; 3],
        translation_to_view: &[f64; 3],
    ) -> [f64; 3] {
        let r = rotation_to_view;
        let t = translation_to_view;
        [
            -r[0][0] * t[0] - r[0][1] * t[1] - r[0][2] * t[2],
            -r[1][0] * t[0] - r[1][1] * t[1] - r[1][2] * t[2],
            -r[2][0] * t[0] - r[2][1] * t[1] - r[2][2] * t[2],
        ]
    }

    /// ## Arguments
    ///
    /// * `quaternion_normalized` - A normalized Hamiltonian quaternion
    ///   **(in scalar-first order, i.e., `[w, x, y, z]`)**.
    ///
    /// ## Returns
    ///
    /// A 3D rotation matrix **(in column-major order)**.
    pub fn rotation(quaternion_normalized: &[f64; 4]) -> [[f64; 3]; 3] {
        let [w, x, y, z] = quaternion_normalized;
        let w_x = w * x * 2.0;
        let w_y = w * y * 2.0;
        let w_z = w * z * 2.0;
        let x_x = x * x * 2.0;
        let x_y = x * y * 2.0;
        let x_z = x * z * 2.0;
        let y_y = y * y * 2.0;
        let y_z = y * z * 2.0;
        let z_z = z * z * 2.0;
        [
            [1.0 - y_y - z_z, x_y + w_z, x_z - w_y],
            [x_y - w_z, 1.0 - x_x - z_z, y_z + w_x],
            [x_z + w_y, y_z - w_x, 1.0 - x_x - y_y],
        ]
    }

    /// ## Arguments
    ///
    /// * `rotation_to_view` - A 3D rotation matrix mapping
    ///   from world space to view space **(in column-major order)**.
    ///
    /// * `translation_to_view` - A 3D translation vector mapping
    ///   from world space to view space.
    ///
    /// ## Returns
    ///
    /// A 3D affine transformation matrix mapping
    ///   from world space to view space **(in column-major order)**.
    ///
    /// ## Details
    ///
    /// ```plaintext
    /// Tr_v = [R_v   | T_v]
    ///        [0 0 0 | 1  ]
    /// ```
    pub fn transform(
        rotation_to_view: &[[f64; 3]; 3],
        translation_to_view: &[f64; 3],
    ) -> [[f64; 4]; 4] {
        let r = rotation_to_view;
        let t = translation_to_view;
        [
            [r[0][0], r[0][1], r[0][2], 0.0],
            [r[1][0], r[1][1], r[1][2], 0.0],
            [r[2][0], r[2][1], r[2][2], 0.0],
            [t[0], t[1], t[2], 1.0],
        ]
    }
}

/// Dimension operations
impl View {
    #[inline]
    pub fn aspect_ratio(&self) -> f32 {
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
    fn position() {
        use super::*;

        let quaternion_normalized = [
            0.9928923624805012,
            0.006208227229002722,
            -0.11837120574960786,
            0.010699163142319695,
        ];
        let translation_to_view =
            [2.1400970808418642, 0.18616441825409558, 4.726341984431894];

        let view_position = View::position(
            &View::rotation(&quaternion_normalized),
            &translation_to_view,
        );
        assert_eq!(
            view_position,
            [-3.194916373379071, -0.18378876753171225, -4.087996124741175]
        );
    }

    #[test]
    fn rotation() {
        use super::*;

        let quaternion_normalized = [
            0.9898446088507,
            0.0712377208478,
            -0.122993928961,
            -0.002308873358,
        ];

        let rotation = View::rotation(&quaternion_normalized);
        assert_eq!(
            rotation,
            [
                [
                    0.9697343250851065000,
                    -0.022094466046466348,
                    0.2431607972553234400,
                ],
                [
                    -0.012952762662725100,
                    0.9898397124644553000,
                    0.1415965026675595000,
                ],
                [
                    -0.243818712758323920,
                    -0.140460593044464320,
                    0.9595953611342949000,
                ]
            ]
        );
    }

    #[test]
    fn transform() {
        use super::*;

        let quaternion_normalized = [
            0.9961499472928047,
            -0.03510862409346388,
            -0.08026977784966388,
            0.003070795788047984,
        ];
        let translation_to_view = [0.129242027423, 0.0, -0.3424233862];

        let view_transform = View::transform(
            &View::rotation(&quaternion_normalized),
            &translation_to_view,
        );
        assert_eq!(
            view_transform,
            [
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
            ]
        );
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
