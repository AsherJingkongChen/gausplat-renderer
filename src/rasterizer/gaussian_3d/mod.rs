use core::f32;

pub use crate::scene::gaussian_3d::*;
pub use gausplat_importer::scene::sparse_view;

use crate::function::{spherical_harmonics::SH_C, tensor::TensorExtension};
use tensor::{Data, Int};

#[derive(Debug, Module)]
pub struct Gaussian3dRasterizer<B: backend::Backend> {
    pub scene: Gaussian3dScene<B>,
}

impl<B: backend::Backend> Gaussian3dRasterizer<B> {
    pub fn forward(
        &self,
        view: &sparse_view::View,
    ) {
        const FILTER_LOW_PASS: f32 = 0.3;

        let image_pixels_x = 16;
        let image_pixels_y = 16;
        debug_assert_ne!(image_pixels_x, 0, "image_pixels_x != 0");
        debug_assert_ne!(image_pixels_y, 0, "image_pixels_y != 0");

        let image_height = view.image_height as i64;
        let image_width = view.image_width as i64;
        let image_tiles_x = (image_width + image_pixels_x - 1) / image_pixels_x;
        let image_tiles_y =
            (image_height + image_pixels_y - 1) / image_pixels_y;

        // [P, (D + 1) ^ 2, 3]
        let colors_sh = self.scene.colors_sh();

        // [P, 1]
        let opacities = self.scene.opacities();

        // [P, 3]
        let positions = self.scene.positions();

        // [P, 4]
        let rotations = self.scene.rotations();

        // [P, 3]
        let scalings = self.scene.scalings();

        let device = positions.device();
        let point_count = positions.dims()[0];

        // Outputs
        let mut image_rendered =
            Tensor::<B, 3>::zeros(vec![image_height, image_width, 3], &device);
        let mut radii = Tensor::<B, 1, Int>::zeros([point_count], &device);
        let mut point_count_rendered = point_count;

        // Point-wise states
        let point_offsets = Tensor::<B, 1, Int>::zeros([point_count], &device);

        // Pixel-wise states
        let opacities_blended =
            Tensor::<B, 2>::zeros(vec![image_height, image_width], &device);
        let point_counts_rendered = Tensor::<B, 2, Int>::zeros(
            vec![image_height, image_width],
            &device,
        );
        let point_key_ranges = Tensor::<B, 3, Int>::zeros(
            vec![image_height, image_width, 2],
            &device,
        );

        // Sorting states
        let point_keys =
            Tensor::<B, 1, Int>::zeros([point_count_rendered], &device);
        let point_values =
            Tensor::<B, 1, Int>::zeros([point_count_rendered], &device);

        let mut duration = std::time::Instant::now();

        // [4, 4]
        let view_transform = Tensor::from_data(
            Data::<f64, 2>::from(view.view_transform).convert(),
            &device,
        );

        // [3, 3]
        let view_rotation = view_transform.to_owned().slice([0..3, 0..3]);

        // [P, 1] * 3
        let positions_3d_in_view = {
            // [3, 1]
            let view_translation = view_transform.slice([0..3, 3..4]);

            // [P, 1] * 3 = [P, 3] * [3, 3] + [1, 3]
            let mut positions_3d_in_view = positions
                .to_owned()
                .matmul(view_rotation.to_owned().transpose())
                .add(view_translation.transpose())
                .iter_dim(1);

            // [P, 1] * 3
            (
                positions_3d_in_view
                    .next()
                    .expect("positions_3d_in_view.dims()[1] == 3"),
                positions_3d_in_view
                    .next()
                    .expect("positions_3d_in_view.dims()[1] == 3"),
                positions_3d_in_view
                    .next()
                    .expect("positions_3d_in_view.dims()[1] == 3"),
            )
        };

        // [P, 1]
        let depths = positions_3d_in_view.2.clamp_min(f32::EPSILON);

        println!("1: {:?}", duration.elapsed());

        duration = std::time::Instant::now();

        // [P, 1]
        let is_in_frustum = depths.to_owned().greater_elem(0.2);

        // [P, 3, 3] (Symmetric)
        let covariances_3d = {
            // [P, 3, 3]
            let rotations = {
                // [P, 1] * 4
                let mut r = rotations.iter_dim(1);
                let r = (
                    r.next().expect("rotations.dims() == 4"),
                    r.next().expect("rotations.dims() == 4"),
                    r.next().expect("rotations.dims() == 4"),
                    r.next().expect("rotations.dims() == 4"),
                );

                // [P, 1] * 9
                let r0_r1 = r.0.to_owned() * r.1.to_owned() * 2.0;
                let r0_r2 = r.0.to_owned() * r.2.to_owned() * 2.0;
                let r0_r3 = r.0 * r.3.to_owned() * 2.0;
                let r1_r1 = r.1.to_owned() * r.1.to_owned() * 2.0;
                let r1_r2 = r.1.to_owned() * r.2.to_owned() * 2.0;
                let r1_r3 = r.1 * r.3.to_owned() * 2.0;
                let r2_r2 = r.2.to_owned() * r.2.to_owned() * 2.0;
                let r2_r3 = r.2 * r.3.to_owned() * 2.0;
                let r3_r3 = r.3.to_owned() * r.3 * 2.0;

                // [P, 3, 3]
                Tensor::cat(
                    [
                        -r2_r2.to_owned() - r3_r3.to_owned() + 1.0,
                        r1_r2.to_owned() - r0_r3.to_owned(),
                        r1_r3.to_owned() + r0_r2.to_owned(),
                        r1_r2 + r0_r3,
                        -r1_r1.to_owned() - r3_r3 + 1.0,
                        r2_r3.to_owned() - r0_r1.to_owned(),
                        r1_r3 - r0_r2,
                        r2_r3 + r0_r1,
                        -r1_r1 - r2_r2 + 1.0,
                    ]
                    .into(),
                    1,
                )
                .reshape([point_count, 3, 3])
            };

            // [P, 1, 3]
            let scalings = scalings.reshape([point_count, 1, 3]);

            // [P, 3, 3] = [P, 3, 3] * [P, 1, 3]
            let transforms = rotations * scalings;

            // [P, 3, 3] = [P, 3, 3] * [P, 3, 3]
            transforms.to_owned().matmul_batched(transforms.transpose())
        };

        println!("2: {:?}", duration.elapsed());

        duration = std::time::Instant::now();

        // [P, 1]
        let focal_lengths_x_normalized = Tensor::from_data(
            Data::<f64, 2>::from([[view.focal_length_x]]).convert(),
            &device,
        )
        .div(depths.to_owned());

        // [P, 1]
        let focal_lengths_y_normalized = Tensor::from_data(
            Data::<f64, 2>::from([[view.focal_length_y]]).convert(),
            &device,
        )
        .div(depths.to_owned());

        // [P, 2, 2] (Symmetric)
        let covariances_2d = {
            let filter_low_pass = FILTER_LOW_PASS as f64 + 1.0;
            let bound_x = image_width as f64 / view.focal_length_x / 2.0
                * filter_low_pass;
            let bound_y = image_height as f64 / view.focal_length_y / 2.0
                * filter_low_pass;

            // [P, 1]
            let nulls = Tensor::zeros([point_count, 1], &device);

            // [P, 1]
            let x_normalized = positions_3d_in_view
                .0
                .to_owned()
                .div(depths.to_owned())
                .clamp(-bound_x, bound_x);

            // [P, 1]
            let y_normalized = positions_3d_in_view
                .1
                .to_owned()
                .div(depths.to_owned())
                .clamp(-bound_y, bound_y);

            // [P, 2, 3]
            let projections_jacobian = Tensor::cat(
                [
                    focal_lengths_x_normalized.to_owned(),
                    nulls.to_owned(),
                    -focal_lengths_x_normalized.to_owned() * x_normalized,
                    nulls,
                    focal_lengths_y_normalized.to_owned(),
                    -focal_lengths_y_normalized.to_owned() * y_normalized,
                ]
                .into(),
                1,
            )
            .reshape([point_count, 2, 3]);

            // [P, 2, 3] = [P, 2, 3] * [1, 3, 3]
            let transforms = projections_jacobian
                .matmul_batched(view_rotation.unsqueeze::<3>());

            // [P, 2, 2] = [P, 2, 3] * [P, 3, 3] * [P, 3, 2]
            let covariances = transforms
                .to_owned()
                .matmul_batched(covariances_3d)
                .matmul_batched(transforms.transpose());

            // [1, 2, 2]
            let filter_low_pass = Tensor::from_floats(
                [[[FILTER_LOW_PASS, 0.0], [0.0, FILTER_LOW_PASS]]],
                &device,
            );

            // [P, 2, 2] = [P, 2, 2] + [1, 2, 2]
            covariances.add(filter_low_pass)
        };

        println!("3: {:?}", duration.elapsed());

        duration = std::time::Instant::now();

        // [P, 1] * 3
        let covariances_2d = (
            covariances_2d
                .to_owned()
                .slice([0..point_count, 0..1, 0..1])
                .squeeze::<2>(2),
            covariances_2d
                .to_owned()
                .slice([0..point_count, 0..1, 1..2])
                .squeeze::<2>(2),
            covariances_2d
                .slice([0..point_count, 1..2, 1..2])
                .squeeze::<2>(2),
        );

        // [P, 1]
        let covariances_2d_det = covariances_2d.0.to_owned()
            * covariances_2d.2.to_owned()
            - covariances_2d.1.to_owned() * covariances_2d.1.to_owned();

        // [P, 1]
        let is_covariances_2d_invertible =
            covariances_2d_det.to_owned().not_equal_elem(0.0);

        // [P, 3]
        let conics = Tensor::cat(
            [
                covariances_2d.2.to_owned() / covariances_2d_det.to_owned(),
                -covariances_2d.1 / covariances_2d_det.to_owned(),
                covariances_2d.0.to_owned() / covariances_2d_det.to_owned(),
            ]
            .into(),
            1,
        );

        println!("4: {:?}", duration.elapsed());

        duration = std::time::Instant::now();

        // [P, 1]
        let radii_max = {
            // [P, 1]
            let middle = (covariances_2d.0 + covariances_2d.2) / 2.0;

            // [P, 1]
            let bound = (middle.to_owned() * middle.to_owned())
                .sub(covariances_2d_det)
                .sqrt()
                .clamp_min(FILTER_LOW_PASS);

            // [P, 1]
            let extents_max = middle
                .to_owned()
                .add(bound.to_owned())
                .max_pair(middle.sub(bound));

            // [P, 1]
            (extents_max.sqrt() * 3.0).int() + 1
        };

        let is_radii_max_positive = radii_max.to_owned().greater_elem(0);

        println!("5: {:?}", duration.elapsed());

        duration = std::time::Instant::now();

        // [P, 2] (View -> Clipped + Normalized -> Screen)
        let positions_2d_in_screen = Tensor::cat(
            [
                positions_3d_in_view
                    .0
                    .mul(focal_lengths_x_normalized)
                    .add_scalar((image_width as f64 - 1.0) / 2.0),
                positions_3d_in_view
                    .1
                    .mul(focal_lengths_y_normalized)
                    .add_scalar((image_height as f64 - 1.0) / 2.0),
            ]
            .into(),
            1,
        );

        println!("6: {:?}", duration.elapsed());

        duration = std::time::Instant::now();

        // [P, 1]
        let tile_counts_touched = {
            // [P, 1]
            let radii_max = radii_max.to_owned().float();

            // [P, 1] * 2
            let mut positions_2d_in_screen =
                positions_2d_in_screen.to_owned().iter_dim(1);
            let positions_2d_in_screen = (
                positions_2d_in_screen
                    .next()
                    .expect("positions_2d_in_screen.dims()[1] == 2"),
                positions_2d_in_screen
                    .next()
                    .expect("positions_2d_in_screen.dims()[1] == 2"),
            );

            // [P, 1]
            let bounds_max_x = positions_2d_in_screen
                .0
                .to_owned()
                .add(radii_max.to_owned())
                .add_scalar(image_pixels_x - 1)
                .div_scalar(image_pixels_x)
                .int()
                .clamp(0, image_tiles_x);

            // [P, 1]
            let bounds_min_x = positions_2d_in_screen
                .0
                .sub(radii_max.to_owned())
                .div_scalar(image_pixels_x)
                .int()
                .clamp(0, image_tiles_x);

            // [P, 1]
            let bounds_max_y = positions_2d_in_screen
                .1
                .to_owned()
                .add(radii_max.to_owned())
                .add_scalar(image_pixels_y - 1)
                .div_scalar(image_pixels_y)
                .int()
                .clamp(0, image_tiles_y);

            // [P, 1]
            let bounds_min_y = positions_2d_in_screen
                .1
                .sub(radii_max.to_owned())
                .div_scalar(image_pixels_y)
                .int()
                .clamp(0, image_tiles_y);

            // [P, 1]
            (bounds_max_x - bounds_min_x) * (bounds_max_y - bounds_min_y)
        };

        // [P, 1]
        let is_tiles_touched = tile_counts_touched.to_owned().not_equal_elem(0);

        println!("7: {:?}", duration.elapsed());

        duration = std::time::Instant::now();

        // [P, 3]
        let colors_rgb = {
            // [1, 3]
            let view_position = Tensor::<B, 2>::from_data(
                Data::<f64, 2>::from([view.view_position]).convert(),
                &device,
            );

            // [P, 1, 1] * 3
            let mut directions = {
                // [P, 3, 1]
                let directions =
                    (positions - view_position).unsqueeze_dim::<3>(2);

                // [P, 1, 1]
                let directions_norm = directions
                    .to_owned()
                    .mul(directions.to_owned())
                    .sum_dim(1)
                    .sqrt();

                // [P, 1, 1] * 3
                (directions / directions_norm).iter_dim(1)
            };

            // [P, 1, 1] * 10
            let mut x = Tensor::empty([1, 1, 1], &device);
            let mut y = Tensor::empty([1, 1, 1], &device);
            let mut z = Tensor::empty([1, 1, 1], &device);
            let mut xx = Tensor::empty([1, 1, 1], &device);
            let mut yy = Tensor::empty([1, 1, 1], &device);
            let mut zz = Tensor::empty([1, 1, 1], &device);
            let mut xy = Tensor::empty([1, 1, 1], &device);
            let mut xz = Tensor::empty([1, 1, 1], &device);
            let mut yz = Tensor::empty([1, 1, 1], &device);
            let mut zz_5_1 = Tensor::empty([1, 1, 1], &device);

            // [P, 1, 3] (Iterator::Item)
            let mut colors_sh = colors_sh.iter_dim(1);

            // [P, 1, 3]
            let mut colors_rgb =
                colors_sh.next().expect("colors_sh.dims()[1] >= 1")
                    * SH_C[0][0];

            if let Some(colors_sh) = colors_sh.next() {
                y = directions.next().expect("directions.dims()[1] == 3");
                colors_rgb = colors_rgb + colors_sh * y.to_owned() * SH_C[1][0];
            }
            if let Some(colors_sh) = colors_sh.next() {
                z = directions.next().expect("directions.dims()[1] == 3");
                colors_rgb = colors_rgb + colors_sh * z.to_owned() * SH_C[1][1];
            }
            if let Some(colors_sh) = colors_sh.next() {
                x = directions.next().expect("directions.dims()[1] == 3");
                colors_rgb = colors_rgb + colors_sh * x.to_owned() * SH_C[1][2];
            }
            if let Some(colors_sh) = colors_sh.next() {
                xy = x.to_owned() * y.to_owned();
                colors_rgb =
                    colors_rgb + colors_sh * xy.to_owned() * SH_C[2][0];
            }
            if let Some(colors_sh) = colors_sh.next() {
                yz = y.to_owned() * z.to_owned();
                colors_rgb =
                    colors_rgb + colors_sh * yz.to_owned() * SH_C[2][1];
            }
            if let Some(colors_sh) = colors_sh.next() {
                zz = z.to_owned() * z.to_owned();
                let zz_3_1 = zz.to_owned() * 3.0 - 1.0;
                colors_rgb = colors_rgb + colors_sh * zz_3_1 * SH_C[2][2];
            }
            if let Some(colors_sh) = colors_sh.next() {
                xz = x.to_owned() * z.to_owned();
                colors_rgb =
                    colors_rgb + colors_sh * xz.to_owned() * SH_C[2][3];
            }
            if let Some(colors_sh) = colors_sh.next() {
                xx = x.to_owned() * x.to_owned();
                yy = y.to_owned() * y.to_owned();
                let xx_yy = xx.to_owned() - yy.to_owned();
                colors_rgb = colors_rgb + colors_sh * xx_yy * SH_C[2][4];
            }
            if let Some(colors_sh) = colors_sh.next() {
                let y_xx_3_yy =
                    y.to_owned() * (xx.to_owned() * 3.0 - yy.to_owned());
                colors_rgb = colors_rgb + colors_sh * y_xx_3_yy * SH_C[3][0];
            }
            if let Some(colors_sh) = colors_sh.next() {
                let z_xy = z.to_owned() * xy;
                colors_rgb = colors_rgb + colors_sh * z_xy * SH_C[3][1];
            }
            if let Some(colors_sh) = colors_sh.next() {
                zz_5_1 = zz * 5.0 - 1.0;
                let y_zz_5_1 = y * zz_5_1.to_owned();
                colors_rgb = colors_rgb + colors_sh * y_zz_5_1 * SH_C[3][2];
            }
            if let Some(colors_sh) = colors_sh.next() {
                let z_zz_5_3 = z.to_owned() * (zz_5_1.to_owned() - 2.0);
                colors_rgb = colors_rgb + colors_sh * z_zz_5_3 * SH_C[3][3];
            }
            if let Some(colors_sh) = colors_sh.next() {
                let x_zz_5_1 = x.to_owned() * zz_5_1;
                colors_rgb = colors_rgb + colors_sh * x_zz_5_1 * SH_C[3][4];
            }
            if let Some(colors_sh) = colors_sh.next() {
                let z_xx_yy = z * (xx.to_owned() - yy.to_owned());
                colors_rgb = colors_rgb + colors_sh * z_xx_yy * SH_C[3][5];
            }
            if let Some(colors_sh) = colors_sh.next() {
                let x_xx_yy_3 = x * (xx - yy * 3.0);
                colors_rgb = colors_rgb + colors_sh * x_xx_yy_3 * SH_C[3][6];
            }

            // [P, 1, 3]
            colors_rgb = colors_rgb + 0.5;
            // let is_colors_rgb_clamped = colors_rgb.to_owned().lower_elem(0.0);

            // [P, 3]
            colors_rgb.clamp_min(0.0).squeeze::<2>(1)
        };

        println!("8: {:?}", duration.elapsed());

        let colors_rgb = colors_rgb;
        let depths = depths;
        let radii_max = radii_max;
        let positions_2d_in_screen = positions_2d_in_screen;
        let conics = conics;
        let opacities = opacities;
        let tile_counts_touched = tile_counts_touched;
    }
}
