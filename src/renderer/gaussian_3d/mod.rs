pub use crate::scene::gaussian_3d::*;
use gausplat_importer::scene;
pub use gausplat_importer::scene::sparse_view;
use kernel::RenderGaussian3dForward1Arguments;
use tensor::ops::IntTensor;
use tensor::Int;

use crate::error::Error;
use crate::function::{spherical_harmonics::SH_C, tensor_extensions::*};
use burn::backend::wgpu::{
    into_contiguous, kernel_wgsl, AutoGraphicsApi, JitBackend, Kernel,
    SourceKernel, WgpuRuntime, WorkGroup, WorkgroupSize,
};
use bytemuck::bytes_of;
use rayon::slice::ParallelSliceMut;
use std::iter::repeat_with;
use tensor::{ops::FloatTensor, Data};

pub trait Gaussian3dRenderer: backend::Backend {
    type Error;

    fn render_gaussian_3d(
        scene: &Gaussian3dScene<Self>,
        view: &sparse_view::View,
        colors_sh_degree_max: usize,
    ) -> Result<Gaussian3dRendererResultOk<Self>, Self::Error>;
}

#[derive(Clone, Debug)]
pub struct Gaussian3dRendererResultOk<B: backend::Backend> {
    // `[H, W, 3]`
    pub colors_rgb_2d: Tensor<B, 3>,
}

mod kernel {
    use burn::backend::wgpu::kernel_wgsl;
    use derive_new::new;

    kernel_wgsl!(
        RenderGaussian3dForward1,
        "./render_gaussian_3d_forward_1.wgsl"
    );

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct RenderGaussian3dForward1Arguments {
        pub colors_sh_degree_max: u32,
        pub filter_low_pass: f32,
        pub focal_length_x: f32,
        pub focal_length_y: f32,

        /// `I_X`
        pub image_size_x: u32,

        /// `I_Y`
        pub image_size_y: u32,

        /// `I_X / 2`
        pub image_size_half_x: f32,

        /// `I_Y / 2`
        pub image_size_half_y: f32,

        /// `P`
        pub point_count: u32,

        /// `I_X / T_X`
        pub tile_count_x: u32,

        /// `I_Y / T_Y`
        pub tile_count_y: u32,

        /// `T_X`
        pub tile_size_x: u32,

        /// `T_Y`
        pub tile_size_y: u32,

        pub view_bound_x: f32,
        pub view_bound_y: f32,
    }
}

pub type Wgpu = JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;

impl Gaussian3dRenderer for Wgpu {
    type Error = Error;

    fn render_gaussian_3d(
        scene: &Gaussian3dScene<Self>,
        view: &sparse_view::View,
        colors_sh_degree_max: usize,
    ) -> Result<Gaussian3dRendererResultOk<Self>, Self::Error> {
        const FILTER_LOW_PASS: f64 = 0.3;

        if colors_sh_degree_max > 3 {
            return Err(Error::Gaussian3dRenderer(format!(
                "colors_sh_degree_max should be no more than 3, but got {}",
                colors_sh_degree_max
            )));
        }

        println!(
            "SH_C: {:#?}",
            SH_C.iter()
                .map(|r| r.iter().map(|c| *c as f32).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );

        let mut duration = std::time::Instant::now();

        // T_X
        let tile_size_x = 16;

        // T_Y
        let tile_size_y = 16;

        debug_assert_ne!(tile_size_x, 0, "tile_size_x != 0");
        debug_assert_ne!(tile_size_y, 0, "tile_size_y != 0");

        // I_X
        let image_size_x = view.image_width as usize;

        // I_Y
        let image_size_y = view.image_height as usize;

        // I_X / T_X
        let tile_count_x = (image_size_x + tile_size_x - 1) / tile_size_x;

        // I_Y / T_Y
        let tile_count_y = (image_size_y + tile_size_y - 1) / tile_size_y;

        // [P, 16, 3]
        let colors_sh = scene.colors_sh();

        // [P, 1]
        let opacities = scene.opacities();

        // [P, 3]
        let positions = scene.positions();

        // [P, 4]
        let rotations = scene.rotations();

        // [P, 3]
        let scalings = scene.scalings();

        // P
        let point_count = positions.dims()[0];

        let device = positions.device();

        let field_of_view_x_half_tan = (view.field_of_view_x / 2.0).tan();
        let field_of_view_y_half_tan = (view.field_of_view_y / 2.0).tan();

        // ([P, 3], [P, 2, 2], [P], [P, 2], [P], [P])
        let (
            colors_rgb_3d,
            conics,
            depths,
            positions_2d_in_screen,
            radii,
            tile_touched_counts,
        ) = {
            // [P, 16, 3]
            let colors_sh =
                into_contiguous(colors_sh.to_owned().into_primitive());

            let client = colors_sh.client;

            let arguments = RenderGaussian3dForward1Arguments {
                colors_sh_degree_max: colors_sh_degree_max as u32,
                filter_low_pass: FILTER_LOW_PASS as f32,
                focal_length_x: (image_size_x as f64
                    / field_of_view_x_half_tan
                    / 2.0) as f32,
                focal_length_y: (image_size_y as f64
                    / field_of_view_y_half_tan
                    / 2.0) as f32,

                // I_X
                image_size_x: image_size_x as u32,

                // I_Y
                image_size_y: image_size_y as u32,

                // I_X / 2
                image_size_half_x: (image_size_x as f64 / 2.0) as f32,

                // I_Y / 2
                image_size_half_y: (image_size_y as f64 / 2.0) as f32,

                // P
                point_count: point_count as u32,

                // I_X / T_X
                tile_count_x: tile_count_x as u32,

                // I_Y / T_Y
                tile_count_y: tile_count_y as u32,

                // T_X
                tile_size_x: tile_size_x as u32,

                // T_Y
                tile_size_y: tile_size_y as u32,

                view_bound_x: (field_of_view_x_half_tan
                    * (FILTER_LOW_PASS + 1.0))
                    as f32,
                view_bound_y: (field_of_view_y_half_tan
                    * (FILTER_LOW_PASS + 1.0))
                    as f32,
            };
            let arguments_handle = client.create(bytes_of(&arguments));

            println!("arguments: {:#?}", arguments);

            // [P, 3]
            let positions =
                into_contiguous(positions.to_owned().into_primitive());

            // [P, 4]
            let rotations =
                into_contiguous(rotations.to_owned().into_primitive());

            // [P, 3]
            let scalings =
                into_contiguous(scalings.to_owned().into_primitive());

            // [3]
            let view_position_handle =
                client.create(bytes_of(&view.view_position.map(|c| c as f32)));

            // [4, 4]
            let view_transform_handle = client.create(bytes_of(
                &view.view_transform.map(|r| r.map(|c| c as f32)),
            ));

            // [P, 3]
            let colors_rgb_3d = into_contiguous(FloatTensor::<Self, 2>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count, 3].into(),
                client.empty(point_count * 3 * std::mem::size_of::<f32>()),
            ));

            // [P, 2, 2]
            let conics = into_contiguous(FloatTensor::<Self, 3>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count, 2, 2].into(),
                client.empty(point_count * 2 * 2 * std::mem::size_of::<f32>()),
            ));

            // [P]
            let depths = into_contiguous(FloatTensor::<Self, 1>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count].into(),
                client.empty(point_count * std::mem::size_of::<f32>()),
            ));

            // [P, 2]
            let positions_2d_in_screen =
                into_contiguous(FloatTensor::<Self, 2>::new(
                    client.to_owned(),
                    device.to_owned(),
                    [point_count, 2].into(),
                    client.empty(point_count * 2 * std::mem::size_of::<f32>()),
                ));

            // [P]
            let radii = into_contiguous(IntTensor::<Self, 1>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count].into(),
                client.empty(point_count * std::mem::size_of::<u32>()),
            ));

            // [P]
            let tile_touched_counts =
                into_contiguous(IntTensor::<Self, 1>::new(
                    client.to_owned(),
                    device.to_owned(),
                    [point_count].into(),
                    client.empty(point_count * std::mem::size_of::<u32>()),
                ));

            client.execute(
                Kernel::Custom(Box::new(SourceKernel::new(
                    kernel::RenderGaussian3dForward1::new(),
                    WorkGroup {
                        x: (point_count as u32 + 256 - 1) / 256,
                        y: 1,
                        z: 1,
                    },
                    WorkgroupSize { x: 256, y: 1, z: 1 },
                ))),
                &[
                    &arguments_handle,
                    &colors_sh.handle,
                    &positions.handle,
                    &rotations.handle,
                    &scalings.handle,
                    &view_position_handle,
                    &view_transform_handle,
                    &colors_rgb_3d.handle,
                    &conics.handle,
                    &depths.handle,
                    &positions_2d_in_screen.handle,
                    &radii.handle,
                    &tile_touched_counts.handle,
                ],
            );

            // ([P, 3], [P, 2, 2], [P], [P, 2], [P], [P])
            (
                Tensor::<Self, 2>::from_primitive(colors_rgb_3d),
                Tensor::<Self, 3>::from_primitive(conics),
                Tensor::<Self, 1>::from_primitive(depths),
                Tensor::<Self, 2>::from_primitive(positions_2d_in_screen),
                Tensor::<Self, 1, Int>::from_primitive(radii),
                Tensor::<Self, 1, Int>::from_primitive(tile_touched_counts),
            )
        };

        println!("wgpu: {:?}", duration.elapsed());

        duration = std::time::Instant::now();

        // [[START]]

        {
            // [4, 4]
            let view_transform = Tensor::<Self, 2>::from_data(
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
            let depths = positions_3d_in_view.2 + f32::EPSILON;

            // [P, 1]
            let is_in_frustum = depths.to_owned().greater_elem(0.2);

            // [P, 1]
            let positions_3d_in_view_x_normalized =
                positions_3d_in_view.0.to_owned().div(depths.to_owned());

            // [P, 1]
            let positions_3d_in_view_y_normalized =
                positions_3d_in_view.1.to_owned().div(depths.to_owned());

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
                        vec![
                            -r2_r2.to_owned() - r3_r3.to_owned() + 1.0,
                            r1_r2.to_owned() - r0_r3.to_owned(),
                            r1_r3.to_owned() + r0_r2.to_owned(),
                            r1_r2 + r0_r3,
                            -r1_r1.to_owned() - r3_r3 + 1.0,
                            r2_r3.to_owned() - r0_r1.to_owned(),
                            r1_r3 - r0_r2,
                            r2_r3 + r0_r1,
                            -r1_r1 - r2_r2 + 1.0,
                        ],
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

            // [P, 2, 2] (Symmetric)
            let covariances_2d = {
                let filter_low_pass = FILTER_LOW_PASS + 1.0;

                let focal_length_x =
                    image_size_x as f64 / field_of_view_x_half_tan / 2.0;
                let focal_length_y =
                    image_size_y as f64 / field_of_view_y_half_tan / 2.0;
                let bound_x = field_of_view_x_half_tan * filter_low_pass;
                let bound_y = field_of_view_y_half_tan * filter_low_pass;

                // [P, 1]
                let focal_lengths_x_normalized = Tensor::from_data(
                    Data::<f64, 2>::from([[focal_length_x]]).convert(),
                    &device,
                )
                .div(depths.to_owned());

                // [P, 1]
                let focal_lengths_y_normalized = Tensor::from_data(
                    Data::<f64, 2>::from([[focal_length_y]]).convert(),
                    &device,
                )
                .div(depths.to_owned());

                // [P, 1]
                let nulls = Tensor::zeros([point_count, 1], &device);

                // [P, 1]
                let x_normalized_clamped = positions_3d_in_view_x_normalized
                    .to_owned()
                    .clamp(-bound_x, bound_x);

                // [P, 1]
                let y_normalized_clamped = positions_3d_in_view_y_normalized
                    .to_owned()
                    .clamp(-bound_y, bound_y);

                // [P, 2, 3]
                let projections_jacobian = Tensor::cat(
                    vec![
                        focal_lengths_x_normalized.to_owned(),
                        nulls.to_owned(),
                        focal_lengths_x_normalized
                            .to_owned()
                            .mul(x_normalized_clamped)
                            .neg(),
                        nulls,
                        focal_lengths_y_normalized.to_owned(),
                        focal_lengths_y_normalized
                            .to_owned()
                            .mul(y_normalized_clamped)
                            .neg(),
                    ],
                    1,
                )
                .reshape([point_count, 2, 3]);

                // [P, 2, 3] = [P, 2, 3] * [1, 3, 3]
                let transforms = projections_jacobian
                    .matmul_batched(view_rotation.unsqueeze::<3>());

                // [P, 2, 2] = [P, 2, 3] * [P, 3, 3] * [P, 3, 2]
                let covariances = transforms
                    .to_owned()
                    .matmul_batched(covariances_3d.to_owned())
                    .matmul_batched(transforms.transpose());

                // [1, 2, 2]
                let filter_low_pass = Tensor::from_floats(
                    [[
                        [FILTER_LOW_PASS as f32, 0.0],
                        [0.0, FILTER_LOW_PASS as f32],
                    ]],
                    &device,
                );

                // [P, 2, 2] = [P, 2, 2] + [1, 2, 2]
                covariances.add(filter_low_pass)
            };

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

            // [P, 1]
            let conics_1 =
                -covariances_2d.1.to_owned() / covariances_2d_det.to_owned();

            // [P, 2, 2]
            let conics = Tensor::cat(
                vec![
                    covariances_2d.2.to_owned() / covariances_2d_det.to_owned(),
                    conics_1.to_owned(),
                    conics_1,
                    covariances_2d.0.to_owned() / covariances_2d_det.to_owned(),
                ],
                1,
            )
            .reshape([point_count, 2, 2]);

            // [P, 1] (No grad)
            let radii = {
                // [P, 1]
                let middle = (covariances_2d.0 + covariances_2d.2) / 2.0;

                // [P, 1]
                let bounds = (middle.to_owned() * middle.to_owned())
                    .sub(covariances_2d_det)
                    .sqrt()
                    .clamp_min(FILTER_LOW_PASS);

                // [P, 1]
                let extents_max = middle
                    .to_owned()
                    .add(bounds.to_owned())
                    .max_pair(middle.sub(bounds));

                // [P, 1]
                (extents_max.sqrt() * 3.0).int() + 1
            };

            // // [P, 1]
            let is_visible = radii.to_owned().greater_elem(0);

            // println!("5: {:?}", duration.elapsed());

            // duration = std::time::Instant::now();

            // // [P, 2] (View -> Clipped -> Screen)
            let positions_2d_in_screen = {
                let image_size_y_half = image_size_y as f64 / 2.0;
                let image_size_x_half = image_size_x as f64 / 2.0;

                Tensor::cat(
                    vec![
                        positions_3d_in_view_x_normalized
                            .mul_scalar(
                                image_size_x_half / field_of_view_x_half_tan,
                            )
                            .add_scalar(image_size_x_half - 0.5),
                        positions_3d_in_view_y_normalized
                            .mul_scalar(
                                image_size_y_half / field_of_view_y_half_tan,
                            )
                            .add_scalar(image_size_y_half - 0.5),
                    ],
                    1,
                )
            };

            // println!("6: {:?}", duration.elapsed());

            // duration = std::time::Instant::now();

            // [P, 1] * 4 (No grad)
            let (
                tiles_touched_x_max,
                tiles_touched_x_min,
                tiles_touched_y_max,
                tiles_touched_y_min,
            ) = {
                let tile_size_x = tile_size_x as u32;
                let tile_size_y = tile_size_y as u32;
                let tile_count_x = tile_count_x as u32;
                let tile_count_y = tile_count_y as u32;

                // [P, 1]
                let radii = radii.to_owned().float();

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

                // [P, 1] * 4
                (
                    positions_2d_in_screen
                        .0
                        .to_owned()
                        .add(radii.to_owned())
                        .add_scalar(tile_size_x - 1)
                        .div_scalar(tile_size_x)
                        .int()
                        .clamp(0, tile_count_x),
                    positions_2d_in_screen
                        .0
                        .sub(radii.to_owned())
                        .div_scalar(tile_size_x)
                        .int()
                        .clamp(0, tile_count_x),
                    positions_2d_in_screen
                        .1
                        .to_owned()
                        .add(radii.to_owned())
                        .add_scalar(tile_size_y - 1)
                        .div_scalar(tile_size_y)
                        .int()
                        .clamp(0, tile_count_y),
                    positions_2d_in_screen
                        .1
                        .sub(radii.to_owned())
                        .div_scalar(tile_size_y)
                        .int()
                        .clamp(0, tile_count_y),
                )
            };

            // [P, 1] (No grad)
            let tile_touched_counts = (tiles_touched_x_max.to_owned()
                - tiles_touched_x_min.to_owned())
                * (tiles_touched_y_max.to_owned()
                    - tiles_touched_y_min.to_owned());

            // [P] * 4
            let tiles_touched_x_max =
                tiles_touched_x_max.into_data().convert::<u32>().value;
            let tiles_touched_x_min =
                tiles_touched_x_min.into_data().convert::<u32>().value;
            let tiles_touched_y_max =
                tiles_touched_y_max.into_data().convert::<u32>().value;
            let tiles_touched_y_min =
                tiles_touched_y_min.into_data().convert::<u32>().value;

            // [P, 1]
            let is_tiles_touched =
                tile_touched_counts.to_owned().greater_elem(0);

            // println!("7: {:?}", duration.elapsed());

            // duration = std::time::Instant::now();

            // [P, 3]
            let colors_rgb_3d = {
                // [1, 3]
                let view_position = Tensor::<Self, 2>::from_data(
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

                // [P, 1, 1] * 8
                let mut x = Tensor::empty([1, 1, 1], &device);
                let mut y = Tensor::empty([1, 1, 1], &device);
                let mut z = Tensor::empty([1, 1, 1], &device);
                let mut xx = Tensor::empty([1, 1, 1], &device);
                let mut yy = Tensor::empty([1, 1, 1], &device);
                let mut zz = Tensor::empty([1, 1, 1], &device);
                let mut xy = Tensor::empty([1, 1, 1], &device);
                let mut zz_5_1 = Tensor::empty([1, 1, 1], &device);

                // [P, 1, 3] * ((D + 1) ^ 2)
                let mut colors_sh = colors_sh.iter_dim(1);

                // [P, 1, 3] (D >= 0)
                let mut colors_rgb_3d =
                    colors_sh.next().expect("colors_sh.dims()[1] >= 1")
                        * SH_C[0][0];

                // (D >= 1)
                if let Some(colors_sh) = colors_sh.next() {
                    y = directions.next().expect("directions.dims()[1] == 3");
                    colors_rgb_3d =
                        colors_rgb_3d + colors_sh * y.to_owned() * SH_C[1][0];
                }
                if let Some(colors_sh) = colors_sh.next() {
                    z = directions.next().expect("directions.dims()[1] == 3");
                    colors_rgb_3d =
                        colors_rgb_3d + colors_sh * z.to_owned() * SH_C[1][1];
                }
                if let Some(colors_sh) = colors_sh.next() {
                    x = directions.next().expect("directions.dims()[1] == 3");
                    colors_rgb_3d =
                        colors_rgb_3d + colors_sh * x.to_owned() * SH_C[1][2];
                }

                // (D >= 2)
                if let Some(colors_sh) = colors_sh.next() {
                    xy = x.to_owned() * y.to_owned();
                    colors_rgb_3d =
                        colors_rgb_3d + colors_sh * xy.to_owned() * SH_C[2][0];
                }
                if let Some(colors_sh) = colors_sh.next() {
                    let yz = y.to_owned() * z.to_owned();
                    colors_rgb_3d = colors_rgb_3d + colors_sh * yz * SH_C[2][1];
                }
                if let Some(colors_sh) = colors_sh.next() {
                    zz = z.to_owned() * z.to_owned();
                    let zz_3_1 = zz.to_owned() * 3.0 - 1.0;
                    colors_rgb_3d =
                        colors_rgb_3d + colors_sh * zz_3_1 * SH_C[2][2];
                }
                if let Some(colors_sh) = colors_sh.next() {
                    let xz = x.to_owned() * z.to_owned();
                    colors_rgb_3d = colors_rgb_3d + colors_sh * xz * SH_C[2][3];
                }
                if let Some(colors_sh) = colors_sh.next() {
                    xx = x.to_owned() * x.to_owned();
                    yy = y.to_owned() * y.to_owned();
                    let xx_yy = xx.to_owned() - yy.to_owned();
                    colors_rgb_3d =
                        colors_rgb_3d + colors_sh * xx_yy * SH_C[2][4];
                }

                // (D >= 3)
                if let Some(colors_sh) = colors_sh.next() {
                    let y_xx_3_yy =
                        y.to_owned() * (xx.to_owned() * 3.0 - yy.to_owned());
                    colors_rgb_3d =
                        colors_rgb_3d + colors_sh * y_xx_3_yy * SH_C[3][0];
                }
                if let Some(colors_sh) = colors_sh.next() {
                    let z_xy = z.to_owned() * xy;
                    colors_rgb_3d =
                        colors_rgb_3d + colors_sh * z_xy * SH_C[3][1];
                }
                if let Some(colors_sh) = colors_sh.next() {
                    zz_5_1 = zz * 5.0 - 1.0;
                    let y_zz_5_1 = y * zz_5_1.to_owned();
                    colors_rgb_3d =
                        colors_rgb_3d + colors_sh * y_zz_5_1 * SH_C[3][2];
                }
                if let Some(colors_sh) = colors_sh.next() {
                    let z_zz_5_3 = z.to_owned() * (zz_5_1.to_owned() - 2.0);
                    colors_rgb_3d =
                        colors_rgb_3d + colors_sh * z_zz_5_3 * SH_C[3][3];
                }
                if let Some(colors_sh) = colors_sh.next() {
                    let x_zz_5_1 = x.to_owned() * zz_5_1;
                    colors_rgb_3d =
                        colors_rgb_3d + colors_sh * x_zz_5_1 * SH_C[3][4];
                }
                if let Some(colors_sh) = colors_sh.next() {
                    let z_xx_yy = z * (xx.to_owned() - yy.to_owned());
                    colors_rgb_3d =
                        colors_rgb_3d + colors_sh * z_xx_yy * SH_C[3][5];
                }
                if let Some(colors_sh) = colors_sh.next() {
                    let x_xx_yy_3 = x * (xx - yy * 3.0);
                    colors_rgb_3d =
                        colors_rgb_3d + colors_sh * x_xx_yy_3 * SH_C[3][6];
                }

                // [P, 1, 3]
                colors_rgb_3d = colors_rgb_3d + 0.5;

                // [P, 3]
                colors_rgb_3d.clamp_min(0.0).squeeze::<2>(1)
            };

            // println!("8: {:?}", duration.elapsed());

            // duration = std::time::Instant::now();

            // [P, 1]
            let mask = Tensor::cat(
                vec![
                    is_in_frustum,
                    is_covariances_2d_invertible,
                    is_tiles_touched,
                ],
                1,
            )
            .all_dim(1);

            // [P, 3]
            let colors_rgb_3d = colors_rgb_3d.zeros_like().mask_where(
                mask.to_owned().expand([point_count, 3]),
                colors_rgb_3d,
            );

            // [P, 2, 2]
            let conics = conics.zeros_like().mask_where(
                mask.to_owned().unsqueeze_dim::<3>(2).expand([
                    point_count,
                    2,
                    2,
                ]),
                conics,
            );

            // [P, 1]
            let depths =
                depths.zeros_like().mask_where(mask.to_owned(), depths);

            // [P, 1]
            let opacities = opacities
                .zeros_like()
                .mask_where(mask.to_owned(), opacities);

            // [P, 2]
            let positions_2d_in_screen =
                positions_2d_in_screen.zeros_like().mask_where(
                    mask.to_owned().expand([point_count, 2]),
                    positions_2d_in_screen,
                );

            // [P, 1] (No grad)
            let radii = radii.zeros_like().mask_where(mask.to_owned(), radii);

            // [P, 1] (No grad)
            let tile_touched_counts = tile_touched_counts
                .zeros_like()
                .mask_where(mask.to_owned(), tile_touched_counts);

            println!("9: {:?}", duration.elapsed());

            duration = std::time::Instant::now();

            // [P], T (No grad)
            let (tile_touched_offsets, tile_touched_count) = {
                // [P]
                let counts = tile_touched_counts
                    .to_owned()
                    .into_data()
                    .convert::<u32>()
                    .value;

                let mut count = *counts.last().unwrap_or(&0) as usize;

                // [P]
                let offsets = counts
                    .into_iter()
                    .scan(0, |state, x| {
                        let y = *state;
                        *state += x;
                        Some(y)
                    })
                    .collect::<Vec<_>>();

                count += *offsets.last().unwrap_or(&0) as usize;

                // [P]
                (offsets, count)
            };

            println!("10: {:?}", duration.elapsed());
            println!("tile_touched_count: {:?}", tile_touched_count);

            duration = std::time::Instant::now();

            // [T] * 2 (No grad)
            let (point_keys, point_indexs) = {
                let tile_touched_offsets = tile_touched_offsets;

                // [P]
                let mask = Tensor::cat(vec![mask, is_visible], 1)
                    .all_dim(1)
                    .into_data()
                    .value;

                // [P] (f32 -> u32)
                let depths = bytemuck::cast_vec::<f32, u32>(
                    depths.into_data().convert().value,
                );

                // [T, 2]
                let mut keys_and_indexs = (0..point_count).fold(
                    vec![(0, 0); tile_touched_count],
                    |mut keys_and_indexs, index| {
                        if !mask[index] {
                            return keys_and_indexs;
                        }

                        let mut offset = tile_touched_offsets[index] as usize;

                        for tile_y in tiles_touched_y_min[index]
                            ..tiles_touched_y_max[index]
                        {
                            for tile_x in tiles_touched_x_min[index]
                                ..tiles_touched_x_max[index]
                            {
                                let tile = (tile_y * tile_count_x as u32
                                    + tile_x)
                                    as u64;
                                let depth = depths[index] as u64;
                                keys_and_indexs[offset] =
                                    (tile << 32 | depth, index as u32);

                                offset += 1;
                            }
                        }

                        keys_and_indexs
                    },
                );

                keys_and_indexs.par_sort_unstable_by_key(|(key, _)| *key);

                // [T] * 2
                keys_and_indexs.into_iter().unzip::<_, _, Vec<_>, Vec<_>>()
            };

            println!("11: {:?}", duration.elapsed());

            duration = std::time::Instant::now();

            // [(H / T_H) * (W / T_W), 2] (No grad)
            let tile_point_ranges = {
                // [(H / T_H) * (W / T_W), 2]
                let mut ranges = vec![0..0; tile_count_x * tile_count_y];

                if !point_keys.is_empty() {
                    let tile = (point_keys.first().unwrap() >> 32) as usize;
                    ranges[tile].start = 0;

                    let tile = (point_keys.last().unwrap() >> 32) as usize;
                    ranges[tile].end = tile_touched_count;
                }

                // [T]
                for point_offset in 1..tile_touched_count {
                    let tile_current =
                        (point_keys[point_offset] >> 32) as usize;
                    let tile_previous =
                        (point_keys[point_offset - 1] >> 32) as usize;
                    if tile_current != tile_previous {
                        ranges[tile_current].start = point_offset;
                        ranges[tile_previous].end = point_offset;
                    }
                }

                // [(H / T_H) * (W / T_W), 2]
                ranges
            };

            println!("12: {:?}", duration.elapsed());
            println!("tile_point_ranges.len(): {:?}", tile_point_ranges.len());

            duration = std::time::Instant::now();

            // [1, 1, P, 3]
            let colors_rgb_3d = colors_rgb_3d.unsqueeze::<4>();

            // [1, 1, P, 2, 2]
            let conics = conics.unsqueeze::<5>();

            // [1, 1, P, 1]
            let opacities = opacities.unsqueeze::<4>();

            // [1, 1, P, 2]
            let positions_2d_in_screen =
                positions_2d_in_screen.unsqueeze::<4>();

            // [T]
            let point_indexs = Tensor::<Self, 1, Int>::from_data(
                Data::new(point_indexs, [tile_touched_count].into()).convert(),
                &device,
            );

            // tile_touched_counts: [339405]

            println!(
                "point_indexs: {:?}",
                point_indexs.to_owned().slice([240..260]).into_data().value
            );
            println!(
                "tile_touched_counts: {:?}",
                tile_touched_counts.to_owned().sum().into_data().value
            );
        }

        // [H, W, 3] (Output)
        let mut colors_rgb_2d = Tensor::<Self, 3>::zeros(
            vec![image_size_y, image_size_x, 3],
            &device,
        );

        // let mut duration_mass = std::time::Duration::default();

        // // [(H / T_H) * (W / T_W)]
        // for (tile_index, tile_point_range) in
        //     tile_point_ranges.into_iter().enumerate()
        // {
        //     // R
        //     let tile_point_count =
        //         tile_point_range.end - tile_point_range.start;

        //     // [R]
        //     let tile_point_indexs =
        //         point_indexs.to_owned().slice([tile_point_range]);

        //     // [1, 1, R, 3]
        //     let tile_colors_rgb_3d = colors_rgb_3d
        //         .to_owned()
        //         .select(2, tile_point_indexs.to_owned());

        //     // [1, 1, R, 2, 2]
        //     let tile_conics =
        //         conics.to_owned().select(2, tile_point_indexs.to_owned());

        //     // [1, 1, R, 1]
        //     let tile_opacities =
        //         opacities.to_owned().select(2, tile_point_indexs.to_owned());

        //     // [1, 1, R, 2]
        //     let tile_positions_2d_in_screen = positions_2d_in_screen
        //         .to_owned()
        //         .select(2, tile_point_indexs.to_owned());

        //     let tile_pixel_x_min = tile_index % tile_count_x * tile_size_x;
        //     let tile_pixel_y_min = tile_index / tile_count_x * tile_size_y;
        //     let tile_pixel_x_max =
        //         (tile_pixel_x_min + tile_size_x).min(image_size_x);
        //     let tile_pixel_y_max =
        //         (tile_pixel_y_min + tile_size_y).min(image_size_y);

        //     // T_W (Clamped)
        //     let tile_size_x = tile_pixel_x_max - tile_pixel_x_min;

        //     // T_H (Clamped)
        //     let tile_size_y = tile_pixel_y_max - tile_pixel_y_min;

        //     // [T_H, T_W, 1, 2]
        //     let tile_pixel_positions_2d = Tensor::<Self, 2>::stack::<3>(
        //         vec![
        //             Tensor::arange(
        //                 tile_pixel_x_min as i64..tile_pixel_x_max as i64,
        //                 &device,
        //             )
        //             .float()
        //             .unsqueeze_dim::<2>(0)
        //             .repeat(0, tile_size_y),
        //             Tensor::arange(
        //                 tile_pixel_y_min as i64..tile_pixel_y_max as i64,
        //                 &device,
        //             )
        //             .float()
        //             .unsqueeze_dim::<2>(1)
        //             .repeat(1, tile_size_x),
        //         ],
        //         2,
        //     )
        //     .unsqueeze_dim::<4>(2);

        //     // [T_H, T_W, R, 1, 2] =
        //     // [T_H, T_W, R, 2] = [1, 1, R, 2] - [T_H, T_W, 1, 2]
        //     let tile_pixel_directions_2d = tile_positions_2d_in_screen
        //         .sub(tile_pixel_positions_2d)
        //         .unsqueeze_dim::<5>(3);

        //     // [T_H, T_W, R, 1] =
        //     // [T_H, T_W, R, 1, 1] =
        //     // [.., R, 1, 2] * [.., R, 2, 2] * [.., R, 2, 1]
        //     let tile_pixel_weights = tile_pixel_directions_2d
        //         .to_owned()
        //         .matmul_batched(tile_conics.expand([
        //             tile_size_y,
        //             tile_size_x,
        //             tile_point_count,
        //             2,
        //             2,
        //         ]))
        //         .matmul_batched(tile_pixel_directions_2d.transpose())
        //         .mul_scalar(-0.5)
        //         .exp()
        //         .squeeze::<4>(4);

        //     // [T_H, T_W, R, 1] = [1, 1, R, 1] * [T_H, T_W, R, 1]
        //     let tile_pixel_opacities =
        //         tile_opacities.mul(tile_pixel_weights).clamp_max(0.99);

        //     let duration_mass_elapsed = std::time::Instant::now();

        //     // [T_H, T_W, R, 1]
        //     let tile_pixel_transmittances = tile_pixel_opacities
        //         .to_owned()
        //         .neg()
        //         .add_scalar(1.0)
        //         .prod_cumulative_exclusive(2);

        //     duration_mass += duration_mass_elapsed.elapsed();

        //     // [T_H, T_W, 3] =
        //     // [T_H, T_W, R, 3] = [1, 1, R, 3] * [T_H, T_W, R, 1] * [T_H, T_W, R, 1]
        //     let tile_pixel_colors_rgb_3d = tile_colors_rgb_3d
        //         .mul(tile_pixel_opacities)
        //         .mul(tile_pixel_transmittances)
        //         .sum_dim(2)
        //         .squeeze::<3>(2);

        //     colors_rgb_2d = colors_rgb_2d.slice_assign(
        //         [
        //             tile_pixel_y_min..tile_pixel_y_max,
        //             tile_pixel_x_min..tile_pixel_x_max,
        //             0..3,
        //         ],
        //         tile_pixel_colors_rgb_3d,
        //     );
        // }

        // println!("13: {:?}", duration.elapsed());
        // println!("duration_mass: {:?}", duration_mass);

        Ok(Gaussian3dRendererResultOk { colors_rgb_2d })
    }
}
