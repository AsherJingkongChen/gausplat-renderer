mod kernel;

pub use super::*;

use crate::consts::render::*;
use burn::backend::wgpu::{Kernel, SourceKernel, WorkGroup, WorkgroupSize};
use bytemuck::bytes_of;
use kernel::*;

pub fn render_gaussian_3d_scene(
    state: backward::RendererInput<Wgpu>,
    grad: FloatTensor<Wgpu, 3>,
) -> backward::RendererOutput<Wgpu> {
    let mut duration = std::time::Instant::now();

    let client = state.colors_rgb_3d.client.to_owned();
    let colors_sh_degree_max = state.options.colors_sh_degree_max;
    let device = state.colors_rgb_3d.device.to_owned();
    let focal_length_x = state.focal_length_x as f32;
    let focal_length_y = state.focal_length_y as f32;
    // I_X
    let image_size_x = state.image_size_x;
    // I_Y
    let image_size_y = state.image_size_y;
    // I_X / 2.0
    let image_size_half_x = (image_size_x as f64 / 2.0) as f32;
    // I_Y / 2.0
    let image_size_half_y = (image_size_y as f64 / 2.0) as f32;
    // P
    let point_count = state.colors_rgb_3d.shape.dims[0];
    // T_X
    let tile_size_x = GROUP_SIZE_X;
    // T_Y
    let tile_size_y = GROUP_SIZE_Y;
    // I_X / T_X
    let tile_count_x = (image_size_x as u32 + tile_size_x - 1) / tile_size_x;
    // I_Y / T_Y
    let tile_count_y = (image_size_y as u32 + tile_size_y - 1) / tile_size_y;

    // Performing the backward pass #1

    let arguments = client.create(bytes_of(&Kernel1Arguments {
        image_size_x: image_size_x as u32,
        image_size_y: image_size_y as u32,
    }));
    // [P, 2, 2] (Symmetric)
    let conics = state.conics.handle;
    // [I_Y, I_X, 3]
    let colors_rgb_2d_grad = grad.handle;
    // [P, 3 (+ 1)] (0.0 ~ 1.0)
    let colors_rgb_3d = state.colors_rgb_3d.handle;
    // [P, 1] (0.0 ~ 1.0)
    let opacities_3d = state.opacities_3d.handle;
    // [T] (0 ~ P)
    let point_indexes = state.point_indexes.handle;
    // [I_Y, I_X]
    let point_rendered_counts = state.point_rendered_counts.handle;
    // [P, 2]
    let positions_2d = state.positions_2d.handle;
    // [(I_X / T_X) * (I_Y / T_Y), 2]
    let tile_point_ranges = state.tile_point_ranges.handle;
    // [I_Y, I_X] (0.0 ~ 1.0)
    let transmittances = state.transmittances.handle;
    // [P, 3 (+ 1)]
    let colors_rgb_3d_grad =
        Tensor::<Wgpu, 2>::zeros([point_count, (3 + 1)], &device)
            .into_primitive()
            .handle;
    // [P, 2, 2]
    let conics_grad = Tensor::<Wgpu, 3>::zeros([point_count, 2, 2], &device)
        .into_primitive()
        .handle;
    // [P, 1]
    let opacities_3d_grad = Tensor::<Wgpu, 2>::zeros([point_count, 1], &device)
        .into_primitive()
        .handle;
    // [P, 2]
    let positions_2d_grad = Tensor::<Wgpu, 2>::zeros([point_count, 2], &device)
        .into_primitive()
        .handle;

    client.execute(
        Kernel::Custom(Box::new(SourceKernel::new(
            Kernel1WgslSource,
            WorkGroup {
                x: tile_count_x,
                y: tile_count_y,
                z: 1,
            },
            WorkgroupSize {
                x: tile_size_x,
                y: tile_size_y,
                z: 1,
            },
        ))),
        &[
            &arguments,
            &conics,
            &colors_rgb_2d_grad,
            &colors_rgb_3d,
            &opacities_3d,
            &point_indexes,
            &point_rendered_counts,
            &positions_2d,
            &tile_point_ranges,
            &transmittances,
            &colors_rgb_3d_grad,
            &conics_grad,
            &opacities_3d_grad,
            &positions_2d_grad,
        ],
    );

    client.sync();
    println!("Duration (Backward 1): {:?}", duration.elapsed());
    duration = std::time::Instant::now();

    // Performing the backward pass #2

    let arguments = client.create(bytes_of(&Kernel2Arguments {
        colors_sh_degree_max,
        focal_length_x,
        focal_length_y,
        image_size_half_x,
        image_size_half_y,
        point_count: point_count as u32,
    }));
    // [P, 16, 3]
    let colors_sh = state.colors_sh.handle;
    // [P, 3 (+ 1), 3] (Symmetric)
    let covariances_3d = state.covariances_3d.handle;
    // [P]
    let depths = state.depths.handle;
    // [P, 3 (+ 1)] (0.0 ~ 1.0)
    let is_colors_rgb_3d_not_clamped = state.is_colors_rgb_3d_not_clamped.handle;
    // [P, 2]
    let positions_3d_in_normalized = state.positions_3d_in_normalized.handle;
    // [P, 2]
    let positions_3d_in_normalized_clamped =
        state.positions_3d_in_normalized_clamped.handle;
    // [P]
    let radii = state.radii.handle;
    // [P, 4]
    let rotations = state.rotations.handle;
    // [P, 3 (+ 1), 3]
    let rotations_matrix = state.rotations_matrix.handle;
    // [P, 3 (+ 1), 3]
    let rotation_scalings = state.rotation_scalings.handle;
    // [P, 3]
    let scalings = state.scalings.handle;
    // [P, 2, 3]
    let transforms_2d = state.transforms_2d.handle;
    // [P, 3 (+ 1)]
    let view_directions = state.view_directions.handle;
    // [P, 3 (+ 1)]
    let view_offsets = state.view_offsets.handle;
    // [3 (+ 1), 3]
    let view_transform_rotation = state.view_transform.handle;
    // [P, 16, 3]
    let colors_sh_grad =
        Tensor::<Wgpu, 3>::zeros([point_count, 16, 3], &device)
            .into_primitive()
            .handle;
    // [P]
    let positions_2d_grad_norm =
        Tensor::<Wgpu, 1>::zeros([point_count], &device)
            .into_primitive()
            .handle;
    // [P, 3]
    let positions_3d_grad = Tensor::<Wgpu, 2>::zeros([point_count, 3], &device)
        .into_primitive()
        .handle;
    // [P, 4]
    let rotations_grad = Tensor::<Wgpu, 2>::zeros([point_count, 4], &device)
        .into_primitive()
        .handle;
    // [P, 3]
    let scalings_grad = Tensor::<Wgpu, 2>::zeros([point_count, 3], &device)
        .into_primitive()
        .handle;

    client.execute(
        Kernel::Custom(Box::new(SourceKernel::new(
            Kernel2WgslSource,
            WorkGroup {
                x: (point_count as u32 + GROUP_SIZE - 1) / GROUP_SIZE,
                y: 1,
                z: 1,
            },
            WorkgroupSize {
                x: GROUP_SIZE_X,
                y: GROUP_SIZE_Y,
                z: 1,
            },
        ))),
        &[
            &arguments,
            &colors_rgb_3d_grad,
            &colors_sh,
            &conics,
            &conics_grad,
            &covariances_3d,
            &depths,
            &is_colors_rgb_3d_not_clamped,
            &positions_2d_grad,
            &positions_3d_in_normalized,
            &positions_3d_in_normalized_clamped,
            &radii,
            &rotations,
            &rotations_matrix,
            &rotation_scalings,
            &scalings,
            &transforms_2d,
            &view_directions,
            &view_offsets,
            &view_transform_rotation,
            &colors_sh_grad,
            &positions_2d_grad_norm,
            &positions_3d_grad,
            &rotations_grad,
            &scalings_grad,
        ],
    );

    // client.sync();
    println!("Duration (Backward 2): {:?}", duration.elapsed());

    backward::RendererOutput {
        // [P, 16, 3]
        colors_sh_grad: FloatTensor::<Wgpu, 3>::new(
            client.to_owned(),
            device.to_owned(),
            [point_count, 16, 3].into(),
            colors_sh_grad,
        ),
        // [P, 1]
        opacities_grad: FloatTensor::<Wgpu, 2>::new(
            client.to_owned(),
            device.to_owned(),
            [point_count, 1].into(),
            opacities_3d_grad,
        ),
        // [P, 3]
        positions_grad: FloatTensor::<Wgpu, 2>::new(
            client.to_owned(),
            device.to_owned(),
            [point_count, 3].into(),
            positions_3d_grad,
        ),
        // [P]
        positions_2d_grad_norm: FloatTensor::<Wgpu, 1>::new(
            client.to_owned(),
            device.to_owned(),
            [point_count].into(),
            positions_2d_grad_norm,
        ),
        // [P, 4]
        rotations_grad: FloatTensor::<Wgpu, 2>::new(
            client.to_owned(),
            device.to_owned(),
            [point_count, 4].into(),
            rotations_grad,
        ),
        // [P, 3]
        scalings_grad: FloatTensor::<Wgpu, 2>::new(
            client.to_owned(),
            device.to_owned(),
            [point_count, 3].into(),
            scalings_grad,
        ),
    }
}
