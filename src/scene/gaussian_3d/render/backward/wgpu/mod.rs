mod kernel;

pub use super::*;

use crate::consts::gaussian_3d::*;
use burn::backend::wgpu::SourceKernel;
use bytemuck::bytes_of;
use kernel::*;

pub fn render_gaussian_3d_scene(
    state: backward::RendererInput<Wgpu>,
    // [I_Y, I_X, 3]
    colors_rgb_2d_grad: <Wgpu as Backend>::FloatTensorPrimitive<3>,
) -> backward::RendererOutput<Wgpu> {
    let mut duration = std::time::Instant::now();

    let client = &state.colors_sh.client;
    let colors_sh_degree_max = state.options.colors_sh_degree_max;
    let device = &state.colors_sh.device;
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
    let point_count = state.colors_sh.shape.dims[0];
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
    let colors_rgb_3d_grad =
        Tensor::<Wgpu, 2>::zeros([point_count, 3 + 1], device)
            .into_primitive()
            .tensor();
    let conics_grad = Tensor::<Wgpu, 3>::zeros([point_count, 2, 2], device)
        .into_primitive()
        .tensor();
    let opacities_3d_grad = Tensor::<Wgpu, 2>::zeros([point_count, 1], device)
        .into_primitive()
        .tensor();
    let positions_2d_grad = Tensor::<Wgpu, 2>::zeros([point_count, 2], device)
        .into_primitive()
        .tensor();

    client.execute(
        Box::new(SourceKernel::new(
            Kernel1WgslSource,
            cubecl::CubeDim {
                x: tile_size_x,
                y: tile_size_y,
                z: 1,
            },
        )),
        cubecl::CubeCount::Static(tile_count_x, tile_count_y, 1),
        vec![
            arguments.binding(),
            state.conics.handle.to_owned().binding(),
            colors_rgb_2d_grad.handle.binding(),
            state.colors_rgb_3d.handle.binding(),
            state.opacities_3d.handle.binding(),
            state.point_indexes.handle.binding(),
            state.point_rendered_counts.handle.binding(),
            state.positions_2d.handle.binding(),
            state.tile_point_ranges.handle.binding(),
            state.transmittances.handle.binding(),
            colors_rgb_3d_grad.handle.to_owned().binding(),
            conics_grad.handle.to_owned().binding(),
            opacities_3d_grad.handle.to_owned().binding(),
            positions_2d_grad.handle.to_owned().binding(),
        ],
    );

    // client.sync(cubecl::client::SyncType::Wait);
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
    let colors_sh_grad = Tensor::<Wgpu, 3>::zeros([point_count, 16, 3], device)
        .into_primitive()
        .tensor();
    let positions_2d_grad_norm =
        Tensor::<Wgpu, 1>::zeros([point_count], device)
            .into_primitive()
            .tensor();
    let positions_3d_grad = Tensor::<Wgpu, 2>::zeros([point_count, 3], device)
        .into_primitive()
        .tensor();
    let rotations_grad = Tensor::<Wgpu, 2>::zeros([point_count, 4], device)
        .into_primitive()
        .tensor();
    let scalings_grad = Tensor::<Wgpu, 2>::zeros([point_count, 3], device)
        .into_primitive()
        .tensor();

    client.execute(
        Box::new(SourceKernel::new(
            Kernel2WgslSource,
            cubecl::CubeDim {
                x: GROUP_SIZE_X,
                y: GROUP_SIZE_Y,
                z: 1,
            },
        )),
        cubecl::CubeCount::Static(
            (point_count as u32 + GROUP_SIZE - 1) / GROUP_SIZE,
            1,
            1,
        ),
        vec![
            arguments.binding(),
            colors_rgb_3d_grad.handle.binding(),
            state.colors_sh.handle.binding(),
            state.conics.handle.binding(),
            conics_grad.handle.binding(),
            state.covariances_3d.handle.binding(),
            state.depths.handle.binding(),
            state.is_colors_rgb_3d_not_clamped.handle.binding(),
            positions_2d_grad.handle.binding(),
            state.positions_3d_in_normalized.handle.binding(),
            state.positions_3d_in_normalized_clamped.handle.binding(),
            state.radii.handle.binding(),
            state.rotations.handle.binding(),
            state.rotations_matrix.handle.binding(),
            state.rotation_scalings.handle.binding(),
            state.scalings.handle.binding(),
            state.transforms_2d.handle.binding(),
            state.view_directions.handle.binding(),
            state.view_offsets.handle.binding(),
            state.view_transform_rotation.handle.binding(),
            colors_sh_grad.handle.to_owned().binding(),
            positions_2d_grad_norm.handle.to_owned().binding(),
            positions_3d_grad.handle.to_owned().binding(),
            rotations_grad.handle.to_owned().binding(),
            scalings_grad.handle.to_owned().binding(),
        ],
    );

    // client.sync(cubecl::client::SyncType::Wait);
    println!("Duration (Backward 2): {:?}", duration.elapsed());

    backward::RendererOutput {
        colors_sh_grad,
        opacities_grad: opacities_3d_grad,
        positions_grad: positions_3d_grad,
        positions_2d_grad_norm,
        rotations_grad,
        scalings_grad,
    }
}
