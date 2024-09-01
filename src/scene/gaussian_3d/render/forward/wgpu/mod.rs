mod kernel;
mod point;

pub use super::*;

use crate::preset::render::*;
use backend::Wgpu;
use burn::backend::wgpu::{into_contiguous, SourceKernel};
use bytemuck::{bytes_of, cast_slice, cast_slice_mut};
use kernel::*;
use point::*;
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

pub fn render_gaussian_3d_scene(
    input: forward::RenderInput<Wgpu>,
    view: &View,
    options: RenderOptions,
) -> forward::RenderOutput<Wgpu> {
    // Specifying the parameters

    #[cfg(debug_assertions)]
    {
        log::debug!(
            target: "gausplat_renderer::scene",
            "Gaussian3dRenderer::<Wgpu>::render_forward",
        );
    }

    let colors_sh_degree_max = options.colors_sh_degree_max;
    let field_of_view_x_half_tan = (view.field_of_view_x / 2.0).tan();
    let field_of_view_y_half_tan = (view.field_of_view_y / 2.0).tan();
    let filter_low_pass = FILTER_LOW_PASS as f32;
    // I_X
    let image_size_x = view.image_width as usize;
    // I_Y
    let image_size_y = view.image_height as usize;
    let focal_length_x =
        (image_size_x as f64 / field_of_view_x_half_tan / 2.0) as f32;
    let focal_length_y =
        (image_size_y as f64 / field_of_view_y_half_tan / 2.0) as f32;
    // I_X / 2.0
    let image_size_half_x = (image_size_x as f64 / 2.0) as f32;
    // I_Y / 2.0
    let image_size_half_y = (image_size_y as f64 / 2.0) as f32;
    // T_X
    let tile_size_x = GROUP_SIZE_X;
    // T_Y
    let tile_size_y = GROUP_SIZE_Y;
    // I_X / T_X
    let tile_count_x = (image_size_x as u32 + tile_size_x - 1) / tile_size_x;
    // I_Y / T_Y
    let tile_count_y = (image_size_y as u32 + tile_size_y - 1) / tile_size_y;
    // (I_X / T_X) * (I_Y / T_Y)
    let tile_count = (tile_count_x * tile_count_y) as usize;
    let view_bound_x =
        (field_of_view_x_half_tan * (FILTER_LOW_PASS + 1.0)) as f32;
    let view_bound_y =
        (field_of_view_y_half_tan * (FILTER_LOW_PASS + 1.0)) as f32;
    // [P, 16, 3]
    let colors_sh = into_contiguous(input.colors_sh);
    let client = &colors_sh.client;
    let device = &colors_sh.device;
    // P
    let point_count = colors_sh.shape.dims[0];

    // Performing the forward pass #1

    let arguments = client.create(bytes_of(&Kernel1Arguments {
        colors_sh_degree_max,
        filter_low_pass,
        focal_length_x,
        focal_length_y,
        image_size_x: image_size_x as u32,
        image_size_y: image_size_y as u32,
        image_size_half_x,
        image_size_half_y,
        point_count: point_count as u32,
        tile_count_x,
        tile_count_y,
        tile_size_x,
        tile_size_y,
        view_bound_x,
        view_bound_y,
    }));
    // [P, 1]
    let opacities_3d = into_contiguous(input.opacities);
    // [P, 3]
    let positions_3d = into_contiguous(input.positions);
    // [P, 4]
    let rotations = into_contiguous(input.rotations);
    // [P, 3]
    let scalings = into_contiguous(input.scalings);
    // [3]
    let view_position =
        Tensor::<Wgpu, 1>::from_data(view.view_position, device)
            .into_primitive()
            .tensor();
    // [4, 4]
    let view_transform =
        Tensor::<Wgpu, 2>::from_data(view.view_transform, device)
            .into_primitive()
            .tensor();
    let colors_rgb_3d = Tensor::<Wgpu, 2>::empty([point_count, 3 + 1], device)
        .into_primitive()
        .tensor();
    let conics = Tensor::<Wgpu, 3>::empty([point_count, 2, 2], device)
        .into_primitive()
        .tensor();
    let covariances_3d =
        Tensor::<Wgpu, 3>::empty([point_count, 3 + 1, 3], device)
            .into_primitive()
            .tensor();
    let depths = Tensor::<Wgpu, 1>::empty([point_count], device)
        .into_primitive()
        .tensor();
    let is_colors_rgb_3d_not_clamped =
        Tensor::<Wgpu, 2>::empty([point_count, 3 + 1], device)
            .into_primitive()
            .tensor();
    let positions_2d = Tensor::<Wgpu, 2>::empty([point_count, 2], device)
        .into_primitive()
        .tensor();
    let positions_3d_in_normalized =
        Tensor::<Wgpu, 2>::empty([point_count, 2], device)
            .into_primitive()
            .tensor();
    let positions_3d_in_normalized_clamped =
        Tensor::<Wgpu, 2>::empty([point_count, 2], device)
            .into_primitive()
            .tensor();
    let radii =
        Tensor::<Wgpu, 1, Int>::zeros([point_count], device).into_primitive();
    let rotations_matrix =
        Tensor::<Wgpu, 3>::empty([point_count, 3 + 1, 3], device)
            .into_primitive()
            .tensor();
    let rotation_scalings =
        Tensor::<Wgpu, 3>::empty([point_count, 3 + 1, 3], device)
            .into_primitive()
            .tensor();
    let tile_touched_counts =
        Tensor::<Wgpu, 1, Int>::zeros([point_count], device).into_primitive();
    let tiles_touched_max =
        Tensor::<Wgpu, 2, Int>::empty([point_count, 2], device)
            .into_primitive();
    let tiles_touched_min =
        Tensor::<Wgpu, 2, Int>::empty([point_count, 2], device)
            .into_primitive();
    let transforms_2d = Tensor::<Wgpu, 3>::empty([point_count, 2, 3], device)
        .into_primitive()
        .tensor();
    let view_directions =
        Tensor::<Wgpu, 2>::empty([point_count, 3 + 1], device)
            .into_primitive()
            .tensor();
    let view_offsets = Tensor::<Wgpu, 2>::empty([point_count, 3 + 1], device)
        .into_primitive()
        .tensor();

    client.execute(
        Box::new(SourceKernel::new(
            Kernel1WgslSource,
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
            colors_sh.handle.to_owned().binding(),
            positions_3d.handle.to_owned().binding(),
            rotations.handle.to_owned().binding(),
            scalings.handle.to_owned().binding(),
            view_position.handle.binding(),
            view_transform.handle.to_owned().binding(),
            colors_rgb_3d.handle.to_owned().binding(),
            conics.handle.to_owned().binding(),
            covariances_3d.handle.to_owned().binding(),
            depths.handle.to_owned().binding(),
            is_colors_rgb_3d_not_clamped.handle.to_owned().binding(),
            positions_2d.handle.to_owned().binding(),
            positions_3d_in_normalized.handle.to_owned().binding(),
            positions_3d_in_normalized_clamped
                .handle
                .to_owned()
                .binding(),
            radii.handle.to_owned().binding(),
            rotations_matrix.handle.to_owned().binding(),
            rotation_scalings.handle.to_owned().binding(),
            tile_touched_counts.handle.to_owned().binding(),
            tiles_touched_max.handle.to_owned().binding(),
            tiles_touched_min.handle.to_owned().binding(),
            transforms_2d.handle.to_owned().binding(),
            view_directions.handle.to_owned().binding(),
            view_offsets.handle.to_owned().binding(),
        ],
    );

    #[cfg(debug_assertions)]
    {
        client.sync(cubecl::client::SyncType::Wait);
        log::debug!(
            target: "gausplat_renderer::scene",
            "Gaussian3dRenderer::<Wgpu>::render_forward > 1",
        );
    }

    // Performing the forward pass #2

    // (T, [P])
    let (tile_touched_count, tile_touched_offsets) = {
        let b = tile_touched_counts.handle.binding();
        let counts = &client.read(b);
        let counts = cast_slice::<u8, u32>(counts);

        let mut count = *counts.last().unwrap_or(&0);
        let offsets = counts
            .into_iter()
            .scan(0, |state, count| {
                let offset = *state;
                *state += count;
                Some(offset)
            })
            .collect::<Vec<_>>();
        count += *offsets.last().unwrap_or(&0);

        (
            count as usize,
            Tensor::<Wgpu, 1, Int>::from_data(offsets.as_slice(), device)
                .into_primitive(),
        )
    };

    #[cfg(debug_assertions)]
    {
        client.sync(cubecl::client::SyncType::Wait);
        log::debug!(
            target: "gausplat_renderer::scene",
            "Gaussian3dRenderer::<Wgpu>::render_forward > 2",
        );
    }

    // Performing the forward pass #3

    let arguments = client.create(bytes_of(&Kernel3Arguments {
        point_count: point_count as u32,
        tile_count_x,
    }));
    let point_infos =
        Tensor::<Wgpu, 2, Int>::empty([tile_touched_count, 3], device)
            .into_primitive();

    client.execute(
        Box::new(SourceKernel::new(
            Kernel3WgslSource,
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
            depths.handle.to_owned().binding(),
            radii.handle.to_owned().binding(),
            tile_touched_offsets.handle.binding(),
            tiles_touched_max.handle.binding(),
            tiles_touched_min.handle.binding(),
            point_infos.handle.to_owned().binding(),
        ],
    );

    #[cfg(debug_assertions)]
    {
        client.sync(cubecl::client::SyncType::Wait);
        log::debug!(
            target: "gausplat_renderer::scene",
            "Gaussian3dRenderer::<Wgpu>::render_forward > 3",
        );
    }

    // Performing the forward pass #4

    // ([T], [T])
    let (point_indexes, point_tile_indexes) = {
        let point_infos = &mut client.read(point_infos.handle.binding());
        let point_infos = cast_slice_mut::<u8, PointInfo>(point_infos);

        point_infos.par_sort_unstable();

        point_infos
            .into_par_iter()
            .map(|point| (point.index, point.tile_index()))
            .unzip::<_, _, Vec<_>, Vec<_>>()
    };

    #[cfg(debug_assertions)]
    {
        client.sync(cubecl::client::SyncType::Wait);
        log::debug!(
            target: "gausplat_renderer::scene",
            "Gaussian3dRenderer::<Wgpu>::render_forward > 4",
        );
    }

    // Performing the forward pass #5

    let arguments = client.create(bytes_of(&Kernel5Arguments {
        tile_touched_count: tile_touched_count as u32,
    }));
    // [I_Y / T_Y, I_X / T_X, 2]
    let tile_point_ranges = {
        let mut ranges = Tensor::<Wgpu, 2, Int>::zeros([tile_count, 2], device);

        if !point_tile_indexes.is_empty() {
            let tile_index_first =
                *point_tile_indexes.first().unwrap() as usize;
            let tile_index_last = *point_tile_indexes.last().unwrap() as usize;
            ranges = ranges.slice_assign(
                [tile_index_first..tile_index_first + 1, 0..1],
                Tensor::from_data([[0]], device),
            );
            ranges = ranges.slice_assign(
                [tile_index_last..tile_index_last + 1, 1..2],
                Tensor::from_data([[tile_touched_count as u32]], device),
            );
        }

        ranges
            .reshape([tile_count_y as i32, tile_count_x as i32, 2])
            .into_primitive()
    };
    // [T]
    let point_tile_indexes = Tensor::<Wgpu, 1, Int>::from_data(
        point_tile_indexes.as_slice(),
        device,
    )
    .into_primitive();

    client.execute(
        Box::new(SourceKernel::new(
            Kernel5WgslSource,
            cubecl::CubeDim {
                x: GROUP_SIZE_X,
                y: GROUP_SIZE_Y,
                z: 1,
            },
        )),
        cubecl::CubeCount::Static(
            (tile_touched_count as u32 + GROUP_SIZE - 1) / GROUP_SIZE,
            1,
            1,
        ),
        vec![
            arguments.binding(),
            point_tile_indexes.handle.binding(),
            tile_point_ranges.handle.to_owned().binding(),
        ],
    );

    #[cfg(debug_assertions)]
    {
        client.sync(cubecl::client::SyncType::Wait);
        log::debug!(
            target: "gausplat_renderer::scene",
            "Gaussian3dRenderer::<Wgpu>::render_forward > 5",
        );
    }

    // Performing the forward pass #6

    let arguments = client.create(bytes_of(&Kernel6Arguments {
        image_size_x: image_size_x as u32,
        image_size_y: image_size_y as u32,
    }));
    // [T]
    let point_indexes =
        Tensor::<Wgpu, 1, Int>::from_data(point_indexes.as_slice(), device)
            .into_primitive();
    let colors_rgb_2d =
        Tensor::<Wgpu, 3>::empty([image_size_y, image_size_x, 3], device)
            .into_primitive()
            .tensor();
    let point_rendered_counts =
        Tensor::<Wgpu, 2, Int>::empty([image_size_y, image_size_x], device)
            .into_primitive();
    let transmittances =
        Tensor::<Wgpu, 2>::empty([image_size_y, image_size_x], device)
            .into_primitive()
            .tensor();

    client.execute(
        Box::new(SourceKernel::new(
            Kernel6WgslSource,
            cubecl::CubeDim {
                x: tile_size_x,
                y: tile_size_y,
                z: 1,
            },
        )),
        cubecl::CubeCount::Static(tile_count_x, tile_count_y, 1),
        vec![
            arguments.binding(),
            colors_rgb_3d.handle.to_owned().binding(),
            conics.handle.to_owned().binding(),
            opacities_3d.handle.to_owned().binding(),
            point_indexes.handle.to_owned().binding(),
            positions_2d.handle.to_owned().binding(),
            tile_point_ranges.handle.to_owned().binding(),
            colors_rgb_2d.handle.to_owned().binding(),
            point_rendered_counts.handle.to_owned().binding(),
            transmittances.handle.to_owned().binding(),
        ],
    );

    #[cfg(debug_assertions)]
    {
        client.sync(cubecl::client::SyncType::Wait);
        log::debug!(
            target: "gausplat_renderer::scene",
            "Gaussian3dRenderer::<Wgpu>::render_forward > 6",
        );
        log::debug!(
            target: "gausplat_renderer::scene",
            "Gaussian3dRenderer::<Wgpu>::render_forward > tile_touched_count ({tile_touched_count})",
        );
    }

    // Specifying the results

    forward::RenderOutput {
        colors_rgb_2d,
        state: backward::RenderInput {
            colors_rgb_3d,
            colors_sh,
            conics,
            covariances_3d,
            depths,
            focal_length_x: focal_length_x as f64,
            focal_length_y: focal_length_y as f64,
            image_size_x,
            image_size_y,
            is_colors_rgb_3d_not_clamped,
            opacities_3d,
            options,
            point_indexes,
            point_rendered_counts,
            positions_2d,
            positions_3d,
            positions_3d_in_normalized,
            positions_3d_in_normalized_clamped,
            radii,
            rotations,
            rotations_matrix,
            rotation_scalings,
            scalings,
            tile_point_ranges,
            transforms_2d,
            transmittances,
            view_directions,
            view_offsets,
            view_transform_rotation: view_transform,
        },
    }
}
