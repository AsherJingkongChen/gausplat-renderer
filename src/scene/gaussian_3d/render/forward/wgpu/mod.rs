mod kernel;
mod point;

pub use super::*;

use crate::preset::render::*;
use backend::Wgpu;
use burn::{
    backend::wgpu::{into_contiguous, SourceKernel},
    tensor::Shape,
};
use burn_jit::cubecl::{CubeCount, CubeDim};
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
    options: &Gaussian3dRendererOptions,
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
    // I_x
    let image_size_x = view.image_width;
    // I_y
    let image_size_y = view.image_height;
    let focal_length_x =
        (image_size_x as f64 / field_of_view_x_half_tan / 2.0) as f32;
    let focal_length_y =
        (image_size_y as f64 / field_of_view_y_half_tan / 2.0) as f32;
    // I_x / 2.0
    let image_size_half_x = (image_size_x as f64 / 2.0) as f32;
    // I_y / 2.0
    let image_size_half_y = (image_size_y as f64 / 2.0) as f32;
    // T_x
    let tile_size_x = GROUP_SIZE_X;
    // T_y
    let tile_size_y = GROUP_SIZE_Y;
    // I_x / T_x
    let tile_count_x = (image_size_x + tile_size_x - 1) / tile_size_x;
    // I_y / T_y
    let tile_count_y = (image_size_y + tile_size_y - 1) / tile_size_y;
    let view_bound_x =
        (field_of_view_x_half_tan * (FILTER_LOW_PASS + 1.0)) as f32;
    let view_bound_y =
        (field_of_view_y_half_tan * (FILTER_LOW_PASS + 1.0)) as f32;
    // [P, 16 * 3]
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
        image_size_x,
        image_size_y,
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
            CubeDim {
                x: GROUP_SIZE_X,
                y: GROUP_SIZE_Y,
                z: 1,
            },
        )),
        CubeCount::Static(
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

    // Performing the forward pass #2

    // (T, [P])
    let (tile_touched_count, tile_touched_offsets) = {
        let counts = &client.read(tile_touched_counts.handle.binding());
        let counts = cast_slice::<u8, u32>(counts);

        let mut count = *counts.last().unwrap_or(&0);
        let offsets = counts
            .iter()
            .scan(0, |state, count| {
                let offset = *state;
                *state += count;
                Some(offset)
            })
            .collect::<Vec<_>>();
        count += *offsets.last().unwrap_or(&0);

        debug_assert!(count != 0, "No point is touched by any tile");

        (
            count,
            Tensor::<Wgpu, 1, Int>::from_data(
                TensorData::new(offsets, [point_count]),
                device,
            )
            .into_primitive(),
        )
    };

    // Performing the forward pass #3

    let arguments = client.create(bytes_of(&Kernel3Arguments {
        point_count: point_count as u32,
        tile_count_x,
    }));
    let point_infos =
        Tensor::<Wgpu, 2, Int>::empty([tile_touched_count as usize, 3], device)
            .into_primitive();

    client.execute(
        Box::new(SourceKernel::new(
            Kernel3WgslSource,
            CubeDim {
                x: GROUP_SIZE_X,
                y: GROUP_SIZE_Y,
                z: 1,
            },
        )),
        CubeCount::Static(
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

    // Performing the forward pass #5

    let arguments =
        client.create(bytes_of(&Kernel5Arguments { tile_touched_count }));
    // [I_y / T_y, I_x / T_x, 2]
    let tile_point_ranges = {
        let ranges_shape =
            Shape::from([tile_count_y as usize, tile_count_x as usize, 2]);
        let mut ranges = vec![0; ranges_shape.num_elements()];

        if !point_tile_indexes.is_empty() {
            let tile_index_last =
                *point_tile_indexes.last().expect("Unreachable") as usize;
            ranges[tile_index_last * 2 + 1] = tile_touched_count;
        }

        Tensor::<Wgpu, 3, Int>::from_data(
            TensorData::new(ranges, ranges_shape),
            device,
        )
        .into_primitive()
    };

    // [T]
    let point_tile_indexes = Tensor::<Wgpu, 1, Int>::from_data(
        TensorData::new(point_tile_indexes, [tile_touched_count as usize]),
        device,
    )
    .into_primitive();

    client.execute(
        Box::new(SourceKernel::new(
            Kernel5WgslSource,
            CubeDim {
                x: GROUP_SIZE_X,
                y: GROUP_SIZE_Y,
                z: 1,
            },
        )),
        CubeCount::Static(
            (tile_touched_count + GROUP_SIZE - 1) / GROUP_SIZE,
            1,
            1,
        ),
        vec![
            arguments.binding(),
            point_tile_indexes.handle.binding(),
            tile_point_ranges.handle.to_owned().binding(),
        ],
    );

    // Performing the forward pass #6

    let arguments = client.create(bytes_of(&Kernel6Arguments {
        image_size_x,
        image_size_y,
    }));
    // [T]
    let point_indexes = Tensor::<Wgpu, 1, Int>::from_data(
        TensorData::new(point_indexes, [tile_touched_count as usize]),
        device,
    )
    .into_primitive();
    let image_size = [image_size_y as usize, image_size_x as usize];
    let colors_rgb_2d =
        Tensor::<Wgpu, 3>::empty([image_size[0], image_size[1], 3], device)
            .into_primitive()
            .tensor();
    let point_rendered_counts =
        Tensor::<Wgpu, 2, Int>::empty(image_size, device).into_primitive();
    let transmittances = Tensor::<Wgpu, 2>::empty(image_size, device)
        .into_primitive()
        .tensor();

    client.execute(
        Box::new(SourceKernel::new(
            Kernel6WgslSource,
            CubeDim {
                x: tile_size_x,
                y: tile_size_y,
                z: 1,
            },
        )),
        CubeCount::Static(tile_count_x, tile_count_y, 1),
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

    // Specifying the results

    forward::RenderOutput {
        colors_rgb_2d,
        state: backward::RenderInput {
            colors_rgb_3d,
            colors_sh,
            colors_sh_degree_max,
            conics,
            covariances_3d,
            depths,
            focal_length_x: focal_length_x as f64,
            focal_length_y: focal_length_y as f64,
            image_size_x,
            image_size_y,
            is_colors_rgb_3d_not_clamped,
            opacities_3d,
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
            view_rotation: view_transform,
        },
    }
}
