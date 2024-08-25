mod kernel;
mod point;

pub use super::*;

use crate::consts::render::*;
use burn::{
    backend::wgpu::{
        into_contiguous, Kernel, SourceKernel, WorkGroup, WorkgroupSize,
    },
    tensor::ops::IntTensor,
};
use bytemuck::{bytes_of, cast_slice, cast_slice_mut};
use kernel::*;
use point::*;
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

pub fn render_gaussian_3d_scene(
    input: forward::RendererInput<Wgpu>,
    view: &sparse_view::View,
    options: RendererOptions,
) -> forward::RendererOutput<Wgpu> {
    // Specifying the parameters

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
    // ([P, 16, 3], P)
    let (client, colors_sh, device, point_count) = {
        let c = into_contiguous(input.colors_sh);
        (c.client, c.handle, c.device, c.shape.dims[0])
    };

    // Performing the forward pass #1

    let mut duration = std::time::Instant::now();

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

    // [P, 3]
    let positions_3d = into_contiguous(input.positions).handle;
    // [P, 1]
    let opacities_3d = into_contiguous(input.opacities).handle;
    // [P, 4]
    let rotations = into_contiguous(input.rotations).handle;
    // [P, 3]
    let scalings = into_contiguous(input.scalings).handle;
    // [3]
    let view_position =
        client.create(bytes_of(&view.view_position.map(|c| c as f32)));
    // [3 (+ 1), 4]
    let view_transform = client
        .create(bytes_of(&view.view_transform.map(|c| c.map(|c| c as f32))));
    // [P, 3 (+ 1)] (the alignment of vec3<f32> is 16 = (3 + 1) * 4 bytes)
    let colors_rgb_3d = client.empty(point_count * (3 + 1) * 4);
    // [P, 2, 2]
    let conics = client.empty(point_count * 2 * 2 * 4);
    // [P, 3 (+ 1), 3] (the alignment of mat3x3<f32> is 16 = (3 + 1) * 4 bytes)
    let covariances_3d = client.empty(point_count * (3 + 1) * 3 * 4);
    // [P]
    let depths = client.empty(point_count * 4);
    // [P, 3 (+ 1)]
    let is_colors_rgb_3d_clamped = client.empty(point_count * (3 + 1) * 4);
    // [P, 2]
    let positions_2d = client.empty(point_count * 2 * 4);
    // [P, 2]
    let positions_3d_in_normalized = client.empty(point_count * 2 * 4);
    // [P, 2]
    let positions_3d_in_normalized_clamped = client.empty(point_count * 2 * 4);
    // [P]
    let radii = client.empty(point_count * 4);
    // [P, 3 (+ 1), 3]
    let rotations_matrix = client.empty(point_count * (3 + 1) * 3 * 4);
    // [P, 3 (+ 1), 3]
    let rotation_scalings = client.empty(point_count * (3 + 1) * 3 * 4);
    // [P]
    let tile_touched_counts = client.empty(point_count * 4);
    // [P, 2]
    let tiles_touched_max = client.empty(point_count * 2 * 4);
    // [P, 2]
    let tiles_touched_min = client.empty(point_count * 2 * 4);
    // [P, 2, 3]
    let transforms_2d = client.empty(point_count * 2 * 3 * 4);
    // [P, 3 (+ 1)]
    let view_directions = client.empty(point_count * (3 + 1) * 4);
    // [P, 3 (+ 1)]
    let view_offsets = client.empty(point_count * (3 + 1) * 4);

    client.execute(
        Kernel::Custom(Box::new(SourceKernel::new(
            Kernel1WgslSource,
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
            &colors_sh,
            &positions_3d,
            &rotations,
            &scalings,
            &view_position,
            &view_transform,
            &colors_rgb_3d,
            &conics,
            &covariances_3d,
            &depths,
            &is_colors_rgb_3d_clamped,
            &positions_2d,
            &positions_3d_in_normalized,
            &positions_3d_in_normalized_clamped,
            &radii,
            &rotations_matrix,
            &rotation_scalings,
            &tile_touched_counts,
            &tiles_touched_max,
            &tiles_touched_min,
            &transforms_2d,
            &view_directions,
            &view_offsets,
        ],
    );

    client.sync();
    println!("Duration (Forward 1): {:?}", duration.elapsed());
    duration = std::time::Instant::now();

    // Performing the forward pass #2

    // (T, [P])
    let (tile_touched_count, tile_touched_offsets) = {
        let counts = &client.read(&tile_touched_counts).read();
        let counts = cast_slice::<u8, u32>(counts).to_vec();

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

        (count as usize, client.create(cast_slice(&offsets)))
    };

    println!("tile_touched_count (wgsl): {:?}", tile_touched_count);

    println!("Duration (Forward 2): {:?}", duration.elapsed());
    duration = std::time::Instant::now();

    // Performing the forward pass #3

    let arguments = client.create(bytes_of(&Kernel3Arguments {
        point_count: point_count as u32,
        tile_count_x,
    }));
    let point_keys_and_indexes = client.empty(tile_touched_count * 3 * 4);

    client.execute(
        Kernel::Custom(Box::new(SourceKernel::new(
            Kernel3WgslSource,
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
            &depths,
            &radii,
            &tile_touched_offsets,
            &tiles_touched_max,
            &tiles_touched_min,
            &point_keys_and_indexes,
        ],
    );

    client.sync();
    println!("Duration (Forward 3): {:?}", duration.elapsed());
    duration = std::time::Instant::now();

    // Performing the forward pass #4

    // ([T], [T])
    let (point_indexes, point_tile_indexes) = {
        let point_keys_and_indexes =
            &mut client.read(&point_keys_and_indexes).read();
        let point_keys_and_indexes =
            cast_slice_mut::<u8, PointKeyAndIndex>(point_keys_and_indexes);

        point_keys_and_indexes.par_sort_unstable();

        point_keys_and_indexes
            .into_par_iter()
            .map(|point| (point.index, point.tile_index()))
            .unzip::<_, _, Vec<_>, Vec<_>>()
    };

    println!("Duration (Forward 4): {:?}", duration.elapsed());
    duration = std::time::Instant::now();

    // Performing the forward pass #5

    let arguments = client.create(bytes_of(&Kernel5Arguments {
        tile_touched_count: tile_touched_count as u32,
    }));
    // [I_Y / T_Y, I_X / T_X, 2]
    let tile_point_ranges = {
        let mut ranges = vec![[0; 2]; tile_count];

        if !point_tile_indexes.is_empty() {
            let tile_index_first =
                *point_tile_indexes.first().unwrap() as usize;
            let tile_index_last = *point_tile_indexes.last().unwrap() as usize;
            ranges[tile_index_first][0] = 0;
            ranges[tile_index_last][1] = tile_touched_count as u32;
        }

        client.create(cast_slice::<[u32; 2], u8>(&ranges))
    };
    // [T]
    let point_tile_indexes =
        client.create(cast_slice::<u32, u8>(&point_tile_indexes));

    client.execute(
        Kernel::Custom(Box::new(SourceKernel::new(
            Kernel5WgslSource,
            WorkGroup {
                x: (tile_touched_count as u32 + GROUP_SIZE - 1) / GROUP_SIZE,
                y: 1,
                z: 1,
            },
            WorkgroupSize {
                x: GROUP_SIZE_X,
                y: GROUP_SIZE_Y,
                z: 1,
            },
        ))),
        &[&arguments, &point_tile_indexes, &tile_point_ranges],
    );

    client.sync();
    println!("Duration (Forward 5): {:?}", duration.elapsed());

    // Performing the forward pass #6

    let arguments = client.create(bytes_of(&Kernel6Arguments {
        image_size_x: image_size_x as u32,
        image_size_y: image_size_y as u32,
    }));
    // [T]
    let point_indexes = client.create(cast_slice::<u32, u8>(&point_indexes));
    // [I_Y, I_X, 3]
    let colors_rgb_2d = client.empty(image_size_y * image_size_x * 3 * 4);
    // [I_Y, I_X]
    let point_rendered_counts = client.empty(image_size_y * image_size_x * 4);
    // [I_Y, I_X]
    let transmittances = client.empty(image_size_y * image_size_x * 4);

    client.execute(
        Kernel::Custom(Box::new(SourceKernel::new(
            Kernel6WgslSource,
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
            &colors_rgb_3d,
            &conics,
            &opacities_3d,
            &point_indexes,
            &positions_2d,
            &tile_point_ranges,
            &colors_rgb_2d,
            &point_rendered_counts,
            &transmittances,
        ],
    );

    client.sync();
    println!("Duration (Forward 6): {:?}", duration.elapsed());

    // Specifying the results

    forward::RendererOutput {
        // [I_Y, I_X, 3]
        colors_rgb_2d: FloatTensor::<Wgpu, 3>::new(
            client.to_owned(),
            device.to_owned(),
            [image_size_y, image_size_x, 3].into(),
            colors_rgb_2d,
        ),
        state: backward::RendererInput {
            // [P, 3 (+ 1)]
            colors_rgb_3d: FloatTensor::<Wgpu, 2>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count, 3 + 1].into(),
                colors_rgb_3d,
            ),
            // [P, 16, 3]
            colors_sh: FloatTensor::<Wgpu, 3>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count, 16, 3].into(),
                colors_sh,
            ),
            // [P, 2, 2]
            conics: FloatTensor::<Wgpu, 3>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count, 2, 2].into(),
                conics,
            ),
            // [P, 3 (+ 1), 3]
            covariances_3d: FloatTensor::<Wgpu, 3>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count, 3 + 1, 3].into(),
                covariances_3d,
            ),
            // [P]
            depths: FloatTensor::<Wgpu, 1>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count].into(),
                depths,
            ),
            focal_length_x: focal_length_x as f64,
            focal_length_y: focal_length_y as f64,
            // I_X
            image_size_x,
            // I_Y
            image_size_y,
            // I_X / 2.0
            image_size_half_x: image_size_half_x as f64,
            // I_Y / 2.0
            image_size_half_y: image_size_half_y as f64,
            // [P, 3 (+ 1)]
            is_colors_rgb_3d_clamped: FloatTensor::<Wgpu, 2>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count, 3 + 1].into(),
                is_colors_rgb_3d_clamped,
            ),
            // [P, 1]
            opacities_3d: FloatTensor::<Wgpu, 2>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count, 1].into(),
                opacities_3d,
            ),
            options,
            // P
            point_count,
            // [T]
            point_indexes: IntTensor::<Wgpu, 1>::new(
                client.to_owned(),
                device.to_owned(),
                [tile_touched_count].into(),
                point_indexes,
            ),
            // [I_Y, I_X]
            point_rendered_counts: IntTensor::<Wgpu, 2>::new(
                client.to_owned(),
                device.to_owned(),
                [image_size_y, image_size_x].into(),
                point_rendered_counts,
            ),
            // [P, 2]
            positions_2d: FloatTensor::<Wgpu, 2>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count, 2].into(),
                positions_2d,
            ),
            // [P, 3]
            positions_3d: FloatTensor::<Wgpu, 2>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count, 3].into(),
                positions_3d,
            ),
            // [P, 2]
            positions_3d_in_normalized: FloatTensor::<Wgpu, 2>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count, 2].into(),
                positions_3d_in_normalized,
            ),
            // [P, 2]
            positions_3d_in_normalized_clamped: FloatTensor::<Wgpu, 2>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count, 2].into(),
                positions_3d_in_normalized_clamped,
            ),
            // [P]
            radii: IntTensor::<Wgpu, 1>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count].into(),
                radii,
            ),
            // [P, 4]
            rotations: FloatTensor::<Wgpu, 2>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count, 4].into(),
                rotations,
            ),
            // [P, 3 (+ 1), 3]
            rotations_matrix: FloatTensor::<Wgpu, 3>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count, 3 + 1, 3].into(),
                rotations_matrix,
            ),
            // [P, 3 (+ 1), 3]
            rotation_scalings: FloatTensor::<Wgpu, 3>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count, 3 + 1, 3].into(),
                rotation_scalings,
            ),
            // [P, 3]
            scalings: FloatTensor::<Wgpu, 2>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count, 3].into(),
                scalings,
            ),
            // [I_Y / T_Y, I_X / T_X, 2]
            tile_point_ranges: IntTensor::<Wgpu, 2>::new(
                client.to_owned(),
                device.to_owned(),
                [tile_count, 2].into(),
                tile_point_ranges,
            ),
            // [P, 2, 3]
            transforms_2d: FloatTensor::<Wgpu, 3>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count, 2, 3].into(),
                transforms_2d,
            ),
            // [I_Y, I_X]
            transmittances: FloatTensor::<Wgpu, 2>::new(
                client.to_owned(),
                device.to_owned(),
                [image_size_y, image_size_x].into(),
                transmittances,
            ),
            // [P, 3 (+ 1)]
            view_directions: FloatTensor::<Wgpu, 2>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count, 3 + 1].into(),
                view_directions,
            ),
            // [P, 3 (+ 1)]
            view_offsets: FloatTensor::<Wgpu, 2>::new(
                client.to_owned(),
                device.to_owned(),
                [point_count, 3 + 1].into(),
                view_offsets,
            ),
            // [3 (+ 1), 4]
            view_transform: FloatTensor::<Wgpu, 2>::new(
                client.to_owned(),
                device.to_owned(),
                [3 + 1, 4].into(),
                view_transform,
            ),
        },
    }
}
