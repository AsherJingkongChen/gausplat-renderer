mod kernel;
mod point;

pub use super::*;

use crate::preset::render::*;
use backend::Wgpu;
use burn::tensor::Shape;
use burn_jit::{
    cubecl::{CubeCount, CubeDim},
    kernel::into_contiguous,
    template::SourceKernel,
};
use bytemuck::{bytes_of, cast_slice_mut, from_bytes};
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
    #[cfg(debug_assertions)]
    log::debug!(
        target: "gausplat_renderer::scene",
        "Gaussian3dRenderer::<Wgpu>::render_forward",
    );

    // Specifying the parameters

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

    debug_assert!(
        colors_sh_degree_max <= SH_DEGREE_MAX,
        "colors_sh_degree_max should be no more than {SH_DEGREE_MAX}",
    );
    debug_assert!(
        image_size_x * image_size_y <= PIXEL_COUNT_MAX,
        "Pixel count should be no more than {PIXEL_COUNT_MAX}",
    );
    debug_assert_ne!(image_size_x, 0);
    debug_assert_ne!(image_size_y, 0);
    debug_assert_ne!(point_count, 0);

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
                x: GROUP_SIZE,
                y: 1,
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

    // Computing the offsets of touched tiles

    let tile_touched_offsets = tile_touched_counts;
    // T
    let tile_touched_count = scan_add(&tile_touched_offsets);
    debug_assert_ne!(tile_touched_count, 0);

    // Performing the forward pass #3

    let arguments = client.create(bytes_of(&Kernel3Arguments {
        point_count: point_count as u32,
        tile_count_x,
    }));
    let point_infos =
        Tensor::<Wgpu, 2, Int>::empty([tile_touched_count as usize, 2], device)
            .into_primitive();

    client.execute(
        Box::new(SourceKernel::new(
            Kernel3WgslSource,
            CubeDim {
                x: GROUP_SIZE,
                y: 1,
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

    client.sync(burn::tensor::backend::SyncType::Wait);
    let time = std::time::Instant::now();

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

    client.sync(burn::tensor::backend::SyncType::Wait);
    let time = time.elapsed().as_secs_f64();
    print!("{time} ");

    // Performing the forward pass #5

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
                x: GROUP_SIZE,
                y: 1,
                z: 1,
            },
        )),
        CubeCount::Static(
            (tile_touched_count + GROUP_SIZE * GROUP_SIZE - 1)
                / (GROUP_SIZE * GROUP_SIZE),
            GROUP_SIZE,
            1,
        ),
        vec![
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

/// Performing an exclusive scan-and-add operation on the given tensor in place.
///
/// ## Arguments
///
/// - `sums`: The tensor to scan in place.
///
/// ## Returns
///
/// The total sum of all elements in `sums`.
pub fn scan_add(
    // [N]
    sums: &<Wgpu as Backend>::IntTensorPrimitive<1>,
) -> u32 {
    const GROUP_SIZE: u32 = 256;

    // Specifying the parameters

    let client = &sums.client;
    let device = &sums.device;
    // N
    let count = sums.shape.dims[0];
    // N'
    let count_next = (count as u32 + GROUP_SIZE - 1) / GROUP_SIZE;
    // [N']
    let sums_next =
        Tensor::<Wgpu, 1, Int>::empty([count_next as usize], device)
            .into_primitive();

    let cube_count = CubeCount::Static(count_next, 1, 1);
    let cube_dim = CubeDim {
        x: GROUP_SIZE,
        y: 1,
        z: 1,
    };

    // Scanning

    client.execute(
        Box::new(SourceKernel::new(KernelScanAddScan, cube_dim)),
        cube_count.to_owned(),
        vec![
            sums.handle.to_owned().binding(),
            sums_next.handle.to_owned().binding(),
        ],
    );

    // Recursing if there is more than one remaining group
    if count_next > 1 {
        let sum = scan_add(&sums_next);

        // Adding

        client.execute(
            Box::new(SourceKernel::new(KernelScanAddAdd, cube_dim)),
            cube_count,
            vec![sums.handle.to_owned().binding(), sums_next.handle.binding()],
        );

        sum
    } else {
        debug_assert_eq!(count_next, 1);
        *from_bytes::<u32>(&client.read(sums_next.handle.binding()))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn radix_sort() {
        use super::*;

        // 2^R
        const RADIX: u32 = 1 << RADIX_BIT_COUNT;
        // R
        const RADIX_BIT_COUNT: u32 = 8;

        let device = &WgpuDevice::default();
        let radix = RADIX as usize;
        let mut keys_input = Tensor::<Wgpu, 1, Int>::from_ints(
            [
                0x100, 0x200, 0x103, 0x102, 0x905, 0x904, 0x907, 0x306, 0x302,
                0x308,
            ],
            device,
        )
        .into_primitive();
        let values_input = Tensor::<Wgpu, 1, Int>::from_ints(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            device,
        )
        .into_primitive();
        let mut keys_output =
            Tensor::<Wgpu, 1, Int>::zeros([10], device).into_primitive();
        let values_output =
            Tensor::<Wgpu, 1, Int>::zeros([10], device).into_primitive();

        let client = &values_output.client;

        let cube_dim = CubeDim {
            x: RADIX,
            y: 1,
            z: 1,
        };
        let cube_count = CubeCount::Static(
            (keys_input.shape.dims[0] as u32 + RADIX - 1) / RADIX,
            1,
            1,
        );

        println!(
            "keys_input: {:04x?}",
            bytemuck::cast_slice::<u8, u32>(
                &client.read(keys_input.handle.to_owned().binding()),
            )
        );

        for pass in 0..4 {
            let arguments =
                client.create(bytes_of(&KernelRadixSortArguments {
                    radix_bit_offset: 8 * pass,
                }));
            let radix_counts =
                Tensor::<Wgpu, 1, Int>::zeros([radix], device).into_primitive();

            let time = std::time::Instant::now();
            client.execute(
                Box::new(SourceKernel::new(
                    KernelRadixSortCountRadix,
                    cube_dim,
                )),
                cube_count.to_owned(),
                vec![
                    arguments.to_owned().binding(),
                    keys_input.handle.to_owned().binding(),
                    radix_counts.handle.to_owned().binding(),
                ],
            );
            client.sync(burn::tensor::backend::SyncType::Wait);
            println!("Sort Pass {pass} - 1: {:?}", time.elapsed());

            let time = std::time::Instant::now();
            client.execute(
                Box::new(SourceKernel::new(
                    KernelRadixSortScanRadix,
                    CubeDim { x: 1, y: 1, z: 1 },
                )),
                CubeCount::Static(1, 1, 1),
                vec![radix_counts.handle.to_owned().binding()],
            );
            // scan_add(&radix_counts);
            let radix_offsets = radix_counts;
            client.sync(burn::tensor::backend::SyncType::Wait);
            println!("Sort Pass {pass} - 2: {:?}", time.elapsed());

            let time = std::time::Instant::now();
            client.execute(
                Box::new(SourceKernel::new(
                    KernelRadixSortScatterKey,
                    cube_dim,
                )),
                cube_count.to_owned(),
                vec![
                    arguments.to_owned().binding(),
                    keys_input.handle.to_owned().binding(),
                    keys_output.handle.to_owned().binding(),
                    radix_offsets.handle.to_owned().binding(),
                ],
            );
            client.sync(burn::tensor::backend::SyncType::Wait);
            println!("Sort Pass {pass} - 3: {:?}", time.elapsed());

            println!(
                "keys_output: {:04x?}",
                bytemuck::cast_slice::<u8, u32>(
                    &client.read(keys_output.handle.to_owned().binding()),
                )
            );

            std::mem::swap(&mut keys_input, &mut keys_output);
        }
    }

    #[test]
    fn scan_add_small() {
        use super::*;
        use bytemuck::cast_slice;

        let device = &Default::default();

        let sums = Tensor::<Wgpu, 1, Int>::from_ints(
            [0, 3, 0, 2, 4, 1, 3, 2, 9],
            device,
        )
        .into_primitive();

        let sum_value = scan_add(&sums);
        let sums_value = sums.client.read(sums.handle.to_owned().binding());
        let sums_value = cast_slice::<u8, u32>(&sums_value);

        let sum_target = 24;
        let sums_target = [0, 0, 3, 3, 5, 9, 10, 13, 15];

        assert_eq!(sum_value, sum_target);
        sums_value.iter().zip(&sums_target).enumerate().for_each(
            |(index, (&value, &target))| {
                assert_eq!(value, target, "index: {index}");
            },
        );
    }

    #[test]
    fn scan_add_random() {
        use super::*;
        use burn::tensor::Distribution;
        use bytemuck::cast_slice;

        let device = &Default::default();

        let sums = Tensor::<Wgpu, 1, Int>::random(
            [1 << 15],
            Distribution::Uniform(0.0, 15.0),
            device,
        )
        .into_primitive();
        let sums_source = sums.client.read(sums.handle.to_owned().binding());
        let sums_source = cast_slice::<u8, u32>(&sums_source);

        let sum_value = scan_add(&sums);
        let sums_value = sums.client.read(sums.handle.to_owned().binding());
        let sums_value = cast_slice::<u8, u32>(&sums_value);

        let sum_target = sums_source.iter().sum::<u32>();
        let sums_target = sums_source
            .iter()
            .scan(0, |state, &sum| {
                let value = *state;
                *state += sum;
                Some(value)
            })
            .collect::<Vec<_>>();

        assert_eq!(sum_value, sum_target);
        sums_value.iter().zip(&sums_target).enumerate().for_each(
            |(index, (&value, &target))| {
                assert_eq!(value, target, "index: {index}");
            },
        );
    }
}
