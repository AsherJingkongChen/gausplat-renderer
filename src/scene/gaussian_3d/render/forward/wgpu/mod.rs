mod kernel;

pub use super::*;

use crate::preset::render::*;
use burn_jit::{
    cubecl::{CubeCount, CubeDim},
    kernel::into_contiguous,
    template::SourceKernel,
};
use bytemuck::{bytes_of, from_bytes};
use kernel::*;

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
        Tensor::<Wgpu, 1, Int>::empty([point_count], device).into_primitive();
    let rotations_matrix =
        Tensor::<Wgpu, 3>::empty([point_count, 3 + 1, 3], device)
            .into_primitive()
            .tensor();
    let rotation_scalings =
        Tensor::<Wgpu, 3>::empty([point_count, 3 + 1, 3], device)
            .into_primitive()
            .tensor();
    let tile_touched_counts =
        Tensor::<Wgpu, 1, Int>::empty([point_count], device).into_primitive();
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

    let ScanAddOutput {
        sum: tile_touched_count,
        sums: tile_touched_offsets,
    } = scan_add(tile_touched_counts);
    // T
    let tile_touched_count =
        *from_bytes::<u32>(&client.read(tile_touched_count.handle.binding()));
    debug_assert_ne!(tile_touched_count, 0);

    // Performing the forward pass #3

    let arguments = client.create(bytes_of(&Kernel3Arguments {
        point_count: point_count as u32,
        tile_count_x,
    }));
    let point_indices =
        Tensor::<Wgpu, 1, Int>::empty([tile_touched_count as usize], device)
            .into_primitive();
    let point_orders =
        Tensor::<Wgpu, 1, Int>::empty([tile_touched_count as usize], device)
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
            point_indices.handle.to_owned().binding(),
            point_orders.handle.to_owned().binding(),
        ],
    );

    // Sorting the points by tile index and depth

    // ([T], [T])
    let SortStableOutput {
        keys: point_orders,
        values: point_indices,
    } = sort_stable(point_orders, point_indices);

    // Performing the forward pass #5

    // [I_y / T_y, I_x / T_x, 2]
    let tile_point_ranges = Tensor::<Wgpu, 3, Int>::empty(
        [tile_count_y as usize, tile_count_x as usize, 2],
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
            point_orders.handle.binding(),
            tile_point_ranges.handle.to_owned().binding(),
        ],
    );

    // Performing the forward pass #6

    let arguments = client.create(bytes_of(&Kernel6Arguments {
        image_size_x,
        image_size_y,
    }));
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
            point_indices.handle.to_owned().binding(),
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
            point_indices,
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

#[derive(Clone, Debug)]
pub struct ScanAddOutput {
    /// The sum of scanned values.
    pub sum: <Wgpu as Backend>::IntTensorPrimitive<1>,
    /// The exclusive sum of scanned values.
    pub sums: <Wgpu as Backend>::IntTensorPrimitive<1>,
}

/// Scan-and-add exclusively.
///
/// ## Arguments
///
/// - `sums`: The tensor to scan in place.
pub fn scan_add(
    // [N]
    sums: <Wgpu as Backend>::IntTensorPrimitive<1>,
) -> ScanAddOutput {
    const GROUP_SIZE: u32 = 256;

    // Specifying the parameters

    let client = &sums.client;
    let device = &sums.device;
    // N
    let count = sums.shape.dims[0];
    // N / N'
    let group_size = GROUP_SIZE as usize;
    // N'
    let count_next = (count + group_size - 1) / group_size;
    // [N']
    let sums_next =
        Tensor::<Wgpu, 1, Int>::empty([count_next], device).into_primitive();

    let cube_count = CubeCount::Static(count_next as u32, 1, 1);
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
        let ScanAddOutput {
            sum,
            sums: sums_next,
        } = scan_add(sums_next);

        // Adding

        client.execute(
            Box::new(SourceKernel::new(KernelScanAddAdd, cube_dim)),
            cube_count,
            vec![sums.handle.to_owned().binding(), sums_next.handle.binding()],
        );

        ScanAddOutput { sum, sums }
    } else {
        debug_assert_eq!(count_next, 1);
        ScanAddOutput {
            sum: sums_next,
            sums,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SortStableOutput {
    /// The keys of sorted items.
    pub keys: <Wgpu as Backend>::IntTensorPrimitive<1>,
    /// The values of sorted items.
    pub values: <Wgpu as Backend>::IntTensorPrimitive<1>,
}

/// Sort stably.
///
/// ## Arguments
///
/// - `keys`: The keys to sort.
pub fn sort_stable(
    // [N]
    keys: <Wgpu as Backend>::IntTensorPrimitive<1>,
    // [N]
    values: <Wgpu as Backend>::IntTensorPrimitive<1>,
) -> SortStableOutput {
    const GROUP_SIZE: u32 = 256;
    const RADIX_BIT_COUNT: u32 = 2;
    const RADIX: u32 = 1 << RADIX_BIT_COUNT;

    // Specifying the parameters

    let client = &keys.client.to_owned();
    let device = &keys.device.to_owned();
    // N
    let count = keys.shape.dims[0];
    // G
    let group_size = GROUP_SIZE as usize;
    // N / G
    let group_count = (count + group_size - 1) / group_size;
    // R
    let radix = RADIX as usize;
    // log2(R)
    let radix_bit_count = RADIX_BIT_COUNT as usize;

    // [N]
    let mut keys_input = keys;
    // [N]
    let mut values_input = values;
    // [N]
    let mut keys_output =
        Tensor::<Wgpu, 1, Int>::empty([count], device).into_primitive();
    // [N]
    let mut values_output =
        Tensor::<Wgpu, 1, Int>::empty([count], device).into_primitive();
    // [R, N / G]
    let counts_group_radix =
        Tensor::<Wgpu, 1, Int>::empty([radix * group_count], device)
            .into_primitive();
    // [N]
    let offsets_local =
        Tensor::<Wgpu, 1, Int>::empty([count], device).into_primitive();

    let cube_count = CubeCount::Static(group_count as u32, 1, 1);
    let cube_dim = CubeDim {
        x: GROUP_SIZE,
        y: 1,
        z: 1,
    };

    let mut is_swapped = false;

    for radix_bit_offset in (0..32).step_by(radix_bit_count) {
        let arguments = client
            .create(bytes_of(&KernelRadixSortArguments { radix_bit_offset }));

        client.execute(
            Box::new(SourceKernel::new(KernelRadixSortScanLocal, cube_dim)),
            cube_count.to_owned(),
            vec![
                arguments.to_owned().binding(),
                keys_input.handle.to_owned().binding(),
                counts_group_radix.handle.to_owned().binding(),
                offsets_local.handle.to_owned().binding(),
            ],
        );

        let offsets_group = scan_add(counts_group_radix.to_owned()).sums;

        client.execute(
            Box::new(SourceKernel::new(KernelRadixSortScatterKey, cube_dim)),
            cube_count.to_owned(),
            vec![
                arguments.to_owned().binding(),
                keys_input.handle.to_owned().binding(),
                values_input.handle.to_owned().binding(),
                offsets_group.handle.to_owned().binding(),
                offsets_local.handle.to_owned().binding(),
                keys_output.handle.to_owned().binding(),
                values_output.handle.to_owned().binding(),
            ],
        );

        (keys_input, keys_output) = (keys_output, keys_input);
        (values_input, values_output) = (values_output, values_input);
        is_swapped = !is_swapped;
    }

    let (keys, values) = if is_swapped {
        (keys_output, values_output)
    } else {
        (keys_input, values_input)
    };

    SortStableOutput { keys, values }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sort_stable() {
        use super::*;
        use rayon::slice::ParallelSliceMut;

        let device = &Default::default();
        Wgpu::seed(0);

        let keys = if false {
            Tensor::<Wgpu, 1, Int>::from_ints(
                [
                    0x12100, 0x21200, 0x32103, 0x23102, 0x13905, 0x31904,
                    0x31907, 0x23306, 0x10302, 0x20308, 0x12100, 0x21200,
                    0x32103, 0x23102, 0x13905, 0x31904, 0x31907, 0x23306,
                ],
                device,
            )
        } else {
            Tensor::<Wgpu, 1>::random(
                [1 << 22],
                burn::tensor::Distribution::Uniform(0.0, i32::MAX as f64),
                device,
            )
            .int()
        }
        .into_primitive();
        let values = Tensor::<Wgpu, 1, Int>::arange(
            0..keys.shape.dims[0] as i64,
            device,
        )
        .into_primitive();

        keys.client.sync(burn::tensor::backend::SyncType::Wait);
        let time = std::time::Instant::now();
        let keys_target = &keys.client.read(keys.handle.to_owned().binding());
        let keys_target = bytemuck::cast_slice::<u8, u32>(keys_target);
        let values_target =
            &values.client.read(values.handle.to_owned().binding());
        let values_target = bytemuck::cast_slice::<u8, u32>(values_target);
        let mut items_target = keys_target
            .iter()
            .zip(values_target)
            .map(|(&key, &value)| (key, value))
            .collect::<Vec<_>>();
        // keys_target.par_sort_unstable_by_key(|p| p.0);
        items_target.par_sort_by_key(|p| p.0);
        let (keys_target, values_target) =
            items_target.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
        println!("Sorting on CPU: {:?}", time.elapsed());

        keys.client.sync(burn::tensor::backend::SyncType::Wait);
        let time = std::time::Instant::now();
        let SortStableOutput { keys, values } = sort_stable(keys, values);
        keys.client.sync(burn::tensor::backend::SyncType::Wait);
        println!("Sorting on GPU: {:?}", time.elapsed());

        let keys_output = &keys.client.read(keys.handle.to_owned().binding());
        let keys_output = bytemuck::cast_slice::<u8, u32>(keys_output);

        let values_output =
            &values.client.read(values.handle.to_owned().binding());
        let values_output = bytemuck::cast_slice::<u8, u32>(values_output);

        keys_output.iter().zip(keys_target).enumerate().for_each(
            |(index, (&value, target))| {
                assert_eq!(value, target, "index: {index}");
            },
        );
        values_output
            .iter()
            .zip(values_target)
            .enumerate()
            .for_each(|(index, (&value, target))| {
                assert_eq!(value, target, "index: {index}");
            });
    }

    #[test]
    fn bench() {
        (0..100).for_each(|_| {
            sort_stable();
        });
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

        let sum_target = 24;
        let sums_target = [0, 0, 3, 3, 5, 9, 10, 13, 15];

        let ScanAddOutput { sum, sums } = scan_add(sums);
        let sum_output = *from_bytes::<u32>(
            &sums.client.read(sum.handle.to_owned().binding()),
        );
        let sums_output = sums.client.read(sums.handle.to_owned().binding());
        let sums_output = cast_slice::<u8, u32>(&sums_output);

        assert_eq!(sum_output, sum_target);
        sums_output.iter().zip(&sums_target).enumerate().for_each(
            |(index, (output, target))| {
                assert_eq!(output, target, "index: {index}");
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
            [1 << 23],
            Distribution::Uniform(0.0, 256.0),
            device,
        )
        .into_primitive();

        let sums_target = sums.client.read(sums.handle.to_owned().binding());
        let sums_target = cast_slice::<u8, u32>(&sums_target);
        let sum_target = sums_target.iter().sum::<u32>();
        let sums_target = sums_target
            .iter()
            .scan(0, |state, &sum| {
                let output = *state;
                *state += sum;
                Some(output)
            })
            .collect::<Vec<_>>();

        let ScanAddOutput { sum, sums } = scan_add(sums);
        let sum_output = *from_bytes::<u32>(
            &sums.client.read(sum.handle.to_owned().binding()),
        );
        let sums_output = sums.client.read(sums.handle.to_owned().binding());
        let sums_output = cast_slice::<u8, u32>(&sums_output);

        assert_eq!(sum_output, sum_target);
        sums_output.iter().zip(&sums_target).enumerate().for_each(
            |(index, (output, target))| {
                assert_eq!(output, target, "index: {index}");
            },
        );
    }
}
