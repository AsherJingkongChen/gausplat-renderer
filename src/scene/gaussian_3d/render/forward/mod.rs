mod wgpu;

pub use super::*;

use crate::consts::render::*;
use burn::{
    backend::wgpu::{
        into_contiguous, Kernel, SourceKernel, WorkGroup, WorkgroupSize,
    },
    tensor::ops::{FloatTensor, IntTensor},
};
use bytemuck::{bytes_of, cast_slice, cast_slice_mut};
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

#[derive(Clone, Debug)]
pub struct RendererOutput<B: Backend> {
    /// `[I_Y, I_X, 3]`
    pub colors_rgb_2d: B::FloatTensorPrimitive<3>,
    /// `[P, 3 (+ 1)]`
    pub colors_rgb_3d: B::FloatTensorPrimitive<2>,
    /// `[P, 16, 3]`
    pub colors_sh: B::FloatTensorPrimitive<3>,
    /// `[P, 2, 2]`
    pub conics: B::FloatTensorPrimitive<3>,
    /// `[P, 3 (+ 1), 3]`
    pub covariances_3d: B::FloatTensorPrimitive<3>,
    /// `I_X`
    pub image_size_x: u32,
    /// `I_Y`
    pub image_size_y: u32,
    /// `[P, 3 (+ 1)]`
    pub is_colors_rgb_3d_clamped: B::FloatTensorPrimitive<2>,
    /// `[P, 1]`
    pub opacities_3d: B::FloatTensorPrimitive<2>,
    /// `P`
    pub point_count: u32,
    /// `[T]`
    pub point_indexes: B::IntTensorPrimitive<1>,
    /// `[I_Y, I_X]`
    pub point_rendered_counts: B::IntTensorPrimitive<2>,
    /// `[P, 2]`
    pub positions_2d: B::FloatTensorPrimitive<2>,
    /// `[P, 3]`
    pub positions_3d: B::FloatTensorPrimitive<2>,
    /// `[P]`
    pub radii: B::IntTensorPrimitive<1>,
    /// `[P, 4]`
    pub rotations: B::FloatTensorPrimitive<2>,
    /// `[P, 3]`
    pub scalings: B::FloatTensorPrimitive<2>,
    /// `[(I_X / T_X) * (I_Y / T_Y), 2]`
    pub tile_point_ranges: B::IntTensorPrimitive<2>,
    /// `[I_Y, I_X]`
    pub transmittances: B::FloatTensorPrimitive<2>,
    pub view_bound_x: f32,
    pub view_bound_y: f32,
    /// `[3]`
    pub view_position: B::FloatTensorPrimitive<1>,
    /// `[4, 4]`
    pub view_transform: B::FloatTensorPrimitive<2>,
}

pub(super) fn render_gaussian_3d_scene_wgpu(
    scene: &Gaussian3dScene<Wgpu>,
    view: &sparse_view::View,
    options: &RendererOptions,
) -> forward::RendererOutput<Wgpu> {
    use wgpu::*;

    // Specifying the parameters

    let colors_sh_degree_max = options.colors_sh_degree_max;
    let field_of_view_x_half_tan = (view.field_of_view_x / 2.0).tan();
    let field_of_view_y_half_tan = (view.field_of_view_y / 2.0).tan();
    let filter_low_pass = FILTER_LOW_PASS as f32;
    // I_X
    let image_size_x = view.image_width as usize;
    // I_Y
    let image_size_y = view.image_height as usize;
    // I_X / 2
    let image_size_half_x = (image_size_x as f64 / 2.0) as f32;
    // I_Y / 2
    let image_size_half_y = (image_size_y as f64 / 2.0) as f32;
    let focal_length_x =
        (image_size_x as f64 / field_of_view_x_half_tan / 2.0) as f32;
    let focal_length_y =
        (image_size_y as f64 / field_of_view_y_half_tan / 2.0) as f32;
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
        let p = into_contiguous(scene.colors_sh().into_primitive());
        (p.client, p.handle, p.device, p.shape.dims[0])
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
    let positions_3d = into_contiguous(scene.positions().into_primitive()).handle;
    // [P, 1]
    let opacities_3d = into_contiguous(scene.opacities().into_primitive()).handle;
    // [P, 4]
    let rotations = into_contiguous(scene.rotations().into_primitive()).handle;
    // [P, 3]
    let scalings = into_contiguous(scene.scalings().into_primitive()).handle;
    // [3]
    let view_position =
        client.create(bytes_of(&view.view_position.map(|c| c as f32)));
    // [4, 4]
    let view_transform = {
        let view_transform_in_column_major = [
            view.view_transform.map(|r| r[0] as f32),
            view.view_transform.map(|r| r[1] as f32),
            view.view_transform.map(|r| r[2] as f32),
            view.view_transform.map(|r| r[3] as f32),
        ];
        client.create(bytes_of(&view_transform_in_column_major))
    };
    // [P, 3 (+ 1)] (the alignment of vec3<f32> is 16 = (3 + 1) * 4 bytes)
    let colors_rgb_3d = client.empty(point_count * (3 + 1) * 4);
    // [P, 2, 2]
    let conics = client.empty(point_count * 2 * 2 * 4);
    // [P, 3 (+ 1), 3] (the alignment of mat3x3<f32> is 16 = (3 + 1) * 4 bytes)
    let covariances_3d = client.empty(point_count * (3 + 1) * 3 * 4);
    // [P]
    let depths = client.empty(point_count * 4);
    // [P, 3 (+ 1)] (the alignment of vec3<f32> is 16 = (3 + 1) * 4 bytes)
    let is_colors_rgb_3d_clamped = client.empty(point_count * (3 + 1) * 4);
    // [P, 2]
    let positions_2d = client.empty(point_count * 2 * 4);
    // [P]
    let radii = client.empty(point_count * 4);
    // [P]
    let tile_touched_counts = client.empty(point_count * 4);
    // [P, 2]
    let tiles_touched_max = client.empty(point_count * 2 * 4);
    // [P, 2]
    let tiles_touched_min = client.empty(point_count * 2 * 4);

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
            &radii,
            &tile_touched_counts,
            &tiles_touched_max,
            &tiles_touched_min,
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

    let output = RendererOutput {
        // [I_Y, I_X, 3]
        colors_rgb_2d: FloatTensor::<Wgpu, 3>::new(
            client.to_owned(),
            device.to_owned(),
            [image_size_y, image_size_x, 3].into(),
            colors_rgb_2d,
        ),
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
        // I_X
        image_size_x: image_size_x as u32,
        // I_Y
        image_size_y: image_size_y as u32,
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
        // P
        point_count: point_count as u32,
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
        // [I_Y, I_X]
        transmittances: FloatTensor::<Wgpu, 2>::new(
            client.to_owned(),
            device.to_owned(),
            [image_size_y, image_size_x].into(),
            transmittances,
        ),
        view_bound_x,
        view_bound_y,
        // [3]
        view_position: FloatTensor::<Wgpu, 1>::new(
            client.to_owned(),
            device.to_owned(),
            [3].into(),
            view_position,
        ),
        // [4, 4]
        view_transform: FloatTensor::<Wgpu, 2>::new(
            client.to_owned(),
            device.to_owned(),
            [4, 4].into(),
            view_transform,
        ),
    };

    output
}
