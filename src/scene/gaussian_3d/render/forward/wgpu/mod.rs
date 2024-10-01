pub mod kernel;

pub use super::*;

use crate::preset::{backend::*, render::*};
use burn::tensor::ops::FloatTensorOps;
use burn_jit::{
    cubecl::{CubeCount, CubeDim},
    kernel::into_contiguous,
    template::SourceKernel,
};
use bytemuck::{bytes_of, from_bytes};
use kernel::*;

pub fn render_gaussian_3d_scene(
    mut input: forward::RenderInput<Wgpu>,
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
    // TODO: Import it from the scene
    let point_count = colors_sh.shape.dims[0] as u32;

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

    // Specifying the inputs

    input.colors_sh = into_contiguous(input.colors_sh);
    input.opacities = into_contiguous(input.opacities);
    input.positions = into_contiguous(input.positions);
    input.rotations = into_contiguous(input.rotations);
    input.scalings = into_contiguous(input.scalings);
    let view_position =
        Wgpu::float_from_data(view.view_position.into(), device);
    let view_transform =
        Wgpu::float_from_data(view.view_transform.into(), device);

    // Launching the kernels

    let outputs_transform = transform::main(
        transform::Arguments {
            colors_sh_degree_max,
            filter_low_pass,
            focal_length_x,
            focal_length_y,
            image_size_half_x,
            image_size_half_y,
            point_count,
            tile_count_x,
            tile_count_y,
            view_bound_x,
            view_bound_y,
        },
        transform::Inputs {
            colors_sh: input.colors_sh,
            positions_3d: input.positions,
            rotations: input.rotations,
            scalings: input.scalings,
            view_position,
            view_transform: view_transform.to_owned(),
        },
    );

    // Scanning the counts of the touched tiles into offsets

    let outputs_scan = scan::add::main(scan::add::Inputs {
        values: outputs_transform.tile_touched_counts,
    });

    // T
    let tile_point_count = *from_bytes::<u32>(
        &outputs_scan
            .total
            .client
            .read(outputs_scan.total.handle.binding()),
    );

    #[cfg(debug_assertions)]
    log::debug!(
        target: "gausplat_renderer::scene",
        "Gaussian3dRenderer::<Wgpu>::render_forward > tile_point_count ({tile_point_count})",
    );

    let outputs_rank = rank::main(
        rank::Arguments {
            point_count,
            tile_count_x,
            tile_point_count,
        },
        rank::Inputs {
            depths: outputs_transform.depths,
            radii: outputs_transform.radii,
            tile_touched_offsets: outputs_scan.values,
            tiles_touched_max: outputs_transform.tiles_touched_max,
            tiles_touched_min: outputs_transform.tiles_touched_min,
        },
    );

    // Sorting the points by its tile index and depth

    let outputs_sort = sort::radix::main(sort::radix::Inputs {
        keys: outputs_rank.point_indices,
        values: outputs_rank.point_orders,
    });

    let outputs_segment = segment::main(
        segment::Arguments {
            tile_count_x,
            tile_count_y,
            tile_point_count,
        },
        segment::Inputs {
            point_orders: outputs_sort.keys,
        },
    );
    
    let outputs_rasterize = rasterize::main(
        rasterize::Arguments {
            image_size_x,
            image_size_y,
            tile_count_x,
            tile_count_y,
        },
        rasterize::Inputs {
            colors_rgb_3d: outputs_transform.colors_rgb_3d,
            conics: outputs_transform.conics,
            opacities_3d: input.opacities,
            point_indices: outputs_sort.values,
            positions_2d: outputs_transform.positions_2d,
            tile_point_ranges: outputs_segment.tile_point_ranges,
        },
    );

    // Specifying the results of forward rendering

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
