pub mod kernel;

pub use super::{backward, forward, Gaussian3dRenderOptions, View};
pub use crate::{
    backend::jit::{FloatElement, IntElement, JitBackend, JitRuntime},
    scene::gaussian_3d::SH_DEGREE_MAX,
};
pub use rank::TILE_COUNT_MAX;
pub use rasterize::{TILE_SIZE_X, TILE_SIZE_Y};
pub use transform::FILTER_LOW_PASS;

use burn::tensor::{ops::FloatTensorOps, TensorData};
use burn_jit::kernel::into_contiguous;
use kernel::*;

/// Maximum of `I_y * I_x`
pub const PIXEL_COUNT_MAX: u32 = TILE_SIZE_X * TILE_SIZE_Y * TILE_COUNT_MAX;

pub fn forward<R: JitRuntime, F: FloatElement, I: IntElement>(
    mut input: forward::RenderInput<JitBackend<R, F, I>>,
    view: &View,
    options: &Gaussian3dRenderOptions,
) -> forward::RenderOutput<JitBackend<R, F, I>> {
    #[cfg(debug_assertions)]
    log::debug!(target: "gausplat::renderer::gaussian_3d::forward", "start");

    // Specifying the arguments

    let colors_sh_degree_max = options.colors_sh_degree_max;
    // tan(Fov_x / 2)
    let field_of_view_x_half_tan = (view.field_of_view_x / 2.0).tan();
    // tan(Fov_y / 2)
    let field_of_view_y_half_tan = (view.field_of_view_y / 2.0).tan();
    // I_x
    let image_size_x = view.image_width;
    // I_y
    let image_size_y = view.image_height;
    // F_x <- I_x / tan(Fov_x / 2) / 2
    let focal_length_x =
        (image_size_x as f64 / field_of_view_x_half_tan / 2.0) as f32;
    // F_y <- I_y / tan(Fov_y / 2) / 2
    let focal_length_y =
        (image_size_y as f64 / field_of_view_y_half_tan / 2.0) as f32;
    // I_x / 2
    let image_size_half_x = (image_size_x as f64 / 2.0) as f32;
    // I_y / 2
    let image_size_half_y = (image_size_y as f64 / 2.0) as f32;
    // P
    let point_count = input.point_count as u32;
    // T_x
    let tile_size_x = TILE_SIZE_X;
    // T_y
    let tile_size_y = TILE_SIZE_Y;
    // I_x / T_x
    let tile_count_x = image_size_x.div_ceil(tile_size_x);
    // I_y / T_y
    let tile_count_y = image_size_y.div_ceil(tile_size_y);
    // tan(Fov_x / 2) * (C_f + 1)
    let view_bound_x =
        (field_of_view_x_half_tan * (FILTER_LOW_PASS + 1.0)) as f32;
    // tan(Fov_y / 2) * (C_f + 1)
    let view_bound_y =
        (field_of_view_y_half_tan * (FILTER_LOW_PASS + 1.0)) as f32;

    // TODO: These should be errors, not panics.
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

    let view_position = view.view_position;
    let view_transform = view
        .view_transform
        .iter()
        .flatten()
        .chain(&[view_position[0], view_position[1], view_position[2], 0.0])
        .copied()
        .collect::<Vec<f64>>();

    // [3 (+ 1), 3 + 1 + 1]
    let view_transform = JitBackend::<R, F, I>::float_from_data(
        TensorData::new(view_transform, [3 + 1, 3 + 1 + 1]),
        &input.device,
    );

    // Launching the kernels

    let outputs_transform = transform::main::<R, F, I>(
        transform::Arguments {
            colors_sh_degree_max,
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
            colors_sh: input.colors_sh.to_owned(),
            positions_3d: input.positions.to_owned(),
            rotations: input.rotations.to_owned(),
            scalings: input.scalings.to_owned(),
            view_transform: view_transform.to_owned(),
        },
    );
    #[cfg(debug_assertions)]
    log::debug!(target: "gausplat::renderer::gaussian_3d::forward", "transform");

    // Scanning the counts of the touched tiles into offsets

    let outputs_scan = scan::add::main::<R, F, I>(scan::add::Inputs {
        values: outputs_transform.tile_touched_counts,
    });

    #[cfg(debug_assertions)]
    log::info!(target: "gausplat::renderer::gaussian_3d::forward", "scan");

    let outputs_rank = rank::main::<R, F, I>(
        rank::Arguments {
            point_count,
            tile_count_x,
        },
        rank::Inputs {
            depths: outputs_transform.depths.to_owned(),
            radii: outputs_transform.radii.to_owned(),
            tile_touched_offsets: outputs_scan.values,
            tiles_touched_max: outputs_transform.tiles_touched_max,
            tiles_touched_min: outputs_transform.tiles_touched_min,
        },
    );
    #[cfg(debug_assertions)]
    log::debug!(target: "gausplat::renderer::gaussian_3d::forward", "rank");

    // Sorting the points by its tile index and depth

    let outputs_sort = sort::radix::main::<R, F, I>(sort::radix::Inputs {
        count: outputs_scan.total.to_owned(),
        keys: outputs_rank.point_orders,
        values: outputs_rank.point_indices,
    });
    #[cfg(debug_assertions)]
    log::debug!(target: "gausplat::renderer::gaussian_3d::forward", "sort");

    let outputs_segment = segment::main::<R, F, I>(
        segment::Arguments {
            tile_count_x,
            tile_count_y,
        },
        segment::Inputs {
            point_orders: outputs_sort.keys,
            tile_point_count: outputs_scan.total,
        },
    );
    #[cfg(debug_assertions)]
    log::debug!(target: "gausplat::renderer::gaussian_3d::forward", "segment");

    let outputs_rasterize = rasterize::main::<R, F, I>(
        rasterize::Arguments {
            image_size_x,
            image_size_y,
            tile_count_x,
            tile_count_y,
        },
        rasterize::Inputs {
            colors_rgb_3d: outputs_transform.colors_rgb_3d.to_owned(),
            conics: outputs_transform.conics.to_owned(),
            opacities_3d: input.opacities.to_owned(),
            point_indices: outputs_sort.values.to_owned(),
            positions_2d: outputs_transform.positions_2d.to_owned(),
            tile_point_ranges: outputs_segment.tile_point_ranges.to_owned(),
        },
    );
    #[cfg(debug_assertions)]
    log::debug!(target: "gausplat::renderer::gaussian_3d::forward", "rasterize");

    forward::RenderOutput {
        colors_rgb_2d: outputs_rasterize.colors_rgb_2d,
        state: backward::RenderInput {
            colors_rgb_3d: outputs_transform.colors_rgb_3d,
            colors_sh: input.colors_sh,
            colors_sh_degree_max,
            conics: outputs_transform.conics,
            depths: outputs_transform.depths,
            focal_length_x,
            focal_length_y,
            image_size_half_x,
            image_size_half_y,
            image_size_x,
            image_size_y,
            is_colors_rgb_3d_not_clamped: outputs_transform
                .is_colors_rgb_3d_not_clamped,
            opacities_3d: input.opacities,
            point_count,
            point_indices: outputs_sort.values,
            point_rendered_counts: outputs_rasterize.point_rendered_counts,
            positions_2d: outputs_transform.positions_2d,
            positions_3d: input.positions,
            radii: outputs_transform.radii,
            rotations: input.rotations,
            scalings: input.scalings,
            tile_count_x,
            tile_count_y,
            tile_point_ranges: outputs_segment.tile_point_ranges,
            transforms_2d: outputs_transform.transforms_2d,
            transmittances: outputs_rasterize.transmittances,
            view_bound_x,
            view_bound_y,
            view_directions: outputs_transform.view_directions,
            view_offsets: outputs_transform.view_offsets,
            view_rotation: view_transform,
        },
    }
}

/// ## Arguments
///
/// * `colors_rgb_2d_grad` - `[I_y, I_x, 3]`
pub fn backward<R: JitRuntime, F: FloatElement, I: IntElement>(
    state: backward::RenderInput<JitBackend<R, F, I>>,
    mut colors_rgb_2d_grad: JitTensor<R, F>,
) -> backward::RenderOutput<JitBackend<R, F, I>> {
    #[cfg(debug_assertions)]
    log::debug!(target: "gausplat::renderer::gaussian_3d::backward", "start");

    // Specifying the inputs

    colors_rgb_2d_grad = into_contiguous(colors_rgb_2d_grad);

    // Launching the kernels

    let outputs_rasterize_backward = rasterize_backward::main(
        rasterize_backward::Arguments {
            image_size_x: state.image_size_x,
            image_size_y: state.image_size_y,
            point_count: state.point_count,
            tile_count_x: state.tile_count_x,
            tile_count_y: state.tile_count_y,
        },
        rasterize_backward::Inputs {
            colors_rgb_2d_grad,
            colors_rgb_3d: state.colors_rgb_3d,
            conics: state.conics.to_owned(),
            opacities_3d: state.opacities_3d,
            point_indices: state.point_indices,
            point_rendered_counts: state.point_rendered_counts,
            positions_2d: state.positions_2d,
            tile_point_ranges: state.tile_point_ranges,
            transmittances: state.transmittances,
        },
    );
    #[cfg(debug_assertions)]
    log::debug!(target: "gausplat::renderer::gaussian_3d::backward", "rasterize_backward");

    let outputs_transform_backward = transform_backward::main(
        transform_backward::Arguments {
            colors_sh_degree_max: state.colors_sh_degree_max,
            focal_length_x: state.focal_length_x,
            focal_length_y: state.focal_length_y,
            image_size_half_x: state.image_size_half_x,
            image_size_half_y: state.image_size_half_y,
            point_count: state.point_count,
            view_bound_x: state.view_bound_x,
            view_bound_y: state.view_bound_y,
        },
        transform_backward::Inputs {
            colors_rgb_3d_grad: outputs_rasterize_backward.colors_rgb_3d_grad,
            colors_sh: state.colors_sh,
            conics: state.conics,
            conics_grad: outputs_rasterize_backward.conics_grad,
            depths: state.depths,
            is_colors_rgb_3d_not_clamped: state.is_colors_rgb_3d_not_clamped,
            positions_2d_grad: outputs_rasterize_backward.positions_2d_grad,
            positions_3d: state.positions_3d,
            radii: state.radii,
            rotations: state.rotations,
            scalings: state.scalings,
            transforms_2d: state.transforms_2d,
            view_directions: state.view_directions,
            view_offsets: state.view_offsets,
            view_rotation: state.view_rotation,
        },
    );
    #[cfg(debug_assertions)]
    log::debug!(target: "gausplat::renderer::gaussian_3d::backward", "transform_backward");

    backward::RenderOutput {
        colors_sh_grad: outputs_transform_backward.colors_sh_grad,
        opacities_grad: outputs_rasterize_backward.opacities_3d_grad,
        positions_2d_grad_norm: outputs_transform_backward
            .positions_2d_grad_norm,
        positions_grad: outputs_transform_backward.positions_3d_grad,
        rotations_grad: outputs_transform_backward.rotations_grad,
        scalings_grad: outputs_transform_backward.scalings_grad,
    }
}
