//! JIT implementation of the 3DGS renderer.

pub mod kernel;

pub use super::{backward, forward, Gaussian3dRenderOptions, View};
pub use crate::error::Error;
pub use crate::{
    backend::jit::{FloatElement, IntElement, JitBackend, JitRuntime},
    scene::gaussian_3d::SH_DEGREE_MAX,
};
pub use rank::TILE_COUNT_MAX;
pub use rasterize::{TILE_SIZE_X, TILE_SIZE_Y};
pub use transform::FILTER_LOW_PASS;

use burn_jit::kernel::into_contiguous;
use kernel::*;

/// Maximum of `I_y * I_x`
pub const PIXEL_COUNT_MAX: u32 = TILE_SIZE_X * TILE_SIZE_Y * TILE_COUNT_MAX;

/// Forward pass for rendering the 3DGS.
pub fn forward<R: JitRuntime, F: FloatElement, I: IntElement, B: BoolElement>(
    mut input: forward::RenderInput<JitBackend<R, F, I, B>>,
    view: &View,
    options: &Gaussian3dRenderOptions,
) -> Result<forward::RenderOutput<JitBackend<R, F, I, B>>, Error> {
    #[cfg(all(debug_assertions, not(test)))]
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
    let focal_length_x = (image_size_x as f64 / field_of_view_x_half_tan / 2.0) as f32;
    // F_y <- I_y / tan(Fov_y / 2) / 2
    let focal_length_y = (image_size_y as f64 / field_of_view_y_half_tan / 2.0) as f32;
    // I_x / 2
    let image_size_half_x = (image_size_x as f64 / 2.0) as f32;
    // I_y / 2
    let image_size_half_y = (image_size_y as f64 / 2.0) as f32;
    // I_y * I_x
    let pixel_count = image_size_x as usize * image_size_y as usize;
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
    let view_bound_x = (field_of_view_x_half_tan * (FILTER_LOW_PASS + 1.0)) as f32;
    // tan(Fov_y / 2) * (C_f + 1)
    let view_bound_y = (field_of_view_y_half_tan * (FILTER_LOW_PASS + 1.0)) as f32;
    let view_position = view.view_position.map(|c| c as f32);
    let view_transform = view.view_transform.map(|c| c.map(|c| c as f32));

    if colors_sh_degree_max > SH_DEGREE_MAX {
        return Err(Error::UnsupportedSphericalHarmonicsDegree(
            colors_sh_degree_max,
        ));
    }
    if pixel_count == 0 || pixel_count > PIXEL_COUNT_MAX as usize {
        return Err(Error::InvalidPixelCount(pixel_count));
    }
    if point_count == 0 {
        return Err(Error::MismatchedPointCount(0, "non-zero".into()));
    }

    // Specifying the inputs

    input.colors_sh = into_contiguous(input.colors_sh);
    input.opacities = into_contiguous(input.opacities);
    input.positions = into_contiguous(input.positions);
    input.rotations = into_contiguous(input.rotations);
    input.scalings = into_contiguous(input.scalings);

    // Launching the kernels

    let outputs_transform = transform::main::<R, F, I, B>(
        transform::Arguments {
            colors_sh_degree_max,
            focal_length_x,
            focal_length_y,
            image_size_half_x,
            image_size_half_y,
            point_count,
            tile_count_x: tile_count_x as i32,
            tile_count_y: tile_count_y as i32,
            view_bound_x,
            view_bound_y,
            view_position,
            view_transform,
            _padding_1: Default::default(),
            _padding_2: Default::default(),
        },
        transform::Inputs {
            colors_sh: input.colors_sh.to_owned(),
            positions_3d: input.positions.to_owned(),
            rotations: input.rotations.to_owned(),
            scalings: input.scalings.to_owned(),
        },
    );
    #[cfg(all(debug_assertions, not(test)))]
    log::debug!(target: "gausplat::renderer::gaussian_3d::forward", "transform");

    // Scanning the counts of the touched tiles into offsets

    let outputs_scan = scan::add::main::<R, F, I, B>(scan::add::Inputs {
        values: outputs_transform.tile_touched_counts,
    });

    #[cfg(all(debug_assertions, not(test)))]
    log::info!(
        target: "gausplat::renderer::gaussian_3d::forward",
        "scan > total ({})",
        bytemuck::from_bytes::<u32>(
            &outputs_scan.total.client
                .read(vec![outputs_scan.total.handle.to_owned().binding()])[0],
        )
    );

    let outputs_rank = rank::main::<R, F, I, B>(
        rank::Arguments {
            point_count,
            tile_count_x,
        },
        rank::Inputs {
            depths: outputs_transform.depths.to_owned(),
            point_tile_bounds: outputs_transform.point_tile_bounds,
            radii: outputs_transform.radii.to_owned(),
            tile_touched_offsets: outputs_scan.values,
        },
    );
    #[cfg(all(debug_assertions, not(test)))]
    log::debug!(target: "gausplat::renderer::gaussian_3d::forward", "rank");

    // Sorting the points by its tile index and depth

    let outputs_sort = sort::radix::main::<R, F, I, B>(sort::radix::Inputs {
        count: outputs_scan.total.to_owned(),
        keys: outputs_rank.point_orders,
        values: outputs_rank.point_indices,
    });
    #[cfg(all(debug_assertions, not(test)))]
    log::debug!(target: "gausplat::renderer::gaussian_3d::forward", "sort");

    let outputs_segment = segment::main::<R, F, I, B>(
        segment::Arguments {
            tile_count_x,
            tile_count_y,
        },
        segment::Inputs {
            point_orders: outputs_sort.keys,
            tile_point_count: outputs_scan.total,
        },
    );
    #[cfg(all(debug_assertions, not(test)))]
    log::debug!(target: "gausplat::renderer::gaussian_3d::forward", "segment");

    let outputs_rasterize = rasterize::main::<R, F, I, B>(
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
    #[cfg(all(debug_assertions, not(test)))]
    log::debug!(target: "gausplat::renderer::gaussian_3d::forward", "rasterize");

    Ok(forward::RenderOutput {
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
            is_colors_rgb_3d_not_clamped: outputs_transform.is_colors_rgb_3d_not_clamped,
            opacities_3d: input.opacities,
            point_count,
            point_indices: outputs_sort.values,
            point_rendered_counts: outputs_rasterize.point_rendered_counts,
            positions_2d: outputs_transform.positions_2d,
            positions_3d: input.positions,
            positions_3d_in_normalized: outputs_transform.positions_3d_in_normalized,
            radii: outputs_transform.radii,
            rotations: input.rotations,
            rotations_matrix: outputs_transform.rotations_matrix,
            scalings: input.scalings,
            tile_count_x,
            tile_count_y,
            tile_point_ranges: outputs_segment.tile_point_ranges,
            transmittances: outputs_rasterize.transmittances,
            view_bound_x,
            view_bound_y,
            view_position,
            view_transform,
        },
    })
}

/// Backward pass for rendering the 3DGS.
///
/// It takes the outputs of the forward pass.
pub fn backward<R: JitRuntime, F: FloatElement, I: IntElement, B: BoolElement>(
    state: backward::RenderInput<JitBackend<R, F, I, B>>,
    mut colors_rgb_2d_grad: JitTensor<R>,
) -> backward::RenderOutput<JitBackend<R, F, I, B>> {
    #[cfg(all(debug_assertions, not(test)))]
    log::debug!(target: "gausplat::renderer::gaussian_3d::backward", "start");

    // Specifying the inputs

    colors_rgb_2d_grad = into_contiguous(colors_rgb_2d_grad);

    // Launching the kernels

    let outputs_rasterize_backward = rasterize_backward::main::<R, F, I, B>(
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
    #[cfg(all(debug_assertions, not(test)))]
    log::debug!(target: "gausplat::renderer::gaussian_3d::backward", "rasterize_backward");

    let outputs_transform_backward = transform_backward::main::<R, F, I, B>(
        transform_backward::Arguments {
            colors_sh_degree_max: state.colors_sh_degree_max,
            focal_length_x: state.focal_length_x,
            focal_length_y: state.focal_length_y,
            image_size_half_x: state.image_size_half_x,
            image_size_half_y: state.image_size_half_y,
            point_count: state.point_count,
            view_bound_x: state.view_bound_x,
            view_bound_y: state.view_bound_y,
            view_position: state.view_position,
            view_transform: state.view_transform,
            _padding_1: Default::default(),
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
            positions_3d_in_normalized: state.positions_3d_in_normalized,
            radii: state.radii,
            rotations: state.rotations,
            rotations_matrix: state.rotations_matrix,
            scalings: state.scalings,
        },
    );
    #[cfg(all(debug_assertions, not(test)))]
    log::debug!(target: "gausplat::renderer::gaussian_3d::backward", "transform_backward");

    backward::RenderOutput {
        colors_sh_grad: outputs_transform_backward.colors_sh_grad,
        opacities_grad: outputs_rasterize_backward.opacities_3d_grad,
        positions_2d_grad_norm: outputs_transform_backward.positions_2d_grad_norm,
        positions_grad: outputs_transform_backward.positions_3d_grad,
        rotations_grad: outputs_transform_backward.rotations_grad,
        scalings_grad: outputs_transform_backward.scalings_grad,
    }
}
