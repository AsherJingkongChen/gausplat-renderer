//! Rasterizing the point to the image.

pub use super::*;

use burn::tensor::ops::{FloatTensorOps, IntTensorOps};
use bytemuck::{bytes_of, Pod, Zeroable};

/// Arguments.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Arguments {
    /// `I_x`
    pub image_size_x: u32,
    /// `I_y`
    pub image_size_y: u32,

    /// `I_x / T_x`
    pub tile_count_x: u32,
    /// `I_y / T_y`
    pub tile_count_y: u32,
}

/// Inputs.
#[derive(Clone, Debug)]
pub struct Inputs<R: JitRuntime> {
    /// `[P, 3]`
    pub colors_rgb_3d: JitTensor<R>,
    /// `[P, 3]`
    pub conics: JitTensor<R>,
    /// `[P]`
    pub opacities_3d: JitTensor<R>,
    /// `[T]`
    pub point_indices: JitTensor<R>,
    /// `[P, 2]`
    pub positions_2d: JitTensor<R>,
    /// `[I_y / T_y, I_x / T_x, 2]`
    pub tile_point_ranges: JitTensor<R>,
}

/// Outputs.
#[derive(Clone, Debug)]
pub struct Outputs<R: JitRuntime> {
    /// `[I_y, I_x, 3]`
    pub colors_rgb_2d: JitTensor<R>,
    /// `[I_y, I_x]`
    pub point_rendered_counts: JitTensor<R>,
    /// `[I_y, I_x]`
    pub transmittances: JitTensor<R>,
}

/// `T_x`
pub const TILE_SIZE_X: u32 = 16;
/// `T_y`
pub const TILE_SIZE_Y: u32 = 16;

/// Rasterizing the point to the image.
pub fn main<R: JitRuntime, F: FloatElement, I: IntElement, B: BoolElement>(
    arguments: Arguments,
    inputs: Inputs<R>,
) -> Outputs<R> {
    impl_kernel_source!(Kernel, "kernel.wgsl");

    // Specifying the parameters

    let client = &inputs.colors_rgb_3d.client;
    let device = &inputs.colors_rgb_3d.device;
    // I_x
    let image_size_x = arguments.image_size_x as usize;
    // I_y
    let image_size_y = arguments.image_size_y as usize;

    // [I_x, I_y, 3]
    let colors_rgb_2d = JitBackend::<R, F, I, B>::float_empty(
        [image_size_y, image_size_x, 3].into(),
        device,
    );
    // [I_x, I_y]
    let point_rendered_counts =
        JitBackend::<R, F, I, B>::int_empty([image_size_y, image_size_x].into(), device);
    // [I_x, I_y]
    let transmittances = JitBackend::<R, F, I, B>::float_empty(
        [image_size_y, image_size_x].into(),
        device,
    );

    // Launching the kernel

    client.execute(
        Box::new(SourceKernel::new(
            Kernel,
            CubeDim {
                x: TILE_SIZE_X,
                y: TILE_SIZE_Y,
                z: 1,
            },
        )),
        CubeCount::Static(arguments.tile_count_x, arguments.tile_count_y, 1),
        vec![
            client.create(bytes_of(&arguments)).binding(),
            inputs.colors_rgb_3d.handle.binding(),
            inputs.conics.handle.binding(),
            inputs.opacities_3d.handle.binding(),
            inputs.point_indices.handle.binding(),
            inputs.positions_2d.handle.binding(),
            inputs.tile_point_ranges.handle.binding(),
            colors_rgb_2d.handle.to_owned().binding(),
            point_rendered_counts.handle.to_owned().binding(),
            transmittances.handle.to_owned().binding(),
        ],
    );

    Outputs {
        colors_rgb_2d,
        point_rendered_counts,
        transmittances,
    }
}
