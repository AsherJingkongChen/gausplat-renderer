pub use super::*;
pub use bytemuck::{Pod, Zeroable};
pub use rasterize::{TILE_SIZE_X, TILE_SIZE_Y};

use burn::tensor::ops::FloatTensorOps;
use bytemuck::bytes_of;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Arguments {
    /// `I_x`
    pub image_size_x: u32,
    /// `I_y`
    pub image_size_y: u32,

    /// `P`
    pub point_count: u32,
    /// `I_x / T_x`
    pub tile_count_x: u32,
    /// `I_y / T_y`
    pub tile_count_y: u32,
}

#[derive(Clone, Debug)]
pub struct Inputs<R: JitRuntime, F: FloatElement, I: IntElement> {
    /// `[I_y, I_x, 3]`
    pub colors_rgb_2d_grad: JitTensor<R, F>,
    /// `[P, 3]`
    pub colors_rgb_3d: JitTensor<R, F>,
    /// `[P, 3]`
    pub conics: JitTensor<R, F>,
    /// `[P]`
    pub opacities_3d: JitTensor<R, F>,
    /// `[T]`
    pub point_indices: JitTensor<R, I>,
    /// `[I_y, I_x]`
    pub point_rendered_counts: JitTensor<R, I>,
    /// `[P, 2]`
    pub positions_2d: JitTensor<R, F>,
    /// `[I_y / T_y, I_x / T_x, 2]`
    pub tile_point_ranges: JitTensor<R, I>,
    /// `[I_y, I_x]`
    pub transmittances: JitTensor<R, F>,
}

#[derive(Clone, Debug)]
pub struct Outputs<R: JitRuntime, F: FloatElement> {
    /// `[P, 3]`
    pub colors_rgb_3d_grad: JitTensor<R, F>,
    /// `[P, 3]`
    pub conics_grad: JitTensor<R, F>,
    /// `[P]`
    pub opacities_3d_grad: JitTensor<R, F>,
    /// `[P, 2]`
    pub positions_2d_grad: JitTensor<R, F>,
}

/// Rasterizing the point to the image.
pub fn main<R: JitRuntime, F: FloatElement, I: IntElement>(
    arguments: Arguments,
    inputs: Inputs<R, F, I>,
) -> Outputs<R, F> {
    impl_kernel_source!(Kernel, "kernel.wgsl");

    // Specifying the parameters

    let client = &inputs.colors_rgb_2d_grad.client;
    let device = &inputs.colors_rgb_2d_grad.device;
    // P
    let point_count = arguments.point_count as usize;

    let colors_rgb_3d_grad =
        JitBackend::<R, F, I>::float_zeros([point_count, 3].into(), device);
    let conics_grad =
        JitBackend::<R, F, I>::float_zeros([point_count, 3].into(), device);
    let opacities_3d_grad =
        JitBackend::<R, F, I>::float_zeros([point_count, 1].into(), device);
    let positions_2d_grad =
        JitBackend::<R, F, I>::float_zeros([point_count, 2].into(), device);

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
            inputs.colors_rgb_2d_grad.handle.binding(),
            inputs.colors_rgb_3d.handle.binding(),
            inputs.conics.handle.binding(),
            inputs.opacities_3d.handle.binding(),
            inputs.point_indices.handle.binding(),
            inputs.point_rendered_counts.handle.binding(),
            inputs.positions_2d.handle.binding(),
            inputs.tile_point_ranges.handle.binding(),
            inputs.transmittances.handle.binding(),
            colors_rgb_3d_grad.handle.to_owned().binding(),
            conics_grad.handle.to_owned().binding(),
            opacities_3d_grad.handle.to_owned().binding(),
            positions_2d_grad.handle.to_owned().binding(),
        ],
    );

    Outputs {
        colors_rgb_3d_grad,
        conics_grad,
        opacities_3d_grad,
        positions_2d_grad,
    }
}
