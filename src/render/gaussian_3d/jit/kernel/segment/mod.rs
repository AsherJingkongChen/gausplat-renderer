pub use super::*;
pub use bytemuck::{Pod, Zeroable};

use burn::tensor::ops::IntTensorOps;
use bytemuck::bytes_of;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Arguments {
    /// `T`
    pub tile_point_count: u32,
    /// `I_x / T_x`
    pub tile_count_x: u32,
    /// `I_y / T_y`
    pub tile_count_y: u32,
}

#[derive(Clone, Debug)]
pub struct Inputs<R: JitRuntime, I: IntElement> {
    /// `[T]`
    pub point_orders: JitTensor<R, I>,
}

#[derive(Clone, Debug)]
pub struct Outputs<R: JitRuntime, I: IntElement> {
    /// `[I_y / T_y, I_x / T_x, 2]`
    pub tile_point_ranges: JitTensor<R, I>,
}

pub const GROUP_SIZE: u32 = 256;
pub const GROUP_SIZE2: u32 = GROUP_SIZE * GROUP_SIZE;

/// Segmenting the points into tiles.
pub fn main<R: JitRuntime, F: FloatElement, I: IntElement>(
    arguments: Arguments,
    inputs: Inputs<R, I>,
) -> Outputs<R, I> {
    impl_kernel_source!(Kernel, "kernel.wgsl");

    // Specifying the parameters

    let client = &inputs.point_orders.client;
    let device = &inputs.point_orders.device;

    // [I_y / T_y, I_x / T_x, 2]
    let tile_point_ranges = JitBackend::<R, F, I>::int_zeros(
        [
            arguments.tile_count_y as usize,
            arguments.tile_count_x as usize,
            2,
        ]
        .into(),
        device,
    );

    // Launching the kernel

    client.execute(
        Box::new(SourceKernel::new(
            Kernel,
            CubeDim {
                x: GROUP_SIZE,
                y: 1,
                z: 1,
            },
        )),
        CubeCount::Static(
            arguments.tile_point_count.div_ceil(GROUP_SIZE2),
            GROUP_SIZE,
            1,
        ),
        vec![
            client.create(bytes_of(&arguments)).binding(),
            inputs.point_orders.handle.binding(),
            tile_point_ranges.handle.to_owned().binding(),
        ],
    );

    Outputs { tile_point_ranges }
}
