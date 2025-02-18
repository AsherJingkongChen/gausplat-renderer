//! Segmenting the points into tiles.

pub use super::*;

use burn::tensor::ops::IntTensorOps;
use bytemuck::{Pod, Zeroable};

/// Arguments.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Arguments {
    /// `I_x / T_x`
    pub tile_count_x: u32,
    /// `I_y / T_y`
    pub tile_count_y: u32,
}

/// Inputs.
#[derive(Clone, Debug)]
pub struct Inputs<R: JitRuntime> {
    /// `[T]`
    pub point_orders: JitTensor<R>,
    /// `T`
    pub tile_point_count: JitTensor<R>,
}

/// Outputs.
#[derive(Clone, Debug)]
pub struct Outputs<R: JitRuntime> {
    /// `[I_y / T_y, I_x / T_x, 2]`
    pub tile_point_ranges: JitTensor<R>,
}

/// `G`
pub const GROUP_SIZE: u32 = 256;
/// `G ^ 2`
pub const GROUP_SIZE2: u32 = GROUP_SIZE * GROUP_SIZE;

/// Segment the points into tiles.
pub fn main<R: JitRuntime, F: FloatElement, I: IntElement, B: BoolElement>(
    arguments: Arguments,
    inputs: Inputs<R>,
) -> Outputs<R> {
    impl_kernel_source!(Kernel1, "kernel.1.wgsl");
    impl_kernel_source!(Kernel2, "kernel.2.wgsl");

    // Specifying the parameters

    let client = &inputs.point_orders.client;
    let device = &inputs.point_orders.device;

    // (T / G^2, G, 1)
    let group_count = JitBackend::<R, F, I, B>::int_empty([3].into(), device);
    // [I_y / T_y, I_x / T_x, 2]
    let tile_point_ranges = JitBackend::<R, F, I, B>::int_zeros(
        [
            arguments.tile_count_y as usize,
            arguments.tile_count_x as usize,
            2,
        ]
        .into(),
        device,
    );

    // Launching the kernel 1

    client.execute(
        Box::new(SourceKernel::new(Kernel1, CubeDim { x: 1, y: 1, z: 1 })),
        CubeCount::Static(1, 1, 1),
        vec![
            inputs.tile_point_count.handle.to_owned().binding(),
            group_count.handle.to_owned().binding(),
        ],
    );

    // Launching the kernel 2

    client.execute(
        Box::new(SourceKernel::new(
            Kernel2,
            CubeDim {
                x: GROUP_SIZE,
                y: 1,
                z: 1,
            },
        )),
        CubeCount::Dynamic(group_count.handle.binding()),
        vec![
            inputs.tile_point_count.handle.binding(),
            inputs.point_orders.handle.binding(),
            tile_point_ranges.handle.to_owned().binding(),
        ],
    );

    Outputs { tile_point_ranges }
}
