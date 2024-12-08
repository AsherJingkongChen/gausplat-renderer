//! Ranking the points.

pub use super::*;

use burn::tensor::ops::IntTensorOps;
use bytemuck::{bytes_of, Pod, Zeroable};

/// Arguments.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Arguments {
    /// `P`
    pub point_count: u32,
    /// `I_x / T_x`
    pub tile_count_x: u32,
}

/// Inputs.
#[derive(Clone, Debug)]
pub struct Inputs<R: JitRuntime> {
    /// `[P]`
    pub depths: JitTensor<R>,
    /// `[P, 4]`
    pub point_tile_bounds: JitTensor<R>,
    /// `[P]`
    pub radii: JitTensor<R>,
    /// `[P]`
    pub tile_touched_offsets: JitTensor<R>,
}

/// Outputs.
#[derive(Clone, Debug)]
pub struct Outputs<R: JitRuntime> {
    /// `[T]`
    pub point_indices: JitTensor<R>,
    /// `[T]`
    pub point_orders: JitTensor<R>,
}

/// Group size.
pub const GROUP_SIZE: u32 = 256;
/// Maximum of `(I_y / T_y) * (I_x / T_x)`
pub const TILE_COUNT_MAX: u32 = 1 << 16;
/// `E[T / P]`
pub const FACTOR_TILE_POINT_COUNT: u32 = 65;

/// Ranking the points.
pub fn main<R: JitRuntime, F: FloatElement, I: IntElement, B: BoolElement>(
    arguments: Arguments,
    inputs: Inputs<R>,
) -> Outputs<R> {
    impl_kernel_source!(Kernel, "kernel.wgsl");

    // Specifying the parameters

    let client = &inputs.depths.client;
    let device = &inputs.depths.device;
    // E[T]
    // HACK: The actual tile point count should be less than the estimated value.
    let tile_point_count_estimated =
        (arguments.point_count * FACTOR_TILE_POINT_COUNT) as usize;

    // [T]
    let point_indices =
        JitBackend::<R, F, I, B>::int_empty([tile_point_count_estimated].into(), device);
    // [T]
    let point_orders =
        JitBackend::<R, F, I, B>::int_empty([tile_point_count_estimated].into(), device);

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
        CubeCount::Static(arguments.point_count.div_ceil(GROUP_SIZE), 1, 1),
        vec![
            client.create(bytes_of(&arguments)).binding(),
            inputs.depths.handle.binding(),
            inputs.point_tile_bounds.handle.binding(),
            inputs.radii.handle.binding(),
            inputs.tile_touched_offsets.handle.binding(),
            point_indices.handle.to_owned().binding(),
            point_orders.handle.to_owned().binding(),
        ],
    );

    Outputs {
        point_indices,
        point_orders,
    }
}
