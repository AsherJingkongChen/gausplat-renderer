pub use super::*;

use burn::backend::wgpu::WgpuRuntime;
use burn_jit::cubecl::{client::ComputeClient, Runtime};
use bytemuck::from_bytes;

pub type WgpuClient = ComputeClient<
    <WgpuRuntime as Runtime>::Server,
    <WgpuRuntime as Runtime>::Channel,
>;

/// Performing a scan-and-add operation on the given tensor in place. 
///
/// ## Arguments
/// 
/// - `client`: The client to execute the kernels.
/// - `sums`: The tensor to scan in place.
///
/// ## Returns
/// 
/// The total sum of all elements in `sums`.
pub fn sum_exclusive(
    client: &WgpuClient,
    sums: &<Wgpu as Backend>::IntTensorPrimitive<1>,
) -> u32 {
    let device = &sums.device;
    let group_size = GROUP_SIZE as usize;
    // N
    let count = sums.shape.dims[0];
    // N / G
    let count_next = (count + group_size - 1) / group_size;

    let sums_next =
        Tensor::<Wgpu, 1, Int>::zeros([count_next], device).into_primitive();

    // Scanning

    let cube_count = CubeCount::Static(count_next as u32, 1, 1);
    let cube_dim = CubeDim {
        x: GROUP_SIZE,
        y: 1,
        z: 1,
    };
    let bindings = vec![
        sums.handle.to_owned().binding(),
        sums_next.handle.to_owned().binding(),
    ];

    client.execute(
        Box::new(SourceKernel::new(KernelSumExclusiveScan, cube_dim)),
        cube_count.to_owned(),
        bindings.to_owned(),
    );

    // Recursing if there is more than one remaining group
    if count_next > 1 {
        let sum = sum_exclusive(&client, &sums_next);

        // Adding

        client.execute(
            Box::new(SourceKernel::new(KernelSumExclusiveAdd, cube_dim)),
            cube_count,
            bindings,
        );

        sum
    } else {
        debug_assert_eq!(count_next, 1);
        *from_bytes::<u32>(&client.read(sums_next.handle.binding()))
    }
}
