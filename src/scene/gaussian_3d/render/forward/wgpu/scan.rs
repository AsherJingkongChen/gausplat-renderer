pub use super::*;

use bytemuck::from_bytes;

/// Performing an exclusive scan-and-add operation on the given tensor in place.
///
/// ## Arguments
///
/// - `client`: The client to execute the kernels.
/// - `sums`: The tensor to scan in place.
///
/// ## Returns
///
/// The total sum of all elements in `sums`.
pub fn scan_add_exclusive(
    client: &WgpuClient,
    // [N]
    sums: &<Wgpu as Backend>::IntTensorPrimitive<1>,
) -> u32 {
    // Specifying the parameters

    let device = &sums.device;
    let group_size = GROUP_SIZE as usize;
    // N
    let count = sums.shape.dims[0];
    // N'
    let count_next = (count + group_size - 1) / group_size;
    // [N']
    let sums_next =
        Tensor::<Wgpu, 1, Int>::zeros([count_next], device).into_primitive();

    let cube_count = CubeCount::Static(count_next as u32, 1, 1);
    let cube_dim = CubeDim {
        x: group_size as u32,
        y: 1,
        z: 1,
    };

    // Scanning

    client.execute(
        Box::new(SourceKernel::new(KernelScanAddExclusiveScan, cube_dim)),
        cube_count.to_owned(),
        vec![
            sums.handle.to_owned().binding(),
            sums_next.handle.to_owned().binding(),
        ],
    );

    // Recursing if there is more than one remaining group
    if count_next > 1 {
        let sum = scan_add_exclusive(client, &sums_next);

        // Adding

        client.execute(
            Box::new(SourceKernel::new(KernelScanAddExclusiveAdd, cube_dim)),
            cube_count,
            vec![sums.handle.to_owned().binding(), sums_next.handle.binding()],
        );

        sum
    } else {
        debug_assert_eq!(count_next, 1);
        *from_bytes::<u32>(&client.read(sums_next.handle.binding()))
    }
}
