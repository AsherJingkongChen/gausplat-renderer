//! Radix sort.

pub use super::*;

use burn::tensor::ops::IntTensorOps;
use bytemuck::{bytes_of, Pod, Zeroable};
use std::mem::swap;

/// Arguments.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Arguments {
    /// `(0 ~ 32: +log2(R))`
    pub radix_shift: u32,
}

/// Inputs.
#[derive(Clone, Debug)]
pub struct Inputs<R: JitRuntime> {
    /// The count of items to sort.
    pub count: JitTensor<R>,
    /// The keys of items to sort.
    pub keys: JitTensor<R>,
    /// The values of items to sort.
    pub values: JitTensor<R>,
}

/// Outputs.
#[derive(Clone, Debug)]
pub struct Outputs<R: JitRuntime> {
    /// The keys of sorted items.
    pub keys: JitTensor<R>,
    /// The values of sorted items.
    pub values: JitTensor<R>,
}

/// `B <- G * W`
pub const BLOCK_SIZE: u32 = GROUP_SIZE * ROW_COUNT;
/// `C`
pub const CHUNK_SIZE: u32 = 64;
/// `G`
pub const GROUP_SIZE: u32 = 256;
/// `|Key|`
pub const KEY_BIT_COUNT: usize = size_of::<u32>() << 3;
/// `R`
pub const RADIX_COUNT: usize = 1 << RADIX_COUNT_SHIFT;
/// `log2(R)`
pub const RADIX_COUNT_SHIFT: u32 = 8;
/// `W`
// NOTE: The workgroup memory of the kernel 5 is `(B + R) * 4 = 16384` bytes.
pub const ROW_COUNT: u32 = 15;

/// Sort the items by the keys.
///
/// Radix-Sort-256 is a stable sort.
///
/// ## Details
///
/// The items are split into blocks of `B` items, one block for each group.
/// For each radix of the key:
///
/// 1. Count the radix in each block.
/// 2. Scan the counts into the offsets of each block, in chunks of `C` blocks.
/// 3. Put each block in the radix order in the workgroup memory,
///    and write it to the offsets, so that the threads write next to each other.
pub fn main<R: JitRuntime, F: FloatElement, I: IntElement, B: BoolElement>(
    inputs: Inputs<R>
) -> Outputs<R> {
    // NOTE: The key size is fixed.
    debug_assert_eq!(size_of::<I>(), KEY_BIT_COUNT >> 3);

    impl_kernel_source!(Kernel1, "kernel.1.wgsl");
    impl_kernel_source!(Kernel2, "kernel.2.wgsl");
    impl_kernel_source!(Kernel3, "kernel.3.wgsl");
    impl_kernel_source!(Kernel4, "kernel.4.wgsl");
    impl_kernel_source!(Kernel5, "kernel.5.wgsl");

    // Specifying the parameters

    // [N]
    let mut keys_input = inputs.keys;
    // [N]
    let mut values_input = inputs.values;
    // N
    let count = keys_input.shape.dims[0];
    let device = &keys_input.device.to_owned();
    // N / B
    let group_count_max = count.div_ceil(BLOCK_SIZE as usize).max(1);
    // N / B / C
    let chunk_count_max = group_count_max.div_ceil(CHUNK_SIZE as usize);

    let mut arguments = Arguments { radix_shift: 0 };
    // (N / B, 1, 1)
    let group_count = JitBackend::<R, F, I, B>::int_empty([3].into(), device);
    // (N / B / C, 1, 1)
    let chunk_count = JitBackend::<R, F, I, B>::int_empty([3].into(), device);
    // [N]
    let mut keys_output = JitBackend::<R, F, I, B>::int_empty([count].into(), device);
    // [N]
    let mut values_output = JitBackend::<R, F, I, B>::int_empty([count].into(), device);
    // [N / B, R]
    let counts_radix_group = JitBackend::<R, F, I, B>::int_empty(
        [group_count_max, RADIX_COUNT].into(),
        device,
    );
    // [N / B / C, R]
    let counts_radix_chunk = JitBackend::<R, F, I, B>::int_empty(
        [chunk_count_max, RADIX_COUNT].into(),
        device,
    );

    // Launching the kernel 1

    keys_input.client.execute(
        Box::new(SourceKernel::new(Kernel1, CubeDim { x: 1, y: 1, z: 1 })),
        CubeCount::Static(1, 1, 1),
        vec![
            inputs.count.handle.to_owned().binding(),
            group_count.handle.to_owned().binding(),
            chunk_count.handle.to_owned().binding(),
        ],
    );

    // Launching the kernel 2 to 5 iteratively

    for radix_shift in (0..KEY_BIT_COUNT as u32).step_by(RADIX_COUNT_SHIFT as usize) {
        // Specifying the parameters for the pass

        arguments.radix_shift = radix_shift;

        let client = &keys_input.client;
        let arguments = client.create(bytes_of(&arguments));
        let group_size = CubeDim {
            x: GROUP_SIZE,
            y: 1,
            z: 1,
        };

        // Launching the kernel 2

        client.execute(
            Box::new(SourceKernel::new(Kernel2, group_size)),
            CubeCount::Dynamic(group_count.handle.to_owned().binding()),
            vec![
                arguments.to_owned().binding(),
                inputs.count.handle.to_owned().binding(),
                keys_input.handle.to_owned().binding(),
                counts_radix_group.handle.to_owned().binding(),
            ],
        );

        // Launching the kernel 3

        client.execute(
            Box::new(SourceKernel::new(Kernel3, group_size)),
            CubeCount::Dynamic(chunk_count.handle.to_owned().binding()),
            vec![
                group_count.handle.to_owned().binding(),
                counts_radix_group.handle.to_owned().binding(),
                counts_radix_chunk.handle.to_owned().binding(),
            ],
        );

        // Launching the kernel 4

        client.execute(
            Box::new(SourceKernel::new(Kernel4, group_size)),
            CubeCount::Static(1, 1, 1),
            vec![
                chunk_count.handle.to_owned().binding(),
                counts_radix_chunk.handle.to_owned().binding(),
            ],
        );

        // Launching the kernel 5

        client.execute(
            Box::new(SourceKernel::new(Kernel5, group_size)),
            CubeCount::Dynamic(group_count.handle.to_owned().binding()),
            vec![
                arguments.binding(),
                inputs.count.handle.to_owned().binding(),
                counts_radix_group.handle.to_owned().binding(),
                counts_radix_chunk.handle.to_owned().binding(),
                keys_input.handle.to_owned().binding(),
                values_input.handle.to_owned().binding(),
                keys_output.handle.to_owned().binding(),
                values_output.handle.to_owned().binding(),
            ],
        );

        // Swapping the input and output

        swap(&mut keys_input, &mut keys_output);
        swap(&mut values_input, &mut values_output);
    }

    Outputs {
        keys: keys_input,
        values: values_input,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sort_stable_small() {
        use super::*;
        use crate::backend::{Wgpu, WgpuDevice, WgpuRuntime};
        use burn::tensor::TensorData;
        use bytemuck::cast_slice;

        type B = Wgpu;
        type R = WgpuRuntime;
        type F = f32;
        type I = i32;
        let device = &WgpuDevice::default();

        let count = 14;

        let keys_source = vec![
            0x221a707e, 0x404673dd, 0x08f23dac, 0x79dc4824, 0x60986a48, 0x6f358f8e,
            0x61f1a696, 0x2255a70e, 0x3009911f, 0x3628f9f4, 0x3c95798b, 0x561b9e2e,
            0x41c02344, 0x168ff8d5,
        ];
        let values_source = (0..count as u32 * 10).step_by(10).collect::<Vec<_>>();

        let keys_target = vec![
            0x08f23dac, 0x168ff8d5, 0x221a707e, 0x2255a70e, 0x3009911f, 0x3628f9f4,
            0x3c95798b, 0x404673dd, 0x41c02344, 0x561b9e2e, 0x60986a48, 0x61f1a696,
            0x6f358f8e, 0x79dc4824,
        ];
        let values_target =
            vec![20, 130, 0, 70, 80, 90, 100, 10, 120, 110, 40, 60, 50, 30];

        let keys = B::int_from_data(TensorData::new(keys_source, [count]), device);
        let values = B::int_from_data(TensorData::new(values_source, [count]), device);
        let count = B::int_from_data([count].into(), device);
        let Outputs { keys, values } = main::<R, F, I, u32>(Inputs {
            count,
            keys,
            values,
        });
        let keys_output = &keys.client.read(vec![keys.handle.to_owned().binding()])[0];
        let keys_output = cast_slice::<u8, u32>(keys_output);
        let values_output =
            &values.client.read(vec![values.handle.to_owned().binding()])[0];
        let values_output = cast_slice::<u8, u32>(values_output);

        // keys_output.iter().enumerate().try_fold(
        //     keys_output.iter().next().unwrap(),
        //     |previous, (index, current)| {
        //         let result = (previous <= current).then_some(previous);
        //         assert!(
        //             result.is_some(),
        //             "The key {previous} should be no more than {current}, index: {index}"
        //         );
        //         result
        //     },
        // );
        keys_output.iter().zip(&keys_target).enumerate().for_each(
            |(index, (&output, &target))| {
                assert_eq!(output, target, "key index: {index}");
            },
        );
        values_output
            .iter()
            .zip(&values_target)
            .enumerate()
            .for_each(|(index, (&output, &target))| {
                assert_eq!(output, target, "value index: {index}");
            });
    }

    #[test]
    fn sort_stable_random() {
        use super::*;
        use crate::backend::{Wgpu, WgpuDevice, WgpuRuntime};
        use burn::tensor::TensorData;
        use bytemuck::cast_slice;
        use rand::{rngs::StdRng, Rng, SeedableRng};
        use rayon::slice::ParallelSliceMut;

        type B = Wgpu;
        type R = WgpuRuntime;
        type F = f32;
        type I = i32;
        let device = &WgpuDevice::default();

        let count = (1 << 18) - 1;
        let keys_source = StdRng::from_entropy()
            .sample_iter(rand_distr::Uniform::new(0, i32::MAX as u32))
            .take(count)
            .collect::<Vec<_>>();
        let values_source = (0..count as u32).collect::<Vec<_>>();

        let (keys_target, values_target) = {
            let mut items_source = keys_source
                .iter()
                .zip(&values_source)
                .map(|(&key, &value)| (key, value))
                .collect::<Vec<_>>();
            items_source.par_sort_by_key(|p| p.0);
            items_source.into_iter().collect::<(Vec<_>, Vec<_>)>()
        };

        let keys = B::int_from_data(TensorData::new(keys_source, [count]), device);
        let values = B::int_from_data(TensorData::new(values_source, [count]), device);
        let count = B::int_from_data([count].into(), device);
        let Outputs { keys, values } = main::<R, F, I, u32>(Inputs {
            count,
            keys,
            values,
        });
        let keys_output = &keys.client.read(vec![keys.handle.to_owned().binding()])[0];
        let keys_output = cast_slice::<u8, u32>(keys_output);
        let values_output =
            &values.client.read(vec![values.handle.to_owned().binding()])[0];
        let values_output = cast_slice::<u8, u32>(values_output);

        // keys_output.iter().enumerate().try_fold(
        //     keys_output.iter().next().unwrap(),
        //     |previous, (index, current)| {
        //         let result = (previous <= current).then_some(previous);
        //         assert!(
        //             result.is_some(),
        //             "The key {previous} should be no more than {current}, index: {index}"
        //         );
        //         result
        //     },
        // );
        keys_output.iter().zip(&keys_target).enumerate().for_each(
            |(index, (&output, &target))| {
                assert_eq!(output, target, "key index: {index}");
            },
        );
        values_output
            .iter()
            .zip(&values_target)
            .enumerate()
            .for_each(|(index, (&output, &target))| {
                assert_eq!(output, target, "value index: {index}");
            });
    }
}
