pub use super::*;
pub use bytemuck::{Pod, Zeroable};

use burn::tensor::ops::IntTensorOps;
use bytemuck::bytes_of;
use std::mem::swap;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Arguments {
    /// `N / N'`
    pub block_count_group: u32,
    /// `(0 ~ 32: +log2(R))`
    pub radix_shift: u32,
}

#[derive(Clone, Debug)]
pub struct Inputs<R: JitRuntime, I: IntElement> {
    /// The keys of items to sort.
    pub keys: JitTensor<R, I>,
    /// The values of items to sort.
    pub values: JitTensor<R, I>,
}

#[derive(Clone, Debug)]
pub struct Outputs<R: JitRuntime, I: IntElement> {
    /// The keys of sorted items.
    pub keys: JitTensor<R, I>,
    /// The values of sorted items.
    pub values: JitTensor<R, I>,
}

/// `K`
pub const BLOCK_COUNT_GROUP_SHIFT: u32 = 14;
/// `G <- R`
pub const GROUP_SIZE: u32 = RADIX_COUNT;
/// `R`
pub const RADIX_COUNT: u32 = 1 << RADIX_COUNT_SHIFT;
/// `log2(R)`
pub const RADIX_COUNT_SHIFT: u32 = 8;

/// Sort the keys and values.
pub fn main<R: JitRuntime, F: FloatElement, I: IntElement>(
    inputs: Inputs<R, I>
) -> Outputs<R, I> {
    impl_kernel_source!(Kernel1, "kernel.1.wgsl");
    impl_kernel_source!(Kernel2, "kernel.2.wgsl");

    // Specifying the parameters

    // [N]
    let mut keys_input = inputs.keys;
    // [N]
    let mut values_input = inputs.values;
    let device = &keys_input.device.to_owned();
    // N
    let count = keys_input.shape.dims[0];
    // N / N' <- N / 2^K
    let block_count_group = (count as u32 >> BLOCK_COUNT_GROUP_SHIFT).max(1);
    // G * N / N'
    let block_size = block_count_group * GROUP_SIZE;
    // N' / G <- N / (G * N / N')
    let group_count = (count as u32 + block_size - 1) / block_size;

    let mut arguments = Arguments {
        block_count_group,
        radix_shift: 0,
    };
    // [N]
    let mut keys_output =
        JitBackend::<R, F, I>::int_empty([count].into(), device);
    // [N]
    let mut values_output =
        JitBackend::<R, F, I>::int_empty([count].into(), device);
    // [N' / G, R]
    let counts_radix_group = JitBackend::<R, F, I>::int_empty(
        [group_count as usize, RADIX_COUNT as usize].into(),
        device,
    );

    let pass = |radix_shift: u32| {
        // Specifying the parameters for the pass

        arguments.radix_shift = radix_shift;

        let client = &keys_input.client;
        let arguments = client.create(bytes_of(&arguments));

        // Launching the kernel 1

        client.execute(
            Box::new(SourceKernel::new(
                Kernel1,
                CubeDim {
                    x: GROUP_SIZE,
                    y: 1,
                    z: 1,
                },
            )),
            CubeCount::Static(group_count, 1, 1),
            vec![
                arguments.to_owned().binding(),
                keys_input.handle.to_owned().binding(),
                counts_radix_group.handle.to_owned().binding(),
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
            CubeCount::Static(group_count, 1, 1),
            vec![
                arguments.binding(),
                counts_radix_group.handle.to_owned().binding(),
                keys_input.handle.to_owned().binding(),
                values_input.handle.to_owned().binding(),
                keys_output.handle.to_owned().binding(),
                values_output.handle.to_owned().binding(),
            ],
        );

        // Swapping the input and output

        swap(&mut keys_input, &mut keys_output);
        swap(&mut values_input, &mut values_output);
    };

    (0..32).step_by(RADIX_COUNT_SHIFT as usize).for_each(pass);

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
        use crate::preset::backend::{Wgpu, WgpuDevice, WgpuRuntime};
        use burn::tensor::TensorData;
        use bytemuck::cast_slice;

        type B = Wgpu;
        type R = WgpuRuntime;
        type F = f32;
        type I = i32;
        let device = &WgpuDevice::default();

        let count = 14;

        let keys_source = vec![
            0x221a707e, 0x404673dd, 0x08f23dac, 0x79dc4824, 0x60986a48,
            0x6f358f8e, 0x61f1a696, 0x2255a70e, 0x3009911f, 0x3628f9f4,
            0x3c95798b, 0x561b9e2e, 0x41c02344, 0x168ff8d5,
        ];
        let values_source =
            (0..count as u32 * 10).step_by(10).collect::<Vec<_>>();

        let keys_target = vec![
            0x08f23dac, 0x168ff8d5, 0x221a707e, 0x2255a70e, 0x3009911f,
            0x3628f9f4, 0x3c95798b, 0x404673dd, 0x41c02344, 0x561b9e2e,
            0x60986a48, 0x61f1a696, 0x6f358f8e, 0x79dc4824,
        ];
        let values_target =
            vec![20, 130, 0, 70, 80, 90, 100, 10, 120, 110, 40, 60, 50, 30];

        let keys =
            B::int_from_data(TensorData::new(keys_source, [count]), device);
        let values =
            B::int_from_data(TensorData::new(values_source, [count]), device);
        let Outputs { keys, values } = main::<R, F, I>(Inputs { keys, values });
        let keys_output = &keys.client.read(keys.handle.to_owned().binding());
        let keys_output = cast_slice::<u8, u32>(keys_output);
        let values_output =
            &values.client.read(values.handle.to_owned().binding());
        let values_output = cast_slice::<u8, u32>(values_output);

        keys_output.iter().enumerate().try_fold(
            keys_output.iter().next().unwrap(),
            |previous, (index, current)| {
                let result = (previous <= current).then_some(previous);
                assert!(
                    result.is_some(),
                    "The key {previous} should be no more than {current}, index: {index}"
                );
                result
            },
        );
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
        use crate::preset::backend::{Wgpu, WgpuDevice, WgpuRuntime};
        use burn::tensor::TensorData;
        use bytemuck::cast_slice;
        use rand::{rngs::StdRng, Rng, SeedableRng};
        use rayon::slice::ParallelSliceMut;

        type B = Wgpu;
        type R = WgpuRuntime;
        type F = f32;
        type I = i32;
        let device = &WgpuDevice::default();

        let count = 1 << 25 | 2025;
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
            items_source.into_iter().unzip::<_, _, Vec<_>, Vec<_>>()
        };

        let keys =
            B::int_from_data(TensorData::new(keys_source, [count]), device);
        let values =
            B::int_from_data(TensorData::new(values_source, [count]), device);
        let Outputs { keys, values } = main::<R, F, I>(Inputs { keys, values });
        let keys_output = &keys.client.read(keys.handle.to_owned().binding());
        let keys_output = cast_slice::<u8, u32>(keys_output);
        let values_output =
            &values.client.read(values.handle.to_owned().binding());
        let values_output = cast_slice::<u8, u32>(values_output);

        keys_output.iter().enumerate().try_fold(
            keys_output.iter().next().unwrap(),
            |previous, (index, current)| {
                let result = (previous <= current).then_some(previous);
                assert!(
                    result.is_some(),
                    "The key {previous} should be no more than {current}, index: {index}"
                );
                result
            },
        );
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
