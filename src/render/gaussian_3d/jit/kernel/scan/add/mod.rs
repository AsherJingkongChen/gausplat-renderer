pub use super::*;

use burn::tensor::ops::IntTensorOps;

#[derive(Clone, Debug)]
pub struct Inputs<R: JitRuntime, I: IntElement> {
    /// The values to scan.
    pub values: JitTensor<R, I>,
}

#[derive(Clone, Debug)]
pub struct Outputs<R: JitRuntime, I: IntElement> {
    /// The exclusively scanned values.
    pub values: JitTensor<R, I>,
    /// The total of scanned values.
    pub total: JitTensor<R, I>,
}

/// `N / N'`
pub const GROUP_SIZE: u32 = 256;

/// Scanning the values exclusively.
pub fn main<R: JitRuntime, F: FloatElement, I: IntElement>(
    inputs: Inputs<R, I>
) -> Outputs<R, I> {
    impl_kernel_source!(Kernel1, "kernel.1.wgsl");
    impl_kernel_source!(Kernel2, "kernel.2.wgsl");

    // Specifying the parameters

    // [N]
    let values = inputs.values;
    let client = &values.client;
    let device = &values.device;
    // N
    let count = values.shape.dims[0] as u32;
    // N'
    let count_next = (count + GROUP_SIZE - 1) / GROUP_SIZE;

    // [N']
    let values_next =
        JitBackend::<R, F, I>::int_empty([count_next as usize].into(), device);

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
        CubeCount::Static(count_next, 1, 1),
        vec![
            values.handle.to_owned().binding(),
            values_next.handle.to_owned().binding(),
        ],
    );

    let total = if count_next > 1 {
        // Recursing if there is more than one remaining group

        let Outputs {
            total,
            values: values_next,
        } = main::<R, F, I>(Inputs {
            values: values_next,
        });

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
            CubeCount::Static(count_next, 1, 1),
            vec![
                values.handle.to_owned().binding(),
                values_next.handle.binding(),
            ],
        );

        total
    } else {
        // Returning the next values if there is only one group
        values_next
    };

    Outputs { total, values }
}

#[cfg(test)]
mod tests {
    #[test]
    fn scan_add_small() {
        use super::*;
        use crate::backend::{Wgpu, WgpuDevice, WgpuRuntime};
        use burn::tensor::TensorData;
        use bytemuck::{cast_slice, from_bytes};

        type B = Wgpu;
        type R = WgpuRuntime;
        type F = f32;
        type I = i32;
        let device = &WgpuDevice::default();

        let count = 9;
        let values_source = vec![0, 3, 0, 2, 4, 1, 3, 2, 9];

        let values_target = vec![0, 0, 3, 3, 5, 9, 10, 13, 15];
        let total_target = 24;

        let values =
            B::int_from_data(TensorData::new(values_source, [count]), device);
        let Outputs { total, values } = main::<R, F, I>(Inputs { values });
        let total_output = *from_bytes::<u32>(
            &total.client.read(total.handle.to_owned().binding()),
        );
        let values_output =
            total.client.read(values.handle.to_owned().binding());
        let values_output = cast_slice::<u8, u32>(&values_output);

        assert_eq!(total_output, total_target);
        values_output
            .iter()
            .zip(&values_target)
            .enumerate()
            .for_each(|(index, (output, target))| {
                assert_eq!(output, target, "index: {index}");
            });
    }

    #[test]
    fn scan_add_random() {
        use super::*;
        use crate::backend::{Wgpu, WgpuDevice, WgpuRuntime};
        use burn::tensor::TensorData;
        use bytemuck::{cast_slice, from_bytes};
        use rand::{rngs::StdRng, Rng, SeedableRng};

        type B = Wgpu;
        type R = WgpuRuntime;
        type F = f32;
        type I = i32;
        let device = &WgpuDevice::default();

        let count = (1 << 23) - 1;
        let values_source = StdRng::from_entropy()
            .sample_iter(rand_distr::Uniform::new(0, 1 << 8))
            .take(count)
            .collect::<Vec<_>>();

        let values_target = values_source
            .iter()
            .scan(0, |state, &sum| {
                let output = *state;
                *state += sum;
                Some(output)
            })
            .collect::<Vec<_>>();
        let total_target =
            values_target.last().unwrap() + values_source.last().unwrap();

        let values =
            B::int_from_data(TensorData::new(values_source, [count]), device);
        let Outputs { total, values } = main::<R, F, I>(Inputs { values });
        let total_output = *from_bytes::<u32>(
            &total.client.read(total.handle.to_owned().binding()),
        );
        let values_output =
            total.client.read(values.handle.to_owned().binding());
        let values_output = cast_slice::<u8, u32>(&values_output);

        assert_eq!(total_output, total_target);
        values_output
            .iter()
            .zip(&values_target)
            .enumerate()
            .for_each(|(index, (output, target))| {
                assert_eq!(output, target, "index: {index}");
            });
    }
}
