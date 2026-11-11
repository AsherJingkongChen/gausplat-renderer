//! Sorters under test: the sort in the crate, and the original one (baseline).
//!
//! Written by Anzu on 2026-11-11.
//! The steps tried between the two are on the branch `anzu/radix-experiments`.

#![allow(dead_code)]

pub use burn::tensor::{backend::Backend, ops::IntTensorOps, TensorData};
pub use gausplat_renderer::{
    backend::{Wgpu, WgpuDevice, WgpuRuntime},
    render::gaussian_3d::jit::kernel::sort::radix::{main, Arguments, Inputs, Outputs},
};

use burn_jit::{
    cubecl::{CubeCount, CubeDim},
    template::SourceKernel,
};
use bytemuck::bytes_of;
use gausplat_renderer::render::gaussian_3d::jit::kernel::{
    KernelId, KernelSource, SourceTemplate,
};

pub type B = Wgpu;
pub type R = WgpuRuntime;
pub type F = f32;
pub type I = i32;
pub type Sorter = fn(Inputs<R>, &WgpuDevice) -> Outputs<R>;

macro_rules! impl_kernel_source {
    ($kernel: ident, $source_path: expr) => {
        #[derive(Clone, Copy, Debug, Default)]
        struct $kernel;

        impl KernelSource for $kernel {
            fn source(&self) -> SourceTemplate {
                SourceTemplate::new(include_str!($source_path))
            }

            fn id(&self) -> KernelId {
                KernelId::new::<Self>()
            }
        }
    };
}

impl_kernel_source!(BaselineKernel1, "../radix_baseline/kernel.1.wgsl");
impl_kernel_source!(BaselineKernel2, "../radix_baseline/kernel.2.wgsl");
impl_kernel_source!(BaselineKernel3, "../radix_baseline/kernel.3.wgsl");

/// The sort in the crate.
pub fn sort_current(
    inputs: Inputs<R>,
    _device: &WgpuDevice,
) -> Outputs<R> {
    main::<R, F, I, u32>(inputs)
}

/// The original sort at the commit f6cb361, with its host loop copied as is.
///
/// It needs the subgroup operations of the device.
pub fn sort_baseline(
    inputs: Inputs<R>,
    device: &WgpuDevice,
) -> Outputs<R> {
    let group_size = CubeDim {
        x: 256,
        y: 1,
        z: 1,
    };
    let mut keys_input = inputs.keys;
    let mut values_input = inputs.values;
    let capacity = keys_input.shape.dims[0];
    let group_count = B::int_empty([3].into(), device);
    let mut keys_output = B::int_empty([capacity].into(), device);
    let mut values_output = B::int_empty([capacity].into(), device);
    let counts_radix_group = B::int_empty([128, 256].into(), device);

    keys_input.client.execute(
        Box::new(SourceKernel::new(
            BaselineKernel1,
            CubeDim { x: 1, y: 1, z: 1 },
        )),
        CubeCount::Static(1, 1, 1),
        vec![
            inputs.count.handle.to_owned().binding(),
            group_count.handle.to_owned().binding(),
        ],
    );
    for radix_shift in (0..32).step_by(8) {
        let client = &keys_input.client;
        let arguments = client.create(bytes_of(&Arguments { radix_shift }));
        client.execute(
            Box::new(SourceKernel::new(BaselineKernel2, group_size)),
            CubeCount::Dynamic(group_count.handle.to_owned().binding()),
            vec![
                arguments.to_owned().binding(),
                inputs.count.handle.to_owned().binding(),
                keys_input.handle.to_owned().binding(),
                counts_radix_group.handle.to_owned().binding(),
            ],
        );
        client.execute(
            Box::new(SourceKernel::new(BaselineKernel3, group_size)),
            CubeCount::Dynamic(group_count.handle.to_owned().binding()),
            vec![
                arguments.binding(),
                inputs.count.handle.to_owned().binding(),
                counts_radix_group.handle.to_owned().binding(),
                keys_input.handle.to_owned().binding(),
                values_input.handle.to_owned().binding(),
                keys_output.handle.to_owned().binding(),
                values_output.handle.to_owned().binding(),
            ],
        );
        std::mem::swap(&mut keys_input, &mut keys_output);
        std::mem::swap(&mut values_input, &mut values_output);
    }

    Outputs {
        keys: keys_input,
        values: values_input,
    }
}

/// The sorters to compare, by name.
pub const SORTERS: [(&str, Sorter); 2] = [("baseline", sort_baseline), ("current", sort_current)];
