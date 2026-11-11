//! Sorters under test: the sort in the crate, the original one (baseline 1),
//! and the experimental one with a separate scan kernel (v2).
//!
//! Temporary file written by Anzu on 2026-11-11 for research; not part of the crate.

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
use std::{sync::Mutex, time::Instant};
use gausplat_renderer::render::gaussian_3d::jit::kernel::{
    KernelId, KernelSource, SourceTemplate,
};

pub type B = Wgpu;
pub type R = WgpuRuntime;
pub type F = f32;
pub type I = i32;
pub type Sorter = fn(Inputs<R>, &WgpuDevice) -> Outputs<R>;

/// A kernel source whose `BLOCK_COUNT_GROUP_SHIFT` is replaced by `shift`.
#[derive(Clone, Copy, Debug)]
struct Kernel {
    name: &'static str,
    shift: u32,
    source: &'static str,
}

impl KernelSource for Kernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(self.source.replace(
            "const BLOCK_COUNT_GROUP_SHIFT: u32 = 14;",
            &format!("const BLOCK_COUNT_GROUP_SHIFT: u32 = {};", self.shift),
        ))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.name, self.shift))
    }
}

fn make_kernel(
    name: &'static str,
    shift: u32,
    source: &'static str,
    group_size: u32,
) -> Box<SourceKernel<Kernel>> {
    Box::new(SourceKernel::new(
        Kernel {
            name,
            shift,
            source,
        },
        CubeDim {
            x: group_size,
            y: 1,
            z: 1,
        },
    ))
}

/// The time spent in each stage in ms, as (count, scan, scatter), collected only
/// when `RADIX_STAGE` is set, by syncing after each kernel.
pub static STAGE_TIMES: Mutex<[f64; 3]> = Mutex::new([0.0; 3]);
static STAGE_START: Mutex<Option<Instant>> = Mutex::new(None);

fn lap(
    stage: Option<usize>,
    device: &WgpuDevice,
) {
    if std::env::var("RADIX_STAGE").is_err() {
        return;
    }
    B::sync(device);
    let mut start = STAGE_START.lock().unwrap();
    if let (Some(stage), Some(start)) = (stage, *start) {
        STAGE_TIMES.lock().unwrap()[stage] += start.elapsed().as_secs_f64() * 1e3;
    }
    *start = Some(Instant::now());
}

/// The sort in the crate.
pub fn sort_current(
    inputs: Inputs<R>,
    _device: &WgpuDevice,
) -> Outputs<R> {
    main::<R, F, I, u32>(inputs)
}

/// The original sort at the commit f6cb361, with its host loop copied as is.
pub fn sort_baseline(
    inputs: Inputs<R>,
    device: &WgpuDevice,
) -> Outputs<R> {
    const SHIFT: u32 = 14;

    let mut keys_input = inputs.keys;
    let mut values_input = inputs.values;
    let capacity = keys_input.shape.dims[0];
    let group_count = B::int_empty([3].into(), device);
    let mut keys_output = B::int_empty([capacity].into(), device);
    let mut values_output = B::int_empty([capacity].into(), device);
    let counts_radix_group = B::int_empty([128, 256].into(), device);

    keys_input.client.execute(
        make_kernel(
            "baseline.1",
            SHIFT,
            include_str!("../radix_baseline/kernel.1.wgsl"),
            1,
        ),
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
            make_kernel(
                "baseline.2",
                SHIFT,
                include_str!("../radix_baseline/kernel.2.wgsl"),
                256,
            ),
            CubeCount::Dynamic(group_count.handle.to_owned().binding()),
            vec![
                arguments.to_owned().binding(),
                inputs.count.handle.to_owned().binding(),
                keys_input.handle.to_owned().binding(),
                counts_radix_group.handle.to_owned().binding(),
            ],
        );
        client.execute(
            make_kernel(
                "baseline.3",
                SHIFT,
                include_str!("../radix_baseline/kernel.3.wgsl"),
                256,
            ),
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

/// The experimental sort: count, scan, scatter, with `2^SHIFT / 256` groups or more.
pub fn sort_v2<const SHIFT: u32>(
    inputs: Inputs<R>,
    device: &WgpuDevice,
) -> Outputs<R> {
    sort_with_scan::<SHIFT>(inputs, device, "v2.4", include_str!("v2/kernel.4.wgsl"))
}

/// The experimental sort v2 whose scatter has two barriers for each block, not three.
pub fn sort_v3<const SHIFT: u32>(
    inputs: Inputs<R>,
    device: &WgpuDevice,
) -> Outputs<R> {
    sort_with_scan::<SHIFT>(inputs, device, "v3.4", include_str!("v3/kernel.4.wgsl"))
}

fn sort_with_scan<const SHIFT: u32>(
    inputs: Inputs<R>,
    device: &WgpuDevice,
    scatter_name: &'static str,
    scatter_source: &'static str,
) -> Outputs<R> {
    let mut keys_input = inputs.keys;
    let mut values_input = inputs.values;
    let capacity = keys_input.shape.dims[0];
    let group_count_max = (2usize << SHIFT) / 256;
    let group_count = B::int_empty([3].into(), device);
    let mut keys_output = B::int_empty([capacity].into(), device);
    let mut values_output = B::int_empty([capacity].into(), device);
    let counts_radix_group = B::int_empty([group_count_max, 256].into(), device);

    keys_input.client.execute(
        make_kernel("v2.1", SHIFT, include_str!("v2/kernel.1.wgsl"), 1),
        CubeCount::Static(1, 1, 1),
        vec![
            inputs.count.handle.to_owned().binding(),
            group_count.handle.to_owned().binding(),
        ],
    );
    lap(None, device);
    for radix_shift in (0..32).step_by(8) {
        let client = &keys_input.client;
        let arguments = client.create(bytes_of(&Arguments { radix_shift }));
        client.execute(
            make_kernel("v2.2", SHIFT, include_str!("v2/kernel.2.wgsl"), 256),
            CubeCount::Dynamic(group_count.handle.to_owned().binding()),
            vec![
                arguments.to_owned().binding(),
                inputs.count.handle.to_owned().binding(),
                keys_input.handle.to_owned().binding(),
                counts_radix_group.handle.to_owned().binding(),
            ],
        );
        lap(Some(0), device);
        client.execute(
            make_kernel("v2.3", SHIFT, include_str!("v2/kernel.3.wgsl"), 256),
            CubeCount::Static(1, 1, 1),
            vec![
                group_count.handle.to_owned().binding(),
                counts_radix_group.handle.to_owned().binding(),
            ],
        );
        lap(Some(1), device);
        client.execute(
            make_kernel(scatter_name, SHIFT, scatter_source, 256),
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
        lap(Some(2), device);
        std::mem::swap(&mut keys_input, &mut keys_output);
        std::mem::swap(&mut values_input, &mut values_output);
    }

    Outputs {
        keys: keys_input,
        values: values_input,
    }
}

/// A kernel source whose `BLOCK_COUNT_GROUP_SHIFT` and `RADIX_COUNT_SHIFT` are replaced.
#[derive(Clone, Copy, Debug)]
struct KernelRadix {
    name: &'static str,
    shift: u32,
    bits: u32,
    source: &'static str,
}

impl KernelSource for KernelRadix {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(
            self.source
                .replace(
                    "const BLOCK_COUNT_GROUP_SHIFT: u32 = 14;",
                    &format!("const BLOCK_COUNT_GROUP_SHIFT: u32 = {};", self.shift),
                )
                .replace(
                    "const RADIX_COUNT_SHIFT: u32 = 8;",
                    &format!("const RADIX_COUNT_SHIFT: u32 = {};", self.bits),
                ),
        )
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.name, self.shift, self.bits))
    }
}

/// The experimental sort v3 with `BITS` bits for each pass, not 8.
pub fn sort_v4<const SHIFT: u32, const BITS: u32>(
    inputs: Inputs<R>,
    device: &WgpuDevice,
) -> Outputs<R> {
    let make = |name: &'static str, source: &'static str, group_size: u32| {
        Box::new(SourceKernel::new(
            KernelRadix {
                name,
                shift: SHIFT,
                bits: BITS,
                source,
            },
            CubeDim {
                x: group_size,
                y: 1,
                z: 1,
            },
        ))
    };
    let mut keys_input = inputs.keys;
    let mut values_input = inputs.values;
    let capacity = keys_input.shape.dims[0];
    let group_count_max = (2usize << SHIFT) / 256;
    let group_count = B::int_empty([3].into(), device);
    let mut keys_output = B::int_empty([capacity].into(), device);
    let mut values_output = B::int_empty([capacity].into(), device);
    let counts_radix_group =
        B::int_empty([group_count_max, 1 << BITS].into(), device);

    keys_input.client.execute(
        make("v4.1", include_str!("v4/kernel.1.wgsl"), 1),
        CubeCount::Static(1, 1, 1),
        vec![
            inputs.count.handle.to_owned().binding(),
            group_count.handle.to_owned().binding(),
        ],
    );
    lap(None, device);
    for radix_shift in (0..32).step_by(BITS as usize) {
        let client = &keys_input.client;
        let arguments = client.create(bytes_of(&Arguments { radix_shift }));
        client.execute(
            make("v4.2", include_str!("v4/kernel.2.wgsl"), 256),
            CubeCount::Dynamic(group_count.handle.to_owned().binding()),
            vec![
                arguments.to_owned().binding(),
                inputs.count.handle.to_owned().binding(),
                keys_input.handle.to_owned().binding(),
                counts_radix_group.handle.to_owned().binding(),
            ],
        );
        lap(Some(0), device);
        client.execute(
            make("v4.3", include_str!("v4/kernel.3.wgsl"), 1 << BITS),
            CubeCount::Static(1, 1, 1),
            vec![
                group_count.handle.to_owned().binding(),
                counts_radix_group.handle.to_owned().binding(),
            ],
        );
        lap(Some(1), device);
        client.execute(
            make("v4.4", include_str!("v4/kernel.4.wgsl"), 256),
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
        lap(Some(2), device);
        std::mem::swap(&mut keys_input, &mut keys_output);
        std::mem::swap(&mut values_input, &mut values_output);
    }

    Outputs {
        keys: keys_input,
        values: values_input,
    }
}

/// A kernel source whose `ROW_COUNT` is replaced by `rows`.
#[derive(Clone, Copy, Debug)]
struct KernelRow {
    name: &'static str,
    rows: u32,
    source: &'static str,
}

impl KernelSource for KernelRow {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(self.source.replace(
            "const ROW_COUNT: u32 = 15;",
            &format!("const ROW_COUNT: u32 = {};", self.rows),
        ))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.name, self.rows))
    }
}

/// Anzu's sort: one block of `256 * ROWS` items for each group, put in the radix
/// order in the workgroup memory before it is written, so that the threads write
/// next to each other.
pub fn sort_v5<const ROWS: u32>(
    inputs: Inputs<R>,
    device: &WgpuDevice,
) -> Outputs<R> {
    let make = |name: &'static str, source: &'static str, group_size: u32| {
        Box::new(SourceKernel::new(
            KernelRow {
                name,
                rows: ROWS,
                source,
            },
            CubeDim {
                x: group_size,
                y: 1,
                z: 1,
            },
        ))
    };
    let mut keys_input = inputs.keys;
    let mut values_input = inputs.values;
    let capacity = keys_input.shape.dims[0];
    let group_count_max = capacity.div_ceil(256 * ROWS as usize).max(1);
    let chunk_count_max = group_count_max.div_ceil(64);
    let group_count = B::int_empty([3].into(), device);
    let chunk_count = B::int_empty([3].into(), device);
    let mut keys_output = B::int_empty([capacity].into(), device);
    let mut values_output = B::int_empty([capacity].into(), device);
    let counts_radix_group = B::int_empty([group_count_max, 256].into(), device);
    let counts_radix_chunk = B::int_empty([chunk_count_max, 256].into(), device);

    keys_input.client.execute(
        make("v5.1", include_str!("v5/kernel.1.wgsl"), 1),
        CubeCount::Static(1, 1, 1),
        vec![
            inputs.count.handle.to_owned().binding(),
            group_count.handle.to_owned().binding(),
            chunk_count.handle.to_owned().binding(),
        ],
    );
    lap(None, device);
    for radix_shift in (0..32).step_by(8) {
        let client = &keys_input.client;
        let arguments = client.create(bytes_of(&Arguments { radix_shift }));
        client.execute(
            make("v5.2", include_str!("v5/kernel.2.wgsl"), 256),
            CubeCount::Dynamic(group_count.handle.to_owned().binding()),
            vec![
                arguments.to_owned().binding(),
                inputs.count.handle.to_owned().binding(),
                keys_input.handle.to_owned().binding(),
                counts_radix_group.handle.to_owned().binding(),
            ],
        );
        lap(Some(0), device);
        client.execute(
            make("v5.3", include_str!("v5/kernel.3.wgsl"), 256),
            CubeCount::Dynamic(chunk_count.handle.to_owned().binding()),
            vec![
                group_count.handle.to_owned().binding(),
                counts_radix_group.handle.to_owned().binding(),
                counts_radix_chunk.handle.to_owned().binding(),
            ],
        );
        client.execute(
            make("v5.4", include_str!("v5/kernel.4.wgsl"), 256),
            CubeCount::Static(1, 1, 1),
            vec![
                chunk_count.handle.to_owned().binding(),
                counts_radix_chunk.handle.to_owned().binding(),
            ],
        );
        lap(Some(1), device);
        client.execute(
            make("v5.5", include_str!("v5/kernel.5.wgsl"), 256),
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
        lap(Some(2), device);
        std::mem::swap(&mut keys_input, &mut keys_output);
        std::mem::swap(&mut values_input, &mut values_output);
    }

    Outputs {
        keys: keys_input,
        values: values_input,
    }
}

/// The sorters to compare, by name.
pub const SORTERS: [(&str, Sorter); 4] = [
    ("baseline", sort_baseline),
    ("anzu rows 4", sort_v5::<4>),
    ("anzu rows 8", sort_v5::<8>),
    ("anzu rows 15", sort_v5::<15>),
];
