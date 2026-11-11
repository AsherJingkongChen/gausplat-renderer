//! A probe of what the memory costs for one shape of the scatter, with no ranking.
//!
//! Written by Anzu on 2026-11-11.

use burn::tensor::{backend::Backend, ops::IntTensorOps, TensorData};
use burn_jit::{
    cubecl::{CubeCount, CubeDim},
    template::SourceKernel,
};
use gausplat_renderer::{
    backend::{Wgpu, WgpuDevice},
    render::gaussian_3d::jit::kernel::{KernelId, KernelSource, SourceTemplate},
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

type B = Wgpu;

#[derive(Clone, Copy, Debug)]
struct Permute {
    block_count_group: u32,
    row_count: u32,
    is_barrier: bool,
}

impl KernelSource for Permute {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(
            include_str!("radix_probe/permute.wgsl")
                .replace(
                    "const BLOCK_COUNT_GROUP: u32 = 1;",
                    &format!("const BLOCK_COUNT_GROUP: u32 = {};", self.block_count_group),
                )
                .replace(
                    "const ROW_COUNT: u32 = 1;",
                    &format!("const ROW_COUNT: u32 = {};", self.row_count),
                )
                .replace(
                    "const IS_BARRIER: bool = true;",
                    &format!("const IS_BARRIER: bool = {};", self.is_barrier),
                ),
        )
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.block_count_group, self.row_count, self.is_barrier))
    }
}

#[test]
fn scatter_probe() {
    let device = &WgpuDevice::default();
    let mut rng = StdRng::seed_from_u64(20261111);
    let count = 1usize << 23;
    let repeat_count = 24;

    let keys = (0..count)
        .map(|_| rng.gen_range(0..i32::MAX))
        .collect::<Vec<_>>();
    let values = (0..count as i32).collect::<Vec<_>>();
    // The destinations of a stable sort by the lowest 8 bits, or of no move at all
    let make_destinations = |bits: u32| {
        let mut order = (0..count as u32).collect::<Vec<_>>();
        order.sort_by_key(|&index| keys[index as usize] & ((1 << bits) - 1));
        let mut destinations = vec![0i32; count];
        order
            .iter()
            .enumerate()
            .for_each(|(position, &index)| destinations[index as usize] = position as i32);
        destinations
    };

    let upload = |data: &[i32]| B::int_from_data(TensorData::new(data.to_vec(), [count]), device);
    let count_device = B::int_from_data(TensorData::new(vec![count as i32], [1]), device);
    let keys_input = upload(&keys);
    let values_input = upload(&values);
    let keys_output = B::int_empty([count].into(), device);
    let values_output = B::int_empty([count].into(), device);

    println!("| writes | groups | blocks per group | rows | barrier | min / median (ms) for one pass over 2^23 items |");
    println!("|---|---|---|---|---|---|");
    for (name, bits) in [("in place", 0), ("16 runs", 4), ("256 runs", 8)] {
        let destinations = upload(&make_destinations(bits));
        B::sync(device);
        for (group_count, row_count, is_barrier) in [
            (64, 1, true),
            (64, 1, false),
            (1024, 1, true),
            (1024, 1, false),
            (0, 1, true),
            (0, 1, false),
            (64, 4, true),
            (64, 15, true),
            (64, 15, false),
            (1024, 15, true),
            (0, 4, true),
            (0, 15, true),
            (0, 15, false),
        ] {
            let block_size = 256 * row_count as usize;
            // No group count means one block for each group
            let block_count_group = if group_count == 0 {
                1
            } else {
                count.div_ceil(block_size * group_count)
            };
            let group_count = count.div_ceil(block_size * block_count_group);
            let kernel = Permute {
                block_count_group: block_count_group as u32,
                row_count,
                is_barrier,
            };
            let mut durations = (0..repeat_count)
                .map(|_| {
                    let time = Instant::now();
                    keys_input.client.execute(
                        Box::new(SourceKernel::new(
                            kernel,
                            CubeDim {
                                x: 256,
                                y: 1,
                                z: 1,
                            },
                        )),
                        CubeCount::Static(group_count as u32, 1, 1),
                        vec![
                            count_device.handle.to_owned().binding(),
                            destinations.handle.to_owned().binding(),
                            keys_input.handle.to_owned().binding(),
                            values_input.handle.to_owned().binding(),
                            keys_output.handle.to_owned().binding(),
                            values_output.handle.to_owned().binding(),
                        ],
                    );
                    B::sync(device);
                    time.elapsed().as_secs_f64() * 1e3
                })
                .skip(4)
                .collect::<Vec<_>>();
            durations.sort_by(f64::total_cmp);
            println!(
                "| {name} | {group_count} | {block_count_group} | {row_count} | {is_barrier} | {:.3} / {:.3} |",
                durations[0],
                durations[durations.len() / 2],
            );
        }
    }
}

/// The same destinations, but each block is first put in the order of its radix,
/// so that neighbouring threads write to neighbouring places.
#[test]
fn scatter_probe_sorted_in_block() {
    let device = &WgpuDevice::default();
    let mut rng = StdRng::seed_from_u64(20261111);
    let count = 1usize << 23;
    let repeat_count = 24;

    let keys = (0..count)
        .map(|_| rng.gen_range(0..i32::MAX))
        .collect::<Vec<_>>();
    let mut order = (0..count as u32).collect::<Vec<_>>();
    order.sort_by_key(|&index| keys[index as usize] & 255);
    let mut destinations = vec![0i32; count];
    order
        .iter()
        .enumerate()
        .for_each(|(position, &index)| destinations[index as usize] = position as i32);

    let upload = |data: &[i32]| B::int_from_data(TensorData::new(data.to_vec(), [count]), device);
    let count_device = B::int_from_data(TensorData::new(vec![count as i32], [1]), device);
    let keys_output = B::int_empty([count].into(), device);
    let values_output = B::int_empty([count].into(), device);

    println!("| order in block | groups | blocks per group | rows | min / median (ms) for one pass over 2^23 items, 256 runs |");
    println!("|---|---|---|---|---|");
    for (group_count, row_count) in [(64usize, 1u32), (0, 1), (0, 4), (0, 15), (64, 15)] {
        let block_size = 256 * row_count as usize;
        let block_count_group = if group_count == 0 {
            1
        } else {
            count.div_ceil(block_size * group_count)
        };
        let group_count = count.div_ceil(block_size * block_count_group);
        for is_sorted in [false, true] {
            // The items of each block, in the input order or in the order of the radix
            let mut indices = (0..count).collect::<Vec<_>>();
            if is_sorted {
                indices
                    .chunks_mut(block_size)
                    .for_each(|block| block.sort_by_key(|&index| keys[index] & 255));
            }
            let keys_input = upload(&indices.iter().map(|&index| keys[index]).collect::<Vec<_>>());
            let values_input = upload(&indices.iter().map(|&index| index as i32).collect::<Vec<_>>());
            let destinations_input =
                upload(&indices.iter().map(|&index| destinations[index]).collect::<Vec<_>>());
            B::sync(device);

            let kernel = Permute {
                block_count_group: block_count_group as u32,
                row_count,
                is_barrier: true,
            };
            let mut durations = (0..repeat_count)
                .map(|_| {
                    let time = Instant::now();
                    keys_input.client.execute(
                        Box::new(SourceKernel::new(
                            kernel,
                            CubeDim {
                                x: 256,
                                y: 1,
                                z: 1,
                            },
                        )),
                        CubeCount::Static(group_count as u32, 1, 1),
                        vec![
                            count_device.handle.to_owned().binding(),
                            destinations_input.handle.to_owned().binding(),
                            keys_input.handle.to_owned().binding(),
                            values_input.handle.to_owned().binding(),
                            keys_output.handle.to_owned().binding(),
                            values_output.handle.to_owned().binding(),
                        ],
                    );
                    B::sync(device);
                    time.elapsed().as_secs_f64() * 1e3
                })
                .skip(4)
                .collect::<Vec<_>>();
            durations.sort_by(f64::total_cmp);
            println!(
                "| {} | {group_count} | {block_count_group} | {row_count} | {:.3} / {:.3} |",
                if is_sorted { "radix" } else { "input" },
                durations[0],
                durations[durations.len() / 2],
            );
        }
    }
}
