//! Stress cases for the radix sorters.
//!
//! Written by Anzu on 2026-11-11.

mod radix_sorter;

use bytemuck::cast_slice;
use radix_sorter::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn make_keys(
    kind: &str,
    count: usize,
    rng: &mut StdRng,
) -> Vec<u32> {
    (0..count)
        .map(|index| match kind {
            "full" => rng.gen::<u32>(),
            "duplicate" => rng.gen_range(0..4u32),
            "equal" => 0xdeadbeef,
            "sorted" => index as u32 * 3,
            "reverse" => (count - index) as u32 * 3,
            "rank" => rng.gen_range(0..2000u32) << 16 | rng.gen_range(0..1u32 << 16),
            _ => unreachable!(),
        })
        .collect()
}

fn sort_on_device(
    sorter: Sorter,
    keys_source: &[u32],
    capacity: usize,
) -> (Vec<u32>, Vec<u32>) {
    let device = &WgpuDevice::default();
    let count = keys_source.len();

    // Filling the tail with a sentinel that must stay out of the result
    let mut keys = vec![0x5a5a5a5au32; capacity];
    keys[..count].copy_from_slice(keys_source);
    let keys = cast_slice::<u32, i32>(&keys).to_vec();
    let values = (0..capacity as i32).collect::<Vec<_>>();

    let keys = B::int_from_data(TensorData::new(keys, [capacity]), device);
    let values = B::int_from_data(TensorData::new(values, [capacity]), device);
    let count_device =
        B::int_from_data(TensorData::new(vec![count as i32], [1]), device);
    let Outputs { keys, values } = sorter(
        Inputs {
            count: count_device,
            keys,
            values,
        },
        device,
    );
    let keys_output = &keys.client.read(vec![keys.handle.to_owned().binding()])[0];
    let values_output = &values.client.read(vec![values.handle.to_owned().binding()])[0];

    (
        cast_slice::<u8, u32>(keys_output)[..count].to_vec(),
        cast_slice::<u8, u32>(values_output)[..count].to_vec(),
    )
}

#[test]
fn sort_stress() {
    let counts = [
        1,
        2,
        255,
        256,
        257,
        511,
        513,
        16383,
        16384,
        16385,
        32767,
        32768,
        32769,
        49151,
        3839,
        3840,
        3841,
        65535,
        65536,
        65537,
        100_000,
        245_759,
        245_760,
        245_761,
        491_521,
        (1 << 20) + 7,
        (1 << 22) + 1,
    ];
    let kinds = ["full", "duplicate", "equal", "sorted", "reverse", "rank"];
    let count_max = std::env::var("RADIX_STRESS_COUNT_MAX")
        .map(|count| count.parse::<usize>().unwrap())
        .unwrap_or(usize::MAX);
    let sort_filter = std::env::var("RADIX_STRESS_SORT").unwrap_or_default();
    let mut failures = Vec::new();
    let mut case_count = 0;

    for (name, sorter) in SORTERS
        .into_iter()
        .filter(|(name, _)| name.contains(&sort_filter))
    {
        let mut rng = StdRng::seed_from_u64(20261111);
        for count in counts.into_iter().filter(|&count| count <= count_max) {
            for kind in kinds {
                for capacity in [count, count + 1000] {
                    case_count += 1;
                    let keys_source = make_keys(kind, count, &mut rng);
                    let mut items_target = keys_source
                        .iter()
                        .copied()
                        .zip(0..count as u32)
                        .collect::<Vec<_>>();
                    // NOTE: It is a stable sort.
                    items_target.sort_by_key(|item| item.0);

                    let (keys_output, values_output) =
                        sort_on_device(sorter, &keys_source, capacity);
                    let mismatch = (0..count).find(|&index| {
                        (keys_output[index], values_output[index]) != items_target[index]
                    });
                    if let Some(index) = mismatch {
                        failures.push(format!(
                            "sort: {name}, count: {count}, kind: {kind}, capacity: {capacity}, \
                             first mismatch at {index}: output ({:#x}, {}), target ({:#x}, {})",
                            keys_output[index],
                            values_output[index],
                            items_target[index].0,
                            items_target[index].1,
                        ));
                    }
                }
            }
        }
        println!("sort: {name}, cases so far: {case_count}, failures so far: {}", failures.len());
    }

    failures.iter().take(40).for_each(|failure| println!("{failure}"));
    assert!(failures.is_empty());
}
