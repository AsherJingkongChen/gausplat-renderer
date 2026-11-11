//! Timing of the radix sorters.
//!
//! Written by Anzu on 2026-11-11.
//!
//! The GPU lowers its clocks when the duty cycle is low (P0 -> P3 -> P5 on the
//! RTX 3090), which slows every kernel by about 4 times. So the inputs are made
//! ahead of the timed loop, the sorts run back to back, and each row reports the
//! performance state read from `nvidia-smi` right after its loop.

mod radix_sorter;

use radix_sorter::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{process::Command, time::Instant};

fn make_keys(
    kind: &str,
    count: usize,
    rng: &mut StdRng,
) -> Vec<i32> {
    (0..count)
        .map(|_| match kind {
            "uniform" => rng.gen_range(0..i32::MAX),
            "duplicate" => rng.gen_range(0..4),
            "equal" => 0x2badbeef,
            "rank" => rng.gen_range(0..8160) << 16 | rng.gen_range(0..1 << 16),
            _ => unreachable!(),
        })
        .collect()
}

/// The performance state and the clock of the GPU.
fn read_gpu_state() -> String {
    Command::new("nvidia-smi")
        .args(["--query-gpu=pstate,clocks.gr", "--format=csv,noheader"])
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_default()
}

/// The times of the sort in ms, as (min, median, max), and the GPU state.
fn measure(
    sorter: Sorter,
    keys: &[i32],
    repeat_count: usize,
    device: &WgpuDevice,
) -> ((f64, f64, f64), String) {
    const WARMUP_COUNT: usize = 4;

    let count = keys.len();
    let values = (0..count as i32).collect::<Vec<_>>();
    let inputs = (0..repeat_count + WARMUP_COUNT)
        .map(|_| Inputs {
            count: B::int_from_data(TensorData::new(vec![count as i32], [1]), device),
            keys: B::int_from_data(TensorData::new(keys.to_vec(), [count]), device),
            values: B::int_from_data(TensorData::new(values.to_owned(), [count]), device),
        })
        .collect::<Vec<_>>();
    B::sync(device);

    let mut durations = inputs
        .into_iter()
        .map(|inputs| {
            let time = Instant::now();
            let _outputs = sorter(inputs, device);
            B::sync(device);
            time.elapsed().as_secs_f64() * 1e3
        })
        .skip(WARMUP_COUNT)
        .collect::<Vec<_>>();
    let state = read_gpu_state();
    if std::env::var("RADIX_TIMING_RAW").is_ok() {
        println!(
            "raw: {}",
            durations
                .iter()
                .map(|duration| format!("{duration:.1}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    durations.sort_by(f64::total_cmp);

    (
        (
            durations[0],
            durations[repeat_count / 2],
            durations[repeat_count - 1],
        ),
        state,
    )
}

fn print_header() {
    println!("| N | keys | sort | min / median / max (ms) | max / median | M keys/s | GPU state |");
    println!("|---|---|---|---|---|---|---|");
}

fn print_row(
    count: usize,
    kind: &str,
    name: &str,
    times: (f64, f64, f64),
    state: &str,
) {
    println!(
        "| {count} | {kind} | {name} | {:.3} / {:.3} / {:.3} | {:.2} | {:.0} | {state} |",
        times.0,
        times.1,
        times.2,
        times.2 / times.1,
        count as f64 / times.1 / 1e3,
    );
}

/// The counts come from `RADIX_TIMING_COUNTS`, separated by commas.
fn read_counts(default: &[usize]) -> Vec<usize> {
    std::env::var("RADIX_TIMING_COUNTS")
        .map(|counts| {
            counts
                .split(',')
                .map(|count| count.trim().parse().unwrap())
                .collect()
        })
        .unwrap_or(default.to_vec())
}

#[test]
fn sort_timing() {
    let device = &WgpuDevice::default();
    let mut rng = StdRng::seed_from_u64(20261111);

    print_header();
    for count in read_counts(&[1 << 14, 1 << 18, 1 << 20, 1 << 22, 1 << 24]) {
        let keys = make_keys("uniform", count, &mut rng);
        for (name, sorter) in SORTERS.into_iter().filter(|(name, _)| {
            name.contains(&std::env::var("RADIX_TIMING_SORT").unwrap_or_default())
        }) {
            let (times, state) = measure(sorter, &keys, 15, device);
            print_row(count, "uniform", name, times, &state);
        }
    }
}

#[test]
fn sort_timing_distribution() {
    let device = &WgpuDevice::default();
    let mut rng = StdRng::seed_from_u64(20261111);

    print_header();
    for count in read_counts(&[1 << 20, 1 << 23]) {
        for kind in ["uniform", "rank", "duplicate", "equal"] {
            let keys = make_keys(kind, count, &mut rng);
            for (name, sorter) in SORTERS {
                let (times, state) = measure(sorter, &keys, 15, device);
                print_row(count, kind, name, times, &state);
            }
        }
    }
}


/// The wall time of a sort beside the time the GPU spent on it.
#[test]
fn sort_timing_gpu() {
    fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
        let mut future = std::pin::pin!(future);
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        loop {
            if let std::task::Poll::Ready(output) = future.as_mut().poll(&mut context) {
                return output;
            }
            std::thread::yield_now();
        }
    }

    let device = &WgpuDevice::default();
    let client = B::int_empty([1].into(), device).client;
    let mut rng = StdRng::seed_from_u64(20261111);
    client.enable_timestamps();

    println!("| N | sort | wall median (ms) | GPU median (ms) | wall each (ms) | GPU each (ms) |");
    println!("|---|---|---|---|---|---|");
    for count in read_counts(&[30_000, 100_000, 200_000, 100_000, 200_000]) {
        let keys = make_keys("uniform", count, &mut rng);
        let values = (0..count as i32).collect::<Vec<_>>();
        for (name, sorter) in SORTERS {
            let inputs = (0..12)
                .map(|_| Inputs {
                    count: B::int_from_data(TensorData::new(vec![count as i32], [1]), device),
                    keys: B::int_from_data(TensorData::new(keys.to_vec(), [count]), device),
                    values: B::int_from_data(TensorData::new(values.to_owned(), [count]), device),
                })
                .collect::<Vec<_>>();
            let _ = block_on(client.sync_elapsed());

            let (walls, gpus): (Vec<f64>, Vec<f64>) = inputs
                .into_iter()
                .map(|inputs| {
                    let time = Instant::now();
                    let _outputs = sorter(inputs, device);
                    let gpu = block_on(client.sync_elapsed())
                        .map(|duration| duration.as_secs_f64() * 1e3)
                        .unwrap_or(f64::NAN);
                    (time.elapsed().as_secs_f64() * 1e3, gpu)
                })
                .skip(4)
                .unzip();
            let median = |times: &[f64]| {
                let mut times = times.to_vec();
                times.sort_by(f64::total_cmp);
                times[times.len() / 2]
            };
            let join = |times: &[f64]| {
                times
                    .iter()
                    .map(|time| format!("{time:.2}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            println!(
                "| {count} | {name} | {:.3} | {:.3} | {} | {} |",
                median(&walls),
                median(&gpus),
                join(&walls),
                join(&gpus),
            );
        }
    }
}
