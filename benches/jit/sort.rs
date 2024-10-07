//! ## Usage
//!
//! To run the benchmarks, execute the following command in the console:
//!
//! ```sh
//! cargo bench jit_sort
//! ```

use burn_jit::cubecl::client::SyncType;
use divan::Bencher;
use gausplat_renderer::{
    backend::{Backend, Wgpu, WgpuRuntime},
    render::gaussian_3d::{jit::kernel, Int, Tensor},
};
use rayon::slice::ParallelSliceMut;

fn main() {
    divan::main();
}

#[divan::bench(sample_count = 100, sample_size = 2)]
fn sort_on_cpu(bencher: Bencher) {
    bencher
        .with_inputs(data::random_vec_u32())
        .bench_local_refs(|v| v.par_sort());
}

#[divan::bench(sample_count = 100, sample_size = 2)]
fn sort_on_jit(bencher: Bencher) {
    use kernel::sort::radix::{main, Inputs};

    bencher
        .with_inputs(data::random_tensor_u32_tensor_u32())
        .bench_local_refs(|(k, v)| {
            main::<WgpuRuntime, f32, i32>(Inputs {
                keys: k.to_owned().into_primitive(),
                values: v.to_owned().into_primitive(),
            });
            Wgpu::sync(&Default::default(), SyncType::Wait);
        });
}

mod data {
    use super::*;
    use burn::tensor::Distribution;
    use rand::{distributions::Uniform, rngs::StdRng, Rng, SeedableRng};

    const SIZE: usize = 1 << 23;
    const ELEMENT_MIN: u32 = 0;
    const ELEMENT_MAX: u32 = (1 << 31) - 1;

    pub fn random_vec_u32() -> impl FnMut() -> Vec<u32> {
        || {
            StdRng::seed_from_u64(0)
                .sample_iter(Uniform::new_inclusive(ELEMENT_MIN, ELEMENT_MAX))
                .take(SIZE)
                .collect()
        }
    }

    pub fn random_tensor_u32_tensor_u32(
    ) -> impl Fn() -> (Tensor<Wgpu, 1, Int>, Tensor<Wgpu, 1, Int>) {
        || {
            let result = (
                Tensor::random(
                    [SIZE],
                    Distribution::Uniform(
                        ELEMENT_MIN.into(),
                        ELEMENT_MAX.into(),
                    ),
                    &Default::default(),
                ),
                Tensor::arange(0..SIZE as i64, &Default::default()),
            );
            Wgpu::sync(&Default::default(), SyncType::Wait);
            result
        }
    }
}
