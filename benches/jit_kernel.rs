use divan::Bencher;
use gausplat_renderer::{
    backend::{Backend, Wgpu, WgpuRuntime},
    render::gaussian_3d::{jit::kernel, Int, Tensor},
};
use rayon::slice::ParallelSliceMut;

fn main() {
    divan::main();
}

mod cpu {
    use super::*;

    #[divan::bench(sample_count = 100, sample_size = 2)]
    fn par_sort(bencher: Bencher) {
        bencher
            .with_inputs(data::random_vec_u32_u32())
            .bench_local_refs(|v| v.par_sort_by_key(|c| c.0));
    }

    #[divan::bench(sample_count = 100, sample_size = 1)]
    fn scan_add(bencher: Bencher) {
        bencher
            .with_inputs(data::random_vec_u32())
            .bench_local_refs(|v| {
                v.iter()
                    .scan(0, |state, &x| {
                        let y = *state;
                        *state += x;
                        Some(y)
                    })
                    .collect::<Vec<_>>()
            });
    }
}

mod gpu {
    use super::*;

    #[divan::bench(sample_count = 100, sample_size = 2)]
    fn sort_radix(bencher: Bencher) {
        use kernel::sort::radix::{main, Inputs};

        bencher
            .with_inputs(data::random_tensor_u32_tensor_u32())
            .bench_local_refs(|(k, v)| {
                let output = main::<WgpuRuntime, f32, i32, u32>(Inputs {
                    count: Tensor::<Wgpu, 1, Int>::from_data(
                        [k.shape().num_elements()],
                        &k.device(),
                    )
                    .into_primitive(),
                    keys: k.to_owned().into_primitive(),
                    values: v.to_owned().into_primitive(),
                });
                Wgpu::sync(&Default::default());
                output
            });
    }

    #[divan::bench(sample_count = 100, sample_size = 1)]
    fn scan_add(bencher: Bencher) {
        use kernel::scan::add::{main, Inputs};

        bencher
            .with_inputs(data::random_tensor_u32())
            .bench_local_refs(|v| {
                let output = main::<WgpuRuntime, f32, i32, u32>(Inputs {
                    values: v.to_owned().into_primitive(),
                });
                Wgpu::sync(&Default::default());
                output
            });
    }
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

    pub fn random_vec_u32_u32() -> impl FnMut() -> Vec<(u32, u32)> {
        || random_vec_u32()().into_iter().zip(0..SIZE as u32).collect()
    }

    pub fn random_tensor_u32() -> impl Fn() -> Tensor<Wgpu, 1, Int> {
        || {
            let result = Tensor::random(
                [SIZE],
                Distribution::Uniform(ELEMENT_MIN.into(), ELEMENT_MAX.into()),
                &Default::default(),
            );
            Wgpu::sync(&Default::default());
            result
        }
    }

    pub fn random_tensor_u32_tensor_u32(
    ) -> impl Fn() -> (Tensor<Wgpu, 1, Int>, Tensor<Wgpu, 1, Int>) {
        || {
            let result = (
                random_tensor_u32()(),
                Tensor::arange(0..SIZE as i64, &Default::default()),
            );
            Wgpu::sync(&Default::default());
            result
        }
    }
}
