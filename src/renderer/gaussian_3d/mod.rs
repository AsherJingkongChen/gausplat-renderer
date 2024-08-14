pub use crate::scene::gaussian_3d::*;
pub use gausplat_importer::scene::sparse_view;

use crate::error::Error;
use burn::backend::wgpu::{
    into_contiguous, AutoGraphicsApi, JitBackend, Kernel, SourceKernel,
    WgpuRuntime, WorkGroup, WorkgroupSize,
};
use bytemuck::{
    bytes_of, cast_slice, cast_slice_mut, from_bytes, Pod, Zeroable,
};
use rayon::slice::ParallelSliceMut;
use tensor::Int;

pub trait Gaussian3dRenderer: backend::Backend {
    type Error;

    fn render_gaussian_3d(
        scene: &Gaussian3dScene<Self>,
        view: &sparse_view::View,
        colors_sh_degree_max: u32,
    ) -> Result<Gaussian3dRendererResultOk<Self>, Self::Error>;
}

#[derive(Clone, Debug)]
pub struct Gaussian3dRendererResultOk<B: backend::Backend> {
    // `[H, W, 3]`
    pub colors_rgb_2d: Tensor<B, 3>,
}

pub type Wgpu = JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;

mod kernel {
    use burn::backend::wgpu::kernel_wgsl;
    use bytemuck::{Pod, Zeroable};
    use derive_new::new;

    kernel_wgsl!(
        RenderGaussian3dForward1,
        "./render_gaussian_3d_forward_1.wgsl"
    );

    kernel_wgsl!(
        RenderGaussian3dForward2,
        "./render_gaussian_3d_forward_2.wgsl"
    );

    kernel_wgsl!(
        RenderGaussian3dForward3,
        "./render_gaussian_3d_forward_3.wgsl"
    );

    kernel_wgsl!(
        RenderGaussian3dForward4,
        "./render_gaussian_3d_forward_4.wgsl"
    );

    kernel_wgsl!(
        RenderGaussian3dForward5,
        "./render_gaussian_3d_forward_5.wgsl"
    );

    kernel_wgsl!(
        RenderGaussian3dForward6,
        "./render_gaussian_3d_forward_6.wgsl"
    );

    #[repr(C)]
    #[derive(Copy, Clone, Debug, Pod, Zeroable)]
    pub struct RenderGaussian3dForward1Arguments {
        pub colors_sh_degree_max: u32,
        pub filter_low_pass: f32,
        pub focal_length_x: f32,
        pub focal_length_y: f32,

        /// `I_X`
        pub image_size_x: u32,

        /// `I_Y`
        pub image_size_y: u32,

        /// `I_X / 2`
        pub image_size_half_x: f32,

        /// `I_Y / 2`
        pub image_size_half_y: f32,

        /// `P`
        pub point_count: u32,

        /// `I_X / T_X`
        pub tile_count_x: u32,

        /// `I_Y / T_Y`
        pub tile_count_y: u32,

        /// `T_X`
        pub tile_size_x: u32,

        /// `T_Y`
        pub tile_size_y: u32,

        pub view_bound_x: f32,
        pub view_bound_y: f32,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, Pod, Zeroable)]
    pub struct RenderGaussian3dForward2Arguments {
        pub point_count: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, Pod, Zeroable)]
    pub struct RenderGaussian3dForward3Arguments {
        pub point_count: u32,
        pub tile_count_x: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, Pod, Zeroable)]
    pub struct RenderGaussian3dForward5Arguments {
        // T
        pub tile_touched_count: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, Pod, Zeroable)]
    pub struct RenderGaussian3dForward6Arguments {
        // I_X
        pub image_size_x: u32,

        // I_Y
        pub image_size_y: u32,
    }
}

#[repr(C, packed(4))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
struct Point {
    key: [u32; 2],
    pub index: u32,
}

impl Point {
    #[inline]
    fn tile_index(&self) -> u32 {
        self.key[0]
    }
}

impl Ord for Point {
    fn cmp(
        &self,
        other: &Self,
    ) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

impl PartialOrd for Point {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<std::cmp::Ordering> {
        self.key.partial_cmp(&other.key)
    }
}

impl Gaussian3dRenderer for Wgpu {
    type Error = Error;

    fn render_gaussian_3d(
        scene: &Gaussian3dScene<Self>,
        view: &sparse_view::View,
        colors_sh_degree_max: u32,
    ) -> Result<Gaussian3dRendererResultOk<Self>, Self::Error> {
        const FILTER_LOW_PASS: f64 = 0.3;

        // T_X
        const TILE_SIZE_X: u32 = 16;

        // T_Y
        const TILE_SIZE_Y: u32 = 16;

        if colors_sh_degree_max > 3 {
            return Err(Error::Gaussian3dRenderer(format!(
                "colors_sh_degree_max should be no more than 3, but got {}",
                colors_sh_degree_max
            )));
        }

        let mut duration = std::time::Instant::now();

        // I_X
        let image_size_x = view.image_width;
        // I_Y
        let image_size_y = view.image_height;
        // I_X / 2
        let image_size_half_x = (image_size_x as f64 / 2.0) as f32;
        // I_Y / 2
        let image_size_half_y = (image_size_y as f64 / 2.0) as f32;
        // I_X / T_X
        let tile_count_x = (image_size_x + TILE_SIZE_X - 1) / TILE_SIZE_X;
        // I_Y / T_Y
        let tile_count_y = (image_size_y + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

        let field_of_view_x_half_tan = (view.field_of_view_x / 2.0).tan();
        let field_of_view_y_half_tan = (view.field_of_view_y / 2.0).tan();
        let filter_low_pass = FILTER_LOW_PASS as f32;
        let focal_length_x =
            (image_size_x as f64 / field_of_view_x_half_tan / 2.0) as f32;
        let focal_length_y =
            (image_size_y as f64 / field_of_view_y_half_tan / 2.0) as f32;
        let view_bound_x =
            (field_of_view_x_half_tan * (FILTER_LOW_PASS + 1.0)) as f32;
        let view_bound_y =
            (field_of_view_y_half_tan * (FILTER_LOW_PASS + 1.0)) as f32;

        let scene = scene.to_owned();

        // ([P, 16, 3], P)
        let (client, colors_sh, device, point_count) = {
            let p = into_contiguous(scene.colors_sh().into_primitive());
            (p.client, p.handle, p.device, p.shape.dims[0])
        };

        println!("point_count: {:?}", point_count);

        let arguments = client.create(bytes_of(
            &kernel::RenderGaussian3dForward1Arguments {
                colors_sh_degree_max,
                filter_low_pass,
                focal_length_x,
                focal_length_y,
                image_size_x,
                image_size_y,
                image_size_half_x,
                image_size_half_y,
                point_count: point_count as u32,
                tile_count_x,
                tile_count_y,
                tile_size_x: TILE_SIZE_X,
                tile_size_y: TILE_SIZE_Y,
                view_bound_x,
                view_bound_y,
            },
        ));

        // [P, 3]
        let positions =
            into_contiguous(scene.positions().into_primitive()).handle;

        // [P, 1]
        let opacities =
            into_contiguous(scene.opacities().into_primitive()).handle;

        // [P, 4]
        let rotations =
            into_contiguous(scene.rotations().into_primitive()).handle;

        // [P, 3]
        let scalings =
            into_contiguous(scene.scalings().into_primitive()).handle;

        // [3]
        let view_position =
            client.create(bytes_of(&view.view_position.map(|c| c as f32)));

        // [4, 4]
        let view_transform = {
            let view_transform_in_column_major = [
                view.view_transform.map(|r| r[0] as f32),
                view.view_transform.map(|r| r[1] as f32),
                view.view_transform.map(|r| r[2] as f32),
                view.view_transform.map(|r| r[3] as f32),
            ];
            client.create(bytes_of(&view_transform_in_column_major))
        };

        // [P, 3]
        let colors_rgb_3d = client.empty(point_count * 3 * 4);

        // [P, 2, 2]
        let conics = client.empty(point_count * 2 * 2 * 4);

        // [P]
        let depths = client.empty(point_count * 4);

        // [P, 2]
        let positions_2d_in_screen = client.empty(point_count * 2 * 4);

        // [P]
        let radii = client.empty(point_count * 4);

        // [P]
        let tile_touched_counts = client.empty(point_count * 4);

        // [P, 2]
        let tiles_touched_max = client.empty(point_count * 2 * 4);

        // [P, 2]
        let tiles_touched_min = client.empty(point_count * 2 * 4);

        client.execute(
            Kernel::Custom(Box::new(SourceKernel::new(
                kernel::RenderGaussian3dForward1::new(),
                WorkGroup {
                    x: (point_count as u32 + 256 - 1) / 256,
                    y: 1,
                    z: 1,
                },
                WorkgroupSize { x: 256, y: 1, z: 1 },
            ))),
            &[
                &arguments,
                &colors_sh,
                &positions,
                &rotations,
                &scalings,
                &view_position,
                &view_transform,
                &colors_rgb_3d,
                &conics,
                &depths,
                &positions_2d_in_screen,
                &radii,
                &tile_touched_counts,
                &tiles_touched_max,
                &tiles_touched_min,
            ],
        );

        println!("wgsl 1: {:?}", duration.elapsed());
        duration = std::time::Instant::now();

        let arguments = client.create(bytes_of(
            &kernel::RenderGaussian3dForward2Arguments {
                point_count: point_count as u32,
            },
        ));

        // T
        let tile_touched_count = client.empty(1 * 4);

        // [P]
        let tile_touched_offsets = client.empty(point_count * 4);

        client.execute(
            Kernel::Custom(Box::new(SourceKernel::new(
                kernel::RenderGaussian3dForward2::new(),
                WorkGroup { x: 1, y: 1, z: 1 },
                WorkgroupSize { x: 1, y: 1, z: 1 },
            ))),
            &[
                &arguments,
                &tile_touched_counts,
                &tile_touched_count,
                &tile_touched_offsets,
            ],
        );

        println!("wgsl 2: {:?}", duration.elapsed());
        duration = std::time::Instant::now();

        let arguments = client.create(bytes_of(
            &kernel::RenderGaussian3dForward3Arguments {
                point_count: point_count as u32,
                tile_count_x,
            },
        ));

        // T
        let tile_touched_count =
            *from_bytes::<u32>(&client.read(&tile_touched_count).read())
                as usize;

        println!("tile_touched_count (wgsl): {:?}", tile_touched_count);

        // [T, 3] ([tile_index, depth, point_index])
        let point_keys_and_indexes = client.empty(tile_touched_count * 3 * 4);

        client.execute(
            Kernel::Custom(Box::new(SourceKernel::new(
                kernel::RenderGaussian3dForward3::new(),
                WorkGroup {
                    x: (point_count as u32 + 256 - 1) / 256,
                    y: 1,
                    z: 1,
                },
                WorkgroupSize { x: 256, y: 1, z: 1 },
            ))),
            &[
                &arguments,
                &depths,
                &radii,
                &tile_touched_offsets,
                &tiles_touched_max,
                &tiles_touched_min,
                &point_keys_and_indexes,
            ],
        );

        println!("wgsl 3: {:?}", duration.elapsed());
        duration = std::time::Instant::now();

        // ([T], [T])
        let (point_indexes, point_tile_indexes) = {
            let mut point_keys_and_indexes =
                client.read(&point_keys_and_indexes).read();

            let point_keys_and_indexes =
                cast_slice_mut::<u8, Point>(&mut point_keys_and_indexes);

            point_keys_and_indexes.par_sort_unstable();

            point_keys_and_indexes
                .into_iter()
                .map(|p| (p.index, p.tile_index()))
                .unzip::<_, _, Vec<_>, Vec<_>>()
        };

        println!("wgsl 4: {:?}", duration.elapsed());
        duration = std::time::Instant::now();

        let arguments = client.create(bytes_of(
            &kernel::RenderGaussian3dForward5Arguments {
                tile_touched_count: tile_touched_count as u32,
            },
        ));

        // [(I_X / T_X) * (I_Y / T_Y), 2]
        let tile_point_ranges = {
            let mut ranges =
                vec![[0; 2]; (tile_count_x * tile_count_y) as usize];

            if !point_tile_indexes.is_empty() {
                let tile_index_first =
                    *point_tile_indexes.first().unwrap() as usize;
                let tile_index_last =
                    *point_tile_indexes.last().unwrap() as usize;

                ranges[tile_index_first][0] = 0;
                ranges[tile_index_last][1] = tile_touched_count as u32;
            }

            client.create(cast_slice::<[u32; 2], u8>(&ranges))
        };

        // [T]
        let point_tile_indexes =
            client.create(cast_slice::<u32, u8>(&point_tile_indexes));

        client.execute(
            Kernel::Custom(Box::new(SourceKernel::new(
                kernel::RenderGaussian3dForward5::new(),
                WorkGroup {
                    x: (tile_touched_count as u32 + 256 - 1) / 256,
                    y: 1,
                    z: 1,
                },
                WorkgroupSize { x: 256, y: 1, z: 1 },
            ))),
            &[&arguments, &point_tile_indexes, &tile_point_ranges],
        );

        println!("wgsl 5: {:?}", duration.elapsed());
        duration = std::time::Instant::now();

        let arguments = client.create(bytes_of(
            &kernel::RenderGaussian3dForward6Arguments {
                image_size_x,
                image_size_y,
            },
        ));

        // [I_X * I_Y, 3]
        let colors_rgb_2d =
            client.empty((image_size_y * image_size_x) as usize * 3 * 4);

        // [T]
        let point_indexes =
            client.create(cast_slice::<u32, u8>(&point_indexes));

        client.execute(
            Kernel::Custom(Box::new(SourceKernel::new(
                kernel::RenderGaussian3dForward6::new(),
                WorkGroup {
                    x: tile_count_x,
                    y: tile_count_y,
                    z: 1,
                },
                WorkgroupSize { x: 16, y: 16, z: 1 },
            ))),
            &[
                &arguments,
                &colors_rgb_3d,
                &conics,
                &opacities,
                &positions_2d_in_screen,
                &point_indexes,
                &tile_point_ranges,
                &colors_rgb_2d,
            ],
        );

        println!("wgsl 6: {:?}", duration.elapsed());
        duration = std::time::Instant::now();

        // [I_Y, I_X, 3]
        let colors_rgb_2d =
            Tensor::<Self, 3>::new(Self::FloatTensorPrimitive::<3>::new(
                client.to_owned(),
                device.to_owned(),
                [image_size_y as usize, image_size_x as usize, 3].into(),
                colors_rgb_2d,
            ));

        // // [P]
        // let radii =
        //     Tensor::<Self, 1, Int>::new(Self::IntTensorPrimitive::<1>::new(
        //         client.to_owned(),
        //         device.to_owned(),
        //         [point_count].into(),
        //         radii,
        //     ));

        println!("wgsl finish: {:?}", duration.elapsed());

        Ok(Gaussian3dRendererResultOk { colors_rgb_2d })
    }
}
