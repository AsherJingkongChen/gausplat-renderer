mod kernel;
mod point;

pub use crate::scene::gaussian_3d::*;
pub use gausplat_importer::scene::sparse_view;

use crate::{backend::Wgpu, consts::render::*, error::Error};
use burn::{
    backend::wgpu::{
        into_contiguous, Kernel, SourceKernel, WorkGroup, WorkgroupSize,
    },
    tensor::ops,
};
use bytemuck::{bytes_of, cast_slice, cast_slice_mut, from_bytes};
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

#[derive(Clone, Debug)]
pub(super) struct Gaussian3dRenderResponse<B: backend::Backend> {
    /// `[H, W, 3]`
    pub colors_rgb_2d: Tensor<B, 3>,
}

impl Gaussian3dScene<Wgpu> {
    pub(super) fn render_forward(
        &self,
        view: &sparse_view::View,
        colors_sh_degree_max: u32,
    ) -> Result<Gaussian3dRenderResponse<Wgpu>, Error> {
        // Specifying the constants

        if colors_sh_degree_max > 3 {
            return Err(Error::Gaussian3dRenderer(format!(
                "colors_sh_degree_max should be no more than 3, but got {}",
                colors_sh_degree_max
            )));
        }

        let mut duration = std::time::Instant::now();

        // Specifying the parameters

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
        // T_X
        let tile_size_x = TILE_SIZE_X;
        // T_Y
        let tile_size_y = TILE_SIZE_Y;
        // T_X * T_Y
        let batch_size = TILE_SIZE_X * TILE_SIZE_Y;
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

        // ([P, 16, 3], P)
        let (client, colors_sh, device, point_count) = {
            let p = into_contiguous(self.colors_sh().into_primitive());
            (p.client, p.handle, p.device, p.shape.dims[0])
        };

        // Performing the forward pass #1

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
                tile_size_x,
                tile_size_y,
                view_bound_x,
                view_bound_y,
            },
        ));
        // [P, 3]
        let positions =
            into_contiguous(self.positions().into_primitive()).handle;
        // [P, 1]
        let opacities =
            into_contiguous(self.opacities().into_primitive()).handle;
        // [P, 4]
        let rotations =
            into_contiguous(self.rotations().into_primitive()).handle;
        // [P, 3]
        let scalings = into_contiguous(self.scalings().into_primitive()).handle;
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
        // [P, 3, 3] (the alignment of mat3x3<f32> is 16 = (3 + 1) * 4 bytes)
        let covariances_3d = client.empty(point_count * (3 + 1) * 3 * 4);
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
                    x: (point_count as u32 + batch_size - 1) / batch_size,
                    y: 1,
                    z: 1,
                },
                WorkgroupSize {
                    x: batch_size,
                    y: 1,
                    z: 1,
                },
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
                &covariances_3d,
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

        // Performing the forward pass #2

        let arguments = client.create(bytes_of(
            &kernel::RenderGaussian3dForward2Arguments {
                point_count: point_count as u32,
            },
        ));
        // T
        let tile_touched_count = client.empty(4);
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

        // Performing the forward pass #3

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
        // [T, 3] ([tile_index, depth, point_index])
        let point_keys_and_indexes = client.empty(tile_touched_count * 3 * 4);

        println!("tile_touched_count (wgsl): {:?}", tile_touched_count);

        client.execute(
            Kernel::Custom(Box::new(SourceKernel::new(
                kernel::RenderGaussian3dForward3::new(),
                WorkGroup {
                    x: (point_count as u32 + batch_size - 1) / batch_size,
                    y: 1,
                    z: 1,
                },
                WorkgroupSize {
                    x: batch_size,
                    y: 1,
                    z: 1,
                },
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

        // Performing the forward pass #4

        // ([T], [T])
        let (point_indexes, point_tile_indexes) = {
            let mut point_keys_and_indexes =
                client.read(&point_keys_and_indexes).read();
            let point_keys_and_indexes =
                cast_slice_mut::<u8, point::PointKeyAndIndex>(
                    &mut point_keys_and_indexes,
                );

            point_keys_and_indexes.par_sort_unstable();

            point_keys_and_indexes
                .into_par_iter()
                .map(|point| (point.index, point.tile_index()))
                .unzip::<_, _, Vec<_>, Vec<_>>()
        };

        println!("wgsl 4: {:?}", duration.elapsed());
        duration = std::time::Instant::now();

        // Performing the forward pass #5

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
                    x: (tile_touched_count as u32 + batch_size - 1)
                        / batch_size,
                    y: 1,
                    z: 1,
                },
                WorkgroupSize {
                    x: batch_size,
                    y: 1,
                    z: 1,
                },
            ))),
            &[&arguments, &point_tile_indexes, &tile_point_ranges],
        );

        println!("wgsl 5: {:?}", duration.elapsed());
        duration = std::time::Instant::now();

        // Performing the forward pass #6

        let arguments = client.create(bytes_of(
            &kernel::RenderGaussian3dForward6Arguments {
                image_size_x,
                image_size_y,
            },
        ));

        // [I_Y, I_X, 3]
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
                WorkgroupSize {
                    x: tile_size_x,
                    y: tile_size_y,
                    z: 1,
                },
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

        // Specifying the results

        // [I_Y, I_X, 3]
        let colors_rgb_2d = Tensor::new(ops::FloatTensor::<Wgpu, 3>::new(
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

        println!("wgsl 6: {:?}", duration.elapsed());

        Ok(Gaussian3dRenderResponse { colors_rgb_2d })
    }
}
