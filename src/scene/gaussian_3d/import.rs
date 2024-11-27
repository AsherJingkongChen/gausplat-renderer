pub use super::*;

use gausplat_loader::function::{Decoder, DecoderWith};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{
    io::{BufReader, Read},
    mem::take,
};

/// Scene importers
impl<B: Backend> Gaussian3dScene<B> {
    pub const SEED: u64 = 0x3D65;

    pub fn decode_polygon_3dgs(
        reader: &mut impl Read,
        device: &B::Device,
    ) -> Result<Self, Error> {
        let reader = &mut BufReader::new(reader);

        let header = polygon::Header::decode(reader)?;
        if !header.is_same_order(&POLYGON_HEADER_3DGS) {
            return Err(Error::MismatchedPolygonHeader3DGS(header.into()));
        }
        let payload = polygon::Payload::decode_with(reader, &header)?;
        let mut object = polygon::Object { header, payload };

        // NOTE: The header is validated previously.
        let point_count = object.elem("vertex").unwrap().meta.count;

        let mut take_tensor = |names: &[String], device: &B::Device| {
            let dtype = DType::F32;
            let channel_count = names.len();
            let bytes = names.iter().fold(
                Vec::<u8>::with_capacity(channel_count * point_count * dtype.size()),
                |mut bytes, name| {
                    let data = object.elem_prop_mut("vertex", name).unwrap().data;
                    bytes.extend(take(data));
                    bytes
                },
            );
            let data = TensorData {
                bytes,
                shape: [channel_count, point_count].into(),
                dtype,
            };
            Tensor::<B, 2>::from_data(data, device)
                .swap_dims(0, 1)
                .set_require_grad(true)
        };

        // [P, 48] <- [P, 1, 3] + [P, 3, 15]
        let colors_sh = take_tensor(
            &(0..COLORS_SH_COUNT_MAX)
                .map(|i| {
                    if i < 3 {
                        format!("f_dc_{i}")
                    } else {
                        let i = i / 3 + (i % 3) * (SH_COUNT_MAX - 1) - 1;
                        format!("f_rest_{i}")
                    }
                    .into()
                })
                .collect::<Vec<_>>(),
            device,
        );

        // [P, 1]
        let opacities = take_tensor(&["opacity"].map(Into::into), device);

        // [P, 3]
        let positions = take_tensor(&["x", "y", "z"].map(Into::into), device);

        // [P, 4] (x, y, z, w) <- (w, x, y, z)
        let rotations =
            take_tensor(&[1, 2, 3, 0].map(|i| format!("rot_{i}").into()), device);

        // [P, 3]
        let scalings = take_tensor(
            &(0..3)
                .map(|i| format!("scale_{i}").into())
                .collect::<Vec<_>>(),
            device,
        );

        let mut scene = Self::default();
        scene
            .set_inner_colors_sh(colors_sh)
            .set_inner_opacities(opacities)
            .set_inner_positions(positions)
            .set_inner_rotations(rotations)
            .set_inner_scalings(scalings);

        #[cfg(all(debug_assertions, not(test)))]
        log::debug!(
            target: "gausplat::renderer::gaussian_3d::scene",
            "decode_polygon_3dgs",
        );

        Ok(scene)
    }

    pub fn from_points(
        points: Points,
        device: &B::Device,
    ) -> Self {
        // P
        let point_count = points.len();

        // ([P, 3], [P, 3])
        let (colors_rgb, positions) = points.iter().fold(
            (
                Vec::<f32>::with_capacity(point_count * 3),
                Vec::<f64>::with_capacity(point_count * 3),
            ),
            |(mut colors_rgb, mut positions), point| {
                colors_rgb.extend(point.color_rgb);
                positions.extend(point.position);
                (colors_rgb, positions)
            },
        );

        // [P, 48] <- [P, 16, 3]
        let colors_sh = Param::uninitialized(
            Default::default(),
            move |device, is_require_grad| {
                let mut colors_sh =
                    Tensor::zeros([point_count, COLORS_SH_COUNT_MAX], device);
                let colors_rgb = Tensor::from_data(
                    TensorData::new(colors_rgb.to_owned(), [point_count, 3]),
                    device,
                );

                colors_sh = colors_sh.slice_assign(
                    [0..point_count, 0..3],
                    (colors_rgb - 0.5) / SH_COEF.0[0],
                );

                colors_sh = Self::make_inner_colors_sh(colors_sh)
                    .set_require_grad(is_require_grad);

                #[cfg(all(debug_assertions, not(test)))]
                log::debug!(
                    target: "gausplat::renderer::gaussian_3d::scene",
                    "from_points > colors_sh",
                );

                colors_sh
            },
            device.to_owned(),
            true,
        );

        // [P, 1]
        let opacities = Param::uninitialized(
            Default::default(),
            move |device, is_require_grad| {
                let opacities = Self::make_inner_opacities(Tensor::full(
                    [point_count, 1],
                    0.1,
                    device,
                ))
                .set_require_grad(is_require_grad);

                #[cfg(all(debug_assertions, not(test)))]
                log::debug!(
                    target: "gausplat::renderer::gaussian_3d::scene",
                    "from_points > opacities",
                );

                opacities
            },
            device.to_owned(),
            true,
        );

        // [P, 3]
        let positions = Param::uninitialized(
            Default::default(),
            move |device, is_require_grad| {
                let positions = Self::make_inner_positions(Tensor::from_data(
                    TensorData::new(positions.to_owned(), [point_count, 3]),
                    device,
                ))
                .set_require_grad(is_require_grad);

                #[cfg(all(debug_assertions, not(test)))]
                log::debug!(
                    target: "gausplat::renderer::gaussian_3d::scene",
                    "from_points > positions",
                );

                positions
            },
            device.to_owned(),
            true,
        );

        // [P, 4] (x, y, z, w)
        let rotations = Param::uninitialized(
            Default::default(),
            move |device, is_require_grad| {
                let rotations = Self::make_inner_rotations(Tensor::from_data(
                    TensorData::new(
                        [0.0, 0.0, 0.0, 1.0].repeat(point_count),
                        [point_count, 4],
                    ),
                    device,
                ))
                .set_require_grad(is_require_grad);

                #[cfg(all(debug_assertions, not(test)))]
                log::debug!(
                    target: "gausplat::renderer::gaussian_3d::scene",
                    "from_points > rotations",
                );

                rotations
            },
            device.to_owned(),
            true,
        );

        // [P, 3]
        let scalings = Param::uninitialized(
            Default::default(),
            move |device, is_require_grad| {
                let mut sample_max = f32::EPSILON;
                let samples = StdRng::seed_from_u64(Self::SEED)
                    .sample_iter(
                        rand_distr::LogNormal::new(0.0, std::f32::consts::E).unwrap(),
                    )
                    .take(point_count)
                    .map(|mut sample| {
                        sample = sample.max(f32::EPSILON);
                        sample_max = sample_max.max(sample);
                        sample
                    })
                    .collect();

                let scalings = Self::make_inner_scalings(
                    Tensor::from_data(TensorData::new(samples, [point_count, 1]), device)
                        .div_scalar(sample_max)
                        .sqrt()
                        .clamp_min(f32::EPSILON)
                        .repeat_dim(1, 3),
                )
                .set_require_grad(is_require_grad);

                #[cfg(all(debug_assertions, not(test)))]
                log::debug!(
                    target: "gausplat::renderer::gaussian_3d::scene",
                    "from_points > scalings",
                );

                scalings
            },
            device.to_owned(),
            true,
        );

        Self {
            colors_sh,
            opacities,
            positions,
            rotations,
            scalings,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn from_and_to_points() {
        use super::super::*;
        use burn::backend::NdArray;

        let device = Default::default();
        let source = vec![
            Point {
                color_rgb: [1.0, 0.5, 0.0],
                position: [0.0, -0.5, 0.25],
            },
            Point {
                color_rgb: [0.5, 1.0, 0.25],
                position: [1.0, 0.0, -0.25],
            },
        ];

        let scene =
            Gaussian3dScene::<NdArray<f32>>::from_points(source.to_owned(), &device);

        let colors_sh = scene.get_colors_sh();
        assert_eq!(colors_sh.dims(), [2, 48]);

        let opacities = scene.get_opacities();
        assert_eq!(opacities.dims(), [2, 1]);

        let positions = scene.get_positions();
        assert_eq!(positions.dims(), [2, 3]);

        let rotations = scene.get_rotations();
        assert_eq!(rotations.dims(), [2, 4]);

        let scalings = scene.get_scalings();
        assert_eq!(scalings.dims(), [2, 3]);

        assert_eq!(scene.point_count(), 2);
        assert_eq!(scene.size(), (2 * 48 + 2 + 2 * 3 + 2 * 4 + 2 * 3) * 4);

        let target = source;
        let output = scene.to_points();
        assert_eq!(output, target);
    }
}