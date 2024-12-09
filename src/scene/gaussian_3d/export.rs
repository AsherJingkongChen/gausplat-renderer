//! 3DGS scene export implementation.

pub use super::*;

use gausplat_loader::function::Encoder;
use std::io::{BufWriter, Write};

/// Scene exporters
impl<B: Backend> Gaussian3dScene<B> {
    /// Export the scene in the 3DGS PLY format.
    pub fn encode_polygon(
        &self,
        writer: &mut impl Write,
    ) -> Result<(), Error> {
        let writer = &mut BufWriter::new(writer);

        let point_count = self.point_count();

        // [P, 1, 3] + [P, 3, M - 3] <- [P, M * 3]
        let colors_sh_dc = self.colors_sh.val().slice([0..point_count, 0..3]);
        let colors_sh_rest = self
            .colors_sh
            .val()
            .slice([0..point_count, 3..SH_COUNT_MAX * 3])
            .reshape([point_count, SH_COUNT_MAX - 1, 3])
            .swap_dims(1, 2)
            .flatten(1, 2);

        // [P, 1]
        let opacities = self.opacities.val();

        // [P, 3]
        let positions = self.positions.val();

        // [P, 4] (w, x, y, z) <- (x, y, z, w)
        let rotations_scalar = self.rotations.val().slice([0..point_count, 3..4]);
        let rotations_vector = self.rotations.val().slice([0..point_count, 0..3]);

        // [P, 3]
        let scalings = self.scalings.val();

        // [P, 3] (Unused)
        let normals = Tensor::<B, 2>::zeros([point_count, 3], &self.device());

        // [P, 62] <- [P, 3 + 3 + 3 + 45 + 1 + 3 + 1 + 3]
        let data = Tensor::cat(
            [
                positions,
                normals,
                colors_sh_dc,
                colors_sh_rest,
                opacities,
                scalings,
                rotations_scalar,
                rotations_vector,
            ]
            .into(),
            1,
        )
        .into_data()
        .bytes;

        let mut header = POLYGON_HEADER_3DGS.to_owned();
        // NOTE: The data format is set to binary native-endian.
        header.format = polygon::Format::binary_native_endian();
        header.get_mut("vertex").unwrap().count = point_count;
        header.encode(writer)?;

        writer.write_all(&data)?;

        Ok(())
    }

    /// Export the scene as a point cloud.
    // TODO: It needs a point cloud viewer to validate the function.
    pub fn to_points(&self) -> Points {
        let point_count = self.point_count();

        // NOTE: The data type is converted.
        let colors_rgb = self
            .get_colors_sh()
            .slice([0..point_count, 0..3])
            .mul_scalar(SH_COEF.0[0])
            .add_scalar(0.5)
            .into_data()
            .convert::<f32>()
            .into_vec()
            .unwrap();

        // NOTE: The data type is converted.
        let positions = self
            .get_positions()
            .into_data()
            .convert::<f64>()
            .into_vec()
            .unwrap();

        colors_rgb
            .chunks_exact(3)
            .zip(positions.chunks_exact(3))
            .map(|(color_rgb, position)| Point {
                // NOTE: The slice size is guaranteed to fit.
                color_rgb: color_rgb.try_into().unwrap(),
                position: position.try_into().unwrap(),
            })
            .collect()
    }
}
