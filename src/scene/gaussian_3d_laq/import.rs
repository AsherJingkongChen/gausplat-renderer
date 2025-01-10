//! 3DGS-LAQ import implementation.

pub use super::*;
pub use crate::error::Error;

use burn::record::{BinBytesRecorder, HalfPrecisionSettings, Recorder};
use miniz_oxide::inflate::decompress_to_vec;

impl<B: Backend> Gaussian3dSceneLAQ<B> {
    /// Import from the compressed byte stream made by [`Self::encode_compressed_bytes`].
    pub fn decode_compressed_bytes(
        compressed_bytes: &[u8],
        device: &B::Device,
    ) -> Result<Self, Error> {
        let bytes = decompress_to_vec(compressed_bytes).map_err(Error::Decompress)?;
        let recorder = BinBytesRecorder::<HalfPrecisionSettings>::new();
        let record = recorder.load(bytes, device)?;
        let scene = Gaussian3dSceneLAQ::<B>::default().load_record(record);
        Ok(scene)
    }
}

impl<AB: AutodiffBackend> Gaussian3dSceneLAQ<AB> {
    /// Import from the [scene](Gaussian3dScene).
    pub fn from_scene(scene: &Gaussian3dScene<AB>) -> Self {
        let m_3 = scene.colors_sh.dims()[1];
        let point_count = scene.point_count();
        let colors_sh_dc = scene.colors_sh.val().slice([0..point_count, 0..3]);
        let colors_sh_rest = scene.colors_sh.val().slice([0..point_count, 3..m_3]);
        let opacities = scene.opacities.val();
        let rotations = scene.rotations.val();
        let scalings = scene.scalings.val();

        let colors_sh_dc_q_h = Param::from_tensor(colors_sh_dc.to_owned());
        let colors_sh_rest_q_h = Param::from_tensor(colors_sh_rest.to_owned());
        let opacities_q_h = Param::from_tensor(opacities.to_owned());
        let rotations_q_h = Param::from_tensor(rotations.to_owned());
        let scalings_q_h = Param::from_tensor(scalings.to_owned());

        let dequantizer_colors_sh_dc = Dequantizer::fit(colors_sh_dc, 0.1, 1e-4);
        let dequantizer_colors_sh_rest = Dequantizer::fit(colors_sh_rest, 0.1, 1e-4);
        let dequantizer_opacities = Dequantizer::fit(opacities, 0.1, 1e-6);
        let dequantizer_rotations = Dequantizer::fit(rotations, 0.2, 1e-3);
        let dequantizer_scalings = Dequantizer::fit(scalings, 0.1, 1e-4);

        Self {
            colors_sh_dc_q_h,
            colors_sh_rest_q_h,
            opacities_q_h,
            rotations_q_h,
            scalings_q_h,
            dequantizer_colors_sh_dc,
            dequantizer_colors_sh_rest,
            dequantizer_opacities,
            dequantizer_rotations,
            dequantizer_scalings,
        }
    }
}
