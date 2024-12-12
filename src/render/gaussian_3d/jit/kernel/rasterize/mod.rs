//! Rasterizing the point to the image.

pub use super::*;

use burn::tensor::ops::{FloatTensorOps, IntTensorOps};
use bytemuck::{bytes_of, Pod, Zeroable};

/// Arguments.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Arguments {
    /// $ \text{im}_x $
    pub image_size_x: u32,
    /// $ \text{im}_y $
    pub image_size_y: u32,

    /// $ \frac{\text{im}_x}{\text{t}_x} $
    ///
    /// $ \text{t}_x $ is the tile width.
    pub tile_count_x: u32,
    /// $ \frac{\text{im}_y}{\text{t}_y} $
    ///
    /// $ \text{t}_y $ is the tile height.
    pub tile_count_y: u32,
}

/// Inputs.
#[derive(Clone, Debug)]
pub struct Inputs<R: JitRuntime> {
    /// $ C_{rgb} \in \mathbb{R}^{3} $ of $ p $ points.
    pub colors_rgb_3d: JitTensor<R>,
    /// $ \Sigma^{'-1} \in \mathbb{R}^{2 \times 2} $ of $ p $ points.
    ///
    /// Inverse of the 2D covariance.
    ///
    /// It can be $ \mathbb{R}^{3} $ since it is symmetric.
    pub conics: JitTensor<R>,
    /// $ \alpha \in \mathbb{R} $ of $ p $ points.
    pub opacities_3d: JitTensor<R>,
    /// $ i \in [0, p) $.
    ///
    /// It is point index per tile.
    pub point_indices: JitTensor<R>,
    /// $ P^' \in \mathbb{R}^{2} $ of $ p $ points.
    ///
    /// 2D position in screen space.
    pub positions_2d: JitTensor<R>,
    /// $ [i_{start}, i_{end}) $ of each tile.
    pub tile_point_ranges: JitTensor<R>,
}

/// Outputs.
#[derive(Clone, Debug)]
pub struct Outputs<R: JitRuntime> {
    /// $ C_{rgb}^' \in \mathbb{R}^{3} $ of each image pixel.
    pub colors_rgb_2d: JitTensor<R>,
    /// Rendered point count of each image pixel.
    pub point_rendered_counts: JitTensor<R>,
    /// $ T_{last} $
    ///
    /// Last transmittance of each image pixel.
    pub transmittances: JitTensor<R>,
}

/// $ \text{t}_x $
pub const TILE_SIZE_X: u32 = 16;
/// $ \text{t}_y $
pub const TILE_SIZE_Y: u32 = 16;

/// Rasterize the point to the image.
///
/// For each pixel in each tile, do the following steps:
///
/// 1. Collect the points in the tile onto the shared memory.
///
/// 2. Compute the Gaussian density centered at the pixel position $ P_x $
///    using the parameters of each point $ n $ touched the tile
///    ([$ \Sigma^{'-1} $](Inputs::conics) and [$ P_v^' $](Inputs::positions_2d)):
/// $$ D = P_v^' - P_x \in \mathbb{R}^2 $$
/// $$ \sigma_n = e^{-\frac{1}{2} D^T \Sigma^{'-1} D} $$
///
/// 3. Accumulate the transmittance [$ T_n $](Outputs::transmittances)
///    and [$ C_{rgb}^' $](Outputs::colors_rgb_2d) of each pixel
///    using [$ \alpha_n $](Inputs::opacities_3d)
///    and [$ C_{rgb,n} $](Inputs::colors_rgb_3d) of each point $ n $
///    (Order-dependent transparency blending):
/// $$ \alpha_n^' \leftarrow \alpha_n \sigma_n $$
/// $$ T_{n + 1} \leftarrow T_n (1 - \alpha_n^') $$
/// $$ C_{rgb}^' \leftarrow C_{rgb,n}^' + (C_{rgb} \cdot \alpha_n^' \cdot T_n) $$
pub fn main<R: JitRuntime, F: FloatElement, I: IntElement, B: BoolElement>(
    arguments: Arguments,
    inputs: Inputs<R>,
) -> Outputs<R> {
    impl_kernel_source!(Kernel, "kernel.wgsl");

    // Specifying the parameters

    let client = &inputs.colors_rgb_3d.client;
    let device = &inputs.colors_rgb_3d.device;
    // I_x
    let image_size_x = arguments.image_size_x as usize;
    // I_y
    let image_size_y = arguments.image_size_y as usize;

    // [I_x, I_y, 3]
    let colors_rgb_2d = JitBackend::<R, F, I, B>::float_empty(
        [image_size_y, image_size_x, 3].into(),
        device,
    );
    // [I_x, I_y]
    let point_rendered_counts =
        JitBackend::<R, F, I, B>::int_empty([image_size_y, image_size_x].into(), device);
    // [I_x, I_y]
    let transmittances = JitBackend::<R, F, I, B>::float_empty(
        [image_size_y, image_size_x].into(),
        device,
    );

    // Launching the kernel

    client.execute(
        Box::new(SourceKernel::new(
            Kernel,
            CubeDim {
                x: TILE_SIZE_X,
                y: TILE_SIZE_Y,
                z: 1,
            },
        )),
        CubeCount::Static(arguments.tile_count_x, arguments.tile_count_y, 1),
        vec![
            client.create(bytes_of(&arguments)).binding(),
            inputs.colors_rgb_3d.handle.binding(),
            inputs.conics.handle.binding(),
            inputs.opacities_3d.handle.binding(),
            inputs.point_indices.handle.binding(),
            inputs.positions_2d.handle.binding(),
            inputs.tile_point_ranges.handle.binding(),
            colors_rgb_2d.handle.to_owned().binding(),
            point_rendered_counts.handle.to_owned().binding(),
            transmittances.handle.to_owned().binding(),
        ],
    );

    Outputs {
        colors_rgb_2d,
        point_rendered_counts,
        transmittances,
    }
}
