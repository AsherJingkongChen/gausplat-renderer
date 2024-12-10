//! Transforming the points.

pub use super::*;

use burn::tensor::ops::{FloatTensorOps, IntTensorOps};
use bytemuck::{bytes_of, Pod, Zeroable};

/// Arguments.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Arguments {
    /// $ 0 \sim 3 $
    pub colors_sh_degree_max: u32,
    /// $ \text{fl}_x = \frac{\text{im}_x}{2 \tan\left(\frac{\text{fov}_x}{2}\right)} $
    pub focal_length_x: f32,
    /// $ \text{fl}_y = \frac{\text{im}_y}{2 \tan\left(\frac{\text{fov}_y}{2}\right)} $
    pub focal_length_y: f32,
    /// $ \frac{\text{im}_x}{2} $
    pub image_size_half_x: f32,
    /// $ \frac{\text{im}_y}{2} $
    pub image_size_half_y: f32,
    /// $ p $
    pub point_count: u32,
    /// $ \frac{\text{im}_x}{t_x} $
    ///
    /// $ t_x $ is the tile width.
    pub tile_count_x: i32,
    /// $ \frac{\text{im}_y}{t_y} $
    ///
    /// $ t_y $ is the tile height.
    pub tile_count_y: i32,
    /// $ \tan\left(\frac{\text{fov}_x}{2}\right) \cdot (c_f + 1) $
    ///
    /// $ c_f $ is [`FILTER_LOW_PASS`].
    pub view_bound_x: f32,
    /// $ \tan\left(\frac{\text{fov}_y}{2}\right) \cdot (c_f + 1) $
    ///
    /// $ c_f $ is [`FILTER_LOW_PASS`].
    pub view_bound_y: f32,
    /// Padding.
    pub _padding_1: [u32; 2],
    /// $ V_p \in \mathbb{R}^3 $
    ///
    /// It is the position in world space.
    pub view_position: [f32; 3],
    /// Padding.
    pub _padding_2: [u32; 1],
    /// $ M_v \in \mathbb{R}^{4 \times 4} =
    /// \begin{bmatrix} R_v & T_v \\\ 0 & 1 \end{bmatrix} $
    ///
    /// $ R_v \in \mathbb{R}^{3 \times 3} $ is the rotation
    /// from world space to view space.
    pub view_transform: [[f32; 4]; 4],
}

/// Inputs.
#[derive(Clone, Debug)]
pub struct Inputs<R: JitRuntime> {
    /// $ C_{sh} \in \mathbb{R}^{m \times 3} $ of $ p $ points.
    ///
    /// $ m $ is [`SH_COUNT_MAX`](crate::spherical_harmonics::SH_COUNT_MAX).
    pub colors_sh: JitTensor<R>,
    /// $ P \in \mathbb{R}^3 $ of $ p $ points.
    pub positions_3d: JitTensor<R>,
    /// $ R \in \mathbb{R}^4 $ of $ p $ points.
    pub rotations: JitTensor<R>,
    /// $ S \in \mathbb{R}^3 $ of $ p $ points.
    pub scalings: JitTensor<R>,
}

/// Outputs.
#[derive(Clone, Debug)]
pub struct Outputs<R: JitRuntime> {
    /// $ C_{rgb} \in \mathbb{R}^3 $ of $ p $ points.
    pub colors_rgb_3d: JitTensor<R>,
    /// $ \Sigma^{'-1} \in \mathbb{R}^{2 \times 2} $ of $ p $ points.
    ///
    /// It can be $ \mathbb{R}^{3} $ since it is symmetric.
    pub conics: JitTensor<R>,
    /// $ P_v.z \in \mathbb{R}$ of $ p $ points.
    pub depths: JitTensor<R>,
    /// $ \neg \text{clamped}(C_{rgb}) \in \mathbb{R}^3 $ of $ p $ points.
    pub is_colors_rgb_3d_not_clamped: JitTensor<R>,
    /// $ [x_{\max}\ x_{\min}\ y_{\max}\ y_{\min}] $ of $ p $ points.
    ///
    /// $ x, y \in \mathbb{N} $ are the tile indices.
    pub point_tile_bounds: JitTensor<R>,
    /// $ P^' \in \mathbb{R}^2 $ of $ p $ points.
    pub positions_2d: JitTensor<R>,
    /// $ P_v.xy / P_v.z \in \mathbb{R}^2 $ of $ p $ points.
    pub positions_3d_in_normalized: JitTensor<R>,
    /// $ r \in \mathbb{N} $ of $ p $ points.
    pub radii: JitTensor<R>,
    /// $ R_s \in \mathbb{R}^{3 \times 3} $ of $ p $ points.
    pub rotations_matrix: JitTensor<R>,
    /// $ T \in \mathbb{N} $ of $ p $ points.
    pub tile_touched_counts: JitTensor<R>,
}

/// $ c_f $
pub const FILTER_LOW_PASS: f64 = 0.3;
/// Group size.
pub const GROUP_SIZE: u32 = 256;

/// Transforming the points.
///
/// For each one of the $ p $ points, do the following steps:
///
/// 1. Transform the 3D position [$ P $](Inputs::positions_3d) from world space to view space:
/// $$ P_v = R_v P + T_v \in \mathbb{R}^3 $$
///
/// 2. Perform viewing-frustum culling:
/// $$ \text{Exit if } P_v.z \notin \text{frustum.} $$
///
/// 3. Convert the rotation from quaternion [$ R $](Inputs::rotations)
///    to matrix [$ R_s $](Outputs::rotations_matrix):
/// $$ R = [x\ y\ z\ w] $$
/// $$ R_s = 2 \cdot \begin{bmatrix}
///  \- y^2 - z^2 + \frac{1}{2} & x y - w z & x z + w y
/// \\\ x y + w z & - x^2 - z^2 + \frac{1}{2} & y z - w x
/// \\\ x z - w y & y z + w x & - x^2 - y^2 + \frac{1}{2}
/// \end{bmatrix} $$
///
/// 4. Compute the 3D covariance matrix from the rotation and scaling [$ S $](Inputs::scalings)
///    using inverse single value decomposition (SVD):
/// $$ S_s = \begin{bmatrix}
///     S.x & 0 & 0
/// \\\ 0 & S.y & 0
/// \\\ 0 & 0 & S.z
/// \end{bmatrix} $$
/// $$ \Sigma = R_s S_s^2 R_s^T = (R_s S_s) (R_s S_s)^T \in \mathbb{R}^{3 \times 3} $$
///
/// 5. Project the 3D position [$ P $](Inputs::positions_3d) from view space
///    onto [screen space](Outputs::positions_2d)
///    using focal length [$ \text{fl} $](Arguments::focal_length_x)
///    and image size [$ \text{im} $](Arguments::image_size_half_x):
/// $$ P_v^' = \begin{bmatrix}
///     \frac{P_v.x}{P_v.z} \cdot \text{fl}_x
/// \\\ \frac{P_v.y}{P_v.z} \cdot \text{fl}_y
/// \end{bmatrix} + \begin{bmatrix}
///     \frac{\text{im}_x - 1}{2}
/// \\\ \frac{\text{im}_y - 1}{2}
/// \end{bmatrix} $$
///
/// 6. Project the 3D covariance matrix from world space
///    onto [screen space](Outputs::conics):
/// $$ J = d P_v^' / d P_v = \begin{bmatrix}
///     \frac{\text{fl}_x}{P_v.z} & 0 & - \frac{P_v.x}{P_v.z^2} \cdot \text{fl}_x
/// \\\ 0 & \frac{\text{fl}_y}{P_v.z} & - \frac{P_v.y}{P_v.z^2} \cdot \text{fl}_y
/// \end{bmatrix} $$
/// $$ C = \begin{bmatrix}
///     C_f & 0
/// \\\ 0 & C_f
/// \end{bmatrix} $$
/// $$ \Sigma^' = J R_v \Sigma (J R_v)^T + C \in \mathbb{R}^{2 \times 2} $$
///
/// 7. Estimate the maximum radius [$ r $](Outputs::radii) from the 2D covariance
///    using eigenvalue decomposition:
/// $$ |\Sigma^' - \lambda I| = 0 $$
/// $$ \lambda = \frac{\Sigma_{11}^' + \Sigma_{22}^'}{2}
/// \pm \sqrt{(\frac{\Sigma_{11}^' + \Sigma_{22}^'}{2})^2 - |\Sigma^'|} $$
/// $$ 0.9973 = \int_{-k}^{k} \exp(-\frac{x^2}{2}) dx $$
/// $$ r = k \sqrt{\lambda_{\max}} $$
///
/// 8. Compute the [tile bounds](Outputs::point_tile_bounds)
///    and touched tile count [$ T $](Outputs::tile_touched_counts)
///    using tile size [$ t $](Arguments::tile_count_x):
/// $$ [x_{\max}\ x_{\min}] =
/// \text{clamp}(\frac{[(P_v^'.x - r)\ (P_v^'.x + r)]}{t_x}) $$
/// $$ [y_{\max}\ y_{\min}] =
/// \text{clamp}(\frac{[(P_v^'.y - r)\ (P_v^'.y + r)]}{t_y}) $$
/// $$ T = (x_{\max} - x_{\min}) \cdot (y_{\max} - y_{\min}) $$
///
/// 9. Compute viewing direction in world space
///    using view position [$ V_p $](Arguments::view_position):
/// $$ D_v = \frac{P - V_p}{| P - V_p |} \in \mathbb{R}^3 $$
///
/// 10. Transform color from [SH](Inputs::colors_sh)
///     to [RGB](Outputs::colors_rgb_3d) space:
/// $$ D = f(D_v) \in \mathbb{R}^m $$
/// $$ C_{rgb} = D \cdot C_{sh} \in \mathbb{R}^3 $$
pub fn main<R: JitRuntime, F: FloatElement, I: IntElement, B: BoolElement>(
    arguments: Arguments,
    inputs: Inputs<R>,
) -> Outputs<R> {
    impl_kernel_source!(Kernel, "kernel.wgsl");

    // Specifying the parameters

    let client = &inputs.colors_sh.client;
    let device = &inputs.colors_sh.device;
    // P
    let point_count = arguments.point_count as usize;

    let colors_rgb_3d =
        JitBackend::<R, F, I, B>::float_empty([point_count, 3].into(), device);
    let conics = JitBackend::<R, F, I, B>::float_empty([point_count, 3].into(), device);
    let depths = JitBackend::<R, F, I, B>::float_empty([point_count].into(), device);
    let is_colors_rgb_3d_not_clamped =
        JitBackend::<R, F, I, B>::float_empty([point_count, 3].into(), device);
    let point_tile_bounds =
        JitBackend::<R, F, I, B>::int_empty([point_count, 4].into(), device);
    let positions_2d =
        JitBackend::<R, F, I, B>::float_empty([point_count, 2].into(), device);
    let positions_3d_in_normalized =
        JitBackend::<R, F, I, B>::float_empty([point_count, 2].into(), device);
    let radii = JitBackend::<R, F, I, B>::int_empty([point_count].into(), device);
    let rotations_matrix =
        JitBackend::<R, F, I, B>::float_empty([point_count, 3, 3].into(), device);
    let tile_touched_counts =
        JitBackend::<R, F, I, B>::int_empty([point_count].into(), device);

    // Launching the kernel

    client.execute(
        Box::new(SourceKernel::new(
            Kernel,
            CubeDim {
                x: GROUP_SIZE,
                y: 1,
                z: 1,
            },
        )),
        CubeCount::Static(arguments.point_count.div_ceil(GROUP_SIZE), 1, 1),
        vec![
            client.create(bytes_of(&arguments)).binding(),
            inputs.colors_sh.handle.binding(),
            inputs.positions_3d.handle.binding(),
            inputs.rotations.handle.binding(),
            inputs.scalings.handle.binding(),
            colors_rgb_3d.handle.to_owned().binding(),
            conics.handle.to_owned().binding(),
            depths.handle.to_owned().binding(),
            is_colors_rgb_3d_not_clamped.handle.to_owned().binding(),
            point_tile_bounds.handle.to_owned().binding(),
            positions_2d.handle.to_owned().binding(),
            positions_3d_in_normalized.handle.to_owned().binding(),
            radii.handle.to_owned().binding(),
            rotations_matrix.handle.to_owned().binding(),
            tile_touched_counts.handle.to_owned().binding(),
        ],
    );

    Outputs {
        colors_rgb_3d,
        conics,
        depths,
        is_colors_rgb_3d_not_clamped,
        point_tile_bounds,
        positions_2d,
        positions_3d_in_normalized,
        radii,
        rotations_matrix,
        tile_touched_counts,
    }
}
