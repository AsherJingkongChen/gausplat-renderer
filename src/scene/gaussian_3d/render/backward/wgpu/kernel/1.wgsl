struct Arguments {
    // I_X
    image_size_x: u32,
    // I_Y
    image_size_y: u32,
}

@group(0) @binding(0)
var<storage, read> arguments: Arguments;
// [P, 2, 2] (Symmetric)
@group(0) @binding(1)
var<storage, read> conics: array<mat2x2<f32>>;
// [I_Y, I_X, 3]
@group(0) @binding(2)
var<storage, read> colors_rgb_2d_grad: array<array<f32, 3>>;
// [P, 3 (+ 1)] (0.0 ~ 1.0)
@group(0) @binding(3)
var<storage, read> colors_rgb_3d: array<vec3<f32>>;
// [P, 1] (0.0 ~ 1.0)
@group(0) @binding(4)
var<storage, read> opacities_3d: array<f32>;
// [T] (0 ~ P)
@group(0) @binding(5)
var<storage, read> point_indexes: array<u32>;
// [I_Y, I_X]
@group(0) @binding(6)
var<storage, read> point_rendered_counts: array<u32>;
// [P, 2]
@group(0) @binding(7)
var<storage, read> positions_2d: array<vec2<f32>>;
// [(I_X / T_X) * (I_Y / T_Y), 2]
@group(0) @binding(8)
var<storage, read> tile_point_ranges: array<vec2<u32>>;
// [I_Y, I_X] (0.0 ~ 1.0)
@group(0) @binding(9)
var<storage, read> transmittances: array<f32>;

// [P, 3 (+ 1)]
@group(0) @binding(10)
var<storage, read_write> colors_rgb_3d_grad: array<atomic<u32>>;
// [P, 2, 2] (Symmetric)
@group(0) @binding(11)
var<storage, read_write> conics_grad: array<atomic<u32>>;
// [P, 1]
@group(0) @binding(12)
var<storage, read_write> opacities_3d_grad: array<atomic<u32>>;
// [P, 2]
@group(0) @binding(13)
var<storage, read_write> positions_2d_grad: array<atomic<u32>>;

// [T_X * T_Y, 3]
var<workgroup> batch_colors_rgb_3d: array<vec3<f32>, BATCH_SIZE>;
// [T_X * T_Y, 2, 2]
var<workgroup> batch_conics: array<mat2x2<f32>, BATCH_SIZE>;
// [T_X * T_Y, 1]
var<workgroup> batch_opacities_3d: array<f32, BATCH_SIZE>;
// [T_X * T_Y]
var<workgroup> batch_point_indexes: array<u32, BATCH_SIZE>;
// [T_X * T_Y, 2]
var<workgroup> batch_positions_2d: array<vec2<f32>, BATCH_SIZE>;

const OPACITY_2D_MAX: f32 = 0.99;
const OPACITY_2D_MIN: f32 = 1.0 / 255.0;
const TRANSMITTANCE_MIN: f32 = 1e-4;
// T_X
const GROUP_SIZE_X: u32 = 16;
// T_Y
const GROUP_SIZE_Y: u32 = 16;
// T_X * T_Y
const BATCH_SIZE: u32 = GROUP_SIZE_X * GROUP_SIZE_Y;

@compute @workgroup_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    // (I_X / T_X, I_Y / T_Y)
    @builtin(num_workgroups) group_count: vec3<u32>,
    // (0 ~ T_X * T_Y)
    @builtin(local_invocation_index) local_index: u32,
) {
    // Specifying the parameters

    // (0 ~ I_X, 0 ~ I_Y)
    let pixel = global_id.xy;
    let position_pixel = vec2<f32>(pixel);
    let pixel_index = pixel.y * arguments.image_size_x + pixel.x;
    let is_pixel_valid = pixel.x < arguments.image_size_x && pixel.y < arguments.image_size_y;
    // (0 ~ I_X / T_X, 0 ~ I_Y / T_Y)
    let tile_point_range = tile_point_ranges[group_id.y * group_count.x + group_id.x];
    let batch_count = (tile_point_range.y - tile_point_range.x + BATCH_SIZE - 1) / BATCH_SIZE;
    let color_rgb_2d_grad = vec3<f32>(
        colors_rgb_2d_grad[pixel_index][0],
        colors_rgb_2d_grad[pixel_index][1],
        colors_rgb_2d_grad[pixel_index][2],
    );
    let point_rendered_count = select(0u, point_rendered_counts[pixel_index], is_pixel_valid);

    // R
    var tile_point_count = tile_point_range.y - tile_point_range.x;
    var color_rgb_2d_state = vec3<f32>();
    var color_rgb_3d_state = vec3<f32>();
    var opacity_2d_state = 0.0;
    var point_rendered_state = tile_point_count;
    var transmittance_state = select(0.0, transmittances[pixel_index], is_pixel_valid);

    // Processing batches of points of the tile
    // [R / (T_X * T_Y)]

    for (var batch_index = 0u; batch_index < batch_count; batch_index++) {
        // Specifying the batch parameters

        workgroupBarrier();
        let index = tile_point_range.y - batch_index * BATCH_SIZE - local_index - 1;
        if index >= tile_point_range.x {
            let point_index = point_indexes[index];
            batch_colors_rgb_3d[local_index] = colors_rgb_3d[point_index];
            batch_conics[local_index] = conics[point_index];
            batch_opacities_3d[local_index] = opacities_3d[point_index];
            batch_point_indexes[local_index] = point_index;
            batch_positions_2d[local_index] = positions_2d[point_index];
        }
        workgroupBarrier();

        // Skipping if the pixel is finished rendered

        if !is_pixel_valid {
            continue;
        }

        // Computing the 2D colors in RGB space using the batch parameters
        // [T_X * T_Y]

        let batch_point_count = min(tile_point_count, BATCH_SIZE);
        for (var batch_index = 0u; batch_index < batch_point_count; batch_index++) {
            // Skipping until the point was rendered in the pixel

            if point_rendered_state > point_rendered_count {
                point_rendered_state--;
                continue;
            }

            // Computing the density of the point in the pixel
            // g[1, 1] = d^T[1, 2] * c'^-1[2, 2] * d[2, 1]

            let conic = batch_conics[batch_index];
            let position_2d = batch_positions_2d[batch_index];
            let position_offset = position_2d - position_pixel;
            let density = exp(-0.5 * dot(position_offset * conic, position_offset));

            // Skipping if the density is greater than one

            if density > 1.0 {
                continue;
            }

            // Computing the 2D opacity of the point in the pixel

            let opacity_3d = batch_opacities_3d[batch_index];
            let opacity_2d = min(opacity_3d * density, OPACITY_2D_MAX);

            // Skipping if the 2D opacity is too low

            if opacity_2d < OPACITY_2D_MIN {
                continue;
            }

            // Updating the states

            color_rgb_2d_state *= 1.0 - opacity_2d_state;
            color_rgb_2d_state += color_rgb_3d_state * opacity_2d_state;
            color_rgb_3d_state = batch_colors_rgb_3d[batch_index];
            opacity_2d_state = opacity_2d;
            transmittance_state /= 1.0 - opacity_2d;

            // Computing the gradients of the point

            let color_rgb_3d_grad = opacity_2d * transmittance_state * color_rgb_2d_grad;
            let opacity_2d_grad = transmittance_state * dot(
                (color_rgb_3d_state - color_rgb_2d_state) * color_rgb_2d_grad,
                vec3<f32>(1.0),
            );
            let opacity_3d_grad = density * opacity_2d_grad;
            let density_grad = opacity_3d * opacity_2d_grad;
            let density_density_grad_n = -density * density_grad;
            let conic_grad = 0.5 * density_density_grad_n * mat2x2<f32>(
                position_offset * position_offset.x,
                position_offset * position_offset.y,
            );
            let position_2d_grad = conic * position_offset * density_density_grad_n;

            // Updating the gradients of the point
            // - 3D color in RGB space
            // - Inverse of 2D covariance
            // - 3D opacity
            // - 2D position

            let point_index = batch_point_indexes[batch_index];

            // [P, 3 (+ 1)]
            for (var expected = atomicLoad(&colors_rgb_3d_grad[4 * point_index + 0]);;) {
                let tested = atomicCompareExchangeWeak(
                    &colors_rgb_3d_grad[4 * point_index + 0],
                    expected,
                    bitcast<u32>(bitcast<f32>(expected) + color_rgb_3d_grad[0]),
                );
                if tested.exchanged {
                    break;
                }
                expected = tested.old_value;
            }
            for (var expected = atomicLoad(&colors_rgb_3d_grad[4 * point_index + 1]);;) {
                let tested = atomicCompareExchangeWeak(
                    &colors_rgb_3d_grad[4 * point_index + 1],
                    expected,
                    bitcast<u32>(bitcast<f32>(expected) + color_rgb_3d_grad[1]),
                );
                if tested.exchanged {
                    break;
                }
                expected = tested.old_value;
            }
            for (var expected = atomicLoad(&colors_rgb_3d_grad[4 * point_index + 2]);;) {
                let tested = atomicCompareExchangeWeak(
                    &colors_rgb_3d_grad[4 * point_index + 2],
                    expected,
                    bitcast<u32>(bitcast<f32>(expected) + color_rgb_3d_grad[2]),
                );
                if tested.exchanged {
                    break;
                }
                expected = tested.old_value;
            }
            // [P, 2, 2]
            for (var expected = atomicLoad(&conics_grad[4 * point_index + 0]);;) {
                let tested = atomicCompareExchangeWeak(
                    &conics_grad[4 * point_index + 0],
                    expected,
                    bitcast<u32>(bitcast<f32>(expected) + conic_grad[0][0]),
                );
                if tested.exchanged {
                    break;
                }
                expected = tested.old_value;
            }
            for (var expected = atomicLoad(&conics_grad[4 * point_index + 1]);;) {
                let tested = atomicCompareExchangeWeak(
                    &conics_grad[4 * point_index + 1],
                    expected,
                    bitcast<u32>(bitcast<f32>(expected) + conic_grad[0][1]),
                );
                if tested.exchanged {
                    break;
                }
                expected = tested.old_value;
            }
            for (var expected = atomicLoad(&conics_grad[4 * point_index + 2]);;) {
                let tested = atomicCompareExchangeWeak(
                    &conics_grad[4 * point_index + 2],
                    expected,
                    bitcast<u32>(bitcast<f32>(expected) + conic_grad[1][0]),
                );
                if tested.exchanged {
                    break;
                }
                expected = tested.old_value;
            }
            for (var expected = atomicLoad(&conics_grad[4 * point_index + 3]);;) {
                let tested = atomicCompareExchangeWeak(
                    &conics_grad[4 * point_index + 3],
                    expected,
                    bitcast<u32>(bitcast<f32>(expected) + conic_grad[1][1]),
                );
                if tested.exchanged {
                    break;
                }
                expected = tested.old_value;
            }
            // [P, 1]
            for (var expected = atomicLoad(&opacities_3d_grad[point_index]);;) {
                let tested = atomicCompareExchangeWeak(
                    &opacities_3d_grad[point_index],
                    expected,
                    bitcast<u32>(bitcast<f32>(expected) + opacity_3d_grad),
                );
                if tested.exchanged {
                    break;
                }
                expected = tested.old_value;
            }
            // [P, 2]
            for (var expected = atomicLoad(&positions_2d_grad[2 * point_index + 0]);;) {
                let tested = atomicCompareExchangeWeak(
                    &positions_2d_grad[2 * point_index + 0],
                    expected,
                    bitcast<u32>(bitcast<f32>(expected) + position_2d_grad[0]),
                );
                if tested.exchanged {
                    break;
                }
                expected = tested.old_value;
            }
            for (var expected = atomicLoad(&positions_2d_grad[2 * point_index + 1]);;) {
                let tested = atomicCompareExchangeWeak(
                    &positions_2d_grad[2 * point_index + 1],
                    expected,
                    bitcast<u32>(bitcast<f32>(expected) + position_2d_grad[1]),
                );
                if tested.exchanged {
                    break;
                }
                expected = tested.old_value;
            }
        }

        tile_point_count -= batch_point_count;
    }
}
