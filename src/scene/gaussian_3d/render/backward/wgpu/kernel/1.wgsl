struct Arguments {
    // I_x
    image_size_x: u32,
    // I_y
    image_size_y: u32,
}

@group(0) @binding(0)
var<storage, read_write> arguments: Arguments;
// [P, 2, 2] (Symmetric)
@group(0) @binding(1)
var<storage, read_write> conics: array<mat2x2<f32>>;
// [I_y, I_x, 3]
@group(0) @binding(2)
var<storage, read_write> colors_rgb_2d_grad: array<array<f32, 3>>;
// [P, 3 (+ 1)] (0.0 ~ 1.0)
@group(0) @binding(3)
var<storage, read_write> colors_rgb_3d: array<vec3<f32>>;
// [P, 1] (0.0 ~ 1.0)
@group(0) @binding(4)
var<storage, read_write> opacities_3d: array<f32>;
// [T] (0 ~ P)
@group(0) @binding(5)
var<storage, read_write> point_indexes: array<u32>;
// [I_y, I_x]
@group(0) @binding(6)
var<storage, read_write> point_rendered_counts: array<u32>;
// [P, 2]
@group(0) @binding(7)
var<storage, read_write> positions_2d: array<vec2<f32>>;
// [(I_y / T_y) * (I_x / T_x), 2]
@group(0) @binding(8)
var<storage, read_write> tile_point_ranges: array<vec2<u32>>;
// [I_y, I_x] (0.0 ~ 1.0)
@group(0) @binding(9)
var<storage, read_write> transmittances: array<f32>;

// [P, 3 (+ 1)]
@group(0) @binding(10)
var<storage, read_write> colors_rgb_3d_grad: array<atomic<f32>>;
// [P, 2, 2] (Symmetric)
@group(0) @binding(11)
var<storage, read_write> conics_grad: array<atomic<f32>>;
// [P, 1]
@group(0) @binding(12)
var<storage, read_write> opacities_3d_grad: array<atomic<f32>>;
// [P, 2]
@group(0) @binding(13)
var<storage, read_write> positions_2d_grad: array<atomic<f32>>;

// [T_x * T_y, 3]
var<workgroup> batch_colors_rgb_3d: array<vec3<f32>, BATCH_SIZE>;
// [T_x * T_y, 2, 2]
var<workgroup> batch_conics: array<mat2x2<f32>, BATCH_SIZE>;
// [T_x * T_y, 1]
var<workgroup> batch_opacities_3d: array<f32, BATCH_SIZE>;
// [T_x * T_y]
var<workgroup> batch_point_indexes: array<u32, BATCH_SIZE>;
// [T_x * T_y, 2]
var<workgroup> batch_positions_2d: array<vec2<f32>, BATCH_SIZE>;

const OPACITY_2D_MAX: f32 = 0.99;
const OPACITY_2D_MIN: f32 = 0.5 / 255.0;
const TRANSMITTANCE_MIN: f32 = 1e-4;
// T_x
const GROUP_SIZE_X: u32 = 16;
// T_y
const GROUP_SIZE_Y: u32 = 16;
// T_x * T_y
const BATCH_SIZE: u32 = GROUP_SIZE_X * GROUP_SIZE_Y;

@compute @workgroup_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    // (I_x / T_x, I_y / T_y)
    @builtin(num_workgroups) group_count: vec3<u32>,
    // (0 ~ T_x * T_y)
    @builtin(local_invocation_index) local_index: u32,
) {
    // Specifying the parameters

    // (0 ~ I_x, 0 ~ I_y)
    let pixel = global_id.xy;
    let position_pixel = vec2<f32>(pixel);
    let pixel_index = pixel.y * arguments.image_size_x + pixel.x;
    let is_pixel_valid = pixel.x < arguments.image_size_x && pixel.y < arguments.image_size_y;
    // (0 ~ I_x / T_x, 0 ~ I_y / T_y)
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
    // [R / (T_x * T_y)]

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
        // [T_x * T_y]

        let batch_point_count = min(tile_point_count, BATCH_SIZE);
        for (var batch_index = 0u; batch_index < batch_point_count; batch_index++) {
            // Skipping until the point was rendered in the pixel

            if point_rendered_state > point_rendered_count {
                point_rendered_state--;
                continue;
            }

            // Computing the density of the point in the pixel
            // σ[n] = e^(-0.5 * D^t[1, 2] * Σ'^-1[2, 2] * D[2, 1])[n]

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

            // Updating the states of the pixel
            // 
            // C_rgb'[n] = C_rgb[n + 1] * α'[n + 1] +
            //             C_rgb'[n + 1] * (1 - α'[n + 1])
            // C_rgb[n]  = C_rgb[n]
            // α'[n]     = α[n] * σ[n]
            // t[n]      = t[n + 1] / (1 - α'[n])

            color_rgb_2d_state = color_rgb_3d_state * opacity_2d_state
                               + color_rgb_2d_state * (1.0 - opacity_2d_state);
            color_rgb_3d_state = batch_colors_rgb_3d[batch_index];
            opacity_2d_state = opacity_2d;
            transmittance_state /= 1.0 - opacity_2d;

            // Computing the gradients of the point
            // 
            // ∂L =〈∂L/∂C_rgb', ∂C_rgb'〉
            //    =〈∂L/∂C_rgb', sum(∂C_rgb[n] *  α'[n] *  t[n])〉+
            //     〈∂L/∂C_rgb', sum( C_rgb[n] * ∂α'[n] *  t[n])〉+
            //     〈∂L/∂C_rgb', sum( C_rgb[n] *  α'[n] * ∂t[n])〉
            // 
            // t[n + 1] = t[n] * (1 - α'[n])
            // 
            // ∂L/∂C_rgb[n] = ∂L/∂C_rgb' * α'[n] * t[n]
            // ∂L/∂α'[n]    =〈∂L/∂C_rgb', C_rgb[n] * t[n]〉+
            //               〈∂L/∂C_rgb', C_rgb[n + 1] * α'[n + 1] * t[n] * (1 - α'[n])〉+
            //               〈∂L/∂C_rgb', C_rgb[n + 2] * α'[n + 2] * t[n] * (1 - α'[n]) * (1 - α'[n + 1])〉+ ...
            //              =〈∂L/∂C_rgb', t[n] * C_rgb[n]〉-
            //               〈∂L/∂C_rgb', t[n] * C_rgb[n + 1] * α'[n + 1]〉-
            //               〈∂L/∂C_rgb', t[n] * C_rgb[n + 2] * α'[n + 2] * (1 - α'[n + 1])〉- ...
            //              =〈∂L/∂C_rgb', t[n] * (C_rgb[n] - C_rgb'[n])〉

            let color_rgb_3d_grad = opacity_2d * transmittance_state * color_rgb_2d_grad;
            let opacity_2d_grad = transmittance_state * dot(
                (color_rgb_3d_state - color_rgb_2d_state) * color_rgb_2d_grad,
                vec3<f32>(1.0),
            );

            // Computing the gradients of the point
            // 
            // ∂L/∂α[n] = ∂L/∂α'[n] * σ[n]
            // ∂L/∂σ[n] = ∂L/∂α'[n] * α[n]

            let opacity_3d_grad = density * opacity_2d_grad;
            let density_grad = opacity_3d * opacity_2d_grad;

            // Computing the gradients of the point
            // 
            // ∂L =〈∂L/∂σ, ∂σ〉[n]
            //    =〈∂L/∂σ, ∂e^(-0.5 * D^t * Σ'^-1 * D)〉
            //    =〈∂L/∂σ, ∂(D^t * Σ'^-1 * D) * -0.5 * e^(-0.5 * D^t * Σ'^-1 * D)〉
            //    =〈∂L/∂σ, ∂(D^t * Σ'^-1 * D)〉* -0.5 * σ
            //    =〈∂L/∂σ, (∂D^t * Σ'^-1 * D) + (D^t * ∂Σ'^-1 * D) + (D^t * Σ'^-1 * ∂D)〉* -0.5 * σ
            //    =〈Σ'^-1 * D * ∂L/∂σ, ∂D〉* 1.0 * -σ +〈D * ∂L/∂σ * D^t, ∂Σ'^-1〉* 0.5 * -σ
            // 
            // ∂L/∂Σ'^-1[2, 2] = ∂L/∂σ * -σ * D[2, 1] * D^t[1, 2] * 0.5
            // ∂L/∂P[2, 1]     = ∂L/∂D * ∂D/∂P
            //                 = ∂L/∂D
            //                 = ∂L/∂σ * -σ * Σ'^-1[2, 2] * D[2, 1]
            // 
            // Σ^-1 is symmetric

            let density_density_grad_n = -density * density_grad;
            let conic_grad = 0.5 * density_density_grad_n * mat2x2<f32>(
                position_offset * position_offset.x,
                position_offset * position_offset.y,
            );
            let position_2d_grad = density_density_grad_n * conic * position_offset;

            // Updating the gradients of the point

            let point_index = batch_point_indexes[batch_index];

            // [P, 3 (+ 1)]
            atomicAdd(&colors_rgb_3d_grad[4 * point_index + 0], color_rgb_3d_grad[0]);
            atomicAdd(&colors_rgb_3d_grad[4 * point_index + 1], color_rgb_3d_grad[1]);
            atomicAdd(&colors_rgb_3d_grad[4 * point_index + 2], color_rgb_3d_grad[2]);
            // [P, 2, 2]
            atomicAdd(&conics_grad[4 * point_index + 0], conic_grad[0][0]);
            atomicAdd(&conics_grad[4 * point_index + 1], conic_grad[0][1]);
            atomicAdd(&conics_grad[4 * point_index + 2], conic_grad[1][0]);
            atomicAdd(&conics_grad[4 * point_index + 3], conic_grad[1][1]);
            // [P, 1]
            atomicAdd(&opacities_3d_grad[point_index], opacity_3d_grad);
            // [P, 2]
            atomicAdd(&positions_2d_grad[2 * point_index + 0], position_2d_grad[0]);
            atomicAdd(&positions_2d_grad[2 * point_index + 1], position_2d_grad[1]);
        }

        tile_point_count -= batch_point_count;
    }
}
