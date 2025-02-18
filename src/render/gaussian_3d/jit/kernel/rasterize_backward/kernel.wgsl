struct Arguments {
    // I_x
    image_size_x: u32,
    // I_y
    image_size_y: u32,
}

@group(0) @binding(0)
var<storage, read_write> arguments: Arguments;
// [I_y, I_x, 3]
@group(0) @binding(1)
var<storage, read_write> colors_rgb_2d_grad: array<array<f32, 3>>;
// [P, 3] (0.0 ~ 1.0)
@group(0) @binding(2)
var<storage, read_write> colors_rgb_3d: array<array<f32, 3>>;
// [P, 3] (Symmetric mat2x2)
@group(0) @binding(3)
var<storage, read_write> conics: array<array<f32, 3>>;
// [P, 1] (Inner)
@group(0) @binding(4)
var<storage, read_write> opacities_3d: array<f32>;
// [T] (0 ~ P)
@group(0) @binding(5)
var<storage, read_write> point_indices: array<u32>;
// [I_y, I_x]
@group(0) @binding(6)
var<storage, read_write> point_rendered_counts: array<u32>;
// [P, 2]
@group(0) @binding(7)
var<storage, read_write> positions_2d: array<vec2<f32>>;
// [I_y / T_y, I_x / T_x, 2]
@group(0) @binding(8)
var<storage, read_write> tile_point_ranges: array<vec2<u32>>;
// [I_y, I_x] (0.0 ~ 1.0)
@group(0) @binding(9)
var<storage, read_write> transmittances: array<f32>;

// [P, 3]
@group(0) @binding(10)
var<storage, read_write> colors_rgb_3d_grad: array<atomic<f32>>;
// [P, 3] (Symmetric mat2x2)
@group(0) @binding(11)
var<storage, read_write> conics_grad: array<atomic<f32>>;
// [P, 1] (Inner)
@group(0) @binding(12)
var<storage, read_write> opacities_3d_grad: array<atomic<f32>>;
// [P, 2]
@group(0) @binding(13)
var<storage, read_write> positions_2d_grad: array<atomic<f32>>;

// [T_x * T_y, 3]
var<workgroup> colors_rgb_3d_in_batch: array<vec3<f32>, BATCH_SIZE>;
// [T_x * T_y, 2, 2]
var<workgroup> conics_in_batch: array<mat2x2<f32>, BATCH_SIZE>;
// [T_x * T_y, 1]
var<workgroup> opacities_3d_in_batch: array<f32, BATCH_SIZE>;
// [T_x * T_y]
var<workgroup> point_indices_in_batch: array<u32, BATCH_SIZE>;
// [T_x * T_y, 2]
var<workgroup> positions_2d_in_batch: array<vec2<f32>, BATCH_SIZE>;

const OPACITY_2D_MAX: f32 = 252.0 / 255.0;
const OPACITY_2D_MIN: f32 = 1.0 / 255.0;
// T_x * T_y
const BATCH_SIZE: u32 = TILE_SIZE_X * TILE_SIZE_Y;
// T_x
const TILE_SIZE_X: u32 = 16;
// T_y
const TILE_SIZE_Y: u32 = 16;

@compute @workgroup_size(TILE_SIZE_X, TILE_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    // (0 ~ T_x * T_y)
    @builtin(local_invocation_index) local_index: u32,
    // (0 ~ I_x / T_x, 0 ~ I_y / T_y)
    @builtin(workgroup_id) tile_id: vec3<u32>,
    // (I_x / T_x, I_y / T_y)
    @builtin(num_workgroups) tile_count: vec3<u32>,
) {
    // Specifying the index

    // (0 ~ I_x, 0 ~ I_y)
    let pixel = global_id.xy;
    // (0 ~ I_y * I_x)
    let pixel_index = pixel.y * arguments.image_size_x + pixel.x;
    // (0 ~ (I_y / T_y) * (I_x / T_x))
    let tile_index = tile_id.y * tile_count.x + tile_id.x;

    // Specifying the parameters

    let is_pixel_valid = pixel.x < arguments.image_size_x && pixel.y < arguments.image_size_y;
    let position_pixel = vec2<f32>(pixel);
    var point_range = vec2<u32>();
    // R
    var tile_point_count = 0u;
    if tile_index < arrayLength(&tile_point_ranges) {
        point_range = tile_point_ranges[tile_index];
        if point_range.y > point_range.x {
            tile_point_count = point_range.y - point_range.x;
        }
    }
    // R / (T_x * T_y)
    let batch_count = (tile_point_count + BATCH_SIZE - 1) / BATCH_SIZE;
    let color_rgb_2d_grad = vec3<f32>(
        colors_rgb_2d_grad[pixel_index][0],
        colors_rgb_2d_grad[pixel_index][1],
        colors_rgb_2d_grad[pixel_index][2],
    );
    var color_rgb_2d_state = vec3<f32>();
    var color_rgb_3d_state = vec3<f32>();
    var opacity_2d_state = 0.0;
    var point_rendered_count = u32();
    var point_rendered_state = tile_point_count;
    var transmittance_state = f32();
    if is_pixel_valid {
        point_rendered_count = point_rendered_counts[pixel_index];
        transmittance_state = transmittances[pixel_index];
    }

    // Processing batches of points of the tile
    // (0 ~ R / (T_x * T_y))

    for (var batch_index = 0u; batch_index < batch_count; batch_index++) {
        // Specifying the batch parameters

        workgroupBarrier();
        let index = point_range.y - batch_index * BATCH_SIZE - local_index - 1;
        if index >= point_range.x {
            let point_index = point_indices[index];
            colors_rgb_3d_in_batch[local_index] = vec_from_array_f32_3(colors_rgb_3d[point_index]);
            conics_in_batch[local_index] = mat_sym_from_array_f32_3(conics[point_index]);
            // (Outer)
            opacities_3d_in_batch[local_index] = sigmoid_f32(opacities_3d[point_index]);
            point_indices_in_batch[local_index] = point_index;
            positions_2d_in_batch[local_index] = positions_2d[point_index];
        }
        workgroupBarrier();

        // Skipping if the pixel is finished rendered

        if !is_pixel_valid {
            continue;
        }

        // Computing the 2D colors in RGB space using the batch parameters
        // (0 ~ T_x * T_y)

        let batch_point_count = min(tile_point_count, BATCH_SIZE);
        for (var batch_index = 0u; batch_index < batch_point_count; batch_index++) {
            // Skipping until the point was rendered in the pixel

            if point_rendered_state > point_rendered_count {
                point_rendered_state--;
                continue;
            }

            // Computing the density of the point in the pixel
            // σ[n] = e^(-0.5 * D^t[1, 2] * Σ'^-1[2, 2] * D[2, 1])[n]

            let conic = conics_in_batch[batch_index];
            let position_2d = positions_2d_in_batch[batch_index];
            let position_offset = position_2d - position_pixel;
            let density = exp(-0.5 * dot(position_offset * conic, position_offset));

            // Skipping if the density is greater than one

            if density > 1.0 {
                continue;
            }

            // Computing the 2D opacity of the point in the pixel

            let opacity_3d = opacities_3d_in_batch[batch_index];
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
            color_rgb_3d_state = colors_rgb_3d_in_batch[batch_index];
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
            let opacity_2d_grad_terms = (color_rgb_3d_state - color_rgb_2d_state) * color_rgb_2d_grad;
            let opacity_2d_grad = transmittance_state * (
                opacity_2d_grad_terms[0] +
                opacity_2d_grad_terms[1] +
                opacity_2d_grad_terms[2]
            );

            // Computing the gradients of the point
            // 
            // ∂L/∂α[n] = ∂L/∂α'[n] * σ[n]
            // ∂L/∂σ[n] = ∂L/∂α'[n] * α[n]

            // (Inner)
            let opacity_3d_grad = sigmoid_grad_f32(opacity_3d) * density * opacity_2d_grad;
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
            // ∂L/∂Σ'^-1[2, 2] = (∂L/∂σ * -σ) * D[2, 1] * D^t[1, 2] * 0.5
            // ∂L/∂P[2, 1]     = ∂L/∂D * ∂D/∂P
            //                 = ∂L/∂D
            //                 = (∂L/∂σ * -σ) * Σ'^-1[2, 2] * D[2, 1]
            // 
            // Σ^-1 is symmetric

            let density_density_grad_n = -density * density_grad;
            let conic_grad = 0.5 * density_density_grad_n *
                position_offset.xxy * position_offset.xyy;
            let position_2d_grad = density_density_grad_n * conic * position_offset;

            // Updating the gradients of the point

            let point_index = point_indices_in_batch[batch_index];

            // [P, 3]
            atomicAdd(&colors_rgb_3d_grad[3 * point_index + 0], color_rgb_3d_grad[0]);
            atomicAdd(&colors_rgb_3d_grad[3 * point_index + 1], color_rgb_3d_grad[1]);
            atomicAdd(&colors_rgb_3d_grad[3 * point_index + 2], color_rgb_3d_grad[2]);
            // [P, 3]
            atomicAdd(&conics_grad[3 * point_index + 0], conic_grad[0]);
            atomicAdd(&conics_grad[3 * point_index + 1], conic_grad[1]);
            atomicAdd(&conics_grad[3 * point_index + 2], conic_grad[2]);
            // [P, 1]
            atomicAdd(&opacities_3d_grad[point_index], opacity_3d_grad);
            // [P, 2]
            atomicAdd(&positions_2d_grad[2 * point_index + 0], position_2d_grad[0]);
            atomicAdd(&positions_2d_grad[2 * point_index + 1], position_2d_grad[1]);
        }

        tile_point_count -= batch_point_count;
    }
}

fn array_from_vec_f32_3(v: vec3<f32>) -> array<f32, 3> {
    return array<f32, 3>(v[0], v[1], v[2]);
}

fn mat_sym_from_array_f32_3(a: array<f32, 3>) -> mat2x2<f32> {
    return mat2x2<f32>(a[0], a[1], a[1], a[2]);
}

// y = e^x / (1 + e^x)
fn sigmoid_f32(x: f32) -> f32 {
    let x_exp = exp(x);
    return x_exp / (1.0 + x_exp);
}

// dy/dx = dy/d(e^x) * de^x/dx
//       = (1 + e^x)^-2 * e^x
//       = y * (1 - y)
fn sigmoid_grad_f32(y: f32) -> f32 {
    return y * (1.0 - y);
}

fn vec_from_array_f32_3(a: array<f32, 3>) -> vec3<f32> {
    return vec3<f32>(a[0], a[1], a[2]);
}
