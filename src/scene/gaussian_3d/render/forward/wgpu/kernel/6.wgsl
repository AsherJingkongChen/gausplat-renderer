struct Arguments {
    // I_x
    image_size_x: u32,
    // I_y
    image_size_y: u32,
}

@group(0) @binding(0)
var<storage, read_write> arguments: Arguments;
// [P, 3 (+ 1)] (0.0 ~ 1.0)
@group(0) @binding(1)
var<storage, read_write> colors_rgb_3d: array<vec3<f32>>;
// [P, 2, 2] (Symmetric)
@group(0) @binding(2)
var<storage, read_write> conics: array<mat2x2<f32>>;
// [P, 1] (0.0 ~ 1.0)
@group(0) @binding(3)
var<storage, read_write> opacities_3d: array<f32>;
// [T] (0 ~ P)
@group(0) @binding(4)
var<storage, read_write> point_indices: array<u32>;
// [P, 2]
@group(0) @binding(5)
var<storage, read_write> positions_2d: array<vec2<f32>>;
// [I_y / T_y, I_x / T_x, 2]
@group(0) @binding(6)
var<storage, read_write> tile_point_ranges: array<vec2<u32>>;

// [I_y, I_x, 3] (0.0 ~ 1.0)
@group(0) @binding(7)
var<storage, read_write> colors_rgb_2d: array<array<f32, 3>>;
// [I_y, I_x]
@group(0) @binding(8)
var<storage, read_write> point_rendered_counts: array<u32>;
// [I_y, I_x] (0.0 ~ 1.0)
@group(0) @binding(9)
var<storage, read_write> transmittances: array<f32>;

// [T_x * T_y, 3]
var<workgroup> colors_rgb_3d_in_batch: array<vec3<f32>, BATCH_SIZE>;
// [T_x * T_y, 2, 2]
var<workgroup> conics_in_batch: array<mat2x2<f32>, BATCH_SIZE>;
// [T_x * T_y, 1]
var<workgroup> opacities_3d_in_batch: array<f32, BATCH_SIZE>;
// [T_x * T_y, 2]
var<workgroup> positions_2d_in_batch: array<vec2<f32>, BATCH_SIZE>;
// (0 ~ T_x * T_y)
var<workgroup> pixel_done_count: atomic<u32>;

const OPACITY_2D_MAX: f32 = 1.0 - 5.0 * OPACITY_2D_MIN;
const OPACITY_2D_MIN: f32 = 0.5 / 255.0;
const TRANSMITTANCE_MIN: f32 = pow(5.0 * OPACITY_2D_MIN, 2.0);
// T_x * T_y
const BATCH_SIZE: u32 = GROUP_SIZE_X * GROUP_SIZE_Y;
// T_x
const GROUP_SIZE_X: u32 = 16;
// T_y
const GROUP_SIZE_Y: u32 = 16;

@compute @workgroup_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)
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
    // (0 ~ I_x * I_y)
    let pixel_index = pixel.y * arguments.image_size_x + pixel.x;
    // (0 ~ (I_y / T_y) * (I_x / T_x))
    let tile_index = tile_id.y * tile_count.x + tile_id.x;

    // Specifying the parameters

    let is_pixel_valid = pixel.x < arguments.image_size_x && pixel.y < arguments.image_size_y;
    let position_pixel = vec2<f32>(pixel);
    let tile_point_range = tile_point_ranges[tile_index];
    // R
    var tile_point_count = tile_point_range.y - tile_point_range.x;
    // R / (T_x * T_y)
    let batch_count = (tile_point_count + BATCH_SIZE - 1) / BATCH_SIZE;
    var color_rgb_2d = vec3<f32>();
    var is_pixel_done = !is_pixel_valid;
    var point_rendered_count = 0u;
    var point_rendered_state = 0u;
    var transmittance_state = 1.0;
    var was_pixel_done = false;
    if local_index == 0 {
        pixel_done_count = 0u;
    }

    // Processing batches of points of the tile
    // (0 ~ R / (T_x * T_y))

    for (var batch_index = 0u; batch_index < batch_count; batch_index++) {
        // Specifying the task status of the pixel

        if is_pixel_done && !was_pixel_done {
            was_pixel_done = true;
            atomicAdd(&pixel_done_count, 1u);
        }
        workgroupBarrier();

        // Leaving if all the pixels in the tile finished rendering

        if pixel_done_count == BATCH_SIZE {
            break;
        }

        // Specifying the parameters in the batch

        let index = tile_point_range.x + batch_index * BATCH_SIZE + local_index;
        if index < tile_point_range.y {
            let point_index = point_indices[index];
            colors_rgb_3d_in_batch[local_index] = colors_rgb_3d[point_index];
            conics_in_batch[local_index] = conics[point_index];
            opacities_3d_in_batch[local_index] = opacities_3d[point_index];
            positions_2d_in_batch[local_index] = positions_2d[point_index];
        }
        workgroupBarrier();

        // Skipping if the pixel is finished rendering

        if is_pixel_done {
            continue;
        }

        // Computing the 2D color of the pixel in RGB space using the batch parameters
        // (0 ~ T_x * T_y)

        let batch_point_count = min(tile_point_count, BATCH_SIZE);
        for (var batch_index = 0u; batch_index < batch_point_count; batch_index++) {
            point_rendered_state++;

            // Computing the density of the point in the pixel
            // D[2, 1] = Pv'[2, 1] - Px[2, 1]
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
            // α'[n] = α[n] * σ[n]

            let opacity_3d = opacities_3d_in_batch[batch_index];
            let opacity_2d = min(opacity_3d * density, OPACITY_2D_MAX);

            // Skipping if the 2D opacity is too low

            if opacity_2d < OPACITY_2D_MIN {
                continue;
            }

            // Computing the next transmittance
            // t[n + 1] = t[n] * (1 - α'[n])

            let transmittance = transmittance_state * (1.0 - opacity_2d);

            // Leaving before the transmittance is too low

            if transmittance < TRANSMITTANCE_MIN {
                is_pixel_done = true;
                break;
            }

            // Blending the 3D colors of the pixel into the 2D color in RGB space
            // C_rgb'[n + 1] = C_rgb'[n] + C_rgb[n] * α'[n] * t[n]

            let color_rgb_3d = colors_rgb_3d_in_batch[batch_index];
            color_rgb_2d += color_rgb_3d * opacity_2d * transmittance_state;

            // Updating the states of the pixel

            point_rendered_count = point_rendered_state;
            transmittance_state = transmittance;
        }

        tile_point_count -= batch_point_count;
    }

    // Specifying the results

    if is_pixel_valid {
        // Painting the pixel

        // [I_y, I_x, 3]
        colors_rgb_2d[pixel_index] = array<f32, 3>(
            color_rgb_2d[0],
            color_rgb_2d[1],
            color_rgb_2d[2],
        );

        // Recording the states

        // [I_y, I_x]
        point_rendered_counts[pixel_index] = point_rendered_count;
        // [I_y, I_x]
        transmittances[pixel_index] = transmittance_state;
    }
}
