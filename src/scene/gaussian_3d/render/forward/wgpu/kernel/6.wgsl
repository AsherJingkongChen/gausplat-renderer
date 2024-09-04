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
var<storage, read_write> point_indexes: array<u32>;
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
var<workgroup> batch_colors_rgb_3d: array<vec3<f32>, BATCH_SIZE>;
// [T_x * T_y, 2, 2]
var<workgroup> batch_conics: array<mat2x2<f32>, BATCH_SIZE>;
// [T_x * T_y, 1]
var<workgroup> batch_opacities_3d: array<f32, BATCH_SIZE>;
// [T_x * T_y, 2]
var<workgroup> batch_positions_2d: array<vec2<f32>, BATCH_SIZE>;
// (0 ~ T_x * T_y)
var<workgroup> pixel_done_count: atomic<u32>;

const OPACITY_2D_MAX: f32 = 0.99;
const OPACITY_2D_MIN: f32 = 1.0 / 255.0;
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

    // R
    var tile_point_count = tile_point_range.y - tile_point_range.x;
    var color_rgb_2d = vec3<f32>();
    var is_pixel_done = !is_pixel_valid;
    var point_rendered_count = 0u;
    var point_rendered_state = 0u;
    var transmittance_state = 1.0;
    var was_pixel_done = false;

    // Processing batches of points of the tile
    // [R / (T_x * T_y)]

    for (var batch_index = 0u; batch_index < batch_count; batch_index++) {
        // Specifying the progress state

        if is_pixel_done && !was_pixel_done {
            was_pixel_done = true;
            let count = atomicAdd(&pixel_done_count, 1u) + 1u;

            // Leaving if all the pixels of tile are finished rendering

            if count == BATCH_SIZE {
                break;
            }
        }

        // Specifying the batch parameters

        workgroupBarrier();
        let index = tile_point_range.x + batch_index * BATCH_SIZE + local_index;
        if index < tile_point_range.y {
            let point_index = point_indexes[index];
            batch_colors_rgb_3d[local_index] = colors_rgb_3d[point_index];
            batch_conics[local_index] = conics[point_index];
            batch_opacities_3d[local_index] = opacities_3d[point_index];
            batch_positions_2d[local_index] = positions_2d[point_index];
        }
        workgroupBarrier();

        // Skipping if the pixel is finished rendering

        if is_pixel_done {
            continue;
        }

        // Computing the 2D color of the pixel in RGB space using the batch parameters
        // [T_x * T_y]

        let batch_point_count = min(tile_point_count, BATCH_SIZE);
        for (var batch_index = 0u; batch_index < batch_point_count; batch_index++) {
            point_rendered_state++;

            // Computing the density of the point in the pixel
            // σ = e^(-0.5 * D^t[1, 2] * Σ'^-1[2, 2] * D[2, 1])

            let conic = batch_conics[batch_index];
            let position_2d = batch_positions_2d[batch_index];
            let position_offset = position_2d - position_pixel;
            let density = exp(-0.5 * dot(position_offset * conic, position_offset));

            // Skipping if the density is greater than one

            if density > 1.0 {
                continue;
            }

            // Computing the 2D opacity of the point in the pixel
            // α' = α * σ

            let opacity_3d = batch_opacities_3d[batch_index];
            let opacity_2d = min(opacity_3d * density, OPACITY_2D_MAX);

            // Skipping if the 2D opacity is too low

            if opacity_2d < OPACITY_2D_MIN {
                continue;
            }

            // Computing the transmittance
            // T[n] = T[n-1] * (1 - α')

            let transmittance = transmittance_state * (1.0 - opacity_2d);

            // Leaving before the transmittance is too low

            if transmittance < TRANSMITTANCE_MIN {
                is_pixel_done = true;
                break;
            }

            // Blending the 3D colors of the pixel into the 2D color in RGB space

            let color_rgb_3d = batch_colors_rgb_3d[batch_index];
            color_rgb_2d += opacity_2d * transmittance_state * color_rgb_3d;

            // Updating the states

            point_rendered_count = point_rendered_state;
            transmittance_state = transmittance;
        }

        tile_point_count -= batch_point_count;
    }

    // Specifying the results

    if is_pixel_valid {
        // Paint the pixel
        // [I_y, I_x, 3]
        colors_rgb_2d[pixel_index] = array<f32, 3>(
            color_rgb_2d[0],
            color_rgb_2d[1],
            color_rgb_2d[2],
        );
        // [I_y, I_x]
        point_rendered_counts[pixel_index] = point_rendered_count;
        // [I_y, I_x]
        transmittances[pixel_index] = transmittance_state;
    }
}
