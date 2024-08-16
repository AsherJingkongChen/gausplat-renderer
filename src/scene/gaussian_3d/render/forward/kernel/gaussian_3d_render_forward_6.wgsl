struct Arguments {
    // I_X
    image_size_x: u32,

    // I_Y
    image_size_y: u32,
}

@group(0) @binding(0)
var<storage, read> arguments: Arguments;
// [P, 3]
@group(0) @binding(1)
var<storage, read> colors_rgb_3d: array<vec3<f32>>;
// [P, 2, 2]
@group(0) @binding(2)
var<storage, read> conics: array<mat2x2<f32>>;
// [P]
@group(0) @binding(3)
var<storage, read> opacities: array<f32>;
// [P, 2]
@group(0) @binding(4)
var<storage, read> positions_2d_in_screen: array<vec2<f32>>;
// [T]
@group(0) @binding(5)
var<storage, read> point_indexes: array<u32>;
// [(I_X / T_X) * (I_Y / T_Y), 2]
@group(0) @binding(6)
var<storage, read> tile_point_ranges: array<vec2<u32>>;
// [I_Y, I_X, 3]
@group(0) @binding(7)
var<storage, read_write> colors_rgb_2d: array<array<f32, 3>>;
// [I_Y, I_X]
@group(0) @binding(8)
var<storage, read_write> point_rendered_counts: array<u32>;
// [I_Y, I_X]
@group(0) @binding(9)
var<storage, read_write> transmittances: array<f32>;

// [T_X * T_Y, 3]
var<workgroup> batch_colors_rgb_3d: array<vec3<f32>, BATCH_SIZE>;
// [T_X * T_Y, 2, 2]
var<workgroup> batch_conics: array<mat2x2<f32>, BATCH_SIZE>;
// [T_X * T_Y]
var<workgroup> batch_opacities: array<f32, BATCH_SIZE>;
// [T_X * T_Y, 2]
var<workgroup> batch_positions_2d_in_screen: array<vec2<f32>, BATCH_SIZE>;
// (0 ~ T_X * T_Y)
var<workgroup> pixel_done_count: atomic<u32>;

// T_X
const TILE_SIZE_X: u32 = 16;
// T_Y
const TILE_SIZE_Y: u32 = 16;
// T_X * T_Y
const BATCH_SIZE: u32 = TILE_SIZE_X * TILE_SIZE_Y;
const OPACITY_MAX: f32 = 0.99;
const OPACITY_MIN: f32 = 1.0 / 255.0;
const TRANSMITTANCE_MIN: f32 = 1e-4;

@compute @workgroup_size(TILE_SIZE_X, TILE_SIZE_Y)
fn main(
    // (0 ~ T_X * T_Y)
    @builtin(local_invocation_index) local_index: u32,
    // (0 ~ I_X, 0 ~ I_Y)
    @builtin(global_invocation_id) global_id: vec3<u32>,
    // (0 ~ I_X / T_X, 0 ~ I_Y / T_Y)
    @builtin(workgroup_id) group_id: vec3<u32>,
    // (I_X / T_X, I_Y / T_Y)
    @builtin(num_workgroups) group_count: vec3<u32>,
) {
    // Specifying the parameters

    let pixel = global_id.xy;
    let pixel_f32 = vec2<f32>(pixel);
    let is_pixel_valid = pixel.x < arguments.image_size_x && pixel.y < arguments.image_size_y;
    let tile_point_range = tile_point_ranges[group_id.y * group_count.x + group_id.x];
    let batch_count = (tile_point_range.y - tile_point_range.x + BATCH_SIZE - 1) / BATCH_SIZE;

    var is_pixel_done = !is_pixel_valid;
    var was_pixel_done = false;
    var pixel_color_rgb_2d = vec3<f32>();
    var pixel_transmittance = 1.0;
    var point_rendered_count = 0u;
    var point_rendered_state = 0u;
    // R
    var point_count = tile_point_range.y - tile_point_range.x;

    // Processing batches of points in the tile
    // [R / (T_X * T_Y)]

    for (var batch_index = 0u; batch_index < batch_count; batch_index++) {
        // Specifying the progress state

        if is_pixel_done && !was_pixel_done {
            was_pixel_done = true;
            atomicAdd(&pixel_done_count, 1u);

            // Leaving if all the pixels of tile are done

            if atomicLoad(&pixel_done_count) == BATCH_SIZE {
                break;
            }
        }

        // Specifying the batch parameters

        let index = tile_point_range.x + batch_index * BATCH_SIZE + local_index;
        if index < tile_point_range.y {
            let point_index = point_indexes[index];
            batch_colors_rgb_3d[local_index] = colors_rgb_3d[point_index];
            batch_conics[local_index] = conics[point_index];
            batch_opacities[local_index] = opacities[point_index];
            batch_positions_2d_in_screen[local_index] = positions_2d_in_screen[point_index];
        }
        workgroupBarrier();

        // Skipping if the pixel is done

        if is_pixel_done {
            continue;
        }

        // Computing the 2D colors in RGB space using the batch parameters
        // [T_X * T_Y]

        let batch_point_count = min(point_count, BATCH_SIZE);

        for (
            var batch_point_index = 0u;
            batch_point_index < batch_point_count;
            batch_point_index++
        ) {
            point_rendered_state++;

            // Computing the density of the point
            // a[I_Y, I_X, 1, 1] =
            // d[I_Y, I_X, 1, 2] * c'^-1[I_Y, I_X, 2, 2] * d[I_Y, I_X, 2, 1]

            let position_2d_in_screen = batch_positions_2d_in_screen[batch_point_index];
            let direction_2d = position_2d_in_screen - pixel_f32;
            let conic = batch_conics[batch_point_index];
            let density = exp(-0.5 * dot(direction_2d * conic, direction_2d));

            // Skipping if the Gaussian density is greater than one

            if density > 1.0 {
                continue;
            }

            // Computing the opacity of the point

            let opacity = min(
                batch_opacities[batch_point_index] * density,
                OPACITY_MAX,
            );

            // Skipping if the opacity is too low

            if opacity < OPACITY_MIN {
                continue;
            }

            // Computing the next transmittance

            let pixel_transmittance_next = pixel_transmittance * (1.0 - opacity);

            // Leaving before the transmittance is too low

            if pixel_transmittance_next < TRANSMITTANCE_MIN {
                is_pixel_done = true;
                break;
            }

            // Blending the 3D colors of the pixel into the 2D color in RGB space

            let batch_color_rgb_3d = batch_colors_rgb_3d[batch_point_index];
            pixel_color_rgb_2d += batch_color_rgb_3d * (opacity * pixel_transmittance);

            // Updating the pixel state

            pixel_transmittance = pixel_transmittance_next;
            point_rendered_count = point_rendered_state;
        }

        point_count -= batch_point_count;
    }

    // Specifying the results

    if is_pixel_valid {
        let pixel_index = pixel.y * arguments.image_size_x + pixel.x;

        // Paint the pixel

        // [I_Y, I_X, 3]
        colors_rgb_2d[pixel_index] = array<f32, 3>(
            pixel_color_rgb_2d[0],
            pixel_color_rgb_2d[1],
            pixel_color_rgb_2d[2],
        );
        // [I_Y, I_X]
        point_rendered_counts[pixel_index] = point_rendered_count;
        // [I_Y, I_X]
        transmittances[pixel_index] = pixel_transmittance;
    }
}