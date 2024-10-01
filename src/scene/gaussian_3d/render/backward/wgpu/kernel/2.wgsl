struct Arguments {
    image_size_x: u32,
    image_size_y: u32,
}

const GROUP_SIZE_X: u32 = 16u;
const GROUP_SIZE_Y: u32 = 16u;
const BATCH_SIZE: u32 = 256u;
const OPACITY_2D_MAX: f32 = 0.98039216f;
const OPACITY_2D_MIN: f32 = 0.0019607844f;

@group(0) @binding(0) 
var<storage, read_write> arguments: Arguments;
@group(0) @binding(1) 
var<storage, read_write> conics: array<mat2x2<f32>>;
@group(0) @binding(2) 
var<storage, read_write> colors_rgb_2d_grad: array<array<f32, 3>>;
@group(0) @binding(3) 
var<storage, read_write> colors_rgb_3d: array<vec3<f32>>;
@group(0) @binding(4) 
var<storage, read_write> opacities_3d: array<f32>;
@group(0) @binding(5) 
var<storage, read_write> point_indices: array<u32>;
@group(0) @binding(6) 
var<storage, read_write> point_rendered_counts: array<u32>;
@group(0) @binding(7) 
var<storage, read_write> positions_2d: array<vec2<f32>>;
@group(0) @binding(8) 
var<storage, read_write> tile_point_ranges: array<vec2<u32>>;
@group(0) @binding(9) 
var<storage, read_write> transmittances: array<f32>;
@group(0) @binding(10) 
var<storage, read_write> colors_rgb_3d_grad: array<atomic<f32>>;
@group(0) @binding(11) 
var<storage, read_write> conics_grad: array<atomic<f32>>;
@group(0) @binding(12) 
var<storage, read_write> opacities_3d_grad: array<atomic<f32>>;
@group(0) @binding(13) 
var<storage, read_write> positions_2d_grad: array<atomic<f32>>;
var<workgroup> colors_rgb_3d_in_batch: array<vec3<f32>, 256>;
var<workgroup> conics_in_batch: array<mat2x2<f32>, 256>;
var<workgroup> opacities_3d_in_batch: array<f32, 256>;
var<workgroup> point_indices_in_batch: array<u32, 256>;
var<workgroup> positions_2d_in_batch: array<vec2<f32>, 256>;

@compute @workgroup_size(16, 16, 1) 
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_index) local_index: u32, @builtin(workgroup_id) tile_id: vec3<u32>, @builtin(num_workgroups) tile_count: vec3<u32>) {
    var point_range: vec2<u32> = vec2<u32>();
    var tile_point_count: u32 = 0u;
    var color_rgb_2d_state: vec3<f32> = vec3<f32>();
    var color_rgb_3d_state: vec3<f32> = vec3<f32>();
    var opacity_2d_state: f32 = 0f;
    var point_rendered_count: u32 = u32();
    var point_rendered_state: u32;
    var transmittance_state: f32 = f32();
    var batch_index: u32 = 0u;
    var batch_index_1: u32;

    let pixel = global_id.xy;
    let _e8 = arguments.image_size_x;
    let pixel_index = ((pixel.y * _e8) + pixel.x);
    let tile_index = ((tile_id.y * tile_count.x) + tile_id.x);
    let _e20 = arguments.image_size_x;
    let _e25 = arguments.image_size_y;
    let is_pixel_valid = ((pixel.x < _e20) && (pixel.y < _e25));
    let position_pixel = vec2<f32>(pixel);
    if (tile_index < arrayLength((&tile_point_ranges))) {
        let _e38 = tile_point_ranges[tile_index];
        point_range = _e38;
        let _e40 = point_range.y;
        let _e42 = point_range.x;
        if (_e40 >= _e42) {
            let _e45 = point_range.y;
            let _e47 = point_range.x;
            tile_point_count = (_e45 - _e47);
        }
    }
    let _e49 = tile_point_count;
    let batch_count = (((_e49 + BATCH_SIZE) - 1u) / BATCH_SIZE);
    let _e59 = colors_rgb_2d_grad[pixel_index][0];
    let _e63 = colors_rgb_2d_grad[pixel_index][1];
    let _e67 = colors_rgb_2d_grad[pixel_index][2];
    let color_rgb_2d_grad = vec3<f32>(_e59, _e63, _e67);
    let _e77 = tile_point_count;
    point_rendered_state = _e77;
    if is_pixel_valid {
        let _e83 = point_rendered_counts[pixel_index];
        point_rendered_count = _e83;
        let _e86 = transmittances[pixel_index];
        transmittance_state = _e86;
    }
    loop {
        let _e89 = batch_index;
        if (_e89 < batch_count) {
        } else {
            break;
        }
        {
            workgroupBarrier();
            let _e92 = point_range.y;
            let _e93 = batch_index;
            let index = (((_e92 - (_e93 * BATCH_SIZE)) - local_index) - 1u);
            let _e101 = point_range.x;
            if (index >= _e101) {
                let point_index = point_indices[index];
                let _e110 = colors_rgb_3d[point_index];
                colors_rgb_3d_in_batch[local_index] = _e110;
                let _e115 = conics[point_index];
                conics_in_batch[local_index] = _e115;
                let _e120 = opacities_3d[point_index];
                opacities_3d_in_batch[local_index] = _e120;
                point_indices_in_batch[local_index] = point_index;
                let _e127 = positions_2d[point_index];
                positions_2d_in_batch[local_index] = _e127;
            }
            workgroupBarrier();
            if !(is_pixel_valid) {
                continue;
            }
            let _e129 = tile_point_count;
            let batch_point_count = min(_e129, BATCH_SIZE);
            batch_index_1 = 0u;
            loop {
                let _e134 = batch_index_1;
                if (_e134 < batch_point_count) {
                } else {
                    break;
                }
                {
                    let _e136 = point_rendered_state;
                    let _e137 = point_rendered_count;
                    if (_e136 > _e137) {
                        let _e140 = point_rendered_state;
                        point_rendered_state = (_e140 - 1u);
                        continue;
                    }
                    let _e143 = batch_index_1;
                    let conic = conics_in_batch[_e143];
                    let _e147 = batch_index_1;
                    let position_2d = positions_2d_in_batch[_e147];
                    let position_offset = (position_2d - position_pixel);
                    let density = exp((-0.5f * dot((position_offset * conic), position_offset)));
                    if (density > 1f) {
                        continue;
                    }
                    let _e159 = batch_index_1;
                    let opacity_3d = opacities_3d_in_batch[_e159];
                    let opacity_2d = min((opacity_3d * density), OPACITY_2D_MAX);
                    if (opacity_2d < OPACITY_2D_MIN) {
                        continue;
                    }
                    let _e167 = color_rgb_3d_state;
                    let _e168 = opacity_2d_state;
                    let _e170 = color_rgb_2d_state;
                    let _e171 = opacity_2d_state;
                    color_rgb_2d_state = ((_e167 * _e168) + (_e170 * (1f - _e171)));
                    let _e177 = batch_index_1;
                    let _e179 = colors_rgb_3d_in_batch[_e177];
                    color_rgb_3d_state = _e179;
                    opacity_2d_state = opacity_2d;
                    let _e182 = transmittance_state;
                    transmittance_state = (_e182 / (1f - opacity_2d));
                    let _e184 = transmittance_state;
                    let color_rgb_3d_grad = ((opacity_2d * _e184) * color_rgb_2d_grad);
                    let _e187 = color_rgb_3d_state;
                    let _e188 = color_rgb_2d_state;
                    let opacity_2d_grad_terms = ((_e187 - _e188) * color_rgb_2d_grad);
                    let _e191 = transmittance_state;
                    let opacity_2d_grad = (_e191 * ((opacity_2d_grad_terms.x + opacity_2d_grad_terms.y) + opacity_2d_grad_terms.z));
                    let opacity_3d_grad = (density * opacity_2d_grad);
                    let density_grad = (opacity_3d * opacity_2d_grad);
                    let density_density_grad_n = (-(density) * density_grad);
                    let conic_grad = ((0.5f * density_density_grad_n) * mat2x2<f32>((position_offset * position_offset.x), (position_offset * position_offset.y)));
                    let position_2d_grad = ((density_density_grad_n * conic) * position_offset);
                    let _e213 = batch_index_1;
                    let point_index_1 = point_indices_in_batch[_e213];
                    let _e223 = atomicAdd((&colors_rgb_3d_grad[((4u * point_index_1) + 0u)]), color_rgb_3d_grad.x);
                    let _e231 = atomicAdd((&colors_rgb_3d_grad[((4u * point_index_1) + 1u)]), color_rgb_3d_grad.y);
                    let _e239 = atomicAdd((&colors_rgb_3d_grad[((4u * point_index_1) + 2u)]), color_rgb_3d_grad.z);
                    let _e248 = atomicAdd((&conics_grad[((4u * point_index_1) + 0u)]), conic_grad[0].x);
                    let _e257 = atomicAdd((&conics_grad[((4u * point_index_1) + 1u)]), conic_grad[0].y);
                    let _e266 = atomicAdd((&conics_grad[((4u * point_index_1) + 2u)]), conic_grad[1].x);
                    let _e275 = atomicAdd((&conics_grad[((4u * point_index_1) + 3u)]), conic_grad[1].y);
                    let _e278 = atomicAdd((&opacities_3d_grad[point_index_1]), opacity_3d_grad);
                    let _e286 = atomicAdd((&positions_2d_grad[((2u * point_index_1) + 0u)]), position_2d_grad.x);
                    let _e294 = atomicAdd((&positions_2d_grad[((2u * point_index_1) + 1u)]), position_2d_grad.y);
                }
                continuing {
                    let _e296 = batch_index_1;
                    batch_index_1 = (_e296 + 1u);
                }
            }
            let _e298 = tile_point_count;
            tile_point_count = (_e298 - batch_point_count);
        }
        continuing {
            let _e301 = batch_index;
            batch_index = (_e301 + 1u);
        }
    }
    return;
}
