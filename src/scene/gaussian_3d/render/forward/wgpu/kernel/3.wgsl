struct Arguments {
    // P
    point_count: u32,
    // I_X / T_X
    tile_count_x: u32,
}
struct PointInfo {
    tile_index: u32,
    depth: f32,
    point_index: u32,
}

@group(0) @binding(0)
var<storage, read_write> arguments: Arguments;
// [P] (0.2 ~ )
@group(0) @binding(1)
var<storage, read_write> depths: array<f32>;
// [P]
@group(0) @binding(2)
var<storage, read_write> radii: array<u32>;
// [P]
@group(0) @binding(3)
var<storage, read_write> tile_touched_offsets: array<u32>;
// [P, 2]
@group(0) @binding(4)
var<storage, read_write> tiles_touched_max: array<vec2<u32>>;
// [P, 2]
@group(0) @binding(5)
var<storage, read_write> tiles_touched_min: array<vec2<u32>>;

// [T, 3]
@group(0) @binding(6)
var<storage, read_write> point_infos: array<PointInfo>;

const GROUP_SIZE_X: u32 = 16;
const GROUP_SIZE_Y: u32 = 16;

@compute @workgroup_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) group_count: vec3<u32>,
) {
    // Checking the index

    // (0 ~ P)
    let index = global_id.y * group_count.x * GROUP_SIZE_X + global_id.x;
    if index >= arguments.point_count {
        return;
    }

    // Leaving if the point is invisible

    if radii[index] == 0u {
        return;
    }

    // Specifying the parameters

    var offset = tile_touched_offsets[index];
    let tile_touched_max = tiles_touched_max[index];
    let tile_touched_min = tiles_touched_min[index];

    // Computing the keys and indexes of point

    for (var tile_y = tile_touched_min.y; tile_y < tile_touched_max.y; tile_y++) {
        for (var tile_x = tile_touched_min.x; tile_x < tile_touched_max.x; tile_x++) {
            let tile_index = tile_y * arguments.tile_count_x + tile_x;
            let depth = depths[index];
            point_infos[offset] = PointInfo(tile_index, depth, index);
            offset++;
        }
    }
}
