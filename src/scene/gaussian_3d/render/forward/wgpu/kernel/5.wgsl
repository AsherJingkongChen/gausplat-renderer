struct Arguments {
    // T
    tile_touched_count: u32,
}

@group(0) @binding(0)
var<storage, read> arguments: Arguments;
// [T] (0 ~ (I_X / T_X) * (I_Y / T_Y))
@group(0) @binding(1)
var<storage, read> point_tile_indexes: array<u32>;
// [(I_X / T_X) * (I_Y / T_Y), 2]
@group(0) @binding(2)
var<storage, read_write> tile_point_ranges: array<u32>;

const GROUP_SIZE_X: u32 = 16;
const GROUP_SIZE_Y: u32 = 16;

@compute @workgroup_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) group_count: vec3<u32>,
) {
    // Checking the index

    // (1 ~ T)
    let index = global_id.y * group_count.x * GROUP_SIZE_X + global_id.x;
    if index >= arguments.tile_touched_count || index == 0 {
        return;
    }

    // Computing the ranges of distinct point tile indexes

    let tile_index_current = point_tile_indexes[index];
    let tile_index_previous = point_tile_indexes[index - 1];
    if tile_index_current != tile_index_previous {
        tile_point_ranges[tile_index_current * 2 + 0] = index;
        tile_point_ranges[tile_index_previous * 2 + 1] = index;
    }
}
