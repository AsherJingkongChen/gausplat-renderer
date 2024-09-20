// [T] (0 ~ (I_y / T_y) * (I_x / T_x))
@group(0) @binding(0)
var<storage, read_write> point_tile_indexes: array<u32>;
// [(I_y / T_y) * (I_x / T_x), 2]
@group(0) @binding(1)
var<storage, read_write> tile_point_ranges: array<u32>;

const GROUP_SIZE: u32 = GROUP_SIZE_X * GROUP_SIZE_Y;
const GROUP_SIZE_X: u32 = 16;
const GROUP_SIZE_Y: u32 = 16;

@compute @workgroup_size(GROUP_SIZE, 1, 1)
fn main(
    @builtin(num_workgroups) group_count: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    // Specifying the index

    // (1 ~ T)
    let index = (group_id.y * group_count.x + group_id.x) * GROUP_SIZE + local_index;
    if index >= arrayLength(&point_tile_indexes) || index == 0 {
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
