// [T] (0 ~ (I_y / T_y) * (I_x / T_x) in high order bits)
@group(0) @binding(0)
var<storage, read_write> point_orders: array<u32>;

// [I_y / T_y, I_x / T_x, 2]
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

    // (0 ~ T)
    let global_index = (group_id.y * group_count.x + group_id.x) * GROUP_SIZE + local_index;
    let global_count = arrayLength(&point_orders);
    if global_index >= global_count || global_index == 0 {
        return;
    }

    // Specifying the tile indices

    let tile_index_current = point_orders[global_index] >> 16;
    let tile_index_previous = point_orders[global_index - 1] >> 16;

    // Computing the ranges of each point tile

    if tile_index_current != tile_index_previous {
        tile_point_ranges[tile_index_current * 2 + 0] = global_index;
        tile_point_ranges[tile_index_previous * 2 + 1] = global_index;
    }

    // Specifying the range of the last point tile

    if global_index + 1 == global_count {
        tile_point_ranges[tile_index_current * 2 + 1] = global_count;
        return;
    }
}
