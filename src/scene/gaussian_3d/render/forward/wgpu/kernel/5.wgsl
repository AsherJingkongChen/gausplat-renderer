// [T] (0 ~ (I_y / T_y) * (I_x / T_x) in high order bits)
@group(0) @binding(0)
var<storage, read_write> point_orders: array<u32>;

// [I_y / T_y, I_x / T_x, 2]
@group(0) @binding(1)
var<storage, read_write> tile_point_ranges: array<array<atomic<u32>, 2>>;

const GROUP_SIZE: u32 = 256;

@compute @workgroup_size(GROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) group_count: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    // Specifying the index

    // (0 ~ T)
    // let global_index = (global_id.y * group_count.x * GROUP_SIZE) + global_id.x;
    let global_index = (group_id.y * group_count.x + group_id.x) * GROUP_SIZE + local_index;
    // T
    let global_count = arrayLength(&point_orders);
    if global_index >= global_count {
        return;
    }

    // Specifying the current tile index

    let tile_index_current = point_orders[global_index] >> 16;


    if global_index == 0 {
        // Specifying the range of the first point tile

        tile_point_ranges[tile_index_current][0] = 0u;
    } else {
        // Finding the ranges of each point tile

        let tile_index_previous = point_orders[global_index - 1] >> 16;
        if tile_index_current != tile_index_previous {
            tile_point_ranges[tile_index_previous][1] = global_index;
            tile_point_ranges[tile_index_current][0] = global_index;
        }
    }

    // Specifying the range of the last point tile

    if global_index + 1 == global_count {
        tile_point_ranges[tile_index_current][1] = global_count;
    }
}
