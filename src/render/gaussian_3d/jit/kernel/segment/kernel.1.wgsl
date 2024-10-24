// T
@group(0) @binding(0) var<storage, read_write>
tile_point_count: u32;

// (T / G^2, G, 1)
@group(0) @binding(1) var<storage, read_write>
group_count: vec3<u32>;

// G
const GROUP_SIZE: u32 = 256;
// G ^ 2
const GROUP_SIZE2: u32 = GROUP_SIZE * GROUP_SIZE;

@compute @workgroup_size(1, 1, 1)
fn main() {
    // Specifying the results

    group_count = vec3<u32>((tile_point_count + GROUP_SIZE2 - 1) / GROUP_SIZE2, GROUP_SIZE, 1);
}
