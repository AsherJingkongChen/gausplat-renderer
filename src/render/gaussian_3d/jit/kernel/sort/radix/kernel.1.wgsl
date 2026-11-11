// N
@group(0) @binding(0) var<storage, read_write>
count: u32;

// (N / B, 1, 1)
@group(0) @binding(1) var<storage, read_write>
group_count: vec3<u32>;
// (N / B / C, 1, 1)
@group(0) @binding(2) var<storage, read_write>
chunk_count: vec3<u32>;

// B <- G * W
const BLOCK_SIZE: u32 = GROUP_SIZE * ROW_COUNT;
// C
const CHUNK_SIZE: u32 = 64;
// G
const GROUP_SIZE: u32 = 256;
// W
const ROW_COUNT: u32 = 15;

@compute @workgroup_size(1, 1, 1)
fn main() {
    // Specifying the results

    let group_count_x = (count + BLOCK_SIZE - 1u) / BLOCK_SIZE;
    group_count = vec3<u32>(group_count_x, 1, 1);
    chunk_count = vec3<u32>((group_count_x + CHUNK_SIZE - 1u) / CHUNK_SIZE, 1, 1);
}
