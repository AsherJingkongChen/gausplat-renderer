// N
@group(0) @binding(0) var<storage, read_write>
count: u32;

// (N' / G, 1, 1) <- N / (N / N' * G)
@group(0) @binding(1) var<storage, read_write>
group_count: vec3<u32>;

// log2(N')
const BLOCK_COUNT_GROUP_SHIFT: u32 = 14;
// R
const RADIX_COUNT: u32 = 1u << RADIX_COUNT_SHIFT;
// log2(R)
const RADIX_COUNT_SHIFT: u32 = 8;
// R - 1
const RADIX_MASK: u32 = RADIX_COUNT - 1;
// G <- R
const GROUP_SIZE: u32 = RADIX_COUNT;

@compute @workgroup_size(1, 1, 1)
fn main() {
    // Specifying the parameters

    // N / N'
    let block_count_group = max(count >> BLOCK_COUNT_GROUP_SHIFT, 1u);
    let block_size = block_count_group * GROUP_SIZE;

    // Specifying the results

    group_count = vec3<u32>((count + block_size - 1u) / block_size, 1, 1);
}
