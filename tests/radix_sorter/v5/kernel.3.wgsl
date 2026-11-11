// (N / B, 1, 1)
@group(0) @binding(0) var<storage, read_write>
group_count: vec3<u32>;

// [N / B, R]
@group(0) @binding(1) var<storage, read_write>
counts_radix_group: array<u32>;
// [N / B / C, R]
@group(0) @binding(2) var<storage, read_write>
counts_radix_chunk: array<u32>;

// C
const CHUNK_SIZE: u32 = 64;
// R
const RADIX_COUNT: u32 = 1u << RADIX_COUNT_SHIFT;
// log2(R)
const RADIX_COUNT_SHIFT: u32 = 8;

@compute @workgroup_size(RADIX_COUNT, 1, 1)
fn main(
    @builtin(workgroup_id) chunk_id: vec3<u32>,
    // (0 ~ R)
    @builtin(local_invocation_index) radix: u32,
) {
    // Specifying the index

    // (0 ~ N / B / C)
    let chunk_index = chunk_id.x;
    let group_index_begin = chunk_index * CHUNK_SIZE;
    let group_index_end = min(group_index_begin + CHUNK_SIZE, group_count.x);

    // Scanning radix counts of the groups in the chunk into radix offsets in the chunk

    var count_radix_chunk = 0u;

    // (0 ~ C)
    for (
        var group_index = group_index_begin;
        group_index < group_index_end;
        group_index++
    ) {
        // [N / B, R]
        let index = group_index * RADIX_COUNT + radix;
        let count_radix_group = counts_radix_group[index];
        counts_radix_group[index] = count_radix_chunk;
        count_radix_chunk += count_radix_group;
    }

    // Specifying the result of radix count in the chunk

    // [N / B / C, R]
    counts_radix_chunk[chunk_index * RADIX_COUNT + radix] = count_radix_chunk;
}
