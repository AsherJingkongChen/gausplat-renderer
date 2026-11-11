// (N / B / C, 1, 1)
@group(0) @binding(0) var<storage, read_write>
chunk_count: vec3<u32>;

// [N / B / C, R]
@group(0) @binding(1) var<storage, read_write>
counts_radix_chunk: array<u32>;

// [R]
var<workgroup>
counts_radix_chunks: array<u32, RADIX_COUNT>;

// R
const RADIX_COUNT: u32 = 1u << RADIX_COUNT_SHIFT;
// log2(R)
const RADIX_COUNT_SHIFT: u32 = 8;

@compute @workgroup_size(RADIX_COUNT, 1, 1)
fn main(
    // (0 ~ R)
    @builtin(local_invocation_index) radix: u32,
) {
    // Scanning radix counts of all chunks into radix offsets of the chunk

    var count_radix_chunks = 0u;

    // (0 ~ N / B / C)
    for (var chunk_index = 0u; chunk_index < chunk_count.x; chunk_index++) {
        // [N / B / C, R]
        let index = chunk_index * RADIX_COUNT + radix;
        let count_radix_chunk = counts_radix_chunk[index];
        counts_radix_chunk[index] = count_radix_chunks;
        count_radix_chunks += count_radix_chunk;
    }

    // [R]
    counts_radix_chunks[radix] = count_radix_chunks;
    workgroupBarrier();

    // Scanning radix counts of all chunks into radix offsets

    var offset_radix_chunks = 0u;

    // (0 ~ R)
    for (var index = 0u; index < radix; index++) {
        offset_radix_chunks += counts_radix_chunks[index];
    }

    // Adding the radix offsets to the radix offsets of the chunk

    // (0 ~ N / B / C)
    for (var chunk_index = 0u; chunk_index < chunk_count.x; chunk_index++) {
        // [N / B / C, R]
        counts_radix_chunk[chunk_index * RADIX_COUNT + radix] += offset_radix_chunks;
    }
}
