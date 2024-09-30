struct Arguments {
    // N / N'
    block_count_group: u32,
    // (0 ~ 32: +log2(R))
    radix_shift: u32,
}

@group(0) @binding(0) var<storage, read>
arguments: Arguments;
// [N] = [N' / G, N / N', G]
@group(0) @binding(1) var<storage, read_write>
keys_input: array<u32>;

// [N' / G, R]
@group(0) @binding(2) var<storage, read_write>
counts_radix_group: array<u32>;

// [R]
var<workgroup>
counts_radix_in_group: array<atomic<u32>, RADIX_COUNT>;

// R
const RADIX_COUNT: u32 = 1u << RADIX_COUNT_SHIFT;
// log2(R)
const RADIX_COUNT_SHIFT: u32 = 8u;
// R - 1
const RADIX_MASK: u32 = RADIX_COUNT - 1u;
// G = R
const GROUP_SIZE: u32 = RADIX_COUNT;

@compute @workgroup_size(GROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    // (0 ~ G)
    @builtin(local_invocation_index) local_index: u32,
) {
    // Specifying the index

    // (0 ~ N' / G)
    let group_index = group_id.x;

    // Specifying the parameters

    // N
    let global_count = arrayLength(&keys_input);
    counts_radix_in_group[local_index] = 0u;
    workgroupBarrier();

    // Counting radix in the block of the group

    // (0 ~ N / N')
    for (var block_index = 0u; block_index < arguments.block_count_group; block_index++) {
        // (0 ~ N' / G, 0 ~ N / N', 0 ~ G)
        let input_index =
            (group_index * arguments.block_count_group + block_index) * GROUP_SIZE + local_index;
        if input_index < global_count {
            // (0 ~ R)
            let radix = keys_input[input_index] >> arguments.radix_shift & RADIX_MASK;
            atomicAdd(&counts_radix_in_group[radix], 1u);
        }
    }
    workgroupBarrier();

    // Specifying the result of radix count in the group

    // [N' / G, R] <- [R]
    counts_radix_group[group_index * RADIX_COUNT + local_index] =
        counts_radix_in_group[local_index];
}
