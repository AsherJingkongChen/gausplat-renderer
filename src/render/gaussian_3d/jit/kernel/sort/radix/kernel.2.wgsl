struct Arguments {
    // (0 ~ 32: +log2(R))
    radix_shift: u32,
}

@group(0) @binding(0) var<storage, read_write>
arguments: Arguments;
// N
@group(0) @binding(1) var<storage, read_write>
count: u32;
// [N] = [N / B, W, G]
@group(0) @binding(2) var<storage, read_write>
keys_input: array<u32>;

// [N / B, R]
@group(0) @binding(3) var<storage, read_write>
counts_radix_group: array<u32>;

// [R]
var<workgroup>
counts_radix_in_group: array<atomic<u32>, RADIX_COUNT>;

// B <- G * W
const BLOCK_SIZE: u32 = GROUP_SIZE * ROW_COUNT;
// G
const GROUP_SIZE: u32 = 256;
// R
const RADIX_COUNT: u32 = 1u << RADIX_COUNT_SHIFT;
// log2(R)
const RADIX_COUNT_SHIFT: u32 = 8;
// R - 1
const RADIX_MASK: u32 = RADIX_COUNT - 1;
// W
const ROW_COUNT: u32 = 15;

@compute @workgroup_size(GROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    // (0 ~ G)
    @builtin(local_invocation_index) local_index: u32,
) {
    // Specifying the index

    // (0 ~ N / B)
    let group_index = group_id.x;

    // Specifying the parameters

    let block_offset = group_index * BLOCK_SIZE;
    atomicStore(&counts_radix_in_group[local_index], 0u);
    workgroupBarrier();

    // Counting radix in the block of the group

    // (0 ~ W)
    for (var row_index = 0u; row_index < ROW_COUNT; row_index++) {
        // (0 ~ N / B, 0 ~ W, 0 ~ G)
        let input_index = block_offset + row_index * GROUP_SIZE + local_index;
        if input_index < count {
            // (0 ~ R)
            let radix = keys_input[input_index] >> arguments.radix_shift & RADIX_MASK;
            atomicAdd(&counts_radix_in_group[radix], 1u);
        }
    }
    workgroupBarrier();

    // Specifying the result of radix count in the group

    // [N / B, R] <- [R]
    counts_radix_group[group_index * RADIX_COUNT + local_index] =
        atomicLoad(&counts_radix_in_group[local_index]);
}
