struct Arguments {
    // (0 ~ , + 2^R)
    radix_bit_offset: u32,
}

@group(0) @binding(0) var<storage, read>
arguments: Arguments;
// [N]
@group(0) @binding(1) var<storage, read_write>
keys_input: array<u32>;
// [2^R]
@group(0) @binding(2) var<storage, read_write>
radix_counts: array<atomic<u32>, RADIX>;

// 2^R
const RADIX: u32 = 1u << RADIX_BIT_COUNT;
// R
const RADIX_BIT_COUNT: u32 = 8u;
// 2^R - 1
const RADIX_BIT_MASK: u32 = RADIX - 1u;

@compute @workgroup_size(RADIX, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    // Specifying the index

    let index = global_id.x;

    // Counting the radix

    if index < arrayLength(&keys_input) {
        let key = keys_input[index];
        let radix = key >> arguments.radix_bit_offset & RADIX_BIT_MASK;

        atomicAdd(&radix_counts[radix], 1u);
    }
}
