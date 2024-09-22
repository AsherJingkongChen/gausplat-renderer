struct Arguments {
    // (0 ~ 32, + R)
    radix_bit_offset: u32,
}

@group(0) @binding(0) var<storage, read>
arguments: Arguments;
// [N]
@group(0) @binding(1) var<storage, read_write>
keys_input: array<u32>;
// [N]
@group(0) @binding(2) var<storage, read_write>
values_input: array<u32>;
// [R, N / G]
@group(0) @binding(3) var<storage, read_write>
offsets_group: array<u32>;
// [N]
@group(0) @binding(4) var<storage, read_write>
offsets_local: array<u32>;
// [N]
@group(0) @binding(5) var<storage, read_write>
keys_output: array<u32>;
// [N]
@group(0) @binding(6) var<storage, read_write>
values_output: array<u32>;

// R
const RADIX: u32 = 1u << RADIX_BIT_COUNT;
// log2(R)
const RADIX_BIT_COUNT: u32 = 2u;
// R - 1
const RADIX_BIT_MASK: u32 = RADIX - 1u;
// G
const GROUP_SIZE: u32 = 256u;

@compute @workgroup_size(GROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) group_counts: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    // (0 ~ G)
    @builtin(local_invocation_index) local_index: u32,
) {
    // Specifying the index

    // (0 ~ N)
    let global_index = global_id.x;
    // N / R
    let group_count = group_counts.x;
    // (0 ~ N / R)
    let group_index = group_id.x;

    // Scattering the keys

    if global_index < arrayLength(&keys_input) {
        let key = keys_input[global_index];
        let value = values_input[global_index];
        let radix = key >> arguments.radix_bit_offset & RADIX_BIT_MASK;
        let offset_group = offsets_group[radix * group_count + group_index];
        let offset_local = offsets_local[global_index];
        let offset = offset_group + offset_local;

        keys_output[offset] = key;
        values_output[offset] = value;
    }
}
