struct Arguments {
    // (0 ~ 32, + R)
    radix_bit_offset: u32,
}

@group(0) @binding(0) var<storage, read>
arguments: Arguments;
// [N]
@group(0) @binding(1) var<storage, read_write>
keys_input: array<u32>;
// [R, N / G]
@group(0) @binding(2) var<storage, read_write>
counts_group_radix: array<u32>;
// [N]
@group(0) @binding(3) var<storage, read_write>
offsets_local: array<u32>;

// R
const RADIX: u32 = 1u << RADIX_BIT_COUNT;
// log2(R)
const RADIX_BIT_COUNT: u32 = 2u;
// R - 1
const RADIX_BIT_MASK: u32 = RADIX - 1u;
// G
const GROUP_SIZE: u32 = 256u;

// [G]
var<workgroup>
offsets_in_group: array<u32, GROUP_SIZE>;

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
    // N / G
    let group_count = group_counts.x;
    // (0 ~ N / G)
    let group_index = group_id.x;

    // Specifying the parameters

    let is_invocation_valid = global_index < arrayLength(&keys_input);
    let is_largest_local_index = local_index + 1u == GROUP_SIZE;
    var offset_radix = array<u32, RADIX>();
    var radix = u32();

    if is_invocation_valid {
        let key = keys_input[global_index];
        radix = key >> arguments.radix_bit_offset & RADIX_BIT_MASK;
    }
    workgroupBarrier();

    for (var r = 0u; r < RADIX; r += 1u) {
        let is_current_radix = u32(r == radix && is_invocation_valid);
        var offset = 0u;

        if !is_largest_local_index {
            offsets_in_group[local_index + 1u] = is_current_radix;
        }
        workgroupBarrier();

        // Scanning the sums in the group exclusively

        for (var stride = 1u; stride < GROUP_SIZE; stride <<= 1u) {
            offset = offsets_in_group[local_index];
            if local_index >= stride {
                offset += offsets_in_group[local_index - stride];
            }
            workgroupBarrier();

            offsets_in_group[local_index] = offset;
            workgroupBarrier();
        }

        // Specifying the result of count for the group and radix

        if is_largest_local_index {
            let count = offset + is_current_radix;
            counts_group_radix[r * group_count + group_index] = count;
        }

        // Specifying the result of offset for the radix

        offset_radix[r] = offset;
    }

    // Specifying the result of local offset

    if is_invocation_valid {
        offsets_local[global_index] = offset_radix[radix];
    }
}
