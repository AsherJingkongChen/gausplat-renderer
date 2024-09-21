// [N]
@group(0) @binding(0)
var<storage, read_write> sums: array<u32>;
// [N']
@group(0) @binding(1)
var<storage, read_write> sums_next: array<u32>;

// [N / N']
var<workgroup> sums_in_group: array<u32, GROUP_SIZE>;

// N / N'
const GROUP_SIZE: u32 = 256u;

@compute @workgroup_size(GROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    // Specifying the index

    // (0 ~ N')
    let group_index = group_id.x;
    // (0 ~ N)
    let index = global_id.x;

    // Specifying the parameters

    let is_invocation_valid = index < arrayLength(&sums);
    var sum = 0u;

    if is_invocation_valid {
        sums_in_group[local_index] = sums[index];
    }
    workgroupBarrier();

    // Scanning the sums in the group in an inclusive manner

    for (var offset = 1u; offset < GROUP_SIZE; offset <<= 1u) {
        sum = sums_in_group[local_index];
        if (local_index >= offset) {
            sum += sums_in_group[local_index - offset];
        }
        workgroupBarrier();

        sums_in_group[local_index] = sum;
        workgroupBarrier();
    }

    // Specifying the result of sums for the next pass

    if local_index + 1u == GROUP_SIZE {
        sums_next[group_index] = sum;
    }

    // Specifying the result of sums in an exclusive manner

    if is_invocation_valid {
        if local_index == 0u {
            sums[index] = 0u;
        } else {
            sums[index] = sums_in_group[local_index - 1u];
        }
    }
}
