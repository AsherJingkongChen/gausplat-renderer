// [N]
@group(0) @binding(0)
var<storage, read_write> sums: array<u32>;
// [N']
@group(0) @binding(1)
var<storage, read_write> sums_next: array<u32>;

// [N / N']
var<workgroup> sums_in_group: array<u32, GROUP_SIZE>;

// N / N'
const GROUP_SIZE: u32 = GROUP_SIZE_X * GROUP_SIZE_Y;
const GROUP_SIZE_X: u32 = 16u;
const GROUP_SIZE_Y: u32 = 16u;

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

    // (1 ~ N / N')
    var stride = 1u;

    if index < arrayLength(&sums) {
        sums_in_group[local_index] = sums[index];
    } else {
        sums_in_group[local_index] = 0u;
    }
    workgroupBarrier();

    // Reducing (Up-sweep)

    while stride < GROUP_SIZE {
        let stride_up = stride << 1u;
        let local_index_strided = (local_index + 1u) * stride_up - 1u;
        if local_index_strided < GROUP_SIZE {
            let local_index_up = local_index_strided - stride;
            sums_in_group[local_index_strided] += sums_in_group[local_index_up];
        }
        stride = stride_up;
        workgroupBarrier();
    }

    // Specifying the results: `sums_next`

    if local_index + 1u == GROUP_SIZE {
        sums_next[group_index] = sums_in_group[local_index];
        sums_in_group[local_index] = 0u;
    }
    workgroupBarrier();

    // Scanning (Down-sweep)

    while stride > 1u {
        let stride_down = stride >> 1u;
        let local_index_strided = (local_index + 1u) * stride - 1u;
        if (local_index_strided < GROUP_SIZE) {
            let local_index_down = local_index_strided - stride_down;
            let sum_in_group_local_down = sums_in_group[local_index_down];
            sums_in_group[local_index_down] = sums_in_group[local_index_strided];
            sums_in_group[local_index_strided] += sum_in_group_local_down;
        }
        stride = stride_down;
        workgroupBarrier();
    }

    // Specifying the results: `sums`

    if index < arrayLength(&sums) {
        sums[index] = sums_in_group[local_index];
    }
}
