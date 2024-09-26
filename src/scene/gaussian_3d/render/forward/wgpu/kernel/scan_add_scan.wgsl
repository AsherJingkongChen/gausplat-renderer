// [N]
@group(0) @binding(0) var<storage, read_write>
sums: array<u32>;
// [N']
@group(0) @binding(1) var<storage, read_write>
sums_next: array<u32>;

// [N / N'] <- [G']
var<workgroup>
sums_subgroup: array<u32, GROUP_SIZE>;

// N / N'
const GROUP_SIZE: u32 = 256u;

@compute @workgroup_size(GROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    // (0 ~ G')
    @builtin(subgroup_invocation_id) lane_index: u32,
    // (0 ~ N / N')
    @builtin(local_invocation_index) local_index: u32,
    // G'
    @builtin(subgroup_size) subgroup_size: u32,
) {
    // Specifying the index

    // (0 ~ N)
    let global_index = global_id.x;
    // (0 ~ N')
    let group_index = group_id.x;
    // (0 ~ N / N' / G')
    let subgroup_index = local_index / subgroup_size;

    // Specifying the parameters

    let is_invocation_valid = global_index < arrayLength(&sums);
    var sum = 0u;
    if is_invocation_valid {
        sum = sums[global_index];
    }

    // Scanning the sums in the group exclusively

    let sum_subgroup = subgroupAdd(sum);
    let sum_exclusive_in_subgroup = subgroupExclusiveAdd(sum);

    if lane_index == 0u {
        sums_subgroup[subgroup_index] = sum_subgroup;
    }
    workgroupBarrier();

    let sum_exclusive_subgroup = subgroupBroadcast(
        subgroupExclusiveAdd(sums_subgroup[lane_index]),
        subgroup_index,
    );

    let sum_exclusive_in_group = sum_exclusive_subgroup + sum_exclusive_in_subgroup;

    if is_invocation_valid {
        sums[global_index] = sum_exclusive_in_group;
    }
    if local_index + 1u == GROUP_SIZE {
        sums_next[group_index] = sum_exclusive_in_group + sum;
    }
}
