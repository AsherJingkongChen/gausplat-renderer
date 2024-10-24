// [N]
@group(0) @binding(0) var<storage, read_write>
values: array<u32>;
// [N']
@group(0) @binding(1) var<storage, read_write>
values_next: array<u32>;

// [N / N'] <- [G']
var<workgroup>
values_subgroup: array<u32, GROUP_SIZE>;

// N / N'
const GROUP_SIZE: u32 = 256;

@compute @workgroup_size(GROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    // (0 ~ G')
    @builtin(subgroup_invocation_id) lane_index: u32,
    // (0 ~ N / N')
    @builtin(local_invocation_index) local_index: u32,
    // (0 ~ N / N' / G')
    @builtin(subgroup_id) subgroup_index: u32,
) {
    // Specifying the index

    // (0 ~ N)
    let global_index = global_id.x;
    // (0 ~ N')
    let group_index = group_id.x;

    // Specifying the parameters

    let is_invocation_valid = global_index < arrayLength(&values);
    var value_local = 0u;
    if is_invocation_valid {
        value_local = values[global_index];
    }

    // Scanning the values in the group exclusively

    let value_subgroup = subgroupAdd(value_local);
    let value_exclusive_in_subgroup = subgroupExclusiveAdd(value_local);
    if lane_index == 0u {
        values_subgroup[subgroup_index] = value_subgroup;
    }
    workgroupBarrier();

    let value_exclusive_subgroup = subgroupShuffle(
        subgroupExclusiveAdd(values_subgroup[lane_index]),
        subgroup_index,
    );

    // Specifying the results of the scanned values and next values

    let value_exclusive_local = value_exclusive_subgroup + value_exclusive_in_subgroup;
    if is_invocation_valid {
        values[global_index] = value_exclusive_local;
    }
    if local_index + 1u == GROUP_SIZE {
        values_next[group_index] = value_exclusive_local + value_local;
    }
}
