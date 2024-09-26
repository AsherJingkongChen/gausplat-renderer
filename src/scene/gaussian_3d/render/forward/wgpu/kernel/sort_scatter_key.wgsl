struct Arguments {
    // N / N'
    block_count_group: u32,
    // (0 ~ 32: +log2(R))
    radix_shift: u32,
}

@group(0) @binding(0) var<storage, read>
arguments: Arguments;
// [N' / G, R]
@group(0) @binding(1) var<storage, read_write>
counts_radix_group: array<u32>;
// [N] = [N' / G, N / N', G]
@group(0) @binding(2) var<storage, read_write>
keys_input: array<u32>;
// [N]
@group(0) @binding(3) var<storage, read_write>
values_input: array<u32>;

// [N]
@group(0) @binding(4) var<storage, read_write>
keys_out: array<u32>;
// [N]
@group(0) @binding(5) var<storage, read_write>
values_out: array<u32>;

// [R] <- [R / G']
var<workgroup>
counts_radix_groups_subgroup: array<u32, RADIX_COUNT>;
// [R]
var<workgroup>
offsets_radix_group: array<atomic<u32>, RADIX_COUNT>;
// [R, G / 32] (G / 32 (u32) <- G (bit))
var<workgroup>
mask_radix_in_group: array<array<atomic<u32>, GROUP_MASK_SIZE>, RADIX_COUNT>;

// 32 - 1
const DIV_32_MASK: u32 = (1u << 5u) - 1u;
// G / 32
const GROUP_MASK_SIZE: u32 = GROUP_SIZE >> 5u;
// G = R
const GROUP_SIZE: u32 = RADIX_COUNT;
// R
const RADIX_COUNT: u32 = 1u << RADIX_COUNT_SHIFT;
// log2(R)
const RADIX_COUNT_SHIFT: u32 = 8u;
// R - 1
const RADIX_MASK: u32 = RADIX_COUNT - 1u;

@compute @workgroup_size(GROUP_SIZE, 1, 1)
fn main(
    @builtin(num_workgroups) group_counts: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    // (0 ~ G')
    @builtin(subgroup_invocation_id) lane_index: u32,
    // (0 ~ G)
    @builtin(local_invocation_index) local_index: u32,
    // G'
    @builtin(subgroup_size) subgroup_size: u32,
) {
    // Specifying the index

    // N' / G
    let group_count = group_counts.x;
    // (0 ~ N' / G)
    let group_index = group_id.x;
    // (0 ~ G / G')
    let subgroup_index = local_index / subgroup_size;

    // Scanning radix counts in all groups into global radix offsets of the group

    var count_radix_groups = 0u;
    var offset_radix_group = 0u;

    // (0 ~ N' / G)
    for (var g = 0u; g < group_count; g++) {
        // [N' / G, R]
        let count_radix_group = counts_radix_group[RADIX_COUNT * g + local_index];
        if g == group_index {
            offset_radix_group = count_radix_groups;
        }
        count_radix_groups += count_radix_group;
    }

    let count_radix_groups_subgroup = subgroupAdd(count_radix_groups);
    let offset_radix_groups_in_subgroup = subgroupExclusiveAdd(count_radix_groups);

    if lane_index == 0u {
        // [R / G']
        counts_radix_groups_subgroup[subgroup_index] = count_radix_groups_subgroup;
    }
    workgroupBarrier();

    // [R / G']
    let offset_radix_groups_subgroup = subgroupBroadcast(
        subgroupExclusiveAdd(counts_radix_groups_subgroup[lane_index]),
        subgroup_index,
    );

    // [R]
    let offset_radix_groups = offset_radix_groups_subgroup + offset_radix_groups_in_subgroup;
    offsets_radix_group[local_index] = offset_radix_groups + offset_radix_group;

    // Scattering keys in the group blocks

    let mask_radix_index = local_index >> 5u;
    let mask_radix_local = 1u << (local_index & DIV_32_MASK);

    // (0 ~ N / N')
    for (var block_index = 0u; block_index < arguments.block_count_group; block_index++) {
        // (0 ~ N' / G, 0 ~ N / N', 0 ~ G)
        let input_index =
            (group_index * arguments.block_count_group + block_index) * GROUP_SIZE + local_index;
        let is_input_index_valid = input_index < arrayLength(&keys_input);
        var key_input = 0u;
        var offset_radix_group = 0u;
        var radix = 0u;

        // Initializing the mask

        // (0 ~ G / 32)
        for (var index = 0u; index < (GROUP_SIZE >> 5u); index++) {
            mask_radix_in_group[local_index][index] = 0u;
        }
        workgroupBarrier();

        // Fetching the radix and offset

        if is_input_index_valid {
            key_input = keys_input[input_index];
            radix = key_input >> arguments.radix_shift & RADIX_MASK;
            offset_radix_group = offsets_radix_group[radix];
            // [R, G / 32] <- [R, G]
            atomicOr(&mask_radix_in_group[radix][mask_radix_index], mask_radix_local);
        }
        workgroupBarrier();

        // Scanning the radix offset in the group

        if is_input_index_valid {
            var count_radix_in_group = 0u;
            var offset_radix_in_group = 0u;
            // (0 ~ G / 32)
            for (var index = 0u; index < (GROUP_SIZE >> 5u); index += 1u) {
                // [R, G / 32]
                let mask_radix = mask_radix_in_group[radix][index];
                let count_radix = countOneBits(mask_radix);
                let count_radix_local = countOneBits(mask_radix & (mask_radix_local - 1u));

                if index < mask_radix_index {
                    offset_radix_in_group += count_radix;
                }
                if index == mask_radix_index {
                    offset_radix_in_group += count_radix_local;
                }
                count_radix_in_group += count_radix;
            }

            // Assigning the key to the new position
            // [N]

            let position = offset_radix_group + offset_radix_in_group;
            keys_out[position] = key_input;
            values_out[position] = values_input[input_index];

            // Incrementing the radix offset of the group
            // [R]

            if offset_radix_in_group + 1u == count_radix_in_group {
                atomicAdd(&offsets_radix_group[radix], count_radix_in_group);
            }
        }
        workgroupBarrier();
    }
}