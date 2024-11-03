struct Arguments {
    // (0 ~ 32: +log2(R))
    radix_shift: u32,
}

@group(0) @binding(0) var<storage, read_write>
arguments: Arguments;
// N
@group(0) @binding(1) var<storage, read_write>
count: u32;
// [N' / G, R]
@group(0) @binding(2) var<storage, read_write>
counts_radix_group: array<u32>;
// [N] = [N' / G, N / N', G]
@group(0) @binding(3) var<storage, read_write>
keys_input: array<u32>;
// [N]
@group(0) @binding(4) var<storage, read_write>
values_input: array<u32>;

// [N]
@group(0) @binding(5) var<storage, read_write>
keys_out: array<u32>;
// [N]
@group(0) @binding(6) var<storage, read_write>
values_out: array<u32>;

// [R] <- [R / G']
var<workgroup>
counts_radix_groups_subgroup: array<u32, RADIX_COUNT>;
// [R]
var<workgroup>
offsets_radix_group: array<atomic<u32>, RADIX_COUNT>;
// [R, G / 32] (G / 32 (u32) <- G (bit))
var<workgroup>
masks_radix_in_block: array<array<atomic<u32>, GROUP_MASK_SIZE>, RADIX_COUNT>;

// log2(N')
const BLOCK_COUNT_GROUP_SHIFT: u32 = 14;
// 32 - 1
const DIV_32_MASK: u32 = (1u << 5) - 1;
// G / 32
const GROUP_MASK_SIZE: u32 = GROUP_SIZE >> 5;
// R
const RADIX_COUNT: u32 = 1u << RADIX_COUNT_SHIFT;
// log2(R)
const RADIX_COUNT_SHIFT: u32 = 8;
// R - 1
const RADIX_MASK: u32 = RADIX_COUNT - 1;
// G <- R
const GROUP_SIZE: u32 = RADIX_COUNT;

@compute @workgroup_size(GROUP_SIZE, 1, 1)
fn main(
    @builtin(num_workgroups) group_counts: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    // (0 ~ G')
    @builtin(subgroup_invocation_id) lane_index: u32,
    // (0 ~ G)
    @builtin(local_invocation_index) local_index: u32,
    // (0 ~ G / G')
    @builtin(subgroup_id) subgroup_index: u32,
) {
    // Specifying the index

    // N' / G
    let group_count = group_counts.x;
    // (0 ~ N' / G)
    let group_index = group_id.x;

    // Specifying the parameters

    // N / N'
    let block_count_group = max(count >> BLOCK_COUNT_GROUP_SHIFT, 1u);

    // Scanning radix counts in all groups into radix offsets of the group

    var count_radix_groups = 0u;
    var offset_radix_group = 0u;

    // (0 ~ N' / G)
    for (var index = 0u; index < group_count; index++) {
        // [N' / G, R]
        let count_radix_group = counts_radix_group[index * RADIX_COUNT + local_index];
        if index == group_index {
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
    let offset_radix_groups_subgroup = subgroupShuffle(
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
    for (var block_index = 0u; block_index < block_count_group; block_index++) {
        // (0 ~ N' / G, 0 ~ N / N', 0 ~ G)
        let input_index =
            (group_index * block_count_group + block_index) * GROUP_SIZE + local_index;
        let is_input_index_valid = input_index < count;
        var key_input = 0u;
        var value_input = 0u;
        var offset_radix_group = 0u;
        var radix = 0u;

        // Initializing the mask

        // (0 ~ G / 32)
        for (var index = 0u; index < GROUP_MASK_SIZE; index++) {
            masks_radix_in_block[local_index][index] = 0u;
        }
        workgroupBarrier();

        // Fetching the radix and offset

        if is_input_index_valid {
            key_input = keys_input[input_index];
            value_input = values_input[input_index];
            radix = key_input >> arguments.radix_shift & RADIX_MASK;
            offset_radix_group = offsets_radix_group[radix];
            // [R, G / 32] <- [R, G]
            atomicOr(&masks_radix_in_block[radix][mask_radix_index], mask_radix_local);
        }
        workgroupBarrier();


        if is_input_index_valid {
            // Scanning the radix counts into offsets in the block

            var count_radix_block = 0u;
            var offset_radix_local = 0u;
            // (0 ~ G / 32)
            for (var index = 0u; index < GROUP_MASK_SIZE; index++) {
                // [R, G / 32]
                let mask_radix = masks_radix_in_block[radix][index];
                let count_radix = countOneBits(mask_radix);

                if index < mask_radix_index {
                    offset_radix_local += count_radix;
                }
                count_radix_block += count_radix;

                if index == mask_radix_index {
                    offset_radix_local +=
                        countOneBits(mask_radix & (mask_radix_local - 1u));
                }
            }

            // Assigning the key to the new position
            // [N]

            let position = offset_radix_group + offset_radix_local;
            keys_out[position] = key_input;
            values_out[position] = value_input;

            // Adding the radix count of the block to the radix offset of the group
            // [R]

            if offset_radix_local + 1u == count_radix_block {
                atomicAdd(&offsets_radix_group[radix], count_radix_block);
            }
        }
        workgroupBarrier();
    }
}
