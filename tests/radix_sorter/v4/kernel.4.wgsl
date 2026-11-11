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
offsets_radix_group: array<u32>;
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

// [R]
var<workgroup>
offsets_radix_in_group: array<atomic<u32>, RADIX_COUNT>;
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
// R (no radix)
const RADIX_NONE: u32 = RADIX_COUNT;
// G <- R
const GROUP_SIZE: u32 = 256;

@compute @workgroup_size(GROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    // (0 ~ G)
    @builtin(local_invocation_index) local_index: u32,
) {
    // Specifying the index

    // (0 ~ N' / G)
    let group_index = group_id.x;

    // Specifying the parameters

    // N / N'
    let block_count_group = max(count >> BLOCK_COUNT_GROUP_SHIFT, 1u);

    // Fetching the radix offsets of the group

    if local_index < RADIX_COUNT {
        // [R] <- [N' / G, R]
        atomicStore(
            &offsets_radix_in_group[local_index],
            offsets_radix_group[group_index * RADIX_COUNT + local_index],
        );

        // Initializing the mask

        // (0 ~ G / 32)
        for (var index = 0u; index < GROUP_MASK_SIZE; index++) {
            atomicStore(&masks_radix_in_block[local_index][index], 0u);
        }
    }
    workgroupBarrier();

    // Scattering keys in the group blocks

    let mask_radix_index = local_index >> 5u;
    let mask_radix_local = 1u << (local_index & DIV_32_MASK);
    var radix_previous = RADIX_NONE;

    // (0 ~ N / N')
    for (var block_index = 0u; block_index < block_count_group; block_index++) {
        // (0 ~ N' / G, 0 ~ N / N', 0 ~ G)
        let input_index =
            (group_index * block_count_group + block_index) * GROUP_SIZE + local_index;
        let is_input_index_valid = input_index < count;
        var key_input = 0u;
        var value_input = 0u;
        var offset_radix_group = 0u;
        var radix = RADIX_NONE;

        // Fetching the radix and offset

        if is_input_index_valid {
            key_input = keys_input[input_index];
            value_input = values_input[input_index];
            radix = key_input >> arguments.radix_shift & RADIX_MASK;
            offset_radix_group = atomicLoad(&offsets_radix_in_group[radix]);
        }

        // Moving the mask bit from the previous radix to the radix
        // [R, G / 32] <- [R, G]

        if radix != radix_previous {
            if radix_previous != RADIX_NONE {
                atomicAnd(
                    &masks_radix_in_block[radix_previous][mask_radix_index],
                    ~mask_radix_local,
                );
            }
            if radix != RADIX_NONE {
                atomicOr(&masks_radix_in_block[radix][mask_radix_index], mask_radix_local);
            }
            radix_previous = radix;
        }
        workgroupBarrier();

        if is_input_index_valid {
            // Scanning the radix counts into offsets in the block

            var count_radix_block = 0u;
            var offset_radix_local = 0u;
            // (0 ~ G / 32)
            for (var index = 0u; index < GROUP_MASK_SIZE; index++) {
                // [R, G / 32]
                let mask_radix = atomicLoad(&masks_radix_in_block[radix][index]);
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
                atomicAdd(&offsets_radix_in_group[radix], count_radix_block);
            }
        }
        workgroupBarrier();
    }
}
