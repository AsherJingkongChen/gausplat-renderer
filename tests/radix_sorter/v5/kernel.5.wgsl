struct Arguments {
    // (0 ~ 32: +log2(R))
    radix_shift: u32,
}

@group(0) @binding(0) var<storage, read_write>
arguments: Arguments;
// N
@group(0) @binding(1) var<storage, read_write>
count: u32;
// [N / B, R]
@group(0) @binding(2) var<storage, read_write>
offsets_radix_group: array<u32>;
// [N / B / C, R]
@group(0) @binding(3) var<storage, read_write>
offsets_radix_chunk: array<u32>;
// [N] = [N / B, W, G]
@group(0) @binding(4) var<storage, read_write>
keys_input: array<u32>;
// [N]
@group(0) @binding(5) var<storage, read_write>
values_input: array<u32>;

// [N]
@group(0) @binding(6) var<storage, read_write>
keys_out: array<u32>;
// [N]
@group(0) @binding(7) var<storage, read_write>
values_out: array<u32>;

// [R]
var<workgroup>
counts_radix_in_block: array<atomic<u32>, RADIX_COUNT>;
// [max(B, R * G / 32)]
//
// It holds the masks of each radix in a row, [R, G / 32] (G / 32 (u32) <- G (bit)),
// while the block is ranked, and then the items of the block in the radix order, [B].
var<workgroup>
items_in_block: array<atomic<u32>, ITEM_COUNT>;

// B <- G * W
const BLOCK_SIZE: u32 = GROUP_SIZE * ROW_COUNT;
// C
const CHUNK_SIZE: u32 = 64;
// 32 - 1
const DIV_32_MASK: u32 = (1u << 5) - 1;
// G / 32
const GROUP_MASK_SIZE: u32 = GROUP_SIZE >> 5;
// G
const GROUP_SIZE: u32 = 256;
// max(B, R * G / 32)
const ITEM_COUNT: u32 = max(BLOCK_SIZE, RADIX_COUNT * GROUP_MASK_SIZE);
// R
const RADIX_COUNT: u32 = 1u << RADIX_COUNT_SHIFT;
// log2(R)
const RADIX_COUNT_SHIFT: u32 = 8;
// R - 1
const RADIX_MASK: u32 = RADIX_COUNT - 1;
// R (no radix)
const RADIX_NONE: u32 = RADIX_COUNT;
// W
const ROW_COUNT: u32 = 15;

@compute @workgroup_size(GROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    // (0 ~ G)
    @builtin(local_invocation_index) local_index: u32,
) {
    // Specifying the index

    // (0 ~ N / B)
    let group_index = group_id.x;
    // (0 ~ N / B / C)
    let chunk_index = group_index / CHUNK_SIZE;

    // Specifying the parameters

    let block_offset = group_index * BLOCK_SIZE;
    // (1 ~ B)
    let item_count = min(count - block_offset, BLOCK_SIZE);
    let mask_radix_index = local_index >> 5u;
    let mask_radix_local = 1u << (local_index & DIV_32_MASK);

    // [W]
    var keys: array<u32, ROW_COUNT>;
    // [W]
    var values: array<u32, ROW_COUNT>;
    // [W]
    var positions: array<u32, ROW_COUNT>;

    // Fetching the items of the thread in the block

    // (0 ~ W)
    for (var row_index = 0u; row_index < ROW_COUNT; row_index++) {
        // (0 ~ W, 0 ~ G)
        let item_index = row_index * GROUP_SIZE + local_index;
        if item_index < item_count {
            keys[row_index] = keys_input[block_offset + item_index];
            values[row_index] = values_input[block_offset + item_index];
        }
    }

    // Initializing the radix counts and the masks

    atomicStore(&counts_radix_in_block[local_index], 0u);
    // (0 ~ G / 32)
    for (var index = 0u; index < GROUP_MASK_SIZE; index++) {
        atomicStore(&items_in_block[local_index * GROUP_MASK_SIZE + index], 0u);
    }
    workgroupBarrier();

    // Ranking the items of the same radix in the block, row by row

    var radix_previous = RADIX_NONE;

    // (0 ~ W)
    for (var row_index = 0u; row_index < ROW_COUNT; row_index++) {
        let is_item_valid = row_index * GROUP_SIZE + local_index < item_count;
        var count_radix_rows = 0u;
        var radix = RADIX_NONE;

        // Fetching the radix and its count in the previous rows

        if is_item_valid {
            radix = keys[row_index] >> arguments.radix_shift & RADIX_MASK;
            count_radix_rows = atomicLoad(&counts_radix_in_block[radix]);
        }

        // Moving the mask bit from the previous radix to the radix
        // [R, G / 32] <- [R, G]

        if radix != radix_previous {
            if radix_previous != RADIX_NONE {
                atomicAnd(
                    &items_in_block[radix_previous * GROUP_MASK_SIZE + mask_radix_index],
                    ~mask_radix_local,
                );
            }
            if radix != RADIX_NONE {
                atomicOr(
                    &items_in_block[radix * GROUP_MASK_SIZE + mask_radix_index],
                    mask_radix_local,
                );
            }
            radix_previous = radix;
        }
        workgroupBarrier();

        if is_item_valid {
            // Scanning the radix counts into offsets in the row

            var count_radix_row = 0u;
            var offset_radix_row = 0u;
            // (0 ~ G / 32)
            for (var index = 0u; index < GROUP_MASK_SIZE; index++) {
                // [R, G / 32]
                let mask_radix =
                    atomicLoad(&items_in_block[radix * GROUP_MASK_SIZE + index]);
                let count_radix = countOneBits(mask_radix);

                if index < mask_radix_index {
                    offset_radix_row += count_radix;
                }
                count_radix_row += count_radix;

                if index == mask_radix_index {
                    offset_radix_row +=
                        countOneBits(mask_radix & (mask_radix_local - 1u));
                }
            }

            // Specifying the rank of the item in its radix

            positions[row_index] = count_radix_rows + offset_radix_row;

            // Adding the radix count of the row to the radix count of the block
            // [R]

            if offset_radix_row + 1u == count_radix_row {
                atomicAdd(&counts_radix_in_block[radix], count_radix_row);
            }
        }
        workgroupBarrier();
    }

    // Scanning the radix counts of the block into radix offsets in the block

    var offset_radix_block = 0u;

    // (0 ~ R)
    for (var index = 0u; index < local_index; index++) {
        offset_radix_block += atomicLoad(&counts_radix_in_block[index]);
    }
    workgroupBarrier();

    // [R]
    atomicStore(&counts_radix_in_block[local_index], offset_radix_block);
    workgroupBarrier();

    // Specifying the position of the item in the block, in the radix order

    // (0 ~ W)
    for (var row_index = 0u; row_index < ROW_COUNT; row_index++) {
        if row_index * GROUP_SIZE + local_index < item_count {
            let radix = keys[row_index] >> arguments.radix_shift & RADIX_MASK;
            positions[row_index] += atomicLoad(&counts_radix_in_block[radix]);
        }
    }

    // Reordering the keys of the block in the radix order
    //
    // The thread takes the items next to those of the next thread,
    // so that the threads write next to each other.

    // (0 ~ W)
    for (var row_index = 0u; row_index < ROW_COUNT; row_index++) {
        if row_index * GROUP_SIZE + local_index < item_count {
            atomicStore(&items_in_block[positions[row_index]], keys[row_index]);
        }
    }
    workgroupBarrier();

    // (0 ~ W)
    for (var row_index = 0u; row_index < ROW_COUNT; row_index++) {
        keys[row_index] = atomicLoad(&items_in_block[row_index * GROUP_SIZE + local_index]);
    }
    workgroupBarrier();

    // Reordering the values of the block in the radix order

    // (0 ~ W)
    for (var row_index = 0u; row_index < ROW_COUNT; row_index++) {
        if row_index * GROUP_SIZE + local_index < item_count {
            atomicStore(&items_in_block[positions[row_index]], values[row_index]);
        }
    }
    workgroupBarrier();

    // (0 ~ W)
    for (var row_index = 0u; row_index < ROW_COUNT; row_index++) {
        values[row_index] =
            atomicLoad(&items_in_block[row_index * GROUP_SIZE + local_index]);
    }

    // Assigning the items to the new positions
    // [N]

    // (0 ~ W)
    for (var row_index = 0u; row_index < ROW_COUNT; row_index++) {
        // (0 ~ W, 0 ~ G)
        let item_index = row_index * GROUP_SIZE + local_index;
        if item_index < item_count {
            let radix = keys[row_index] >> arguments.radix_shift & RADIX_MASK;
            let position = offsets_radix_group[group_index * RADIX_COUNT + radix]
                + offsets_radix_chunk[chunk_index * RADIX_COUNT + radix]
                + item_index
                - atomicLoad(&counts_radix_in_block[radix]);
            keys_out[position] = keys[row_index];
            values_out[position] = values[row_index];
        }
    }
}
