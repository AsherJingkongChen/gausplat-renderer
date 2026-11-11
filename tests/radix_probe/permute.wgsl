// A probe, not a sort: move each item to a destination given by the host.
// It measures what the memory costs for one shape of the scatter, with no ranking.

// N
@group(0) @binding(0) var<storage, read_write>
count: u32;
// [N]
@group(0) @binding(1) var<storage, read_write>
destinations: array<u32>;
// [N]
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

// Blocks for each group, run one after another
const BLOCK_COUNT_GROUP: u32 = 1;
// Items for each thread in a block
const ROW_COUNT: u32 = 1;
// Whether a block has its two barriers
const IS_BARRIER: bool = true;
const GROUP_SIZE: u32 = 256;

@compute @workgroup_size(GROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    var keys: array<u32, ROW_COUNT>;
    var values: array<u32, ROW_COUNT>;
    var positions: array<u32, ROW_COUNT>;

    for (var block_index = 0u; block_index < BLOCK_COUNT_GROUP; block_index++) {
        let block_offset =
            (group_id.x * BLOCK_COUNT_GROUP + block_index) * ROW_COUNT * GROUP_SIZE;

        for (var row_index = 0u; row_index < ROW_COUNT; row_index++) {
            let input_index = block_offset + row_index * GROUP_SIZE + local_index;
            if input_index < count {
                keys[row_index] = keys_input[input_index];
                values[row_index] = values_input[input_index];
                positions[row_index] = destinations[input_index];
            }
        }
        if IS_BARRIER {
            workgroupBarrier();
        }

        for (var row_index = 0u; row_index < ROW_COUNT; row_index++) {
            let input_index = block_offset + row_index * GROUP_SIZE + local_index;
            if input_index < count {
                keys_out[positions[row_index]] = keys[row_index];
                values_out[positions[row_index]] = values[row_index];
            }
        }
        if IS_BARRIER {
            workgroupBarrier();
        }
    }
}
