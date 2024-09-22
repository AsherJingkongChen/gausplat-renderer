// Radix Sort Compute Shader in WGSL
// Handles arbitrary buffer lengths without host code coordination
// Processes data in multiple passses

// Constants
const WORKGROUP_SIZE: u32 = 256;
const NUM_BITS: u32 = 32;
const BITS_PER_PASS: u32 = 4;
const RADIX: u32 = 1u << BITS_PER_PASS; // Radix size (16 for 4 bits per passs)
const NUM_PASSES: u32 = NUM_BITS / BITS_PER_PASS;

// Bindings
@group(0) @binding(0) var<storage, read_write> keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> temp_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>; // Size: RADIX * num_workgroups
@group(0) @binding(3) var<storage, read_write> offsets: array<u32>; // Size: RADIX

// Shared variables
var<workgroup> local_keys: array<u32, WORKGROUP_SIZE>;
var<workgroup> local_counts: array<atomic<u32>, RADIX>;
var<workgroup> local_offsets: array<u32, RADIX>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let gid: u32 = global_id.x;
    let lid: u32 = local_id.x;
    let wg_id: u32 = workgroup_id.x;
    let num_elements: u32 = arrayLength(&keys);

    // Load keys into local memory
    var key: u32 = select(keys[gid], 0xFFFFFFFFu, gid >= num_elements);
    local_keys[lid] = key;

    // Synchronize to ensure all keys are loaded
    workgroupBarrier();

    // Loop over each radix passs
    for (var passs: u32 = 0u; passs < NUM_PASSES; passs = passs + 1u) {
        let shift: u32 = passs * BITS_PER_PASS;

        // Initialize local counts
        for (var i: u32 = lid; i < RADIX; i = i + WORKGROUP_SIZE) {
            local_counts[i] = 0u;
        }

        workgroupBarrier();

        // Count occurrences of each digit
        let digit: u32 = (local_keys[lid] >> shift) & (RADIX - 1u);
        atomicAdd(&local_counts[digit], 1u);

        workgroupBarrier();

        // Write local counts to global counts
        for (var i: u32 = lid; i < RADIX; i = i + WORKGROUP_SIZE) {
            let idx: u32 = wg_id * RADIX + i;
            counts[idx] = local_counts[i];
        }

        workgroupBarrier();

        // Compute global counts and offsets (only in workgroup 0)
        if (wg_id == 0u) {
            for (var i: u32 = lid; i < RADIX; i = i + WORKGROUP_SIZE) {
                var sum: u32 = 0u;
                for (var wg: u32 = 0u; wg < num_workgroups.x; wg = wg + 1u) {
                    let idx: u32 = wg * RADIX + i;
                    let count: u32 = counts[idx];
                    counts[idx] = sum;
                    sum = sum + count;
                }
                offsets[i] = sum;
            }
        }

        // Synchronize to ensure global offsets are computed
        workgroupBarrier();

        // Broadcast global offsets to all workgroups
        for (var i: u32 = lid; i < RADIX; i = i + WORKGROUP_SIZE) {
            local_offsets[i] = offsets[i];
        }

        workgroupBarrier();

        // Compute local offsets
        var local_digit_offset: u32 = 0u;
        for (var i: u32 = 0u; i < wg_id; i = i + 1u) {
            local_digit_offset = local_digit_offset + counts[i * RADIX + digit];
        }
        let pos_in_digit: u32 = counts[wg_id * RADIX + digit];

        let pos: u32 = local_offsets[digit] + local_digit_offset + pos_in_digit;

        // Update counts for next thread
        atomicAdd(&counts[wg_id * RADIX + digit], 1u);

        // Scatter keys to temp_keys
        if (pos < arrayLength(&temp_keys)) {
            temp_keys[pos] = local_keys[lid];
        }

        workgroupBarrier();

        // Prepare for next passs
        local_keys[lid] = temp_keys[gid];

        workgroupBarrier();
    }

    // Write sorted keys back to keys buffer
    if (gid < num_elements) {
        keys[gid] = local_keys[lid];
    }
}
