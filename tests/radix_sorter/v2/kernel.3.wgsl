// (N' / G, 1, 1)
@group(0) @binding(0) var<storage, read_write>
group_count: vec3<u32>;

// [N' / G, R]
@group(0) @binding(1) var<storage, read_write>
counts_radix_group: array<u32>;

// [R]
var<workgroup>
counts_radix_groups: array<u32, RADIX_COUNT>;

// R
const RADIX_COUNT: u32 = 1u << RADIX_COUNT_SHIFT;
// log2(R)
const RADIX_COUNT_SHIFT: u32 = 8;

@compute @workgroup_size(RADIX_COUNT, 1, 1)
fn main(
    // (0 ~ R)
    @builtin(local_invocation_index) radix: u32,
) {
    // Scanning radix counts in all groups into radix offsets of the group

    var count_radix_groups = 0u;

    // (0 ~ N' / G)
    for (var group_index = 0u; group_index < group_count.x; group_index++) {
        // [N' / G, R]
        let index = group_index * RADIX_COUNT + radix;
        let count_radix_group = counts_radix_group[index];
        counts_radix_group[index] = count_radix_groups;
        count_radix_groups += count_radix_group;
    }

    // [R]
    counts_radix_groups[radix] = count_radix_groups;
    workgroupBarrier();

    // Scanning radix counts of all groups into radix offsets

    var offset_radix_groups = 0u;

    // (0 ~ R)
    for (var index = 0u; index < radix; index++) {
        offset_radix_groups += counts_radix_groups[index];
    }

    // Adding the radix offsets to the radix offsets of the group

    // (0 ~ N' / G)
    for (var group_index = 0u; group_index < group_count.x; group_index++) {
        // [N' / G, R]
        counts_radix_group[group_index * RADIX_COUNT + radix] += offset_radix_groups;
    }
}
