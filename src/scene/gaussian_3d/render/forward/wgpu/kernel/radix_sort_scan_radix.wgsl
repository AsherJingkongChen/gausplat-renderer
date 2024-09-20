// [2^R] (radix_offsets)
@group(0) @binding(0) var<storage, read_write>
radix_counts: array<u32, RADIX>;

// 2^R
const RADIX: u32 = 1u << RADIX_BIT_COUNT;
// R
const RADIX_BIT_COUNT: u32 = 8u;

@compute @workgroup_size(1, 1, 1)
fn main() {
    // Scanning the radix counts into offsets (exclusive and inplace)

    var previous = 0u;
    for (var index = 0u; index < RADIX; index += 1u) {
        let current = radix_counts[index];
        radix_counts[index] = previous;
        previous += current;
    }
}
