struct Arguments {
    // B
    point_key_byte_capacity: u32,

    // T
    tile_touched_count: u32,
}

@group(0) @binding(0)
var<storage, read> arguments: Arguments;

// [T]
@group(0) @binding(1)
var<storage, read> point_indexes: array<f32>;

// [T, 2] ([tile_index, depth])
@group(0) @binding(2)
var<storage, read> point_keys: array<vec2<u32>>;

// [T]
@group(0) @binding(3)
var<storage, read_write> point_indexes_sorted: array<f32>;

// [T, 2] ([tile_index, depth])
@group(0) @binding(4)
var<storage, read_write> point_keys_sorted: array<vec2<u32>>;

@compute @workgroup_size(1)
fn main() {
    // Sorting the points by the keys
    // [B]

    for (var byte_index = 0u; byte_index < arguments.point_key_byte_capacity; byte_index++) {
        var radixes = array<u32, 256>();

        var bit_index = byte_index << 3;
        var point_key_position = 1u;
        if byte_index >= 4u {
            bit_index -= 32u;
            point_key_position = 0u;
        }

        // Collecting the radixes

        for (var index = 0u; index < arguments.tile_touched_count; index++) {
            let point_key = point_keys[index];
            let radix_index = (point_key[point_key_position] >> bit_index) & 0xFFu;
            radixes[radix_index]++;
        }

        // Computing the exclusive sum of radixes

        var state = 0u;
        for (var index = 0u; index < 256u; index++) {
            let state_original = state;
            state += radixes[index];
            radixes[index] = state_original;
        }

        // Rearranging the points using the radixes

        for (var index = 0u; index < arguments.tile_touched_count; index++) {
            let point_key = point_keys[index];
            let point_index = point_indexes[index];
            let radix_index = (point_key[point_key_position] >> bit_index) & 0xFFu;
            let radix = radixes[radix_index];

            point_keys_sorted[radix] = point_key;
            point_indexes_sorted[radix] = point_index;
            radixes[radix_index]++;
        }
    }
}
