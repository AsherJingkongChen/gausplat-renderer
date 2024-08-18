struct Arguments {
    // P
    point_count: u32,
}

@group(0) @binding(0)
var<storage, read> arguments: Arguments;
// [P]
@group(0) @binding(1)
var<storage, read> tile_touched_counts: array<u32>;
// T
@group(0) @binding(2)
var<storage, read_write> tile_touched_count: u32;
// [P]
@group(0) @binding(3)
var<storage, read_write> tile_touched_offsets: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn main() {
    // Specifying the parameters

    var state = 0u;

    // Computing the offsets of tile touched
    // [P]

    for (var index = 0u; index < arguments.point_count; index++) {
        tile_touched_offsets[index] = state;
        state += tile_touched_counts[index];
    }

    // Computing the count of tile touched
    // T

    tile_touched_count = select(
        0u,
        tile_touched_counts[arguments.point_count - 1] +
        tile_touched_offsets[arguments.point_count - 1],
        arguments.point_count > 0u,
    );
}
