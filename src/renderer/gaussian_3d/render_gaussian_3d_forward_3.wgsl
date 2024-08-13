struct Arguments {
    // P
    point_count: u32,

    // I_X / T_X
    tile_count_x: u32,
}

@group(0) @binding(0)
var<storage, read> arguments: Arguments;

// [P]
@group(0) @binding(1)
var<storage, read> depths: array<f32>;

// [P]
@group(0) @binding(2)
var<storage, read> radii: array<u32>;

// [P]
@group(0) @binding(3)
var<storage, read> tile_touched_offsets: array<u32>;

// [P, 2]
@group(0) @binding(4)
var<storage, read> tiles_touched_max: array<vec2<u32>>;

// [P, 2]
@group(0) @binding(5)
var<storage, read> tiles_touched_min: array<vec2<u32>>;

// [T]
@group(0) @binding(6)
var<storage, read_write> point_indexs: array<u32>;

// [T, 2]
@group(0) @binding(7)
var<storage, read_write> point_keys: array<vec2<u32>>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    // Checking the index

    let index = global_id.x;
    if index >= arguments.point_count {
        return;
    }

    // Leaving if invisible

    let is_visible = radii[index] > 0u;
    if !is_visible {
        return;
    }

    // Specifying the parameters

    var offset = tile_touched_offsets[index];
    let tile_touched_max = tiles_touched_max[index];
    let tile_touched_min = tiles_touched_min[index];

    // Computing the keys and indexs of point

    for (var tile_y = tile_touched_min.y; tile_y < tile_touched_max.y; tile_y++) {
        for (var tile_x = tile_touched_min.x; tile_x < tile_touched_max.x; tile_x++) {
            let tile = tile_y * arguments.tile_count_x + tile_x;
            let depth = bitcast<u32>(depths[index]);
            let key = vec2<u32>(tile, depth);
            point_indexs[offset] = index;
            point_keys[offset] = key;
            offset++;
        }
    }
}
