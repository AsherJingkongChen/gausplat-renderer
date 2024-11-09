struct Arguments {
    // P
    point_count: u32,
    // I_x / T_x
    tile_count_x: u32,
}

@group(0) @binding(0)
var<storage, read_write> arguments: Arguments;
// [P] (0 ~ )
@group(0) @binding(1)
var<storage, read_write> depths: array<f32>;
// [P, 4] (x max, x min, y max, y min)
@group(0) @binding(2)
var<storage, read_write> point_tile_bounds: array<vec4<u32>>;
// [P]
@group(0) @binding(3)
var<storage, read_write> radii: array<u32>;
// [P]
@group(0) @binding(4)
var<storage, read_write> tile_touched_offsets: array<u32>;

// [T]
@group(0) @binding(5)
var<storage, read_write> point_indices: array<u32>;
// [T]
@group(0) @binding(6)
var<storage, read_write> point_orders: array<u32>;

// The difference in bits between depth to depth order (before shifting).
const FACTOR_DEPTH_ORDER: u32 = (4u << 23) + 0xc0000000;
const GROUP_SIZE: u32 = 256;

@compute @workgroup_size(GROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    // Specifying the index

    // (0 ~ P)
    let global_index = global_id.x;
    if global_index >= arguments.point_count {
        return;
    }

    // Leaving if the point is invisible

    if radii[global_index] == 0u {
        return;
    }

    // Specifying the parameters

    var offset = tile_touched_offsets[global_index];
    let point_tile_bound = point_tile_bounds[global_index];

    // Computing the orders and indices of point

    for (var tile_y = point_tile_bound[3]; tile_y < point_tile_bound[2]; tile_y++) {
        for (var tile_x = point_tile_bound[1]; tile_x < point_tile_bound[0]; tile_x++) {
            let tile_index = tile_y * arguments.tile_count_x + tile_x;
            let depth = depths[global_index];
            point_orders[offset] = make_point_order(tile_index, depth);
            point_indices[offset] = global_index;
            offset++;
        }
    }
}

// Computing the point order.
// 
// ## Arguments
// 
// | Tile index | 32 bits | Unsigned integer      |
// | Depth      | 32 bits | Floating-point number |
// 
// ## Returns
// 
// | Tile order  | 16 bits | High order bits |
// | Depth order | 16 bits | Low order bits  |
// 
// ## Details
// 
// * Tile order:
// 
// | Tile index |       |   0b llll_llll_llll_llll_rrrr_rrrr_rrrr_rrrr |
// | Shifting   | << 16 |                                              |
// | Tile order |       | = 0b rrrr_rrrr_rrrr_rrrr_0000_0000_0000_0000 |
// 
// * Depth order range:
// 
// | Depth order min | 2^1  | 2^(-127 + 2^(4) * (8))     |
// | Depth order max | 2^17 | 2^(-127 + 2^(4) * (8 + 1)) |
// 
// * Depth range:
// 
// | Depth order min |   2^1  |   0b 0_1000_0000_000_0000_0000_0000_0000_0000 |
// | Depth order max |   2^17 |   0b 0_1001_0000_000_0000_0000_0000_0000_0000 |
// | Multiplying     | * 2^-4 | - 0b 0_0000_0100_000_0000_0000_0000_0000_0000 |
// | Depth min       |   2^-3 | = 0b 0_0111_1100_000_0000_0000_0000_0000_0000 |
// | Depth max       |   2^13 | = 0b 0_1000_1100_000_0000_0000_0000_0000_0000 |
// 
// * Depth order:
// 
// | Depth       |       |   0b 0_xxxx_xxxx_nnn_nnnn_nnnn_nnnn_nnnn_nnnn |
// | Multiplying | * 2^4 | + 0b 0_0000_0100_000_0000_0000_0000_0000_0000 |
// | Depth order |       | = 0b 0_1000_xxxx_nnn_nnnn_nnnn_nnnn_nnnn_nnnn |
// | Unsetting   |       | + 0b 1_1000_0000_000_0000_0000_0000_0000_0000 |
// | Shifting    | >> 11 |                                               |
// | Depth order |       | = 0b 0_0000_0000_000_0000_xxxx_nnnn_nnnn_nnnn |
// 
fn make_point_order(tile_index: u32, depth: f32) -> u32 {
    return (tile_index << 16) | (bitcast<u32>(depth) + FACTOR_DEPTH_ORDER >> 11);
}
