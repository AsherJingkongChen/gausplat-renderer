struct Arguments {
    colors_sh_degree_max: u32,
    filter_low_pass: f32,
    focal_length_x: f32,
    focal_length_y: f32,
    // I_X
    image_size_x: u32,
    // I_Y
    image_size_y: u32,
    // I_X / 2
    image_size_half_x: f32,
    // I_Y / 2
    image_size_half_y: f32,
    // P
    point_count: u32,
    // I_X / T_X
    tile_count_x: u32,
    // I_Y / T_Y
    tile_count_y: u32,
    // T_X
    tile_size_x: u32,
    // T_Y
    tile_size_y: u32,
    view_bound_x: f32,
    view_bound_y: f32,
}

@group(0) @binding(0)
var<storage, read> arguments: Arguments;
// [P, 16, 3]
@group(0) @binding(1)
var<storage, read> colors_sh: array<array<array<f32, 3>, 16>>;
// [P, 3]
@group(0) @binding(2)
var<storage, read> positions_3d: array<array<f32, 3>>;
// [P, 4] (x, y, z, w) (Normalized)
@group(0) @binding(3)
var<storage, read> rotations: array<vec4<f32>>;
// [P, 3]
@group(0) @binding(4)
var<storage, read> scalings: array<array<f32, 3>>;
// [3]
@group(0) @binding(5)
var<storage, read> view_position: vec3<f32>;
// [4, 4]
@group(0) @binding(6)
var<storage, read> view_transform: mat4x4<f32>;
// [P, 3 (+ 1)] (0.0, 1.0)
@group(0) @binding(7)
var<storage, read_write> colors_rgb_3d: array<vec3<f32>>;
// [P, 2, 2] (Symmetric)
@group(0) @binding(8)
var<storage, read_write> conics: array<mat2x2<f32>>;
// [P, 3 (+ 1), 3] (Symmetric)
@group(0) @binding(9)
var<storage, read_write> covariances_3d: array<mat3x3<f32>>;
// [P] (0.2 ~ )
@group(0) @binding(10)
var<storage, read_write> depths: array<f32>;
// [P, 3 (+ 1)] (0.0, 1.0)
@group(0) @binding(11)
var<storage, read_write> is_colors_rgb_3d_clamped: array<vec3<f32>>;
// [P, 2]
@group(0) @binding(12)
var<storage, read_write> positions_2d: array<vec2<f32>>;
// [P]
@group(0) @binding(13)
var<storage, read_write> radii: array<u32>;
// [P]
@group(0) @binding(14)
var<storage, read_write> tile_touched_counts: array<u32>;
// [P, 2]
@group(0) @binding(15)
var<storage, read_write> tiles_touched_max: array<vec2<u32>>;
// [P, 2]
@group(0) @binding(16)
var<storage, read_write> tiles_touched_min: array<vec2<u32>>;
// [P, 3 (+ 1)]
@group(0) @binding(17)
var<storage, read_write> view_directions: array<vec3<f32>>;
// [P, 3 (+ 1)]
@group(0) @binding(18)
var<storage, read_write> view_offsets: array<vec3<f32>>;

const EPSILON: f32 = 1.1920929e-7;
const SH_C_0: array<f32, 1> = array<f32, 1>(
    0.2820948,
);
const SH_C_1: array<f32, 3> = array<f32, 3>(
    -0.48860252,
    0.48860252,
    -0.48860252,
);
const SH_C_2: array<f32, 5> = array<f32, 5>(
    1.0925485,
    -1.0925485,
    0.31539157,
    -1.0925485,
    0.54627424,
);
const SH_C_3: array<f32, 7> = array<f32, 7>(
    -0.5900436,
    2.8906114,
    -0.4570458,
    0.37317634,
    -0.4570458,
    1.4453057,
    -0.5900436,
);

const GROUP_SIZE_X: u32 = 16;
const GROUP_SIZE_Y: u32 = 16;

@compute @workgroup_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) group_count: vec3<u32>,
) {
    // Checking the index

    // (0 ~ P)
    let index = global_id.y * group_count.x * GROUP_SIZE_X + global_id.x;
    if index >= arguments.point_count {
        return;
    }

    // Initializing the results

    radii[index] = u32();
    tile_touched_counts[index] = u32();
    tiles_touched_max[index] = vec2<u32>();
    tiles_touched_min[index] = vec2<u32>();

    // Transforming 3D positions from world space to view space
    // pv[P, 3] = vr[3, 3] * pw[P, 3] + vt[P, 3]

    let position_3d = vec3<f32>(
        positions_3d[index][0],
        positions_3d[index][1],
        positions_3d[index][2],
    );
    let view_rotation = mat3x3<f32>(
        view_transform[0].xyz,
        view_transform[1].xyz,
        view_transform[2].xyz,
    );
    let view_translation = view_transform[3].xyz;
    let position_3d_in_view = view_rotation * position_3d + view_translation;
    let depth = position_3d_in_view.z + EPSILON;

    // Performing viewing-frustum culling

    if depth <= 0.2 {
        return;
    }

    // Transforming 3D positions from view space (normalized) to screen space (2D)
    // ps[P, 2] = pv[P, 3]

    let position_3d_in_view_x_normalized = position_3d_in_view.x / depth;
    let position_3d_in_view_y_normalized = position_3d_in_view.y / depth;
    let position_2d = vec2<f32>(
        position_3d_in_view_x_normalized * arguments.focal_length_x + arguments.image_size_half_x - 0.5,
        position_3d_in_view_y_normalized * arguments.focal_length_y + arguments.image_size_half_y - 0.5,
    );

    // Converting the quaternion to rotation matrix
    // r[P, 3, 3] (Symmetric) = q[P, 4] (x, y, z, w)

    let quaternion = rotations[index];
    let q_wx = quaternion.w * quaternion.x;
    let q_wy = quaternion.w * quaternion.y;
    let q_wz = quaternion.w * quaternion.z;
    let q_xx = quaternion.x * quaternion.x;
    let q_xy = quaternion.x * quaternion.y;
    let q_xz = quaternion.x * quaternion.z;
    let q_yy = quaternion.y * quaternion.y;
    let q_yz = quaternion.y * quaternion.z;
    let q_zz = quaternion.z * quaternion.z;

    // Computing 3D covariance matrix
    // t[P, 3, 3] = r[P, 3, 3] * s[P, 1, 3]
    // c[P, 3, 3] (Symmetric) = t[P, 3, 3] * t^T[P, 3, 3]

    let rotation = mat3x3<f32>(
        (- q_yy - q_zz) + 0.5, (q_xy + q_wz), (q_xz - q_wy),
        (q_xy - q_wz), (- q_xx - q_zz) + 0.5, (q_yz + q_wx),
        (q_xz + q_wy), (q_yz - q_wx), (- q_xx - q_yy) + 0.5,
    ) * 2.0;
    let scaling = scalings[index];
    let rotation_scaling = mat3x3<f32>(
        rotation[0] * scaling[0],
        rotation[1] * scaling[1],
        rotation[2] * scaling[2],
    );
    let covariance_3d = rotation_scaling * transpose(rotation_scaling);

    // Computing 2D covariance matrix
    // t[P, 2, 3] = j[P, 2, 3] * vr[1, 3, 3]
    // c'[P, 2, 2] (Symmetric) =
    // t[P, 2, 3] * c[P, 3, 3] * t^T[P, 3, 2] + f[1, 2, 2]

    let focal_length_x_normalized = arguments.focal_length_x / depth;
    let focal_length_y_normalized = arguments.focal_length_y / depth;
    let position_3d_in_view_x_normalized_clamped = clamp(
        position_3d_in_view_x_normalized,
        -arguments.view_bound_x,
        arguments.view_bound_x,
    );
    var position_3d_in_view_y_normalized_clamped = clamp(
        position_3d_in_view_y_normalized,
        -arguments.view_bound_y,
        arguments.view_bound_y,
    );

    let projection = mat3x2<f32>(
        focal_length_x_normalized, 0.0,
        0.0, focal_length_y_normalized,
        -focal_length_x_normalized * position_3d_in_view_x_normalized_clamped,
        -focal_length_y_normalized * position_3d_in_view_y_normalized_clamped,
    );
    let transform = projection * view_rotation;
    let covariance_2d = transform * covariance_3d * transpose(transform) + mat2x2<f32>(
        arguments.filter_low_pass, 0.0,
        0.0, arguments.filter_low_pass,
    );

    // Computing the inverse of the 2D covariance matrix
    // c'^-1[P, 2, 2] (Symmetric) = c'[P, 2, 2] (Symmetric)

    let covariance_2d_det = determinant(covariance_2d);

    // Leaving if the 2D covariance matrix is non-invertible

    if covariance_2d_det == 0.0 {
        return;
    }

    let covariance_2d_01_neg = -covariance_2d[0][1];
    let conic = mat2x2<f32>(
        covariance_2d[1][1], covariance_2d_01_neg,
        covariance_2d_01_neg, covariance_2d[0][0],
    ) * (1.0 / covariance_2d_det);

    // Computing the max radius using the 2D covariance matrix
    // r[P] = c'[P, 2, 2] (Symmetric)

    let covariance_2d_middle = (covariance_2d[0][0] + covariance_2d[1][1]) / 2.0;
    let extent_difference = max(
        arguments.filter_low_pass,
        sqrt(covariance_2d_middle * covariance_2d_middle - covariance_2d_det),
    );
    let extent_max = max(
        covariance_2d_middle + extent_difference,
        covariance_2d_middle - extent_difference,
    );
    let radius = ceil(sqrt(extent_max) * 3.0);

    // Finding the tiles touched

    let tile_size_f32 = vec2<f32>(
        f32(arguments.tile_size_x),
        f32(arguments.tile_size_y),
    );
    let tile_touched_max = vec2<u32>(
        clamp(
            u32((position_2d.x + radius + tile_size_f32.x - 1.0) / tile_size_f32.x),
            0u,
            arguments.tile_count_x,
        ),
        clamp(
            u32((position_2d.y + radius + tile_size_f32.y - 1.0) / tile_size_f32.y),
            0u,
            arguments.tile_count_y,
        ),
    );
    let tile_touched_min = vec2<u32>(
        clamp(
            u32((position_2d.x - radius) / tile_size_f32.x),
            0u,
            arguments.tile_count_x,
        ),
        clamp(
            u32((position_2d.y - radius) / tile_size_f32.y),
            0u,
            arguments.tile_count_y,
        ),
    );
    let tile_touched_count = (tile_touched_max.x - tile_touched_min.x) * (tile_touched_max.y - tile_touched_min.y);
    
    // Leaving if no tile is touched

    if tile_touched_count == 0 {
        return;
    }

    // Computing the view direction in world space
    // vo[P, 3] = pw[P, 3] - vw[1, 3]
    // vd[P, 3] = norm(vo)

    let view_offset = position_3d - view_position;
    let view_direction = normalize(view_offset);
    var x = f32();
    var y = f32();
    var z = f32();
    var xx = f32();
    var xy = f32();
    var yy = f32();
    var zz = f32();
    var zz_5_1 = f32();

    // Transforming 3D color from SH space to RGB space
    // c_rgb[P, 3] = (c_sh[P, 16, 3], vd[P, 3])

    let color_sh = array<vec3<f32>, 16>(
        vec3<f32>(colors_sh[index][0u][0], colors_sh[index][0u][1], colors_sh[index][0u][2]),
        vec3<f32>(colors_sh[index][1u][0], colors_sh[index][1u][1], colors_sh[index][1u][2]),
        vec3<f32>(colors_sh[index][2u][0], colors_sh[index][2u][1], colors_sh[index][2u][2]),
        vec3<f32>(colors_sh[index][3u][0], colors_sh[index][3u][1], colors_sh[index][3u][2]),
        vec3<f32>(colors_sh[index][4u][0], colors_sh[index][4u][1], colors_sh[index][4u][2]),
        vec3<f32>(colors_sh[index][5u][0], colors_sh[index][5u][1], colors_sh[index][5u][2]),
        vec3<f32>(colors_sh[index][6u][0], colors_sh[index][6u][1], colors_sh[index][6u][2]),
        vec3<f32>(colors_sh[index][7u][0], colors_sh[index][7u][1], colors_sh[index][7u][2]),
        vec3<f32>(colors_sh[index][8u][0], colors_sh[index][8u][1], colors_sh[index][8u][2]),
        vec3<f32>(colors_sh[index][9u][0], colors_sh[index][9u][1], colors_sh[index][9u][2]),
        vec3<f32>(colors_sh[index][10][0], colors_sh[index][10][1], colors_sh[index][10][2]),
        vec3<f32>(colors_sh[index][11][0], colors_sh[index][11][1], colors_sh[index][11][2]),
        vec3<f32>(colors_sh[index][12][0], colors_sh[index][12][1], colors_sh[index][12][2]),
        vec3<f32>(colors_sh[index][13][0], colors_sh[index][13][1], colors_sh[index][13][2]),
        vec3<f32>(colors_sh[index][14][0], colors_sh[index][14][1], colors_sh[index][14][2]),
        vec3<f32>(colors_sh[index][15][0], colors_sh[index][15][1], colors_sh[index][15][2]),
    );

    var color_rgb_3d = color_sh[0] * (SH_C_0[0]);

    if arguments.colors_sh_degree_max >= 1 {
        x = view_direction.x;
        y = view_direction.y;
        z = view_direction.z;

        color_rgb_3d +=
            color_sh[1] * (SH_C_1[0] * (y)) +
            color_sh[2] * (SH_C_1[1] * (z)) +
            color_sh[3] * (SH_C_1[2] * (x));
    }

    if arguments.colors_sh_degree_max >= 2 {
        xx = x * x;
        xy = x * y;
        yy = y * y;
        zz = z * z;

        color_rgb_3d +=
            color_sh[4] * (SH_C_2[0] * (xy)) +
            color_sh[5] * (SH_C_2[1] * (y * z)) +
            color_sh[6] * (SH_C_2[2] * (zz * 3.0 - 1.0)) +
            color_sh[7] * (SH_C_2[3] * (x * z)) +
            color_sh[8] * (SH_C_2[4] * (xx - yy));
    }

    if arguments.colors_sh_degree_max >= 3 {
        zz_5_1 = zz * 5.0 - 1.0;

        color_rgb_3d +=
            color_sh[9u] * (SH_C_3[0] * (y * (xx * 3.0 - yy))) +
            color_sh[10] * (SH_C_3[1] * (z * (xy))) +
            color_sh[11] * (SH_C_3[2] * (y * (zz_5_1))) +
            color_sh[12] * (SH_C_3[3] * (z * (zz_5_1 - 2.0))) +
            color_sh[13] * (SH_C_3[4] * (x * (zz_5_1))) +
            color_sh[14] * (SH_C_3[5] * (z * (xx - yy))) +
            color_sh[15] * (SH_C_3[6] * (x * (xx - yy * 3.0)));
    }

    color_rgb_3d += 0.5;

    let is_color_rgb_3d_clamped = vec3<f32>(color_rgb_3d < vec3<f32>());
    color_rgb_3d = max(color_rgb_3d, vec3<f32>());

    // Specifying the results

    // [P, 3]
    colors_rgb_3d[index] = color_rgb_3d;
    // [P, 2, 2] (Symmetric)
    conics[index] = conic;
    // [P, 3, 3] (Symmetric)
    covariances_3d[index] = covariance_3d;
    // [P]
    depths[index] = depth;
    // [P, 3]
    is_colors_rgb_3d_clamped[index] = is_color_rgb_3d_clamped;
    // [P, 2]
    positions_2d[index] = position_2d;
    // [P]
    radii[index] = u32(radius);
    // [P]
    tile_touched_counts[index] = tile_touched_count;
    // [P, 2]
    tiles_touched_max[index] = tile_touched_max;
    // [P, 2]
    tiles_touched_min[index] = tile_touched_min;
    // [P, 3]
    view_directions[index] = view_direction;
    // [P, 3]
    view_offsets[index] = view_offset;
}
