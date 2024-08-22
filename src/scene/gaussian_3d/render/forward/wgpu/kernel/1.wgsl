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
// [3 (+ 1), 4]
@group(0) @binding(6)
var<storage, read> view_transform: mat4x3<f32>;
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

    // Transforming the 3D position from world space to view space
    // Pv[3, 1] = Rv[3, 3] * Pw[3, 1] + Tv[3, 1]

    let position_3d = vec_from_array_f32_3(positions_3d[index]);
    let view_rotation = mat3x3<f32>(
        view_transform[0],
        view_transform[1],
        view_transform[2],
    );
    let view_translation = view_transform[3];
    let position_3d_in_view = view_rotation * position_3d + view_translation;
    let depth = position_3d_in_view.z + EPSILON;

    // Performing viewing-frustum culling

    if depth <= 0.2 {
        return;
    }

    // Projecting the 3D position from view space (normalized) to screen space (2D)
    // Pv'[2, 1] = T[2, 3] * Pv[3, 1]
    //
    // T [[f.x / Pv.z, 0,          (I.x - 1) / 2 / Pv.z]
    //    [0,          f.y / Pv.z, (I.y - 1) / 2 / Pv.z]]

    let position_3d_in_view_x_normalized = position_3d_in_view.x / depth;
    let position_3d_in_view_y_normalized = position_3d_in_view.y / depth;
    let position_2d = vec2<f32>(
        position_3d_in_view_x_normalized * arguments.focal_length_x + arguments.image_size_half_x - 0.5,
        position_3d_in_view_y_normalized * arguments.focal_length_y + arguments.image_size_half_y - 0.5,
    );

    // Converting the quaternion to rotation matrix
    // R[3, 3] (Symmetric) = Q[4] (x, y, z, w)

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

    // Computing the 3D covariance matrix from rotation and scaling
    // T[3, 3] = R[3, 3] * S[3, 3]
    // Σ[3, 3] (Symmetric) = T[3, 3] * T^t[3, 3]
    //
    // S [[S.x, 0,   0]
    //    [0,   S.y, 0]
    //    [0,   0,   S.z]]

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

    // Projecting the 3D covariance matrix into 2D covariance matrix
    // T[2, 3] = J[2, 3] * Rv[3, 3]
    // Σ'[2, 2] (Symmetric) = T[2, 3] * Σ[3, 3] * T^t[3, 2] + F[2, 2]
    //
    // J [[f.x / Pv.z, 0,          -f.x * Pv.x_c / Pv.z^2]
    //    [0,          f.y / Pv.z, -f.y * Pv.y_c / Pv.z^2]]
    // F [[0.3, 0]]
    //    [0,   0.3]]
    //
    // Pv.x_c is the clamped Pv.x
    // Pv.y_c is the clamped Pv.y

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
    let transform_3d_to_2d = projection * view_rotation;
    let covariance_2d =
        transform_3d_to_2d * covariance_3d * transpose(transform_3d_to_2d) + mat2x2<f32>(
            arguments.filter_low_pass, 0.0,
            0.0, arguments.filter_low_pass,
        );

    // Computing the inverse of the 2D covariance matrix
    // Σ'^-1[2, 2] (Symmetric) = Σ'[2, 2]

    let covariance_2d_det = determinant(covariance_2d);

    // Leaving if the 2D covariance matrix is non-invertible

    if covariance_2d_det == 0.0 {
        return;
    }

    let covariance_2d_01_n = -covariance_2d[0][1];
    let conic = mat2x2<f32>(
        covariance_2d[1][1], covariance_2d_01_n,
        covariance_2d_01_n, covariance_2d[0][0],
    ) * (1.0 / covariance_2d_det);

    // Computing the max radius using the 2D covariance matrix
    // r <= Σ'[2, 2]

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
            0u, arguments.tile_count_x,
        ),
        clamp(
            u32((position_2d.y + radius + tile_size_f32.y - 1.0) / tile_size_f32.y),
            0u, arguments.tile_count_y,
        ),
    );
    let tile_touched_min = vec2<u32>(
        clamp(
            u32((position_2d.x - radius) / tile_size_f32.x),
            0u, arguments.tile_count_x,
        ),
        clamp(
            u32((position_2d.y - radius) / tile_size_f32.y),
            0u, arguments.tile_count_y,
        ),
    );
    let tile_touched_count =
        (tile_touched_max.x - tile_touched_min.x) *
        (tile_touched_max.y - tile_touched_min.y);
    
    // Leaving if no tile is touched

    if tile_touched_count == 0 {
        return;
    }

    // Computing the view direction in world space
    // Dv[3] <= Ov[3] = Pw[3] - V[3]

    let view_offset = position_3d - view_position;
    let view_direction = normalize(view_offset);
    var vd_x = f32();
    var vd_y = f32();
    var vd_z = f32();
    var vd_xx = f32();
    var vd_xy = f32();
    var vd_yy = f32();
    var vd_zz = f32();
    var vd_zz_5_1 = f32();

    // Computing the 3D color in RGB space from SH space
    // D[16] <= Dv[3]
    // C_rgb[3] = D[1, 16] * C_sh[16, 3] + 0.5
    //
    // C_rgb is clamped

    let color_sh = array<vec3<f32>, 16>(
        vec_from_array_f32_3(colors_sh[index][0u]),
        vec_from_array_f32_3(colors_sh[index][1u]),
        vec_from_array_f32_3(colors_sh[index][2u]),
        vec_from_array_f32_3(colors_sh[index][3u]),
        vec_from_array_f32_3(colors_sh[index][4u]),
        vec_from_array_f32_3(colors_sh[index][5u]),
        vec_from_array_f32_3(colors_sh[index][6u]),
        vec_from_array_f32_3(colors_sh[index][7u]),
        vec_from_array_f32_3(colors_sh[index][8u]),
        vec_from_array_f32_3(colors_sh[index][9u]),
        vec_from_array_f32_3(colors_sh[index][10]),
        vec_from_array_f32_3(colors_sh[index][11]),
        vec_from_array_f32_3(colors_sh[index][12]),
        vec_from_array_f32_3(colors_sh[index][13]),
        vec_from_array_f32_3(colors_sh[index][14]),
        vec_from_array_f32_3(colors_sh[index][15]),
    );

    var color_rgb_3d = color_sh[0] * (SH_C_0[0]);

    if arguments.colors_sh_degree_max >= 1 {
        vd_x = view_direction.x;
        vd_y = view_direction.y;
        vd_z = view_direction.z;

        color_rgb_3d +=
            color_sh[1] * (SH_C_1[0] * (vd_y)) +
            color_sh[2] * (SH_C_1[1] * (vd_z)) +
            color_sh[3] * (SH_C_1[2] * (vd_x));
    }

    if arguments.colors_sh_degree_max >= 2 {
        vd_xx = vd_x * vd_x;
        vd_xy = vd_x * vd_y;
        vd_yy = vd_y * vd_y;
        vd_zz = vd_z * vd_z;

        color_rgb_3d +=
            color_sh[4] * (SH_C_2[0] * (vd_xy)) +
            color_sh[5] * (SH_C_2[1] * (vd_y * vd_z)) +
            color_sh[6] * (SH_C_2[2] * (vd_zz * 3.0 - 1.0)) +
            color_sh[7] * (SH_C_2[3] * (vd_x * vd_z)) +
            color_sh[8] * (SH_C_2[4] * (vd_xx - vd_yy));
    }

    if arguments.colors_sh_degree_max >= 3 {
        vd_zz_5_1 = vd_zz * 5.0 - 1.0;

        color_rgb_3d +=
            color_sh[9u] * (SH_C_3[0] * (vd_y * (vd_xx * 3.0 - vd_yy))) +
            color_sh[10] * (SH_C_3[1] * (vd_z * (vd_xy))) +
            color_sh[11] * (SH_C_3[2] * (vd_y * (vd_zz_5_1))) +
            color_sh[12] * (SH_C_3[3] * (vd_z * (vd_zz_5_1 - 2.0))) +
            color_sh[13] * (SH_C_3[4] * (vd_x * (vd_zz_5_1))) +
            color_sh[14] * (SH_C_3[5] * (vd_z * (vd_xx - vd_yy))) +
            color_sh[15] * (SH_C_3[6] * (vd_x * (vd_xx - vd_yy * 3.0)));
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

fn vec_from_array_f32_3(a: array<f32, 3>) -> vec3<f32> {
    return vec3<f32>(a[0], a[1], a[2]);
}
