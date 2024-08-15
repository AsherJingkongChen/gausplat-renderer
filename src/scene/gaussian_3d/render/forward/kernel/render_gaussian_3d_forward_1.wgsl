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
var<storage, read> positions: array<array<f32, 3>>;

// [P, 4]
@group(0) @binding(3)
var<storage, read> rotations: array<array<f32, 4>>;

// [P, 3]
@group(0) @binding(4)
var<storage, read> scalings: array<array<f32, 3>>;

// [3]
@group(0) @binding(5)
var<storage, read> view_position: vec3<f32>;

// [4, 4]
@group(0) @binding(6)
var<storage, read> view_transform: mat4x4<f32>;

// [P, 3]
@group(0) @binding(7)
var<storage, read_write> colors_rgb_3d: array<array<f32, 3>>;

// [P, 2, 2]
@group(0) @binding(8)
var<storage, read_write> conics: array<mat2x2<f32>>;

// [P, 3, 3]
@group(0) @binding(9)
var<storage, read_write> covariances_3d: array<mat3x3<f32>>;

// [P]
@group(0) @binding(10)
var<storage, read_write> depths: array<f32>;

// [P, 2]
@group(0) @binding(11)
var<storage, read_write> positions_2d_in_screen: array<vec2<f32>>;

// [P]
@group(0) @binding(12)
var<storage, read_write> radii: array<u32>;

// [P]
@group(0) @binding(13)
var<storage, read_write> tile_touched_counts: array<u32>;

// [P, 2]
@group(0) @binding(14)
var<storage, read_write> tiles_touched_max: array<vec2<u32>>;

// [P, 2]
@group(0) @binding(15)
var<storage, read_write> tiles_touched_min: array<vec2<u32>>;

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

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    // Checking the index

    let index = global_id.x;
    if index >= arguments.point_count {
        return;
    }

    // Specifying the parameters

    let position = vec3<f32>(
        positions[index][0],
        positions[index][1],
        positions[index][2],
    );
    let quaternion = vec4<f32>(
        rotations[index][1],
        rotations[index][2],
        rotations[index][3],
        rotations[index][0],
    );
    let scaling = vec3<f32>(
        scalings[index][0],
        scalings[index][1],
        scalings[index][2],
    );

    // Initializing the results

    radii[index] = 0u;
    tile_touched_counts[index] = 0u;
    tiles_touched_max[index] = vec2<u32>();
    tiles_touched_min[index] = vec2<u32>();

    // Transforming 3D positions from world space to view space
    // pv[3, P] = vr[3, 3] * pw[3, P] + vt[3, P]

    let view_rotation = mat3x3<f32>(
        view_transform[0][0], view_transform[0][1], view_transform[0][2],
        view_transform[1][0], view_transform[1][1], view_transform[1][2],
        view_transform[2][0], view_transform[2][1], view_transform[2][2],
    );
    let view_translation = vec3<f32>(
        view_transform[3][0],
        view_transform[3][1],
        view_transform[3][2],
    );
    let position_3d_in_view = view_rotation * position + view_translation;
    let depth = position_3d_in_view.z + EPSILON;
    let position_3d_in_view_x_normalized = position_3d_in_view.x / depth;
    let position_3d_in_view_y_normalized = position_3d_in_view.y / depth;

    // Performing viewing-frustum culling

    if depth <= 0.2 {
        return;
    }

    // Converting the quaternion to rotation matrix
    // r[P, 3, 3] (Symmetric) = q[P, 4] (x, y, z, w)

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
    let rotation_scaling = mat3x3<f32>(
        rotation[0] * scaling[0],
        rotation[1] * scaling[1],
        rotation[2] * scaling[2],
    );
    let covariance_3d = rotation_scaling * transpose(rotation_scaling);

    // Computing 2D covariance matrix
    // t[P, 2, 3] = j[P, 2, 3] * vr[1, 3, 3]
    // c'[P, 2, 2] (Symmetric) = t[P, 2, 3] * c[P, 3, 3] * t^T[P, 3, 2] + f[1, 2, 2]

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

    // [2, 3]
    let projection_jacobian = mat3x2<f32>(
        focal_length_x_normalized,
        0.0,
        0.0,
        focal_length_y_normalized,
        -focal_length_x_normalized * position_3d_in_view_x_normalized_clamped,
        -focal_length_y_normalized * position_3d_in_view_y_normalized_clamped,
    );
    let transform = projection_jacobian * view_rotation;
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
    // r[P]

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

    // Transforming 3D positions from view space (normalized) to screen space (2D)
    // ps[2, P] = pv[3, P]

    let position_2d_in_screen = vec2<f32>(
        position_3d_in_view_x_normalized * arguments.focal_length_x + arguments.image_size_half_x - 0.5,
        position_3d_in_view_y_normalized * arguments.focal_length_y + arguments.image_size_half_y - 0.5,
    );

    // Finding the tiles touched

    let tile_size_f32 = vec2<f32>(
        f32(arguments.tile_size_x),
        f32(arguments.tile_size_y),
    );
    let tile_touched_max = vec2<u32>(
        clamp(
            u32((position_2d_in_screen.x + radius + tile_size_f32.x - 1.0) / tile_size_f32.x),
            0u,
            arguments.tile_count_x,
        ),
        clamp(
            u32((position_2d_in_screen.y + radius + tile_size_f32.y - 1.0) / tile_size_f32.y),
            0u,
            arguments.tile_count_y,
        ),
    );
    let tile_touched_min = vec2<u32>(
        clamp(
            u32((position_2d_in_screen.x - radius) / tile_size_f32.x),
            0u,
            arguments.tile_count_x,
        ),
        clamp(
            u32((position_2d_in_screen.y - radius) / tile_size_f32.y),
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
    // d[3, P] = pw[3, P] - vw[3, 1]

    let view_direction = normalize(position - view_position);

    // Transforming 3D color from SH space to RGB space
    // c_rgb[P, 3] = c_sh[P, 16, 3]

    let color_sh_0 = array<vec3<f32>, 1>(
        vec3<f32>(
            colors_sh[index][0][0],
            colors_sh[index][0][1],
            colors_sh[index][0][2],
        ),
    );
    var color_rgb_3d = color_sh_0[0] * SH_C_0[0];

    var vd_x = 0.0;
    var vd_y = 0.0;
    var vd_z = 0.0;
    var vd_xx = 0.0;
    var vd_xy = 0.0;
    var vd_yy = 0.0;
    var vd_zz = 0.0;
    var vd_zz_5_1 = 0.0;

    if arguments.colors_sh_degree_max >= 1 {
        vd_x = view_direction.x;
        vd_y = view_direction.y;
        vd_z = view_direction.z;

        let color_sh_1 = array<vec3<f32>, 3>(
            vec3<f32>(
                colors_sh[index][1][0],
                colors_sh[index][1][1],
                colors_sh[index][1][2],
            ),
            vec3<f32>(
                colors_sh[index][2][0],
                colors_sh[index][2][1],
                colors_sh[index][2][2],
            ),
            vec3<f32>(
                colors_sh[index][3][0],
                colors_sh[index][3][1],
                colors_sh[index][3][2],
            ),
        );

        color_rgb_3d +=
            color_sh_1[0] * (SH_C_1[0] * (vd_y)) +
            color_sh_1[1] * (SH_C_1[1] * (vd_z)) +
            color_sh_1[2] * (SH_C_1[2] * (vd_x));
    }

    if arguments.colors_sh_degree_max >= 2 {
        vd_xx = vd_x * vd_x;
        vd_xy = vd_x * vd_y;
        vd_yy = vd_y * vd_y;
        vd_zz = vd_z * vd_z;

        let color_sh_2 = array<vec3<f32>, 5>(
            vec3<f32>(
                colors_sh[index][4][0],
                colors_sh[index][4][1],
                colors_sh[index][4][2],
            ),
            vec3<f32>(
                colors_sh[index][5][0],
                colors_sh[index][5][1],
                colors_sh[index][5][2],
            ),
            vec3<f32>(
                colors_sh[index][6][0],
                colors_sh[index][6][1],
                colors_sh[index][6][2],
            ),
            vec3<f32>(
                colors_sh[index][7][0],
                colors_sh[index][7][1],
                colors_sh[index][7][2],
            ),
            vec3<f32>(
                colors_sh[index][8][0],
                colors_sh[index][8][1],
                colors_sh[index][8][2],
            ),
        );

        color_rgb_3d +=
            color_sh_2[0] * (SH_C_2[0] * (vd_xy)) +
            color_sh_2[1] * (SH_C_2[1] * (vd_y * vd_z)) +
            color_sh_2[2] * (SH_C_2[2] * (vd_zz * 3.0 - 1.0)) +
            color_sh_2[3] * (SH_C_2[3] * (vd_x * vd_z)) +
            color_sh_2[4] * (SH_C_2[4] * (vd_xx - vd_yy));
    }

    if arguments.colors_sh_degree_max >= 3 {
        vd_zz_5_1 = vd_zz * 5.0 - 1.0;

        let color_sh_3 = array<vec3<f32>, 7>(
            vec3<f32>(
                colors_sh[index][9][0],
                colors_sh[index][9][1],
                colors_sh[index][9][2],
            ),
            vec3<f32>(
                colors_sh[index][10][0],
                colors_sh[index][10][1],
                colors_sh[index][10][2],
            ),
            vec3<f32>(
                colors_sh[index][11][0],
                colors_sh[index][11][1],
                colors_sh[index][11][2],
            ),
            vec3<f32>(
                colors_sh[index][12][0],
                colors_sh[index][12][1],
                colors_sh[index][12][2],
            ),
            vec3<f32>(
                colors_sh[index][13][0],
                colors_sh[index][13][1],
                colors_sh[index][13][2],
            ),
            vec3<f32>(
                colors_sh[index][14][0],
                colors_sh[index][14][1],
                colors_sh[index][14][2],
            ),
            vec3<f32>(
                colors_sh[index][15][0],
                colors_sh[index][15][1],
                colors_sh[index][15][2],
            ),
        );

        color_rgb_3d +=
            color_sh_3[0] * (SH_C_3[0] * (vd_y * (vd_xx * 3.0 - vd_yy))) +
            color_sh_3[1] * (SH_C_3[1] * (vd_z * vd_xy)) +
            color_sh_3[2] * (SH_C_3[2] * (vd_y * vd_zz_5_1))+
            color_sh_3[3] * (SH_C_3[3] * (vd_z * (vd_zz_5_1 - 2.0))) +
            color_sh_3[4] * (SH_C_3[4] * (vd_x * vd_zz_5_1)) +
            color_sh_3[5] * (SH_C_3[5] * (vd_z * (vd_xx - vd_yy))) +
            color_sh_3[6] * (SH_C_3[6] * (vd_x * (vd_xx - vd_yy * 3.0)));
    }

    color_rgb_3d += 0.5;
    color_rgb_3d.r = max(color_rgb_3d.r, 0.0);
    color_rgb_3d.g = max(color_rgb_3d.g, 0.0);
    color_rgb_3d.b = max(color_rgb_3d.b, 0.0);

    // Specifying the results

    // [P, 3]
    colors_rgb_3d[index] = array<f32, 3>(
        color_rgb_3d.r,
        color_rgb_3d.g,
        color_rgb_3d.b,
    );

    // [P, 2, 2] (Symmetric)
    conics[index] = conic;

    // [P, 3, 3] (Symmetric)
    covariances_3d[index] = covariance_3d;

    // [P]
    depths[index] = depth;

    // [P, 2]
    positions_2d_in_screen[index] = position_2d_in_screen;

    // [P]
    radii[index] = u32(radius);

    // [P]
    tile_touched_counts[index] = tile_touched_count;

    // [P, 2]
    tiles_touched_max[index] = tile_touched_max;

    // [P, 2]
    tiles_touched_min[index] = tile_touched_min;
}
