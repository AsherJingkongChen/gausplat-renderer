struct Arguments {
    // (0 ~ 3)
    colors_sh_degree_max: u32,
    // F_x <- I_x / tan(Fov_x / 2) / 2
    focal_length_x: f32,
    // F_y <- I_y / tan(Fov_y / 2) / 2
    focal_length_y: f32,
    // I_x / 2
    image_size_half_x: f32,
    // I_y / 2
    image_size_half_y: f32,
    // P
    point_count: u32,
    // I_x / T_x (0 ~ )
    tile_count_x: i32,
    // I_y / T_y (0 ~ )
    tile_count_y: i32,
    // tan(Fov_x / 2) * (C_f + 1)
    view_bound_x: f32,
    // tan(Fov_y / 2) * (C_f + 1)
    view_bound_y: f32,
    // V[3]
    view_position: vec3<f32>,
    // Rv[3, 3]
    view_rotation: mat3x3<f32>,
    // Tv[3, 1]
    view_translation: vec3<f32>,
}

@group(0) @binding(0)
var<storage, read_write> arguments: Arguments;
// [P, 16, 3]
@group(0) @binding(1)
var<storage, read_write> colors_sh: array<array<array<f32, 3>, 16>>;
// [P, 3]
@group(0) @binding(2)
var<storage, read_write> positions_3d: array<array<f32, 3>>;
// [P, 4] (x, y, z, w) (Normalized)
@group(0) @binding(3)
var<storage, read_write> rotations: array<vec4<f32>>;
// [P, 3]
@group(0) @binding(4)
var<storage, read_write> scalings: array<array<f32, 3>>;

// [P, 3] (0.0, 1.0)
@group(0) @binding(5)
var<storage, read_write> colors_rgb_3d: array<array<f32, 3>>;
// [P, 2, 2] (Symmetric)
@group(0) @binding(6)
var<storage, read_write> conics: array<mat2x2<f32>>;
// [P] (0 ~ )
@group(0) @binding(7)
var<storage, read_write> depths: array<f32>;
// [P, 3] (0.0, 1.0)
@group(0) @binding(8)
var<storage, read_write> is_colors_rgb_3d_not_clamped: array<array<f32, 3>>;
// [P, 2]
@group(0) @binding(9)
var<storage, read_write> positions_2d: array<vec2<f32>>;
// [P]
@group(0) @binding(10)
var<storage, read_write> radii: array<u32>;
// [P]
@group(0) @binding(11)
var<storage, read_write> tile_touched_counts: array<u32>;
// [P, 4] (x max, x min, y max, y min)
@group(0) @binding(12)
var<storage, read_write> tiles_touched_bound: array<vec4<u32>>;

// The real coefficients of orthonormalized spherical harmonics from degree 0 to 3
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

// The depth range is restricted by 16-bit depth order for sorting
const DEPTH_MAX: f32 = f32(1u << (17 - 4));
const DEPTH_MIN: f32 = 1.0 / f32(1u << (4 - 1));
// The `r` for `OPACITY_2D_MAX = ∫[-r, r] e^(-0.5 * x^2) dx / √2π`
const FACTOR_RADIUS: f32 = 2.5826694;
// C_f
const FILTER_LOW_PASS: f32 = 0.3;
// T_x
const TILE_SIZE_X: f32 = 16.0;
// T_y
const TILE_SIZE_Y: f32 = 16.0;
const GROUP_SIZE: u32 = 256;

@compute @workgroup_size(GROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    // Specifying the index

    // (0 ~ P)
    let index = global_id.x;
    if index >= arguments.point_count {
        return;
    }

    // Initializing the results

    radii[index] = u32();
    tile_touched_counts[index] = u32();

    // Transforming the 3D position from world space to view space
    // Pv[3, 1] = Rv[3, 3] * Pw[3, 1] + Tv[3, 1]

    let position_3d = vec_from_array_f32_3(positions_3d[index]);
    let view_rotation = arguments.view_rotation;
    let position_3d_in_view = view_rotation * position_3d + arguments.view_translation;
    let depth = position_3d_in_view.z;

    // Performing viewing-frustum culling

    if depth < DEPTH_MIN || depth >= DEPTH_MAX {
        return;
    }

    // Converting the quaternion to rotation matrix
    // R[3, 3] (Symmetric) = Q[4] (x, y, z, w)
    // R[3, 3] = [[-y * y - z * z + 0.5,  x * y - w * z,        x * z + w * y      ]
    //            [ x * y + w * z,       -x * x - z * z + 0.5,  y * z - w * x      ]
    //            [ x * z - w * y,        y * z + w * x,       -x * x - y * y + 0.5]] * 2

    let rotation = rotations[index];
    let q_wx = rotation.w * rotation.x;
    let q_wy = rotation.w * rotation.y;
    let q_wz = rotation.w * rotation.z;
    let q_xx = rotation.x * rotation.x;
    let q_xy = rotation.x * rotation.y;
    let q_xz = rotation.x * rotation.z;
    let q_yy = rotation.y * rotation.y;
    let q_yz = rotation.y * rotation.z;
    let q_zz = rotation.z * rotation.z;

    // Computing the 3D covariance matrix from rotation and scaling
    // RS[3, 3] = R[3, 3] * S[3, 3]
    // Σ[3, 3] (Symmetric) = RS[3, 3] * RS^t[3, 3]
    // 
    // S is diagonal
    // 
    // ** Advanced **
    // 
    // This is derived from inverse Principal Component Analysis
    // of the 3D covariance matrix:
    // 
    // Defining covariance matrix using Singular Value Decomposition:
    // 
    // Σ = V * L * V^-1
    // 
    // Defining rotation matrix R as eigenvectors of Σ:
    // 
    // R = V
    // R^t = R^-1 = V^-1
    // 
    // Defining scaling matrix S as square root of eigenvalues of Σ:
    // 
    // S = √L

    let rotation_matrix = 2.0 * mat3x3<f32>(
        (- q_yy - q_zz) + 0.5, (q_xy + q_wz), (q_xz - q_wy),
        (q_xy - q_wz), (- q_xx - q_zz) + 0.5, (q_yz + q_wx),
        (q_xz + q_wy), (q_yz - q_wx), (- q_xx - q_yy) + 0.5,
    );
    let scaling = scalings[index];
    let rotation_scaling = mat3x3<f32>(
        rotation_matrix[0] * scaling[0],
        rotation_matrix[1] * scaling[1],
        rotation_matrix[2] * scaling[2],
    );
    let covariance_3d = rotation_scaling * transpose(rotation_scaling);

    // Transforming the 3D position to 2D position (view => normalized => clip => screen)
    // Pv'[2, 1] <- Pv[3, 1]
    // Pv'[2, 1] = [f.x * Pv.x / Pv.z + (I.x * 0.5 - 0.5)
    //              f.y * Pv.y / Pv.z + (I.y * 0.5 - 0.5)]

    let focal_length = vec2<f32>(arguments.focal_length_x, arguments.focal_length_y);
    let position_3d_in_normalized = position_3d_in_view.xy / depth;
    let position_3d_in_clip = position_3d_in_normalized * focal_length;
    let position_2d = position_3d_in_clip + vec2<f32>(
        arguments.image_size_half_x,
        arguments.image_size_half_y,
    ) - 0.5;

    // Projecting the 3D covariance matrix into 2D covariance matrix
    // T[2, 3] = J[2, 3] * Rv[3, 3]
    // Σ'[2, 2] (Symmetric) = T[2, 3] * Σ[3, 3] * T^t[3, 2] + F[2, 2]
    //
    // J = [[f.x / Pv.z, 0,          -f.x * Pv.x / Pv.z^2]
    //      [0,          f.y / Pv.z, -f.y * Pv.y / Pv.z^2]]
    // F = [[C_f, 0  ]
    //      [0,   C_f]]
    //
    // Pv.x and Pv.y are the clamped

    let focal_length_normalized = focal_length / depth;
    let position_3d_in_normalized_clamped = clamp(
        position_3d_in_normalized,
        vec2<f32>(-arguments.view_bound_x, -arguments.view_bound_y),
        vec2<f32>(arguments.view_bound_x, arguments.view_bound_y),
    );
    let projection_affine = mat3x2<f32>(
        vec2<f32>(focal_length_normalized.x, 0.0),
        vec2<f32>(0.0, focal_length_normalized.y),
        -focal_length_normalized * position_3d_in_normalized_clamped,
    );
    let transform_2d = projection_affine * view_rotation;
    let covariance_2d = transform_2d * covariance_3d * transpose(transform_2d) + mat2x2<f32>(
        FILTER_LOW_PASS, 0.0,
        0.0, FILTER_LOW_PASS,
    );

    // Computing the inverse of the 2D covariance matrix
    // Σ'^-1[2, 2] (Symmetric) <- Σ'[2, 2]

    let covariance_2d_det = determinant(covariance_2d);
    if covariance_2d_det == 0.0 {
        return;
    }
    let covariance_2d_01_n = -covariance_2d[0][1];
    let conic = (1.0 / covariance_2d_det) * mat2x2<f32>(
        covariance_2d[1][1], covariance_2d_01_n,
        covariance_2d_01_n, covariance_2d[0][0],
    );

    // Computing the max radius using the 2D covariance matrix
    // r <- Σ'[2, 2]
    // 
    // ** Advanced **
    // 
    // This is derived from the eigendecomposition of the 2D covariance matrix:
    // 
    // Σ' = [[a, b]
    //       [b, c]]
    // 
    // det(Σ) = a * c - b^2
    // 
    // Deriving eigenvalues:
    // 
    // det(Σ - λ * I) = 0
    // λ = ((a + c) ± √((a + c)^2 - 4 * (a * c - b^2))) / 2
    //   = (a + c) / 2 ± √(((a + c) / 2)^2 - det(Σ))
    // 
    // Defining radius as the square root of the maximum eigenvalue:
    // 
    // r = √λ_max

    let covariance_2d_diag_mean = (covariance_2d[0][0] + covariance_2d[1][1]) / 2.0;
    let eigenvalue_difference2 =
        covariance_2d_diag_mean * covariance_2d_diag_mean - covariance_2d_det;
    let eigenvalue_difference = sqrt(max(eigenvalue_difference2, 0.0));
    let eigenvalue_max = max(
        covariance_2d_diag_mean + eigenvalue_difference,
        covariance_2d_diag_mean - eigenvalue_difference,
    );
    let radius = ceil(sqrt(eigenvalue_max) * FACTOR_RADIUS);

    // Checking the tiles touched
    // (x max, x min, y max, y min)

    let tile_touched_bound = bitcast<vec4<u32>>(
        clamp(
            vec4<i32>(
                i32((position_2d.x + radius + TILE_SIZE_X - 1.0) / TILE_SIZE_X),
                i32((position_2d.x - radius) / TILE_SIZE_X),
                i32((position_2d.y + radius + TILE_SIZE_Y - 1.0) / TILE_SIZE_Y),
                i32((position_2d.y - radius) / TILE_SIZE_Y),
            ),
            vec4<i32>(),
            vec4<i32>(
                arguments.tile_count_x, arguments.tile_count_x,
                arguments.tile_count_y, arguments.tile_count_y,
            ),
        )
    );
    let tile_point_count =
        (tile_touched_bound[0] - tile_touched_bound[1]) *
        (tile_touched_bound[2] - tile_touched_bound[3]);

    // Leaving if no tile is touched

    if tile_point_count == 0 {
        return;
    }

    // Computing the view direction in world space
    // Dv[3] <- Ov[3] = Pw[3] - V[3]

    let view_offset = position_3d - arguments.view_position;
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
    // D[16] <- Dv[3]
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

        color_rgb_3d += color_sh[1] * (SH_C_1[0] * (vd_y));
        color_rgb_3d += color_sh[2] * (SH_C_1[1] * (vd_z));
        color_rgb_3d += color_sh[3] * (SH_C_1[2] * (vd_x));
    }

    if arguments.colors_sh_degree_max >= 2 {
        vd_xx = vd_x * vd_x;
        vd_xy = vd_x * vd_y;
        vd_yy = vd_y * vd_y;
        vd_zz = vd_z * vd_z;

        color_rgb_3d += color_sh[4] * (SH_C_2[0] * (vd_xy));
        color_rgb_3d += color_sh[5] * (SH_C_2[1] * (vd_y * vd_z));
        color_rgb_3d += color_sh[6] * (SH_C_2[2] * (vd_zz * 3.0 - 1.0));
        color_rgb_3d += color_sh[7] * (SH_C_2[3] * (vd_x * vd_z));
        color_rgb_3d += color_sh[8] * (SH_C_2[4] * (vd_xx - vd_yy));
    }

    if arguments.colors_sh_degree_max >= 3 {
        vd_zz_5_1 = vd_zz * 5.0 - 1.0;

        color_rgb_3d += color_sh[9u] * (SH_C_3[0] * (vd_y * (vd_xx * 3.0 - vd_yy)));
        color_rgb_3d += color_sh[10] * (SH_C_3[1] * (vd_z * (vd_xy)));
        color_rgb_3d += color_sh[11] * (SH_C_3[2] * (vd_y * (vd_zz_5_1)));
        color_rgb_3d += color_sh[12] * (SH_C_3[3] * (vd_z * (vd_zz_5_1 - 2.0)));
        color_rgb_3d += color_sh[13] * (SH_C_3[4] * (vd_x * (vd_zz_5_1)));
        color_rgb_3d += color_sh[14] * (SH_C_3[5] * (vd_z * (vd_xx - vd_yy)));
        color_rgb_3d += color_sh[15] * (SH_C_3[6] * (vd_x * (vd_xx - vd_yy * 3.0)));
    }

    color_rgb_3d += 0.5;

    let is_color_rgb_3d_not_clamped = color_rgb_3d >= vec3<f32>();
    color_rgb_3d = select(vec3<f32>(), color_rgb_3d, is_color_rgb_3d_not_clamped);

    // Specifying the results

    // [P, 3]
    colors_rgb_3d[index] = array_from_vec3_f32(color_rgb_3d);
    // [P, 2, 2]
    conics[index] = conic;
    // [P]
    depths[index] = depth;
    // [P, 3]
    is_colors_rgb_3d_not_clamped[index] = array_from_vec3_f32(
        vec3<f32>(is_color_rgb_3d_not_clamped)
    );
    // [P, 2]
    positions_2d[index] = position_2d;
    // [P]
    radii[index] = u32(radius);
    // [P]
    tile_touched_counts[index] = tile_point_count;
    // [P, 4]
    tiles_touched_bound[index] = tile_touched_bound;
}

fn array_from_vec3_f32(v: vec3<f32>) -> array<f32, 3> {
    return array<f32, 3>(v[0], v[1], v[2]);
}

fn vec_from_array_f32_3(a: array<f32, 3>) -> vec3<f32> {
    return vec3<f32>(a[0], a[1], a[2]);
}
