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
// [P, 3]
@group(0) @binding(1)
var<storage, read_write> colors_rgb_3d_grad: array<array<f32, 3>>;
// [P, 16, 3]
@group(0) @binding(2)
var<storage, read_write> colors_sh: array<array<array<f32, 3>, 16>>;
// [P, 3] (Symmetric mat2x2)
@group(0) @binding(3)
var<storage, read_write> conics: array<array<f32, 3>>;
// [P, 3] (Symmetric mat2x2)
@group(0) @binding(4)
var<storage, read_write> conics_grad: array<array<f32, 3>>;
// [P]
@group(0) @binding(5)
var<storage, read_write> depths: array<f32>;
// [P, 3] (0.0, 1.0)
@group(0) @binding(6)
var<storage, read_write> is_colors_rgb_3d_not_clamped: array<array<f32, 3>>;
// [P, 2]
@group(0) @binding(7)
var<storage, read_write> positions_2d_grad: array<vec2<f32>>;
// [P, 3]
@group(0) @binding(8)
var<storage, read_write> positions_3d: array<array<f32, 3>>;
// [P, 2]
@group(0) @binding(9)
var<storage, read_write> positions_3d_in_normalized: array<vec2<f32>>;
// [P]
@group(0) @binding(10)
var<storage, read_write> radii: array<u32>;
// [P, 4] (x, y, z, w) (Inner)
@group(0) @binding(11)
var<storage, read_write> rotations: array<vec4<f32>>;
// [P, 3 (+ 1), 3]
// TODO: Compact layout
@group(0) @binding(12)
var<storage, read_write> rotations_matrix: array<mat3x3<f32>>;
// [P, 3] (Inner)
@group(0) @binding(13)
var<storage, read_write> scalings: array<array<f32, 3>>;

// [P, 16, 3]
@group(0) @binding(14)
var<storage, read_write> colors_sh_grad: array<array<array<f32, 3>, 16>>;
// [P]
@group(0) @binding(15)
var<storage, read_write> positions_2d_grad_norm: array<f32>;
// [P, 3]
@group(0) @binding(16)
var<storage, read_write> positions_3d_grad: array<array<f32, 3>>;
// [P, 4] (x, y, z, w) (Inner)
@group(0) @binding(17)
var<storage, read_write> rotations_grad: array<vec4<f32>>;
// [P, 3] (Inner)
@group(0) @binding(18)
var<storage, read_write> scalings_grad: array<array<f32, 3>>;

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

const GROUP_SIZE: u32 = GROUP_SIZE_X * GROUP_SIZE_Y;
const GROUP_SIZE_X: u32 = 16;
const GROUP_SIZE_Y: u32 = 16;

@compute @workgroup_size(GROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    // Specifying the index

    // (0 ~ P)
    let index = global_id.x;
    if index >= arguments.point_count || radii[index] == 0u {
        return;
    }

    // Computing the gradients
    //
    // ∂I = ∂(Σ'^-1 * Σ')
    //    = Σ'^-1 * ∂Σ' + ∂Σ'^-1 * Σ'
    // ∂Σ'^-1 = -Σ'^-1 * ∂Σ' * Σ'^-1
    // ∂L =〈∂L/∂Σ'^-1, ∂Σ'^-1〉
    //    =〈∂L/∂Σ'^-1, -Σ'^-1 * ∂Σ' * Σ'^-1〉
    //    =〈-Σ'^-1^t * ∂L/∂Σ'^-1 * Σ'^-1^t, ∂Σ'〉
    //
    // ∂L/∂Σ'[2, 2] = -Σ'^-1^t[2, 2] * ∂L/∂Σ'^-1[2, 2] * Σ'^-1^t[2, 2]
    //              = -Σ'^-1[2, 2] * ∂L/∂Σ'^-1[2, 2] * Σ'^-1[2, 2]
    //
    // Σ' and Σ'^-1 are symmetric

    let conic = mat_sym_from_array_f32_3(conics[index]);
    let conic_grad = mat_sym_from_array_f32_3(conics_grad[index]);
    // ∂L/∂Σ'[2, 2]
    let covariance_2d_grad = -1.0 * conic * conic_grad * conic;

    // Computing the 3D covariance matrix from rotation and scaling
    // 
    // RS[3, 3] = R[3, 3] * S[3, 3]
    // Σ[3, 3] (Symmetric) = RS[3, 3] * RS^t[3, 3]
    // 
    // S is diagonal

    let rotation_matrix = rotations_matrix[index];
    // (Outer)
    let scaling = exp(vec_from_array_f32_3(scalings[index]));
    let rotation_scaling = mat3x3<f32>(
        rotation_matrix[0] * scaling[0],
        rotation_matrix[1] * scaling[1],
        rotation_matrix[2] * scaling[2],
    );
    let covariance_3d = rotation_scaling * transpose(rotation_scaling);

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

    let depth = depths[index];
    let focal_length = vec2<f32>(arguments.focal_length_x, arguments.focal_length_y);
    let focal_length_normalized = focal_length / depth;
    let view_bound = vec2<f32>(arguments.view_bound_x, arguments.view_bound_y);
    let position_3d_in_normalized = positions_3d_in_normalized[index];
    // [Pv.x / Pv.z, Pv.y / Pv.z]
    let position_3d_in_normalized_clamped = clamp(
        position_3d_in_normalized,
        -view_bound,
        view_bound,
    );
    let view_rotation = arguments.view_rotation;
    let projection_2d = mat3x2<f32>(
        vec2<f32>(focal_length_normalized.x, 0.0),
        vec2<f32>(0.0, focal_length_normalized.y),
        -focal_length_normalized * position_3d_in_normalized_clamped,
    ) * view_rotation;

    // Computing the gradients
    //
    // ∂L =〈∂L/∂Σ', ∂Σ'〉
    //    =〈∂L/∂Σ', ∂(T * Σ * T^t)〉
    //    =〈∂L/∂Σ', ∂T * Σ * T^t + T * ∂Σ * T^t + T * Σ * ∂T^t〉
    //    =〈∂L/∂Σ' * T * Σ^t, ∂T〉+〈T^t * ∂L/∂Σ' * T, ∂Σ〉+〈Σ^t * T^t * ∂L/∂Σ', ∂T^t〉
    //    =〈∂L/∂Σ' * T * Σ^t + (∂L/∂Σ')^t * T * Σ, ∂T〉+〈T^t * ∂L/∂Σ' * T, ∂Σ〉
    // ∂L =〈∂L/∂T, ∂T〉
    //    =〈∂L/∂T, ∂J * Rv + J * ∂Rv〉
    //    =〈∂L/∂T * Rv^t, ∂J〉+〈J^t * ∂L/∂T, ∂Rv〉
    //
    // ∂L/∂Σ[3, 3] = T^t[3, 2] * ∂L/∂Σ'[2, 2] * T[2, 3]
    // ∂L/∂T[2, 3] = (∂L/∂Σ')[2, 2] * T[2, 3] * Σ^t[3, 3] +
    //               (∂L/∂Σ')^t[2, 2] * T[2, 3] * Σ[3, 3]
    //             = ∂L/∂Σ'[2, 2] * T[2, 3] * Σ[3, 3] * 2
    // ∂L/∂J[2, 3] = ∂L/∂T[2, 3] * Rv^t[3, 3]
    //
    // Σ and Σ' are symmetric

    // ∂L/∂Σ[3, 3]
    let covariance_3d_grad = transpose(projection_2d) * covariance_2d_grad * projection_2d;
    // ∂L/∂T[2, 3]
    let projection_2d_grad = 2.0 * covariance_2d_grad * projection_2d * covariance_3d;
    // ∂L/∂J[2, 3]
    let projection_2d_left_grad = projection_2d_grad * transpose(view_rotation);

    // Computing the gradients
    //
    // ∂L/∂Pv[3] = [∂L/∂Pv.x, ∂L/∂Pv.y, ∂L/∂Pv.z]
    //           = [〈∂L/∂J, ∂J/∂Pv.x〉,〈∂L/∂J, ∂J/∂Pv.y〉,〈∂L/∂J, ∂J/∂Pv.z〉]
    // ∂J/∂Pv.x[2, 3] = [[0, 0, -f.x / Pv.z^2]
    //                   [0, 0,  0           ]]
    // ∂J/∂Pv.x[2, 3] = [[0, 0,  0           ]
    //                   [0, 0, -f.y / Pv.z^2]]
    // ∂J/∂Pv.z[2, 3] = [[-f.x / Pv.z^2,  0,            2 * f.x * Pv.x / Pv.z^3]
    //                   [ 0,            -f.y / Pv.z^2, 2 * f.y * Pv.y / Pv.z^3]]

    let is_position_3d_in_normalized_not_clamped = vec2<f32>(
        position_3d_in_normalized == position_3d_in_normalized_clamped
    );
    // [f.x / Pv.z^2, f.y / Pv.z^2]
    let focal_length_normalized_p2 = focal_length_normalized / depth;
    // [f.x / Pv.z^2 * (∂L/∂J).2,0, f.y / Pv.z^2 * (∂L/∂J).2,1]
    let focal_length_normalized_p2_projection_2d_left_grad_2 =
        focal_length_normalized_p2 * projection_2d_left_grad[2];
    // ∂L/∂Pv[3]
    let position_3d_in_view_grad = vec3<f32>(
        - is_position_3d_in_normalized_not_clamped.x *
          focal_length_normalized_p2_projection_2d_left_grad_2.x,
        - is_position_3d_in_normalized_not_clamped.y *
          focal_length_normalized_p2_projection_2d_left_grad_2.y,
        - focal_length_normalized_p2.x * projection_2d_left_grad[0][0]
        - focal_length_normalized_p2.y * projection_2d_left_grad[1][1]
        + 2 * dot(
            position_3d_in_normalized_clamped,
            focal_length_normalized_p2_projection_2d_left_grad_2,
        ),
    );

    // Computing the gradients
    //
    // ∂L =〈∂L/∂Pv, ∂Pv〉
    //    =〈∂L/∂Pv, ∂(Rv * Pw + Tv)〉
    //    =〈∂L/∂Pv, ∂Rv * Pw + Rv * ∂Pw〉
    //    =〈∂L/∂Pv * Pw^t, ∂Rv〉+〈Rv^t * ∂L/∂Pv, ∂Pw〉
    //
    // ∂L/∂Pw[3, 1]   = Rv^t[3, 3] * ∂L/∂Pv[3, 1]
    // ∂L/∂Pw^t[1, 3] = ∂L/∂Pv^t[1, 3] * Rv[3, 3]

    var position_3d_grad = position_3d_in_view_grad * view_rotation;

    // Computing the gradients
    //
    // ∂L =〈∂L/∂Σ, ∂Σ〉
    //    =〈∂L/∂Σ, ∂(RS * RS^t)〉
    //    =〈∂L/∂Σ, ∂RS * (RS)^t + RS * ∂RS^t〉
    //    =〈∂L/∂Σ * RS, ∂RS〉+〈RS^t * ∂L/∂Σ, ∂RS^t〉
    //    =〈∂L/∂Σ * RS, ∂RS〉+〈(∂L/∂Σ)^t * RS, ∂RS〉
    //    =〈2 * ∂L/∂Σ * RS, ∂RS〉
    //
    // ∂L/∂RS[3, 3] = 2 * ∂L/∂Σ[3, 3] * RS[3, 3]
    //
    // Σ is symmetric

    let rotation_scaling_grad = 2.0 * covariance_3d_grad * rotation_scaling;

    // Computing the gradients
    //
    // ∂L =〈∂L/∂RS, ∂RS〉
    //    =〈∂L/∂RS, ∂R * S + R * ∂S〉
    //    =〈∂L/∂RS * S^t, ∂R〉+〈R^t * ∂L/∂RS, ∂S〉
    //
    // ∂L/∂R[3, 3] = ∂L/∂RS[3, 3] * S^t[3, 3]
    // ∂L/∂S[3, 3] = R^t[3, 3] * ∂L/∂RS[3, 3]
    //
    // S is diagonal

    // ∂L/∂R[3, 3]
    let rotation_matrix_grad = mat3x3<f32>(
        rotation_scaling_grad[0] * scaling[0],
        rotation_scaling_grad[1] * scaling[1],
        rotation_scaling_grad[2] * scaling[2],
    );
    // ∂L/∂S[3, 3] (Inner)
    let scaling_grad = exp_grad_vec_f32_3(scaling) * vec3<f32>(
        dot(rotation_matrix[0], rotation_scaling_grad[0]),
        dot(rotation_matrix[1], rotation_scaling_grad[1]),
        dot(rotation_matrix[2], rotation_scaling_grad[2]),
    );

    // Computing the gradients
    //
    // ∂L/∂Q[4] = [∂L/∂Q.x, ∂L/∂Q.y, ∂L/∂Q.z, ∂L/∂Q.w]
    //          = [∂L/∂R * ∂R/∂Q.x, ∂L/∂R * ∂R/∂Q.y, ∂L/∂R * ∂R/∂Q.z, ∂L/∂R * ∂R/∂Q.w]
    // ∂R/∂Q.x[3, 3] = [[0,    Q.y,      Q.z    ]
    //                  [Q.y, -2 * Q.x, -Q.w    ]
    //                  [Q.z,  Q.w,     -2 * Q.x]] * 2
    // ∂R/∂Q.y[3, 3] = [[-2 * Q.y, Q.x,  Q.w    ]
    //                  [ Q.x,     0,    Q.z    ]
    //                  [-Q.w,     Q.z, -2 * Q.y]] * 2
    // ∂R/∂Q.z[3, 3] = [[-2 * Q.z, -Q.w,     Q.x]
    //                  [ Q.w,     -2 * Q.z, Q.y]
    //                  [ Q.x,      Q.y,     0  ]] * 2
    // ∂R/∂Q.w[3, 3] = [[ 0,  -Q.z, Q.y]
    //                  [ Q.z, 0,  -Q.x]
    //                  [-Q.y, Q.x, 0  ]] * 2

    let rotation_inner = rotations[index];
    // (Outer)
    let q = normalize(rotation_inner);
    let q_n_2 = -2.0 * q;
    let q_n_w = q_n_2.w / 2.0;

    // ∂L/∂Q[4] (Inner)
    let rotation_grad = normalize_grad_vec_f32_4(rotation_inner) * vec4<f32>(
        dot(rotation_matrix_grad[0], vec3<f32>(0.0, q.y, q.z)) +
        dot(rotation_matrix_grad[1], vec3<f32>(q.y, q_n_2.x, q.w)) +
        dot(rotation_matrix_grad[2], vec3<f32>(q.z, q_n_w, q_n_2.x)),

        dot(rotation_matrix_grad[0], vec3<f32>(q_n_2.y, q.x, q_n_w)) +
        dot(rotation_matrix_grad[1], vec3<f32>(q.x, 0.0, q.z)) +
        dot(rotation_matrix_grad[2], vec3<f32>(q.w, q.z, q_n_2.y)),

        dot(rotation_matrix_grad[0], vec3<f32>(q_n_2.z, q.w, q.x)) +
        dot(rotation_matrix_grad[1], vec3<f32>(q_n_w, q_n_2.z, q.y)) +
        dot(rotation_matrix_grad[2], vec3<f32>(q.x, q.y, 0.0)),

        dot(rotation_matrix_grad[0], vec3<f32>(0.0, q.z, -q.y)) +
        dot(rotation_matrix_grad[1], vec3<f32>(-q.z, 0.0, q.x)) +
        dot(rotation_matrix_grad[2], vec3<f32>(q.y, -q.x, 0.0)),
    ) * 2.0;

    // Computing the gradients
    //
    // ∂L/∂Pw[1, 3] = ∂L/∂Pv'[1, 2] * ∂Pv'/∂Pv[2, 3] * ∂Pv/∂Pw[3, 3]
    // ∂Pv/∂Pw[3, 3] = Rv
    // ∂Pv'/∂Pv[2, 3] = [[f.x / Pv.z, 0,          -f.x * Pv.x / Pv.z^2]
    //                   [0,          f.y / Pv.z, -f.y * Pv.y / Pv.z^2]]

    // ∂L/∂Pv'[1, 2]
    let position_2d_grad = positions_2d_grad[index];
    // ∂Pv'/∂Pv[2, 3]
    let position_2d_to_3d_in_view_grad = mat3x2<f32>(
        vec2<f32>(focal_length_normalized.x, 0.0),
        vec2<f32>(0.0, focal_length_normalized.y),
        -focal_length_normalized * position_3d_in_normalized,
    );
    // ∂L/∂Pw[1, 3]
    position_3d_grad +=
        position_2d_grad * position_2d_to_3d_in_view_grad *
        view_rotation;

    // Computing the norm of 2D positions gradient
    // |∂L/∂Pv'[1, 2] * [I_x / 2, I_y / 2]|

    let position_2d_grad_norm = length(
        position_2d_grad *
        vec2<f32>(arguments.image_size_half_x, arguments.image_size_half_y)
    );

    // Computing the view direction in world space
    // Ov[3] = Pw[3] - V[3]
    // Dv[3] = Ov[3] / |Ov|

    let position_3d = vec_from_array_f32_3(positions_3d[index]);
    let view_offset = position_3d - arguments.view_position;
    let view_direction = normalize(view_offset);
    var vd_x = f32();
    var vd_y = f32();
    var vd_z = f32();
    var vd_xx = f32();
    var vd_xy = f32();
    var vd_xz = f32();
    var vd_yy = f32();
    var vd_yz = f32();
    var vd_zz = f32();
    var vd_xx_yy = f32();
    var vd_xx_yy_3 = f32();
    var vd_z_10 = f32();
    var vd_zz_5_1 = f32();

    // Computing the gradients
    //
    // ∂L =〈∂L/∂C_rgb, ∂D * C_sh + D * ∂C_sh〉
    //    =〈∂L/∂C_rgb * C_sh^t, ∂D〉+〈D^t * ∂L/∂C_rgb, ∂C_sh〉
    //
    // ∂L/∂C_sh[16, 3] = D^t[16, 1] * ∂L/∂C_rgb[1, 3]
    //
    // ∂L/∂D[1, 16] = ∂L/∂C_rgb[1, 3] * C_sh^t[3, 16]
    // ∂L/∂Dv[1, 3] = ∂L/∂D[1, 16] * ∂D/∂Dv[16, 3]
    //              = ∂L/∂C_rgb[1, 3] * (C_sh^t[3, 16] * ∂D/∂Dv[16, 3])
    //              = ∂L/∂C_rgb[1, 3] * ∂C_rgb/∂Dv[3, 3]
    // ∂C_rgb/∂Dv[3, 3] = C_sh^t[3, 16] * ∂D/∂Dv[16, 3]
    //
    // C_rgb was clamped

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

    // ∂L/∂C_rgb[1, 3]
    let color_rgb_3d_grad =
        vec_from_array_f32_3(colors_rgb_3d_grad[index]) *
        vec_from_array_f32_3(is_colors_rgb_3d_not_clamped[index]);
    // ∂L/∂C_sh[16, 3]
    var color_sh_grad = array<vec3<f32>, 16>();
    // ∂C_rgb/∂Dv[3, 3]
    var color_rgb_3d_to_view_direction_grad = mat3x3<f32>();

    color_sh_grad[0] = color_rgb_3d_grad * (SH_C_0[0]);

    if arguments.colors_sh_degree_max >= 1 {
        vd_x = view_direction.x;
        vd_y = view_direction.y;
        vd_z = view_direction.z;

        color_sh_grad[1] = color_rgb_3d_grad * (SH_C_1[0] * (vd_y));
        color_sh_grad[2] = color_rgb_3d_grad * (SH_C_1[1] * (vd_z));
        color_sh_grad[3] = color_rgb_3d_grad * (SH_C_1[2] * (vd_x));

        color_rgb_3d_to_view_direction_grad += mat3x3<f32>(
            color_sh[3] * (SH_C_1[2]),
            color_sh[1] * (SH_C_1[0]),
            color_sh[2] * (SH_C_1[1]),
        );
    }

    if arguments.colors_sh_degree_max >= 2 {
        vd_xx = vd_x * vd_x;
        vd_xy = vd_x * vd_y;
        vd_xz = vd_x * vd_z;
        vd_yy = vd_y * vd_y;
        vd_yz = vd_y * vd_z;
        vd_zz = vd_z * vd_z;
        vd_xx_yy = vd_xx - vd_yy;

        color_sh_grad[4] = color_rgb_3d_grad * (SH_C_2[0] * (vd_xy));
        color_sh_grad[5] = color_rgb_3d_grad * (SH_C_2[1] * (vd_yz));
        color_sh_grad[6] = color_rgb_3d_grad * (SH_C_2[2] * (vd_zz * 3.0 - 1.0));
        color_sh_grad[7] = color_rgb_3d_grad * (SH_C_2[3] * (vd_xz));
        color_sh_grad[8] = color_rgb_3d_grad * (SH_C_2[4] * (vd_xx_yy));

        color_rgb_3d_to_view_direction_grad += mat3x3<f32>(
            color_sh[4] * (SH_C_2[0] * (vd_y)) +
            color_sh[7] * (SH_C_2[3] * (vd_z)) +
            color_sh[8] * (SH_C_2[4] * (vd_x * 2.0)),

            color_sh[4] * (SH_C_2[0] * (vd_x)) +
            color_sh[5] * (SH_C_2[1] * (vd_z)) +
            color_sh[8] * (SH_C_2[4] * (vd_y * -2.0)),

            color_sh[5] * (SH_C_2[1] * (vd_y)) +
            color_sh[6] * (SH_C_2[2] * (vd_z * 6.0)) +
            color_sh[7] * (SH_C_2[3] * (vd_x)),
        );
    }

    if arguments.colors_sh_degree_max >= 3 {
        vd_xx_yy_3 = vd_xx_yy * 3.0;
        vd_z_10 = vd_z * 10.0;
        vd_zz_5_1 = vd_zz * 5.0 - 1.0;

        color_sh_grad[9u] = color_rgb_3d_grad * (SH_C_3[0] * (vd_y * (vd_xx * 3.0 - vd_yy)));
        color_sh_grad[10] = color_rgb_3d_grad * (SH_C_3[1] * (vd_z * (vd_xy)));
        color_sh_grad[11] = color_rgb_3d_grad * (SH_C_3[2] * (vd_y * (vd_zz_5_1)));
        color_sh_grad[12] = color_rgb_3d_grad * (SH_C_3[3] * (vd_z * (vd_zz_5_1 - 2.0)));
        color_sh_grad[13] = color_rgb_3d_grad * (SH_C_3[4] * (vd_x * (vd_zz_5_1)));
        color_sh_grad[14] = color_rgb_3d_grad * (SH_C_3[5] * (vd_z * (vd_xx_yy)));
        color_sh_grad[15] = color_rgb_3d_grad * (SH_C_3[6] * (vd_x * (vd_xx - vd_yy * 3.0)));

        color_rgb_3d_to_view_direction_grad += mat3x3<f32>(
            color_sh[9u] * (SH_C_3[0] * (vd_xy * 6.0)) +
            color_sh[10] * (SH_C_3[1] * (vd_yz)) +
            color_sh[13] * (SH_C_3[4] * (vd_zz_5_1)) +
            color_sh[14] * (SH_C_3[5] * (vd_xz * 2.0)) +
            color_sh[15] * (SH_C_3[6] * (vd_xx_yy_3)),

            color_sh[9u] * (SH_C_3[0] * (vd_xx_yy_3)) +
            color_sh[10] * (SH_C_3[1] * (vd_xz)) +
            color_sh[11] * (SH_C_3[2] * (vd_zz_5_1)) +
            color_sh[14] * (SH_C_3[5] * (vd_yz * -2.0)) +
            color_sh[15] * (SH_C_3[6] * (vd_xy * -6.0)),

            color_sh[10] * (SH_C_3[1] * (vd_xy)) +
            color_sh[11] * (SH_C_3[2] * (vd_y * vd_z_10)) +
            color_sh[12] * (SH_C_3[3] * (vd_zz_5_1 * 3.0)) +
            color_sh[13] * (SH_C_3[4] * (vd_x * vd_z_10)) +
            color_sh[14] * (SH_C_3[5] * (vd_xx_yy)),
        );
    }

    // Computing the gradients
    //
    // ∂L/∂Dv[1, 3] = ∂L/∂C_rgb[1, 3] * ∂C_rgb/∂Dv[3, 3]
    // ∂L/∂Pw[1, 3] = ∂L/∂Dv[1, 3] * ∂Dv/∂Ov[3, 3] * ∂Ov/∂Pw[3, 3]
    // ∂Ov/∂Pw[3, 3] = I
    // ∂Dv/∂Ov[3, 3] = ∂(Ov / |Ov|)/∂Ov

    // ∂L/∂Dv[1, 3]
    let view_direction_grad = color_rgb_3d_grad * color_rgb_3d_to_view_direction_grad;
    // ∂L/∂Pw[1, 3]
    position_3d_grad += view_direction_grad * normalize_grad_vec_f32_3(view_offset);

    // Specifying the results

    // [P, 16, 3]
    colors_sh_grad[index] = array<array<f32, 3>, 16>(
        array_from_vec_f32_3(color_sh_grad[0u]),
        array_from_vec_f32_3(color_sh_grad[1u]),
        array_from_vec_f32_3(color_sh_grad[2u]),
        array_from_vec_f32_3(color_sh_grad[3u]),
        array_from_vec_f32_3(color_sh_grad[4u]),
        array_from_vec_f32_3(color_sh_grad[5u]),
        array_from_vec_f32_3(color_sh_grad[6u]),
        array_from_vec_f32_3(color_sh_grad[7u]),
        array_from_vec_f32_3(color_sh_grad[8u]),
        array_from_vec_f32_3(color_sh_grad[9u]),
        array_from_vec_f32_3(color_sh_grad[10]),
        array_from_vec_f32_3(color_sh_grad[11]),
        array_from_vec_f32_3(color_sh_grad[12]),
        array_from_vec_f32_3(color_sh_grad[13]),
        array_from_vec_f32_3(color_sh_grad[14]),
        array_from_vec_f32_3(color_sh_grad[15]),
    );
    // [P]
    positions_2d_grad_norm[index] = position_2d_grad_norm;
    // [P, 3]
    positions_3d_grad[index] = array_from_vec_f32_3(position_3d_grad);
    // [P, 4]
    rotations_grad[index] = rotation_grad;
    // [P, 3]
    scalings_grad[index] = array_from_vec_f32_3(scaling_grad);
}

fn array_from_vec_f32_3(v: vec3<f32>) -> array<f32, 3> {
    return array<f32, 3>(v[0], v[1], v[2]);
}

// y = e^x
// dy/dx = e^x = y
fn exp_grad_vec_f32_3(y: vec3<f32>) -> vec3<f32> {
    return y;
}

fn mat_sym_from_array_f32_3(a: array<f32, 3>) -> mat2x2<f32> {
    return mat2x2<f32>(a[0], a[1], a[1], a[2]);
}

// Y[3] = X[3] / |X|
// dot = X^t * X
// ∂Y/∂X = [[-X.x^2 + dot, -X.y   * X.x, -X.z   * X.x]
//          [-X.x   * X.y, -X.y^2 + dot, -X.z   * X.y]
//          [-X.x   * X.z, -X.y   * X.z, -X.z^2 + dot]]
//       * dot^(-3 / 2)
fn normalize_grad_vec_f32_3(x: vec3<f32>) -> mat3x3<f32> {
    let x_p2 = x * x;
    let x_dot = x_p2.x + x_p2.y + x_p2.z;
    let x_n_c2 = -x.xyz * x.yzx;
    return mat3x3<f32>(
        x_dot - x_p2.x, x_n_c2.x, x_n_c2.z,
        x_n_c2.x, x_dot - x_p2.y, x_n_c2.y,
        x_n_c2.z, x_n_c2.y, x_dot - x_p2.z,
    ) * (inverseSqrt(x_dot) / x_dot);
}

// Y[4] = X[4] / |X|
// dot = X^t * X
// ∂Y/∂X = [[-X.x^2 + dot, -X.y   * X.x, -X.z   * X.x, -X.w   * X.x]
//          [-X.x   * X.y, -X.y^2 + dot, -X.z   * X.y, -X.w   * X.y]
//          [-X.x   * X.z, -X.y   * X.z, -X.z^2 + dot, -X.w   * X.z]
//          [-X.x   * X.w, -X.y   * X.w, -X.z   * X.w, -X.w^2 + dot]]
//       * d^(-3 / 2)
fn normalize_grad_vec_f32_4(x: vec4<f32>) -> mat4x4<f32> {
    let x_p2 = x * x;
    let x_dot = x_p2.x + x_p2.y + x_p2.z + x_p2.w;
    let x_n_c2 = -x.xyz * x.yzx;
    let x_n_c2_w = -x.xyz * x.www;
    return mat4x4<f32>(
        x_dot - x_p2.x, x_n_c2.x, x_n_c2.z, x_n_c2_w.x,
        x_n_c2.x, x_dot - x_p2.y, x_n_c2.y, x_n_c2_w.y,
        x_n_c2.z, x_n_c2.y, x_dot - x_p2.z, x_n_c2_w.z,
        x_n_c2_w.x, x_n_c2_w.y, x_n_c2_w.z, x_dot - x_p2.w,
    ) * (inverseSqrt(x_dot) / x_dot);
}

fn vec_from_array_f32_3(a: array<f32, 3>) -> vec3<f32> {
    return vec3<f32>(a[0], a[1], a[2]);
}
