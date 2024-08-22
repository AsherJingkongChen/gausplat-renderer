struct Arguments {
    colors_sh_degree_max: u32,
    // P
    point_count: u32,
}

@group(0) @binding(0)
var<storage, read> arguments: Arguments;
// [P, 3 (+ 1)]
@group(0) @binding(1)
var<storage, read> colors_rgb_3d_grad: array<vec3<f32>>; // dL_dcolor // HIDE
// [P, 16, 3]
@group(0) @binding(2)
var<storage, read> colors_sh: array<array<array<f32, 3>, 16>>;
// [P, 3 (+ 1)]
@group(0) @binding(3)
var<storage, read> is_colors_rgb_3d_clamped: array<vec3<f32>>;
// [P]
@group(0) @binding(4)
var<storage, read> radii: array<u32>;
// [P, 3 (+ 1)]
@group(0) @binding(5)
var<storage, read> view_directions: array<vec3<f32>>;
// [P, 3 (+ 1)]
@group(0) @binding(6)
var<storage, read> view_offsets: array<vec3<f32>>;
// [P, 16, 3]
@group(0) @binding(7)
var<storage, read_write> colors_sh_grad: array<array<array<f32, 3>, 16>>; // dL_dshs // OUTPUT
// [P, 3]
@group(0) @binding(8)
var<storage, read_write> positions_3d_grad: array<array<f32, 3>>; // dL_dmeans // OUTPUT

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
    if index >= arguments.point_count || radii[index] != 0u {
        return;
    }

    // Loading the view direction in world space
    // Dv[3] <= Ov[3] = Pw[3] - V[3]

    let view_offset = view_offsets[index];
    let view_direction = view_directions[index];
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
    // ∂L =〈∂L/∂C_rgb, D * ∂C_sh + ∂D * C_sh〉
    //    =〈D^t * ∂L/∂C_rgb, ∂C_sh〉+〈∂L/∂C_rgb * C_sh^t, ∂D〉
    //
    // ∂L/∂C_sh[16, 3] = D^t[16, 1] * ∂L/∂C_rgb[1, 3]
    // ∂L/∂D[1, 16] = ∂L/∂C_rgb[1, 3] * C_sh^t[3, 16]
    // ∂L/∂Dv[1, 3] = ∂L/∂D[1, 16] * ∂D/∂Dv[16, 3]
    //              = ∂L/∂C_rgb[1, 3] * (C_sh^t[3, 16] * ∂D/∂Dv[16, 3])
    //              = ∂L/∂C_rgb[1, 3] * ∂C_rgb/∂Dv[3, 3]
    // ∂C_rgb/∂Dv[3, 3] = C_sh^t[3, 16] * ∂D/∂Dv[16, 3]
    // ∂L/∂Pw[1, 3] = ∂L/∂Dv[1, 3] * ∂Dv/∂Ov[3, 3] * ∂Ov/∂Pw[3, 3]
    // ∂Ov/∂Pw[3, 3] = I
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
    let color_rgb_3d_grad = colors_rgb_3d_grad[index] * is_colors_rgb_3d_clamped[index];
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

    // ∂L/∂Dv[1, 3]
    let view_direction_grad = color_rgb_3d_grad * color_rgb_3d_to_view_direction_grad;

    // ∂L/∂Pw[1, 3] = ∂L/∂Dv[1, 3] * ∂Dv/∂Ov[3, 3] * ∂Ov/∂Pw[3, 3]
    let vo_xx = view_offset.x * view_offset.x;
    let vo_yy = view_offset.y * view_offset.y;
    let vo_zz = view_offset.z * view_offset.z;
    let vo_xy_n = -view_offset.x * view_offset.y;
    let vo_xz_n = -view_offset.x * view_offset.z;
    let vo_yz_n = -view_offset.y * view_offset.z;
    let vo_l2_invsqrt3 = pow(inverseSqrt(vo_xx + vo_yy + vo_zz), 3.0);
    let position_3d_grad = view_direction_grad * vo_l2_invsqrt3 * mat3x3<f32>(
        vo_yy + vo_zz, vo_xy_n, vo_xz_n,
        vo_xy_n, vo_xx + vo_zz, vo_yz_n,
        vo_xz_n, vo_yz_n, vo_xx + vo_yy,
    );

    // TODO: End of colors and directions

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

    let conic_grad = conics_grad[index];
    // ∂L/∂Σ'[2, 2]
    let covariance_2d_grad = -conic * conic_grad * conic;

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
    // ∂L/∂T[2, 3] = ∂L/∂Σ'[2, 2] * T[2, 3] * Σ^t[3, 3] + (∂L/∂Σ')^t[2, 2] * T[2, 3] * Σ[3, 3]
    //             = ∂L/∂Σ'[2, 2] * T[2, 3] * Σ[3, 3] * 2
    // ∂L/∂J[2, 3] = ∂L/∂T[2, 3] * Rv^t[3, 3]
    //
    // Σ and Σ' are symmetric

    let covariance_2d_det = covariances_2d_det[index];
    // ∂L/∂Σ[3, 3]
    let covariance_3d_grad = select(
        transpose(covariance_3d_to_2d) * covariance_2d_grad * covariance_3d_to_2d,
        mat3x3<f32>(),
        covariance_2d_det == 0.0,
    );
    // ∂L/∂T[2, 3]
    let covariance_3d_to_2d_grad =
        2.0 * covariance_2d_grad * covariance_3d_to_2d * covariance_3d;
    // ∂L/∂J[2, 3]
    let projection_grad = covariance_3d_to_2d_grad * transpose(view_rotation);

    // Computing the gradients
    //
    // ∂L/∂Pv[3] = (∂L/∂Pv.x, ∂L/∂Pv.y, ∂L/∂Pv.z)
    // ∂L/∂Pv[3] = (〈∂L/∂J, ∂J/∂Pv.x〉,〈∂L/∂J, ∂J/∂Pv.y〉,〈∂L/∂J, ∂J/∂Pv.y〉)
    // ∂J/∂Pv.x[2, 3] = [[0, 0, -f.x / Pv.z^2]
    //                   [0, 0, 0            ]]
    // ∂J/∂Pv.x[2, 3] = [[0, 0, 0            ]
    //                   [0, 0, -f.y / Pv.z^2]]
    // ∂J/∂Pv.z[2, 3] = [[-f.x / Pv.z^2, 0,             2 * f.x * Pv.x / Pv.z^3]
    //                   [0,             -f.y / Pv.z^2, 2 * f.y * Pv.y / Pv.z^3]]
    //
    // Pv.x and Pv.y were clamped

    let focal_length_normalized_depth = focal_length_normalized / depth;
    let position_3d_in_view_grad = vec3<f32>(
        - is_position_3d_in_view_clamped.x * focal_length_normalized_depth.x * projection_grad[2][0],
        - is_position_3d_in_view_clamped.y * focal_length_normalized_depth.y * projection_grad[2][1],
		- focal_length_normalized_depth.x * projection_grad[0][0]
		- focal_length_normalized_depth.y * projection_grad[1][1]
		+ 2 * focal_length_normalized_depth.x *
          position_3d_in_view_normalized_clamped.x * projection_grad[2][0]
		+ 2 * focal_length_normalized_depth.y *
          position_3d_in_view_normalized_clamped.y * projection_grad[2][1],
    );

    // Specifying the results

    // [P, 16, 3]
    colors_sh_grad[index] = array<array<f32, 3>, 16>(
        array_from_vec3_f32(color_sh_grad[0u]),
        array_from_vec3_f32(color_sh_grad[1u]),
        array_from_vec3_f32(color_sh_grad[2u]),
        array_from_vec3_f32(color_sh_grad[3u]),
        array_from_vec3_f32(color_sh_grad[4u]),
        array_from_vec3_f32(color_sh_grad[5u]),
        array_from_vec3_f32(color_sh_grad[6u]),
        array_from_vec3_f32(color_sh_grad[7u]),
        array_from_vec3_f32(color_sh_grad[8u]),
        array_from_vec3_f32(color_sh_grad[9u]),
        array_from_vec3_f32(color_sh_grad[10]),
        array_from_vec3_f32(color_sh_grad[11]),
        array_from_vec3_f32(color_sh_grad[12]),
        array_from_vec3_f32(color_sh_grad[13]),
        array_from_vec3_f32(color_sh_grad[14]),
        array_from_vec3_f32(color_sh_grad[15]),
    );
    // [P, 3]
    positions_3d_grad[index] = array_from_vec3_f32(position_3d_grad);
}

fn array_from_vec3_f32(v: vec3<f32>) -> array<f32, 3> {
    return array<f32, 3>(v[0], v[1], v[2]);
}

fn vec_from_array_f32_3(a: array<f32, 3>) -> vec3<f32> {
    return vec3<f32>(a[0], a[1], a[2]);
}
