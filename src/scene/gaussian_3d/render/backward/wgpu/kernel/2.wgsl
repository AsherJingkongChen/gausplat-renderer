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
    if index >= arguments.point_count {
        return;
    }

    // Load the view direction in world space
    // vo[P, 3] = pw[P, 3] - vw[1, 3]
    // vd[P, 3] = norm(vo)

    let view_offsets = view_offsets[index];
    let view_direction = view_directions[index];
    var x = f32();
    var y = f32();
    var z = f32();
    var xx = f32();
    var xy = f32();
    var xz = f32();
    var yy = f32();
    var yz = f32();
    var zz = f32();
    var xx_yy = f32();
    var xx_yy_3 = f32();
    var z_10 = f32();
    var zz_5_1 = f32();

    // Computing the gradients
    // (d c_sh[P, 16, 3], d vd[P, 3]) = d c_rgb[P, 3]

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

    let color_rgb_3d_grad = colors_rgb_3d_grad[index] * is_colors_rgb_3d_clamped[index];
    var color_sh_grad = array<vec3<f32>, 16>();
    // d/dvd c_rgb[3, 3]
    var color_rgb_3d_to_view_direction_grad = mat3x3<f32>();

    color_sh_grad[0] = color_rgb_3d_grad * (SH_C_0[0]);

    if arguments.colors_sh_degree_max >= 1 {
        x = view_direction.x;
        y = view_direction.y;
        z = view_direction.z;

        color_sh_grad[1] = color_rgb_3d_grad * (SH_C_1[0] * (y));
        color_sh_grad[2] = color_rgb_3d_grad * (SH_C_1[1] * (z));
        color_sh_grad[3] = color_rgb_3d_grad * (SH_C_1[2] * (x));

        color_rgb_3d_to_view_direction_grad += mat3x3<f32>(
            color_sh[3] * (SH_C_1[2]),
            color_sh[1] * (SH_C_1[0]),
            color_sh[2] * (SH_C_1[1]),
        );
    }

    if arguments.colors_sh_degree_max >= 2 {
        xx = x * x;
        xy = x * y;
        xz = x * z;
        yy = y * y;
        yz = y * z;
        zz = z * z;
        xx_yy = xx - yy;

        color_sh_grad[4] = color_rgb_3d_grad * (SH_C_2[0] * (xy));
        color_sh_grad[5] = color_rgb_3d_grad * (SH_C_2[1] * (yz));
        color_sh_grad[6] = color_rgb_3d_grad * (SH_C_2[2] * (zz * 3.0 - 1.0));
        color_sh_grad[7] = color_rgb_3d_grad * (SH_C_2[3] * (xz));
        color_sh_grad[8] = color_rgb_3d_grad * (SH_C_2[4] * (xx_yy));

        color_rgb_3d_to_view_direction_grad += mat3x3<f32>(
            color_sh[4] * (SH_C_2[0] * (y)) +
            color_sh[7] * (SH_C_2[3] * (z)) +
            color_sh[8] * (SH_C_2[4] * (x * 2.0)),

            color_sh[4] * (SH_C_2[0] * (x)) +
            color_sh[5] * (SH_C_2[1] * (z)) +
            color_sh[8] * (SH_C_2[4] * (y * -2.0)),

            color_sh[5] * (SH_C_2[1] * (y)) +
            color_sh[6] * (SH_C_2[2] * (z * 6.0)) +
            color_sh[7] * (SH_C_2[3] * (x)),
        );
    }

    if arguments.colors_sh_degree_max >= 3 {
        xx_yy_3 = xx_yy * 3.0;
        z_10 = z * 10.0;
        zz_5_1 = zz * 5.0 - 1.0;

        color_sh_grad[9u] = color_rgb_3d_grad * (SH_C_3[0] * (y * (xx * 3.0 - yy)));
        color_sh_grad[10] = color_rgb_3d_grad * (SH_C_3[1] * (z * (xy)));
        color_sh_grad[11] = color_rgb_3d_grad * (SH_C_3[2] * (y * (zz_5_1)));
        color_sh_grad[12] = color_rgb_3d_grad * (SH_C_3[3] * (z * (zz_5_1 - 2.0)));
        color_sh_grad[13] = color_rgb_3d_grad * (SH_C_3[4] * (x * (zz_5_1)));
        color_sh_grad[14] = color_rgb_3d_grad * (SH_C_3[5] * (z * (xx_yy)));
        color_sh_grad[15] = color_rgb_3d_grad * (SH_C_3[6] * (x * (xx - yy * 3.0)));

        color_rgb_3d_to_view_direction_grad += mat3x3<f32>(
            color_sh[9u] * (SH_C_3[0] * (xy * 6.0)) +
            color_sh[10] * (SH_C_3[1] * (yz)) +
            color_sh[13] * (SH_C_3[4] * (zz_5_1)) +
            color_sh[14] * (SH_C_3[5] * (xz * 2.0)) +
            color_sh[15] * (SH_C_3[6] * (xx_yy_3)),

            color_sh[9u] * (SH_C_3[0] * (xx_yy_3)) +
            color_sh[10] * (SH_C_3[1] * (xz)) +
            color_sh[11] * (SH_C_3[2] * (zz_5_1)) +
            color_sh[14] * (SH_C_3[5] * (yz * -2.0)) +
            color_sh[15] * (SH_C_3[6] * (xy * -6.0)),

            color_sh[10] * (SH_C_3[1] * (xy)) +
            color_sh[11] * (SH_C_3[2] * (y * z_10)) +
            color_sh[12] * (SH_C_3[3] * (zz_5_1 * 3.0)) +
            color_sh[13] * (SH_C_3[4] * (x * z_10)) +
            color_sh[14] * (SH_C_3[5] * (xx_yy))
        );
    }

    // d vd[1, 3] = d c_rgb[1, 3] * d/dvd c_rgb[3, 3]
    let view_direction_grad = color_rgb_3d_grad * color_rgb_3d_to_view_direction_grad;

    // Specifying the results

    colors_sh_grad[index] = array<array<f32, 3>, 16>(
        array<f32, 3>(color_sh_grad[0u][0], color_sh_grad[0u][1], color_sh_grad[0u][2]),
        array<f32, 3>(color_sh_grad[1u][0], color_sh_grad[1u][1], color_sh_grad[1u][2]),
        array<f32, 3>(color_sh_grad[2u][0], color_sh_grad[2u][1], color_sh_grad[2u][2]),
        array<f32, 3>(color_sh_grad[3u][0], color_sh_grad[3u][1], color_sh_grad[3u][2]),
        array<f32, 3>(color_sh_grad[4u][0], color_sh_grad[4u][1], color_sh_grad[4u][2]),
        array<f32, 3>(color_sh_grad[5u][0], color_sh_grad[5u][1], color_sh_grad[5u][2]),
        array<f32, 3>(color_sh_grad[6u][0], color_sh_grad[6u][1], color_sh_grad[6u][2]),
        array<f32, 3>(color_sh_grad[7u][0], color_sh_grad[7u][1], color_sh_grad[7u][2]),
        array<f32, 3>(color_sh_grad[8u][0], color_sh_grad[8u][1], color_sh_grad[8u][2]),
        array<f32, 3>(color_sh_grad[9u][0], color_sh_grad[9u][1], color_sh_grad[9u][2]),
        array<f32, 3>(color_sh_grad[10][0], color_sh_grad[10][1], color_sh_grad[10][2]),
        array<f32, 3>(color_sh_grad[11][0], color_sh_grad[11][1], color_sh_grad[11][2]),
        array<f32, 3>(color_sh_grad[12][0], color_sh_grad[12][1], color_sh_grad[12][2]),
        array<f32, 3>(color_sh_grad[13][0], color_sh_grad[13][1], color_sh_grad[13][2]),
        array<f32, 3>(color_sh_grad[14][0], color_sh_grad[14][1], color_sh_grad[14][2]),
        array<f32, 3>(color_sh_grad[15][0], color_sh_grad[15][1], color_sh_grad[15][2]),
    );
}
