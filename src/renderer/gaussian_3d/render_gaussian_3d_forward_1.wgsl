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

@group(0)
@binding(0)
var<storage, read> arguments: Arguments;

// [P, 3]
@group(0)
@binding(1)
var<storage, read> positions: array<f32>;

// [P, 4]
@group(0)
@binding(2)
var<storage, read> rotations: array<f32>;

// [P, 3]
@group(0)
@binding(3)
var<storage, read> scalings: array<f32>;

// [4, 4]
@group(0)
@binding(4)
var<storage, read> view_transform: array<f32, 16>;

// [P]
@group(0)
@binding(5)
var<storage, read_write> depths: array<f32>;

// [P, 2, 2]
@group(0)
@binding(6)
var<storage, read_write> conics: array<mat2x2<f32>>;

// [P, 2]
@group(0)
@binding(7)
var<storage, read_write> positions_2d_in_screen: array<vec2<f32>>;

// [P]
@group(0)
@binding(8)
var<storage, read_write> radii: array<u32>;

// [P]
@group(0)
@binding(9)
var<storage, read_write> tile_touched_counts: array<u32>;

// Constants

const EPSILON: f32 = 1.1920929e-7;

@compute
@workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    // Checking the index

    let index = global_id.x;
    if index >= arguments.point_count {
        return;
    }

    // Initializing the results

    radii[index] = 0u;
    tile_touched_counts[index] = 0u;

    // Transforming 3D positions from world space to view space
    // pv[3, P] = vr[3, 3] * pw[3, P] + vt[3, P]

    let view_rotation = mat3x3<f32>(
        view_transform[0], view_transform[4], view_transform[8],
        view_transform[1], view_transform[5], view_transform[9],
        view_transform[2], view_transform[6], view_transform[10],
    );
    let view_translation = vec3<f32>(
        view_transform[3],
        view_transform[7],
        view_transform[11],
    );
    let positions = vec3<f32>(
        positions[index * 3 + 0],
        positions[index * 3 + 1],
        positions[index * 3 + 2],
    );

    let position_3d_in_view = view_rotation * positions + view_translation;
    let depth = position_3d_in_view.z;
    let position_3d_in_view_x_normalized = position_3d_in_view.x / depth;
    let position_3d_in_view_y_normalized = position_3d_in_view.y / depth;

    // Performing viewing-frustum culling

    if depth <= 0.2 {
        return;
    }

    // Converting the quaternion to rotation matrix
    // r[P, 3, 3] (Symmetric) = q[P, 4] (x, y, z, w)

    let quaternion = vec4<f32>(
        rotations[index * 4 + 1],
        rotations[index * 4 + 2],
        rotations[index * 4 + 3],
        rotations[index * 4 + 0],
    );

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
    let scaling = vec3<f32>(
        scalings[index * 3 + 0],
        scalings[index * 3 + 1],
        scalings[index * 3 + 2],
    );
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
    if arguments.view_bound_y == 0.0 {
        position_3d_in_view_y_normalized_clamped = -99.0;
    }

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

    // Specifying the results

    // [P]
    depths[index] = depth;

    // [P, 2, 2]
    conics[index] = conic;

    // [P, 2]
    positions_2d_in_screen[index] = position_2d_in_screen;

    // [P]
    radii[index] = u32(radius);

    // [P]
    tile_touched_counts[index] = tile_touched_count;
}
