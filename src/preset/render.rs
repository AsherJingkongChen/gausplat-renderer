pub const GROUP_COUNT: u32 = GROUP_SIZE_X * GROUP_SIZE_Y;
pub const GROUP_SIZE_X: u32 = 16;
pub const GROUP_SIZE_Y: u32 = 16;
pub const PIXEL_COUNT_MAX: u32 = GROUP_COUNT * TILE_COUNT_MAX;
pub const TILE_COUNT_MAX: u32 = 1 << 16;
