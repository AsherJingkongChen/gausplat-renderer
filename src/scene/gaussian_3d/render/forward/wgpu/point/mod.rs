use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Pod, Zeroable)]
pub struct PointInfo {
    pub order: u32,
    pub index: u32,
}

impl PointInfo {
    #[inline]
    pub fn tile_index(&self) -> u32 {
        self.order >> 16
    }
}

impl Ord for PointInfo {
    #[inline]
    fn cmp(
        &self,
        other: &Self,
    ) -> std::cmp::Ordering {
        self.order.cmp(&other.order)
    }
}

impl PartialEq for PointInfo {
    #[inline]
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.order.eq(&other.order)
    }
}

impl PartialOrd for PointInfo {
    #[inline]
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
