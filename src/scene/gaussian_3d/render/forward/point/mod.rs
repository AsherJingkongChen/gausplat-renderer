use bytemuck::{Pod, Zeroable};

#[repr(C, packed(4))]
#[derive(Copy, Clone, Debug, Eq, Pod, Zeroable)]
pub(super) struct PointKeyAndIndex {
    pub key: [u32; 2],
    pub index: u32,
}

impl PointKeyAndIndex {
    #[inline]
    pub(super) fn tile_index(&self) -> u32 {
        self.key[0]
    }
}

impl Ord for PointKeyAndIndex {
    #[inline]
    fn cmp(
        &self,
        other: &Self,
    ) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

impl PartialEq for PointKeyAndIndex {
    #[inline]
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.key.eq(&other.key)
    }
}

impl PartialOrd for PointKeyAndIndex {
    #[inline]
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<std::cmp::Ordering> {
        self.key.partial_cmp(&other.key)
    }
}
