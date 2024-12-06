//! A collection of views.

pub use super::View;

/// A map of view ID to [`View`].
pub type Views = gausplat_loader::collection::IndexMap<u32, View>;
