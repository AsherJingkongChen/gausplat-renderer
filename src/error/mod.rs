#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Validation Error: {0} should be {1}")]
    Validation(String, String),
}
