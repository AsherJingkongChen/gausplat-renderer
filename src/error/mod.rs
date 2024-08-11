use std::{error, fmt};

#[derive(Debug)]
pub enum Error {
    Gaussian3dRenderer(String),
}

impl fmt::Display for Error {
    fn fmt(
        &self,
        f: &mut fmt::Formatter,
    ) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl error::Error for Error {}
