use std::fmt;
use std::io;

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Nd(ha_ndarray::Error),
    InvalidSchema(String),
    InvalidCoord(String),
    InvalidLayout(String),
    SparseIndex(String),
    UnsupportedSparseIterationOrder {
        requested_order: Vec<usize>,
        base_order: Vec<usize>,
        hint: String,
    },
    Unsupported(String),
    DataMismatch(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<io::Error> for Error {
    fn from(cause: io::Error) -> Self {
        Self::Io(cause)
    }
}

impl From<ha_ndarray::Error> for Error {
    fn from(cause: ha_ndarray::Error) -> Self {
        Self::Nd(cause)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(cause) => cause.fmt(f),
            Self::Nd(cause) => cause.fmt(f),
            Self::InvalidSchema(cause)
            | Self::InvalidCoord(cause)
            | Self::InvalidLayout(cause)
            | Self::SparseIndex(cause)
            | Self::Unsupported(cause)
            | Self::DataMismatch(cause) => f.write_str(cause),
            Self::UnsupportedSparseIterationOrder {
                requested_order,
                base_order,
                hint,
            } => write!(
                f,
                "unsupported sparse iteration order: requested {:?}, base {:?}. {}",
                requested_order, base_order, hint
            ),
        }
    }
}

impl std::error::Error for Error {}
