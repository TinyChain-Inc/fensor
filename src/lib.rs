mod error;
mod schema;
mod stream;
mod tensor;
mod traits;
mod validate;
mod view;
mod wire_tags;

pub use error::{Error, Result};
pub use schema::{
    DType, Layout, SparseIndexSchema, SparseTableSchema, TensorSchema, TensorShape,
    contiguous_strides,
};
pub use stream::{TensorViewDecoder, TensorViewEncoder};
pub use traits::{
    BoxFuture, SparseElementStream, TensorArray, TensorBlockStore, TensorGeometry, TensorMatMul,
    TensorMath, TensorMathScalar, TensorRead, TensorReadBulk, TensorReduce, TensorReduceAll,
    TensorReduceBoolean, TensorSparseIndex, TensorTransform, TensorUnary, TensorViewSemantics,
    TensorWrite, TensorWriteBulk,
};

pub use tensor::{Tensor, TensorElement, TensorFileEntry};

pub type TensorF32<FE> = Tensor<FE, f32>;
pub type TensorF64<FE> = Tensor<FE, f64>;
