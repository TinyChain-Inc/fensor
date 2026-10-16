//! Observe real expression delegation without evaluating tensor payloads.

use std::sync::{Arc, Mutex};

use crate::expression::{Batch, Expression};
use crate::request::BatchRequest;
use crate::{
    Axes, AxisRange, BoxFuture, Error, Layout, NumberType, Range, Result, Shape, TensorGeometry,
    TensorMath, TensorTransform, TensorUnaryBoolean, TensorWhere,
};

#[derive(Clone)]
struct Operand {
    id: usize,
    fail_at: Option<usize>,
    visited: Arc<Mutex<Vec<(usize, usize)>>>,
}

impl TensorGeometry for Operand {
    type DType = u8;

    fn dtype(&self) -> NumberType {
        <u8 as number_general::DType>::dtype()
    }

    fn layout(&self) -> Layout {
        Layout::Dense
    }

    fn shape(&self) -> &[u64] {
        &[2]
    }
}

impl crate::expression::traversal::Plan for Operand {}

impl Expression for Operand {
    fn build<'a>(
        &'a self,
        _: crate::expression::Context<'a>,
        _: std::sync::Arc<BatchRequest>,
    ) -> BoxFuture<'a, Result<Batch<u8>>> {
        Box::pin(async { panic!("transform construction must not evaluate operands") })
    }
}

impl TensorTransform for Operand {
    fn reshape(self, _: Shape) -> Result<Self> {
        Ok(self)
    }

    fn broadcast(self, _: Shape) -> Result<Self> {
        Ok(self)
    }

    fn flip(self, _: usize) -> Result<Self> {
        Ok(self)
    }

    fn squeeze(self, _: Axes) -> Result<Self> {
        Ok(self)
    }

    fn transpose(self, _: Option<Axes>) -> Result<Self> {
        Ok(self)
    }

    fn unsqueeze(self, _: Axes) -> Result<Self> {
        Ok(self)
    }

    fn slice(self, range: Range) -> Result<Self> {
        let AxisRange::Of(indices) = &range[0] else {
            panic!("expected owned indices")
        };
        assert_eq!(indices, &[1, 0, 1]);
        self.visited
            .lock()
            .unwrap()
            .push((self.id, indices.as_ptr() as usize));
        if self.fail_at == Some(self.id) {
            Err(Error::InvalidCoord(format!("operand {}", self.id)))
        } else {
            Ok(self)
        }
    }
}

#[tokio::test]
async fn recursive_transforms_stop_at_first_error_and_move_the_final_argument() {
    for count in 1..=3 {
        for fail_at in std::iter::once(None).chain((0..count).map(Some)) {
            let visited = Arc::new(Mutex::new(Vec::new()));
            let operand = |id| Operand {
                id,
                fail_at,
                visited: visited.clone(),
            };

            let indices = vec![1, 0, 1];
            let original = indices.as_ptr() as usize;
            let range = std::iter::once(AxisRange::Of(indices)).collect();
            let result = match count {
                1 => operand(0).not().await.unwrap().slice(range).map(|_| ()),
                2 => operand(0)
                    .add(&operand(1))
                    .await
                    .unwrap()
                    .not()
                    .await
                    .unwrap()
                    .slice(range)
                    .map(|_| ()),
                3 => operand(0)
                    .cond(&operand(1).not().await.unwrap(), &operand(2))
                    .await
                    .unwrap()
                    .slice(range)
                    .map(|_| ()),
                _ => unreachable!(),
            };

            if let Some(id) = fail_at {
                assert!(
                    matches!(result, Err(Error::InvalidCoord(message)) if message == format!("operand {id}"))
                );
            } else {
                result.unwrap();
            }

            let visited = visited.lock().unwrap();
            let expected = fail_at.map_or(count, |id| id + 1);
            assert_eq!(
                visited.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                (0..expected).collect::<Vec<_>>()
            );

            for &(id, address) in visited.iter() {
                if id == count - 1 {
                    assert_eq!(
                        address, original,
                        "last operand must receive the original indices"
                    );
                } else {
                    assert_ne!(
                        address, original,
                        "preceding operands receive independent indices"
                    );
                }
            }
        }
    }
}
