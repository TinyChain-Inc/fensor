# fensor
A filesystem-backed `Tensor` data structure featuring support for dense and sparse indexing

## Data Integrity Policy

`fensor` is fail-closed on corruption.

- `fensor` does not attempt to repair, recover, or auto-heal corrupted metadata or tensor data.
- If metadata or data is malformed, inconsistent, or unreadable, operations must return a structured error with a clear message.
- Recovery workflows (restore/rebuild/migration) are external operational concerns, not `fensor` runtime behavior.

## Serializing a tensor

`Tensor<FE, T>` has two, independent wire-format surfaces:

- **Schema-only (`Tensor: ToStream`/`FromStream`/`IntoStream`)**: encodes just the `TensorSchema` (dtype/shape/layout). Only a base (identity-view) tensor can be encoded this way; encoding a transformed (sliced/transposed/reshaped) view is rejected, since views are metadata-only and never persisted. Decoding always builds a fresh, empty base tensor at the given directory via `Tensor::create` — no element data is carried.
- **View + data streaming (`Tensor::view_encoder` / `TensorViewDecoder`)**: `tensor.view_encoder()` returns a `TensorViewEncoder<'_, FE, T>` that streams the tensor's *current* view — identity or transformed, dense or sparse — directly to the wire via `destream`'s `ToStream`/`IntoStream` contract. The encoder lazily reads from the tensor's filesystem-backed storage and emits values one at a time, with no full in-memory buffering; only non-default (nonzero) values are transmitted, reducing network traffic for sparse-heavy or mostly-empty tensors. On the receiving end, `TensorViewDecoder<FE, T>` implements `destream`'s `FromStream` and writes each arriving value directly to a fresh, independent, identity base tensor's filesystem storage as it arrives off the wire — also with no full in-memory buffering. There is no trailer or checksum on this wire format: a successful transfer is signaled by natural exhaustion of the pairs sequence, and end-to-end transfer completeness/integrity is left to the transport/caller layer rather than re-implemented here. A read failure on the sending side propagates as a bounded error-code sentinel — a closed classification of which failure shape occurred, deliberately excluding free-text detail (filesystem paths, coordinates, or other sender-machine specifics) since this wire format is meant to cross a machine boundary. On that sentinel, a malformed/truncated stream, or any other decode-time failure after the destination storage is created, the directory is truncated and deleted before a fail-closed error is returned. Call `.into_inner()` on the decoder to extract the reconstructed tensor. Like the design it replaces, this produces a fresh, independent identity base tensor with no link back to the source storage; it remains a distinct, additive wire surface alongside the existing schema-only `Tensor: ToStream/FromStream/IntoStream` contract, which is completely unrelated and still only carries dtype/shape/layout metadata without element data.
