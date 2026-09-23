# Python interoperability

Enable the `pyo3` feature for the Python Array API DLPack protocol and DLPack
1.3 C Exchange API.

## Importing from Python

Use `python::from_dlpack` for the ordinary one-step path. For device-aware
stream negotiation, construct `ImportRequest`, query its device, choose the
consumer stream, and call `import`. Discovery proceeds in this order:

1. DLPack C Exchange API when synchronization can be handled safely.
2. `__dlpack_device__` and `__dlpack__`.
3. A legacy `__dlpack__` producer without separate device discovery.

C Exchange callbacks do not synchronize. If a producer reports pending work,
dlpark uses that path only when the supplied `DlpackStream` can order the
consumer stream after the producer stream. Otherwise it falls back to Python's
`__dlpack__` negotiation.

Capsules are single-use. Successful extraction renames the capsule to
`dltensor_used` or `dltensor_versioned_used`; consuming it again is an error.

After import, a container that adopts the pointer and metadata may call
`ImportedDlpack::into_deleter`. The resulting `AllocationDeleter` releases the
original managed tensor exactly once without requiring the container to retain
the legacy/versioned wrapper type.

## Exporting reusable Python objects

A long-lived Python tensor wrapper should own its buffer and implement
`python::DlpackExporter`. Its `__dlpack__` method parses an `ExportRequest` and
calls `python::export_dlpack`. Each call creates a fresh managed tensor and
single-use capsule while the wrapper remains reusable.

The request handles `stream`, `max_version`, `dl_device`, and `copy`. dlpark
validates device, flags, copy policy, and ABI selection before asking the
backend to prepare the consumer stream. A compatible `max_version` selects the
versioned ABI; an omitted value or a DLPack 0.x maximum selects legacy.

The CUDA and Metal demos contain full exporter implementations. They keep the
native allocation in the Python class and create a fresh DLPack ownership
header for every export.

## C Exchange producers

PyO3 classes may implement `python::DlpackExchangeProducer` and call
`python::install_exchange_api::<T>(py)` during module initialization. dlpark
installs the process-lifetime type attribute and provides callbacks with Python
exception restoration, ownership transfer, and panic containment.

Only set `HAS_DLTENSOR_VIEW` and implement `tensor_view_no_sync` when the class
can provide the optional borrowed-view callback.

Consumers discover an exchange table with
`python::consumer::exchange::ExchangeApi::from_object`. The no-sync import API
returns an `ExchangeTensor` that keeps the producer's current stream attached
to the owned tensor. Ordinary ingestion should prefer `python::from_dlpack` so
the synchronization policy is selected automatically.
