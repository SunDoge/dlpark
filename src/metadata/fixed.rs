use super::{Borrowed, Copied, Error, storage::try_copy};
use crate::{ManagedTensorBase, OpaqueContext, allocation::fixed};

/// Fixed-rank shape and strides metadata.
#[derive(Debug, Clone, Copy)]
pub struct Fixed<const N: usize, Shape, Strides> {
    shape: Shape,
    strides: Strides,
}

impl<const N: usize, Shape, Strides> Fixed<N, Shape, Strides> {
    /// Creates fixed-rank metadata with independently selected shape and
    /// strides storage policies.
    pub fn new(shape: Shape, strides: Strides) -> Self {
        Self { shape, strides }
    }
}

/// A fixed allocation whose shape and strides values have been prepared.
pub struct PreparedFixed<M, const N: usize, Shape, Strides>
where
    M: ManagedTensorBase,
    Shape: fixed::Storage<N>,
    Strides: fixed::Storage<N>,
{
    allocation: fixed::Allocation<M, N, Shape, Strides>,
    shape: *mut i64,
    strides: *mut i64,
}

impl<M, const N: usize, Shape, Strides> PreparedFixed<M, N, Shape, Strides>
where
    M: ManagedTensorBase,
    Shape: fixed::Storage<N>,
    Strides: fixed::Storage<N>,
{
    /// Installs the owning context and metadata pointers into the allocation.
    pub fn initialize<C: OpaqueContext>(self, ctx: C) -> fixed::Initialized<M, N, Shape, Strides> {
        let Self {
            allocation,
            shape,
            strides,
        } = self;
        let mut initialized = allocation.initialize(ctx);
        initialized.tensor_mut().shape = shape;
        initialized.tensor_mut().strides = strides;
        initialized
    }
}

/// Internal fixed-rank metadata writing policy.
#[doc(hidden)]
pub trait FixedPart<const N: usize> {
    type Storage: fixed::Storage<N>;

    fn write(
        self,
        storage: &mut <Self::Storage as fixed::Storage<N>>::Value,
    ) -> Result<*mut i64, usize>;
}

/// A fixed metadata part whose values are owned by the managed allocation.
#[doc(hidden)]
pub trait OwnedFixedPart<const N: usize>: FixedPart<N> {}

impl<T, const N: usize> FixedPart<N> for Copied<[T; N]>
where
    T: Copy + TryInto<i64> + 'static,
{
    type Storage = fixed::Copied;

    fn write(self, storage: &mut [i64; N]) -> Result<*mut i64, usize> {
        unsafe { try_copy(&self.0, storage.as_mut_ptr())? };
        Ok(storage.as_mut_ptr())
    }
}

impl<T, const N: usize> OwnedFixedPart<N> for Copied<[T; N]> where T: Copy + TryInto<i64> + 'static {}

impl<T, const N: usize> FixedPart<N> for Copied<&[T; N]>
where
    T: Copy + TryInto<i64> + 'static,
{
    type Storage = fixed::Copied;

    fn write(self, storage: &mut [i64; N]) -> Result<*mut i64, usize> {
        unsafe { try_copy(self.0, storage.as_mut_ptr())? };
        Ok(storage.as_mut_ptr())
    }
}

impl<T, const N: usize> OwnedFixedPart<N> for Copied<&[T; N]> where T: Copy + TryInto<i64> + 'static {}

impl<const N: usize> FixedPart<N> for Borrowed<&[i64; N]> {
    type Storage = fixed::Borrowed;

    fn write(self, _: &mut ()) -> Result<*mut i64, usize> {
        Ok(self.0.as_ptr().cast_mut())
    }
}

impl<const N: usize, Shape, Strides> Fixed<N, Shape, Strides>
where
    Shape: FixedPart<N>,
    Strides: FixedPart<N>,
{
    fn prepare_inner<M>(
        self,
    ) -> Result<PreparedFixed<M, N, Shape::Storage, Strides::Storage>, Error>
    where
        M: ManagedTensorBase,
    {
        let mut allocation =
            fixed::Allocation::<M, N, Shape::Storage, Strides::Storage>::allocate()?;
        let shape = self
            .shape
            .write(allocation.shape_storage_mut())
            .map_err(|axis| Error::ShapeValueOverflow { axis })?;
        let strides = self
            .strides
            .write(allocation.strides_storage_mut())
            .map_err(|axis| Error::StrideValueOverflow { axis })?;
        Ok(PreparedFixed {
            allocation,
            shape,
            strides,
        })
    }

    /// Prepares metadata which may borrow caller-owned arrays.
    ///
    /// # Safety
    ///
    /// Every value wrapped in [`Borrowed`] must remain alive and immutable
    /// until the resulting managed tensor is dropped.
    pub unsafe fn prepare_unchecked<M>(
        self,
    ) -> Result<PreparedFixed<M, N, Shape::Storage, Strides::Storage>, Error>
    where
        M: ManagedTensorBase,
    {
        self.prepare_inner()
    }
}

impl<const N: usize, Shape, Strides> Fixed<N, Shape, Strides>
where
    Shape: OwnedFixedPart<N>,
    Strides: OwnedFixedPart<N>,
{
    /// Allocates fixed metadata storage and copies shape and strides into it.
    pub fn prepare<M>(self) -> Result<PreparedFixed<M, N, Shape::Storage, Strides::Storage>, Error>
    where
        M: ManagedTensorBase,
    {
        self.prepare_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::DLManagedTensor;

    #[test]
    fn copied_metadata_uses_inline_arrays() {
        let prepared = Fixed::new(Copied([2_u32, 3]), Copied([3_isize, 1]))
            .prepare::<DLManagedTensor>()
            .unwrap();
        let initialized = prepared.initialize(Box::new(()));
        let tensor = unsafe { initialized.finish() };

        assert_eq!(tensor.shape().unwrap(), &[2, 3]);
        assert_eq!(tensor.strides().unwrap().unwrap(), &[3, 1]);
    }

    #[test]
    fn borrowed_shape_allocates_only_strides() {
        let shape = [2_i64, 3];
        let prepared = unsafe {
            Fixed::new(Borrowed(&shape), Copied([3_i64, 1]))
                .prepare_unchecked::<DLManagedTensor>()
                .unwrap()
        };
        let initialized = prepared.initialize(Box::new(()));
        let tensor = unsafe { initialized.finish() };

        assert_eq!(tensor.shape().unwrap(), &shape);
        assert_eq!(tensor.strides().unwrap().unwrap(), &[3, 1]);
    }
}
