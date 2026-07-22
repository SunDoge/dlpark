use super::Error;
use crate::{
    ManagedTensorBase, OpaqueContext,
    allocation::{self, dynamic, fixed},
};
use std::any::TypeId;

#[inline]
unsafe fn try_copy<T>(src: &[T], dst: *mut i64) -> Result<(), usize>
where
    T: Copy + TryInto<i64> + 'static,
{
    if TypeId::of::<T>() == TypeId::of::<i64>() {
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr().cast::<i64>(), dst, src.len()) };
        return Ok(());
    }

    for (axis, &value) in src.iter().enumerate() {
        let value = value.try_into().map_err(|_| axis)?;
        unsafe { dst.add(axis).write(value) };
    }
    Ok(())
}

/// Metadata values copied into the managed tensor allocation.
#[derive(Debug, Clone, Copy)]
pub struct Copied<T>(pub T);

/// Metadata values borrowed from caller-owned `i64` storage.
#[derive(Debug, Clone, Copy)]
pub struct Borrowed<T>(pub T);

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

/// Runtime-rank shape and strides metadata.
#[derive(Debug, Clone, Copy)]
pub struct Dynamic<Shape, Strides> {
    shape: Shape,
    strides: Strides,
}

impl<Shape, Strides> Dynamic<Shape, Strides> {
    /// Creates runtime-rank metadata with independently selected shape and
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

/// A dynamic allocation whose shape and strides values have been prepared.
pub struct PreparedDynamic<M: ManagedTensorBase> {
    allocation: dynamic::Allocation<M>,
    shape: *mut i64,
    strides: *mut i64,
    ndim: usize,
}

impl<M: ManagedTensorBase> PreparedDynamic<M> {
    /// Installs the owning context and metadata pointers into the allocation.
    pub fn initialize<C: OpaqueContext>(
        self,
        ctx: C,
    ) -> Result<dynamic::Initialized<M>, allocation::Error> {
        let Self {
            allocation,
            shape,
            strides,
            ndim,
        } = self;
        let mut initialized = allocation.initialize(ctx, ndim)?;
        initialized.tensor_mut().shape = shape;
        initialized.tensor_mut().strides = strides;
        Ok(initialized)
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

/// Internal runtime-rank metadata writing policy.
#[doc(hidden)]
pub trait DynamicPart {
    const COPIED: bool;
    type Item: Copy + TryInto<i64>;

    fn values(&self) -> &[Self::Item];

    fn write(self, dst: *mut i64) -> Result<*mut i64, usize>;
}

/// A dynamic metadata part whose values are owned by the managed allocation.
#[doc(hidden)]
pub trait OwnedDynamicPart: DynamicPart {}

macro_rules! impl_copied_dynamic {
    ($source:ty) => {
        impl<T> DynamicPart for Copied<$source>
        where
            T: Copy + TryInto<i64> + 'static,
        {
            const COPIED: bool = true;
            type Item = T;

            fn values(&self) -> &[T] {
                &self.0
            }

            fn write(self, dst: *mut i64) -> Result<*mut i64, usize> {
                unsafe { try_copy(&self.0, dst)? };
                Ok(dst)
            }
        }

        impl<T> OwnedDynamicPart for Copied<$source> where T: Copy + TryInto<i64> + 'static {}
    };
}

impl_copied_dynamic!(Vec<T>);
impl_copied_dynamic!(Box<[T]>);
impl_copied_dynamic!(&[T]);

impl DynamicPart for Borrowed<&[i64]> {
    const COPIED: bool = false;
    type Item = i64;

    fn values(&self) -> &[i64] {
        self.0
    }

    fn write(self, _: *mut i64) -> Result<*mut i64, usize> {
        Ok(self.0.as_ptr().cast_mut())
    }
}

impl<Shape, Strides> Dynamic<Shape, Strides>
where
    Shape: DynamicPart,
    Strides: DynamicPart,
{
    fn prepare_inner<M>(self) -> Result<PreparedDynamic<M>, Error>
    where
        M: ManagedTensorBase,
    {
        let shape_len = self.shape.values().len();
        let strides_len = self.strides.values().len();
        if shape_len != strides_len {
            return Err(Error::MismatchedLength {
                shape_len,
                strides_len,
            });
        }
        i32::try_from(shape_len).map_err(|source| Error::NdimOverflow {
            ndim: shape_len,
            source,
        })?;

        let shape_extra = usize::from(Shape::COPIED)
            .checked_mul(shape_len)
            .ok_or(allocation::Error::LayoutOverflow)?;
        let strides_extra = usize::from(Strides::COPIED)
            .checked_mul(strides_len)
            .ok_or(allocation::Error::LayoutOverflow)?;
        let extra = shape_extra
            .checked_add(strides_extra)
            .ok_or(allocation::Error::LayoutOverflow)?;
        let mut allocation = dynamic::Allocation::<M>::allocate(extra)?;
        let extra = allocation.extra_mut().as_mut_ptr();

        let shape = self
            .shape
            .write(extra)
            .map_err(|axis| Error::ShapeValueOverflow { axis })?;
        let strides = self
            .strides
            .write(unsafe { extra.add(shape_extra) })
            .map_err(|axis| Error::StrideValueOverflow { axis })?;

        Ok(PreparedDynamic {
            allocation,
            shape,
            strides,
            ndim: shape_len,
        })
    }

    /// Prepares runtime metadata which may borrow caller-owned slices.
    ///
    /// # Safety
    ///
    /// Every value wrapped in [`Borrowed`] must remain alive and immutable
    /// until the resulting managed tensor is dropped.
    pub unsafe fn prepare_unchecked<M>(self) -> Result<PreparedDynamic<M>, Error>
    where
        M: ManagedTensorBase,
    {
        self.prepare_inner()
    }
}

impl<Shape, Strides> Dynamic<Shape, Strides>
where
    Shape: OwnedDynamicPart,
    Strides: OwnedDynamicPart,
{
    /// Validates runtime rank, allocates copied storage, and writes shape and
    /// strides into it.
    pub fn prepare<M>(self) -> Result<PreparedDynamic<M>, Error>
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
    fn fixed_copied_metadata_uses_inline_arrays() {
        let prepared = Fixed::new(Copied([2_u32, 3]), Copied([3_isize, 1]))
            .prepare::<DLManagedTensor>()
            .unwrap();
        let initialized = prepared.initialize(Box::new(()));
        let tensor = unsafe { initialized.finish() };

        assert_eq!(tensor.shape().unwrap(), &[2, 3]);
        assert_eq!(tensor.strides().unwrap().unwrap(), &[3, 1]);
    }

    #[test]
    fn fixed_borrowed_shape_allocates_only_strides() {
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

    #[test]
    fn dynamic_mixed_storage_uses_only_copied_extra() {
        let shape = [2_i64, 3];
        let prepared = unsafe {
            Dynamic::new(Borrowed(shape.as_slice()), Copied(vec![3_i16, 1]))
                .prepare_unchecked::<DLManagedTensor>()
                .unwrap()
        };
        let initialized = prepared.initialize(Box::new(())).unwrap();
        let tensor = unsafe { initialized.finish() };

        assert_eq!(tensor.shape().unwrap(), &shape);
        assert_eq!(tensor.strides().unwrap().unwrap(), &[3, 1]);
    }
}
