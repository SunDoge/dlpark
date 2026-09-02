use super::{Borrowed, Copied, Error, storage::try_copy};
use crate::{
    ManagedTensorBase, OpaqueContext,
    allocation::{self, dynamic},
};

/// Runtime-rank shape and strides metadata.
#[derive(Debug, Clone, Copy)]
pub struct Dynamic<Shape, Strides> {
    shape: Shape,
    strides: Strides,
}

/// Marker selecting automatically computed, explicitly stored compact strides.
#[derive(Debug, Clone, Copy)]
pub struct Compact;

impl<Shape> Dynamic<Shape, Compact> {
    /// Creates runtime-rank metadata with explicit compact row-major strides
    /// computed from `shape`.
    pub fn compact(shape: Shape) -> Self {
        Self {
            shape,
            strides: Compact,
        }
    }
}

impl<Shape, Strides> Dynamic<Shape, Strides> {
    /// Creates runtime-rank metadata with independently selected shape and
    /// strides storage policies.
    pub fn new(shape: Shape, strides: Strides) -> Self {
        Self { shape, strides }
    }
}

/// A dynamic allocation whose shape and strides values have been prepared.
pub struct PreparedDynamic<M: ManagedTensorBase> {
    allocation: dynamic::Allocation<M>,
    shape: *mut i64,
    strides: *mut i64,
    ndim: i32,
}

impl<M: ManagedTensorBase> PreparedDynamic<M> {
    /// Installs the owning context and metadata pointers into the allocation.
    pub fn initialize<C: OpaqueContext>(self, ctx: C) -> dynamic::Initialized<M> {
        let Self {
            allocation,
            shape,
            strides,
            ndim,
        } = self;
        let mut initialized = allocation.initialize_validated(ctx, ndim);
        initialized.tensor_mut().shape = shape;
        initialized.tensor_mut().strides = strides;
        initialized
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
        let ndim = i32::try_from(shape_len).map_err(|source| Error::NdimOverflow {
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
            ndim,
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

impl<Shape> Dynamic<Shape, Compact>
where
    Shape: DynamicPart,
{
    fn prepare_compact_inner<M>(self) -> Result<PreparedDynamic<M>, Error>
    where
        M: ManagedTensorBase,
    {
        let shape_len = self.shape.values().len();
        let ndim = i32::try_from(shape_len).map_err(|source| Error::NdimOverflow {
            ndim: shape_len,
            source,
        })?;
        let shape_extra = usize::from(Shape::COPIED)
            .checked_mul(shape_len)
            .ok_or(allocation::Error::LayoutOverflow)?;
        let extra_len = shape_extra
            .checked_add(shape_len)
            .ok_or(allocation::Error::LayoutOverflow)?;
        let mut allocation = dynamic::Allocation::<M>::allocate(extra_len)?;
        let extra = allocation.extra_mut().as_mut_ptr();
        let shape = self
            .shape
            .write(extra)
            .map_err(|axis| Error::ShapeValueOverflow { axis })?;
        let strides = unsafe { extra.add(shape_extra) };

        let shape_values = unsafe { std::slice::from_raw_parts(shape, shape_len) };
        let mut stride = 1_i64;
        for axis in (0..shape_len).rev() {
            let dimension = shape_values[axis];
            if dimension < 0 {
                return Err(Error::NegativeShapeValue {
                    axis,
                    value: dimension,
                });
            }
            unsafe { strides.add(axis).write(stride) };
            stride = stride
                .checked_mul(dimension)
                .ok_or(Error::CompactStrideOverflow)?;
        }

        Ok(PreparedDynamic {
            allocation,
            shape,
            strides: if shape_len == 0 {
                std::ptr::null_mut()
            } else {
                strides
            },
            ndim,
        })
    }

    /// Prepares compact metadata which may borrow caller-owned shape values.
    ///
    /// # Safety
    ///
    /// A shape wrapped in [`Borrowed`] must remain alive and immutable until
    /// the resulting managed tensor is dropped.
    pub unsafe fn prepare_unchecked<M>(self) -> Result<PreparedDynamic<M>, Error>
    where
        M: ManagedTensorBase,
    {
        self.prepare_compact_inner()
    }
}

impl<Shape> Dynamic<Shape, Compact>
where
    Shape: OwnedDynamicPart,
{
    /// Allocates owned shape storage and computes explicit compact strides.
    pub fn prepare<M>(self) -> Result<PreparedDynamic<M>, Error>
    where
        M: ManagedTensorBase,
    {
        self.prepare_compact_inner()
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
    fn mixed_storage_uses_only_copied_extra() {
        let shape = [2_i64, 3];
        let prepared = unsafe {
            Dynamic::new(Borrowed(shape.as_slice()), Copied(vec![3_i16, 1]))
                .prepare_unchecked::<DLManagedTensor>()
                .unwrap()
        };
        let mut initialized = prepared.initialize(Box::new(()));
        initialized.set_dtype(crate::ffi::DLDataType::U8);
        let tensor = unsafe { initialized.finish() };

        assert_eq!(tensor.validate().unwrap().shape(), &shape);
        assert_eq!(tensor.validate().unwrap().strides().unwrap(), &[3, 1]);
    }

    #[test]
    fn compact_computes_explicit_strides() {
        let prepared = Dynamic::compact(Copied(vec![2_u64, 3, 4]))
            .prepare::<DLManagedTensor>()
            .unwrap();
        let mut initialized = prepared.initialize(Box::new(()));
        initialized.set_dtype(crate::ffi::DLDataType::U8);
        let tensor = unsafe { initialized.finish() };
        let tensor = tensor.validate().unwrap();

        assert_eq!(tensor.shape(), &[2, 3, 4]);
        assert_eq!(tensor.strides(), Some([12, 4, 1].as_slice()));
    }

    #[test]
    fn compact_scalar_may_omit_strides() {
        let prepared = Dynamic::compact(Copied(Vec::<i64>::new()))
            .prepare::<DLManagedTensor>()
            .unwrap();
        let mut initialized = prepared.initialize(Box::new(()));
        initialized.set_dtype(crate::ffi::DLDataType::U8);
        let tensor = unsafe { initialized.finish() };
        let tensor = tensor.validate().unwrap();

        assert!(tensor.shape().is_empty());
        assert_eq!(tensor.strides(), None);
    }

    #[test]
    fn compact_rejects_invalid_shape() {
        let negative = match Dynamic::compact(Copied(vec![2_i64, -1])).prepare::<DLManagedTensor>()
        {
            Ok(_) => panic!("negative shape must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(
            negative,
            Error::NegativeShapeValue { axis: 1, value: -1 }
        ));

        let overflow =
            match Dynamic::compact(Copied(vec![i64::MAX, 2])).prepare::<DLManagedTensor>() {
                Ok(_) => panic!("overflowing compact strides must be rejected"),
                Err(error) => error,
            };
        assert!(matches!(overflow, Error::CompactStrideOverflow));
    }
}
