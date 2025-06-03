use crate::ffi::DataType;

pub trait InferDataType {
    fn data_type() -> DataType;
}

// impl_infer_data_type!(f32, DataType::F32)
macro_rules! impl_infer_data_type {
    ($t:ty, $data_type:expr) => {
        impl InferDataType for $t {
            fn data_type() -> DataType {
                $data_type
            }
        }
    };
}

impl_infer_data_type!(f32, DataType::F32);
impl_infer_data_type!(f64, DataType::F64);
impl_infer_data_type!(bool, DataType::BOOL);
impl_infer_data_type!(i8, DataType::I8);
impl_infer_data_type!(i16, DataType::I16);
impl_infer_data_type!(i32, DataType::I32);
impl_infer_data_type!(i64, DataType::I64);
impl_infer_data_type!(i128, DataType::I128);
impl_infer_data_type!(u8, DataType::U8);
impl_infer_data_type!(u16, DataType::U16);
impl_infer_data_type!(u32, DataType::U32);
impl_infer_data_type!(u64, DataType::U64);
impl_infer_data_type!(u128, DataType::U128);

#[cfg(feature = "half")]
impl_infer_data_type!(half::f16, DataType::F16);

#[cfg(feature = "half")]
impl_infer_data_type!(half::bf16, DataType::BF16);
