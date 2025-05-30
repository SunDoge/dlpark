use std::ptr::NonNull;

use crate::managed_tensor::{Dlpack, ManagedTensor};

pub struct SafeManagedTensor(Dlpack);
