use std::{
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
};

/// Owns a resource together with a view that may depend on that owner.
///
/// This is a low-level building block for owning, self-referential adapters
/// whose lifetime relationship cannot be represented in `V`. The relationship
/// between `O` and `V` is established by [`Borrowed::new_unchecked`] and is not
/// checked by the Rust type system.
///
/// `Borrowed` may be moved, and always drops the view before the owner. It does
/// not expose mutable access to the owner or allow the pair to be separated,
/// because either operation could invalidate the view.
pub struct Borrowed<O, V> {
    view: ManuallyDrop<V>,
    owner: ManuallyDrop<O>,
}

impl<O, V> Borrowed<O, V> {
    /// Creates an owning dependent view.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    ///
    /// - `view` remains valid while stored in the returned `Borrowed`;
    /// - `owner` keeps every resource referenced by `view` alive;
    /// - moving `owner`, `view`, or the returned `Borrowed` does not invalidate
    ///   the relationship. In particular, `view` must not point into an inline
    ///   part of `owner` whose address changes when `owner` moves;
    /// - dropping `view` before `owner` is valid.
    ///
    /// Owners that contain a stable pointer to a separate allocation, such as
    /// [`Box`] or a managed allocation handle, commonly satisfy the move
    /// requirement.
    pub unsafe fn new_unchecked(owner: O, view: V) -> Self {
        Self {
            view: ManuallyDrop::new(view),
            owner: ManuallyDrop::new(owner),
        }
    }

    /// Returns the owner that keeps the view valid.
    pub fn owner(&self) -> &O {
        &self.owner
    }

    /// Returns the dependent view.
    pub fn view(&self) -> &V {
        &self.view
    }

    /// Returns mutable access to the dependent view.
    ///
    /// Safe operations on `V` must preserve the relationship established by
    /// [`Borrowed::new_unchecked`].
    pub fn view_mut(&mut self) -> &mut V {
        &mut self.view
    }
}

impl<O, V> Deref for Borrowed<O, V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        self.view()
    }
}

impl<O, V> DerefMut for Borrowed<O, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.view_mut()
    }
}

impl<O, V> Drop for Borrowed<O, V> {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.view);
            ManuallyDrop::drop(&mut self.owner);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, rc::Rc};

    struct Tracked {
        name: &'static str,
        drops: Rc<RefCell<Vec<&'static str>>>,
    }

    impl Drop for Tracked {
        fn drop(&mut self) {
            self.drops.borrow_mut().push(self.name);
        }
    }

    #[test]
    fn drops_view_before_owner() {
        let drops = Rc::new(RefCell::new(Vec::new()));
        let owner = Tracked {
            name: "owner",
            drops: drops.clone(),
        };
        let view = Tracked {
            name: "view",
            drops: drops.clone(),
        };

        drop(unsafe { Borrowed::new_unchecked(owner, view) });

        assert_eq!(&*drops.borrow(), &["view", "owner"]);
    }
}
