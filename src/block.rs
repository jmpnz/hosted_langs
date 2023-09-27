use std::ptr::NonNull;

/// Pointer to a memory block.
pub type BlockPtr = NonNull<u8>;
/// Size of a memory block.
pub type BlockSize = usize;

/// Requested memory blocks are defined as a base address and allocation size.
pub struct Block {
    // Base address of the allocated block.
    ptr: BlockPtr,
    // Size of the allocated block.
    size: BlockSize,
}

impl Block {
    /// Allocate a new memory block.
    pub fn new(size: BlockSize) -> Result<Block, BlockError> {
        if !size.is_power_of_two() {
            Err(BlockError::BadRequest)
        } else {
            Ok(Block {
                ptr: internal::alloc_block(size)?,
                size,
            })
        }
    }

    /// Return the size of the allocated block in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Return a raw pointer to the allocated block.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        internal::dealloc_block(self.ptr, self.size)
    }
}

/// Allocation can fail if the requested size isn't aligned or there isn't
/// enough space for the allocation.
#[derive(Debug, PartialEq)]
pub enum BlockError {
    /// Returned if the requested block size isn't aligned.
    BadRequest,
    /// Returned if we're out of memory.
    OOM,
}

pub fn blocks() {
    println!("Calling from block.rs")
}

mod internal {

    use super::{BlockError, BlockPtr, BlockSize};
    use std::alloc::{alloc, dealloc, Layout};
    use std::ptr::NonNull;

    /// Allocate a block of memory.
    pub fn alloc_block(size: BlockSize) -> Result<BlockPtr, BlockError> {
        unsafe {
            let layout = Layout::from_size_align_unchecked(size, size);
            let ptr = alloc(layout);

            if ptr.is_null() {
                Err(BlockError::OOM)
            } else {
                Ok(NonNull::new_unchecked(ptr))
            }
        }
    }

    /// Deallocate and free a block of memory.
    pub fn dealloc_block(ptr: BlockPtr, size: BlockSize) {
        unsafe {
            let layout = Layout::from_size_align_unchecked(size, size);
            dealloc(ptr.as_ptr(), layout);
        }
    }
}

#[cfg(test)]
mod tests {

    use super::{Block, BlockError, BlockSize};

    fn alloc_dealloc(size: BlockSize) -> Result<(), BlockError> {
        let block = Block::new(size)?;

        // the block address bitwise AND the alignment bits (size - 1) should
        // be a mutually exclusive set of bits
        let mask = size - 1;
        assert!((block.ptr.as_ptr() as usize & mask) ^ mask == mask);

        drop(block);
        Ok(())
    }

    #[test]
    fn test_bad_sizealign() {
        assert!(alloc_dealloc(999) == Err(BlockError::BadRequest))
    }

    #[test]
    fn test_4k() {
        assert!(alloc_dealloc(4096).is_ok())
    }

    #[test]
    fn test_32k() {
        assert!(alloc_dealloc(32768).is_ok())
    }

    #[test]
    fn test_16m() {
        assert!(alloc_dealloc(16 * 1024 * 1024).is_ok())
    }
}
