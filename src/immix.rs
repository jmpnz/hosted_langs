//! Wrapping our heads around Immix a mark region based GC.
//!
//! References :
//!
//! - https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/immix/
//! - https://wingolog.org/archives/2022/08/07/coarse-or-lazy
//! - https://wingolog.org/archives/2022/06/20/blocks-and-pages-and-large-objects
//! - https://wingolog.org/archives/2022/06/15/defragmentation
//!
//! GC Terminology:
//!
//! Mutator: The thread of execution that writes and mutates objects on the heap
//!
//! Live objects: The graph of objects the mutator can reach.
//!
//! Dead objects: Objects disconnected from the graph of live objects.
//!
//! Collector: The thread of execution that identifies unreachable objects
//! and marks them as to be freed.
//!
//! Fragmentation: Since objects have different sizes after allocating and
//! freeing many objects gaps of unused memory appear between allocated objects
//! that are too small to be used by most objects but add up to significant
//! wasted space.
//!
//! Evacuation: The process where the collector moves live objects to another
//! block of memory so that the original block can be de-fragmented.
//!
//! Immix :
//!
//! Immix is an automatic memory management scheme (garbage collector) that uses
//! two techniques, region based allocations and opportunistic defragmentation.
//!
//! Mark Region Collection :
//!
//! Mark-Region collectors uses a fixed size region based memory allocation
//! model where objects may not span across regions and bumps allocates into
//! free regions where there are no living objects. Free space is reclaimed
//! during collection.
//!
//! Immix refines regions into coarse-grained blocks of 32kb and fine grained
//! lines of 128 bytes. Objects may span multiple lines and free-lines are
//! also identified in partially free-blocks, during allocation the algorithm
//! tries to find free-lines large enough for the new object otherwise it
//! allocates an entire new block.
//!
//! Opportunistic Evacuation :
//!
//! Evacuation is a somewhat atomic process, when the collector encounters
//! a live object in a candidate block (candidate for defragmentation) the
//! object is evacuated if and only if the collector can perform all of the
//! following, allocate an object similar to the mutator but not in a candidate
//! block, the collector leavs a forwarding pointer that allows live references
//! to be updated to the new location.
//!
//! Implementation details :
//!
//! - Defragmentation is triggered at the beginning of a collection phase, if
//! there is one or more partially free blocks that the allocator didn't mutate
//! in the previous collection.
//!
//! - Candidate blocks for defragmentation are selected using a heuristic for
//! how many gaps exist in a block.
//!
//! - Headroom blocks are free blocks not used for object allocation but for
//! evacuation candidates.
//!
//! - Pinning is supported by skipping pinned objects (they are not evacuated)

use crate::allocator::{
    alloc_size_of, AllocHeader, AllocObject, AllocRaw, ArraySize, Mark, SizeClass,
};
use crate::block::{Block, BlockError};
use crate::rawptr::RawPtr;

/// The block size is fixed at 32 KBytes.
pub const BLOCK_SIZE_BITS: usize = 15;
pub const BLOCK_SIZE: usize = 1 << BLOCK_SIZE_BITS;
/// The line size is fixed at 128 bytes.
pub const LINE_SIZE_BITS: usize = 7;
pub const LINE_SIZE: usize = 1 << LINE_SIZE_BITS;

/// Number of total lines in a single block (256).
pub const LINE_COUNT: usize = BLOCK_SIZE / LINE_SIZE;

/// Capacity of a block is the block size minus `LINE_COUNT` used for tagging
/// lines.
pub const BLOCK_CAPACITY: usize = BLOCK_SIZE - LINE_COUNT;

/// Mask used to align to word boundaries.
use std::mem::size_of;
pub const ALLOC_ALIGN_BYTES: usize = size_of::<usize>();
pub const ALLOC_ALIGN_MASK: usize = !(ALLOC_ALIGN_BYTES - 1);

/// The first line-mark offset.
pub const LINE_MARK_START: usize = BLOCK_CAPACITY;

/// Allocation errors
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AllocError {
    // Allocation failed due unaligned requests.
    BadRequest,
    // Out of memory.
    OOM,
}

impl From<BlockError> for AllocError {
    fn from(error: BlockError) -> AllocError {
        match error {
            BlockError::BadRequest => AllocError::BadRequest,
            BlockError::OOM => AllocError::OOM,
        }
    }
}

/// Implementation of the Immix heap, `H` is a generic argument to designate
/// an object header. Since the heap can be re-usable with different object
/// implementations, user will specify the header tag for his implementation.
pub struct ImmixHeap<H> {
    // We need to wrap `BlockList` in `UnsafeCell` to provide interior
    // mutability.
    blocks: std::cell::UnsafeCell<BlockList>,
    // Marker for header types.
    _header_type: std::marker::PhantomData<*const H>,
}

impl<H> ImmixHeap<H> {
    /// Create a new `ImmixHeap` instance.
    pub fn new() -> ImmixHeap<H> {
        ImmixHeap {
            blocks: std::cell::UnsafeCell::new(BlockList::new()),
            _header_type: std::marker::PhantomData,
        }
    }

    fn find_space(
        &self,
        alloc_size: usize,
        size_class: SizeClass,
    ) -> Result<*const u8, AllocError> {
        let blocks = unsafe { &mut *self.blocks.get() };

        // If the size allocation is for large objects return `BadRequest`.
        if size_class == SizeClass::Large {
            return Err(AllocError::BadRequest);
        }

        // `find_space` targets allocation into the current block.
        let space = match blocks.head {
            // We have an existing block in `head`.
            Some(ref mut head) => {
                // If this is a medium size object, move the allocation
                // to the `overflow` block.
                if size_class == SizeClass::Medium && alloc_size > head.current_hole_size() {
                    return blocks.overflow_alloc(alloc_size);
                }

                // If this is a small size object try allocating in the current
                // block.
                match head.inner_alloc(alloc_size) {
                    // The block has a suitable hole to fit in.
                    Some(space) => space,
                    // The block doesn't have suitable space.
                    None => {
                        // Fetch head so we can store it and replace it with
                        // a fresh block.
                        let prev = std::mem::replace(head, BumpBlock::new()?);
                        // Push old head to rest.
                        blocks.rest.push(prev);
                        // Allocate the object in the current freshly allocated
                        // block.
                        head.inner_alloc(alloc_size).expect("Unexpected error !")
                    }
                }
            }

            // We have no existing blocks to use.
            None => {
                let mut head = BumpBlock::new()?;
                let space = head
                    .inner_alloc(alloc_size)
                    .expect("Expected {alloc_size} to find in single block.");
                blocks.head = Some(head);
                space
            }
        } as *const u8;
        Ok(space)
    }
}

/// Implementation of the Allocator API for `ImmixHeap`.
impl<H: AllocHeader> AllocRaw for ImmixHeap<H> {
    type Header = H;

    fn alloc<T>(&self, object: T) -> Result<RawPtr<T>, AllocError>
    where
        T: AllocObject<<Self::Header as AllocHeader>::TypeId>,
    {
        // Compute the object and header sizes.
        let header_size = std::mem::size_of::<Self::Header>();
        let object_size = std::mem::size_of::<T>();
        // Total size to allocate.
        let total_size = header_size + object_size;
        // TODO: Fix `alloc_size_of` to handle header sizes.
        let alloc_size = alloc_size_of(total_size);
        // Get the range of the allocation.
        let size_class = SizeClass::get_for_size(alloc_size)?;

        let space = self.find_space(alloc_size, size_class)?;
        // Create an object header, this is flaky.
        let header = Self::Header::new::<T>(object_size as ArraySize, size_class, Mark::Allocated);
        // Write header into the front of the allocation.
        unsafe {
            std::ptr::write(space as *mut Self::Header, header);
        }

        // Write the object right after the header.
        let object_space = unsafe { space.offset(header_size as isize) };
        unsafe {
            std::ptr::write(object_space as *mut T, object);
        }

        // Return raw pointer to the written object.
        Ok(RawPtr::new(object_space as *const T))
    }
    fn alloc_array(&self, size_bytes: ArraySize) -> Result<RawPtr<u8>, AllocError> {
        // calculate the total size of the array and it's header
        let header_size = size_of::<Self::Header>();
        let total_size = header_size + size_bytes as usize;

        // round the size to the next word boundary to keep objects aligned and get the size class
        let alloc_size = alloc_size_of(total_size);
        let size_class = SizeClass::get_for_size(alloc_size)?;

        // attempt to allocate enough space for the header and the array
        let space = self.find_space(alloc_size, size_class)?;

        // instantiate an object header for an array, setting the mark bit to "allocated"
        let header = Self::Header::new_array(size_bytes, size_class, Mark::Allocated);

        // write the header into the front of the allocated space
        unsafe {
            std::ptr::write(space as *mut Self::Header, header);
        }

        // calculate where the array will begin after the header
        let array_space = unsafe { space.offset(header_size as isize) };

        // Initialize object_space to zero here.
        // If using the system allocator for any objects (SizeClass::Large, for example),
        // the memory may already be zeroed.
        let array =
            unsafe { std::slice::from_raw_parts_mut(array_space as *mut u8, size_bytes as usize) };
        // The compiler should recognize this as optimizable
        for byte in array {
            *byte = 0;
        }

        // return a pointer to the array in the allocated space
        Ok(RawPtr::new(array_space as *const u8))
    }
    fn get_header(object: std::ptr::NonNull<()>) -> std::ptr::NonNull<Self::Header> {
        unsafe {
            std::ptr::NonNull::new_unchecked(object.cast::<Self::Header>().as_ptr().offset(-1))
        }
    }

    fn get_object(header: std::ptr::NonNull<Self::Header>) -> std::ptr::NonNull<()> {
        unsafe { std::ptr::NonNull::new_unchecked(header.as_ptr().offset(1).cast::<()>()) }
    }
}

/// BlockList manages multiple bump allocated blocks.
pub struct BlockList {
    // Head of the block list, where the current allocations happen.
    head: Option<BumpBlock>,
    // Block kept for handling overflow objects that that don't fit in the free
    // space available in `head`.
    overflow: Option<BumpBlock>,
    // Rest of the blocks that have been allocated into but aren't recycled yet.
    rest: Vec<BumpBlock>,
}

impl BlockList {
    /// Create a new empty blocklist.
    pub fn new() -> Self {
        Self {
            head: None,
            overflow: None,
            rest: Vec::new(),
        }
    }

    /// Allocate into the overflow block.
    pub fn overflow_alloc(&mut self, alloc_size: usize) -> Result<*const u8, AllocError> {
        // Allocation size must fit in a block.
        assert!(alloc_size <= BLOCK_CAPACITY);

        let space = match self.overflow {
            Some(ref mut overflow) => {
                // We have an overflow block.
                match overflow.inner_alloc(alloc_size) {
                    // The block has enough space.
                    Some(space) => space,
                    // The block doesn't have enough space, push this overflow
                    // block to rest and allocate a new one.
                    None => {
                        let prev = std::mem::replace(overflow, BumpBlock::new()?);
                        self.rest.push(prev);
                        overflow
                            .inner_alloc(alloc_size)
                            .expect("Unexpected allocation error !")
                    }
                }
            }
            None => {
                // We don't have an overflow block yet.
                let mut overflow = BumpBlock::new()?;

                // We expect the allocation to succeed since the assertion
                // was checked.
                let space = overflow
                    .inner_alloc(alloc_size)
                    .expect("We expected {alloc_size} to fit !");

                space
            }
        } as *const u8;
        Ok(space)
    }
}

/// Bump allocated blocks, wraps `block::Block` with a bump pointer & metadata.
/// Bump allocation is done from high to low addresses :
/// https://fitzgeraldnick.com/2019/11/01/always-bump-downwards.html
pub struct BumpBlock {
    // Bump pointer that indexes in the block where the last object was written.
    cursor: *const u8,
    // Pointer to the last available space in the block, since we use only
    // `BLOCK_CAPACITY` for actual allocations not the entire 32 KB.
    limit: *const u8,
    // Actualy heap allocated block that we use for bump allocations.
    block: Block,
    // Block metadata such as pointers to the mark line bytes.
    meta: BlockMeta,
}

impl BumpBlock {
    /// Allocate a new block of heap space and it's metadata, placing a
    /// pointer to the metadata in the first word of the block.
    pub fn new() -> Result<Self, AllocError> {
        let inner_block = Block::new(BLOCK_SIZE)?;
        let block_ptr = inner_block.as_ptr();

        Ok(Self {
            // Cursor points at capacity offset.
            cursor: unsafe { block_ptr.add(BLOCK_CAPACITY) },
            limit: block_ptr,
            block: inner_block,
            meta: BlockMeta::new(block_ptr),
        })
    }

    /// Write an object into the block at the given offset.
    ///
    /// # Unsafe
    ///
    /// Will panic if offset overflows.
    unsafe fn write<T>(&mut self, object: T, offset: usize) -> *const T {
        let p = self.block.as_ptr().add(offset) as *mut T;
        std::ptr::write(p, object);
        p
    }

    /// Allocate `alloc_size` bytes in this block, returning a pointer to
    /// start of the allocation space.
    pub fn inner_alloc(&mut self, alloc_size: usize) -> Option<*const u8> {
        let ptr = self.cursor as usize;
        let limit = self.limit as usize;

        // Bump allocation is done downwards at a word boundary.
        let next_ptr = ptr.checked_sub(alloc_size)? & ALLOC_ALIGN_MASK;

        // The allocation doesn't fit within this block which means we can't
        // allocate an object in this block.
        if next_ptr < limit {
            let block_relative_limit =
                unsafe { self.limit.sub(self.block.as_ptr() as usize) } as usize;

            if block_relative_limit > 0 {
                if let Some((cursor, limit)) = self
                    .meta
                    .find_next_available_hole(block_relative_limit, alloc_size)
                {
                    self.cursor = unsafe { self.block.as_ptr().add(cursor) };
                    self.limit = unsafe { self.block.as_ptr().add(limit) };
                    return self.inner_alloc(alloc_size);
                }
            }

            None
        } else {
            self.cursor = next_ptr as *const u8;
            Some(self.cursor)
        }
    }

    /// Return the size of the current hole.
    pub fn current_hole_size(&self) -> usize {
        self.cursor as usize - self.limit as usize
    }
}

/// Bump allocated block metadata used for keeping track of gaps in the block
/// that can be used for new allocations. The metadata holds pointers to the
/// mark lines used to tag lines in the block and the block itself.
pub struct BlockMeta {
    lines: *mut u8,
}

impl BlockMeta {
    /// Creates a new metadata instance.
    pub fn new(block_ptr: *const u8) -> Self {
        let mut meta = Self {
            lines: unsafe { block_ptr.add(LINE_MARK_START) as *mut u8 },
        };

        meta.reset();

        meta
    }

    /// Reset all mark flags to unmarked.
    pub fn reset(&mut self) {
        unsafe {
            for i in 0..LINE_COUNT {
                *self.lines.add(i) = 0;
            }
        }
    }

    /// Use the entry as a block mark bit space.
    ///
    /// # Unsafe
    /// Overflow isn't checked.
    unsafe fn as_block_mark(&mut self) -> &mut u8 {
        // Use the last byte of the block since no object will occupy
        // the line associated with it.
        &mut *self.lines.add(LINE_COUNT - 1)
    }

    /// Use the entry as a line mark bit space.
    ///
    /// # Unsafe
    /// Overflow isn't checked.
    unsafe fn as_line_mark(&mut self, line: usize) -> &mut u8 {
        &mut *self.lines.add(line)
    }

    /// Mark the indexed line.
    pub fn mark_line(&mut self, index: usize) {
        unsafe { *self.as_line_mark(index) = 1 }
    }

    /// Indicate the entire block as marked.
    pub fn mark_block(&mut self) {
        unsafe { *self.as_block_mark() = 1 }
    }

    /// Finds the next available space for an allocation, we consider available
    /// space to be a gap of unmarked lines to fit `alloc_size`.
    /// If there's such a gap we return a tuple of `(cursor, limit)` where
    /// `cusor` is the new bump pointer value and `limit` is the lower bound
    /// of the available gap. If no space is found it returns `None`.
    /// `starting_at` is the offset into the block where we start the search
    pub fn find_next_available_hole(
        &self,
        starting_at: usize,
        alloc_size: usize,
    ) -> Option<(usize, usize)> {
        // Count of available successive unmarked lines.
        let mut count = 0;
        // Line where we start looking.
        let starting_line = starting_at / LINE_SIZE;
        // Number of lines required for this allocation.
        let lines_required = (alloc_size + LINE_SIZE - 1) / LINE_SIZE;
        // End starts at `starting_line` and is decremented to point towards
        // the end of the space.
        let mut end = starting_line;

        // We iterate over lines in descending order.
        for index in (0..starting_line).rev() {
            // Fetch the mark bit into `marked`.
            let marked = unsafe { *self.lines.add(index) };

            if marked == 0 {
                // if the current line is unmarked we increment the `count`.
                count += 1;

                // We reached line 0 and found enough lines to fit `alloc_size`.
                if index == 0 && count >= lines_required {
                    // Convert the line offsets back to byte offsets.
                    let limit = index * LINE_SIZE;
                    let cursor = end * LINE_SIZE;
                    return Some((cursor, limit));
                }
            } else {
                // This block is marked and we've reached the end of the current gap.
                if count > lines_required {
                    // At least two previous blocks were not marked, return this gap.
                    // 1 for walking back to the current marked line.
                    // 1 for walking back to the previously marked line.
                    let limit = (index + 2) * LINE_SIZE;
                    let cursor = end * LINE_SIZE;
                    return Some((cursor, limit));
                }

                // If this line is marked and we haven't returned a new cursor, limit pair
                // reset the search state.
                count = 0;
                end = index;
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::{AllocObject, AllocTypeId, Mark, SizeClass};
    use crate::block::Block;
    use std::slice::from_raw_parts;

    struct TestHeader {
        _size_class: SizeClass,
        _mark: Mark,
        type_id: TestTypeId,
        _size_bytes: u32,
    }

    #[derive(PartialEq, Copy, Clone)]
    enum TestTypeId {
        Biggish,
        Stringish,
        Usizeish,
        Array,
    }

    impl AllocTypeId for TestTypeId {}

    impl AllocHeader for TestHeader {
        type TypeId = TestTypeId;

        fn new<O: AllocObject<Self::TypeId>>(size: u32, size_class: SizeClass, mark: Mark) -> Self {
            TestHeader {
                _size_class: size_class,
                _mark: mark,
                type_id: O::TYPE_ID,
                _size_bytes: size,
            }
        }

        fn new_array(size: u32, size_class: SizeClass, mark: Mark) -> Self {
            TestHeader {
                _size_class: size_class,
                _mark: mark,
                type_id: TestTypeId::Array,
                _size_bytes: size,
            }
        }

        fn mark(&mut self) {}

        fn is_marked(&self) -> bool {
            true
        }

        fn size_class(&self) -> SizeClass {
            SizeClass::Small
        }

        fn size(&self) -> u32 {
            8
        }

        fn type_id(&self) -> TestTypeId {
            self.type_id
        }
    }

    struct Big {
        _huge: [u8; BLOCK_SIZE + 1],
    }

    impl Big {
        fn make() -> Big {
            Big {
                _huge: [0u8; BLOCK_SIZE + 1],
            }
        }
    }

    impl AllocObject<TestTypeId> for Big {
        const TYPE_ID: TestTypeId = TestTypeId::Biggish;
    }

    impl AllocObject<TestTypeId> for String {
        const TYPE_ID: TestTypeId = TestTypeId::Stringish;
    }

    impl AllocObject<TestTypeId> for usize {
        const TYPE_ID: TestTypeId = TestTypeId::Usizeish;
    }

    #[test]
    fn immix_heap_can_allocate_memory() {
        let mem = ImmixHeap::<TestHeader>::new();

        match mem.alloc(String::from("foo")) {
            Ok(s) => {
                let orig = unsafe { s.as_ref() };
                assert!(*orig == String::from("foo"));
            }

            Err(_) => panic!("Allocation failed"),
        }
    }

    #[test]
    fn immix_heap_handles_large_allocs() {
        let mem = ImmixHeap::<TestHeader>::new();
        assert!(mem.alloc(Big::make()) == Err(AllocError::BadRequest));
    }

    #[test]
    fn immix_heap_can_allocate_many_objects() {
        let mem = ImmixHeap::<TestHeader>::new();

        let mut obs = Vec::new();

        // allocate a sequence of numbers
        for i in 0..(BLOCK_SIZE * 3) {
            match mem.alloc(i as usize) {
                Err(_) => assert!(false, "Allocation failed unexpectedly"),
                Ok(ptr) => obs.push(ptr),
            }
        }

        // check that all values of allocated words match the original
        // numbers written, that no heap corruption occurred
        for (i, ob) in obs.iter().enumerate() {
            println!("{} {}", i, unsafe { ob.as_ref() });
            assert!(i == unsafe { *ob.as_ref() })
        }
    }

    #[test]
    fn immix_heap_can_allocate_arrays() {
        let mem = ImmixHeap::<TestHeader>::new();

        let size = 2048;

        match mem.alloc_array(size) {
            Err(_) => assert!(false, "Array allocation failed unexpectedly"),

            Ok(ptr) => {
                // Validate that array is zero initialized all the way through
                let ptr = ptr.as_ptr();

                let array = unsafe { from_raw_parts(ptr, size as usize) };

                for byte in array {
                    assert!(*byte == 0);
                }
            }
        }
    }

    #[test]
    fn immix_heap_handles_headers() {
        let mem = ImmixHeap::<TestHeader>::new();

        match mem.alloc(String::from("foo")) {
            Ok(s) => {
                let untyped_ptr = s.as_untyped();
                let header_ptr = ImmixHeap::<TestHeader>::get_header(untyped_ptr);
                dbg!(header_ptr);
                let header = unsafe { &*header_ptr.as_ptr() as &TestHeader };

                assert!(header.type_id() == TestTypeId::Stringish);
            }

            Err(_) => panic!("Allocation failed"),
        }
    }

    // Helper function: given the Block, fill all holes with u32 values
    // and return the number of values allocated.
    // Also assert that all allocated values are unchanged as allocation
    // proceeds.
    fn loop_check_allocate(b: &mut BumpBlock) -> usize {
        let mut v = Vec::new();
        let mut index = 0;

        loop {
            //println!("cursor={}, limit={}", b.cursor, b.limit);
            if let Some(ptr) = b.inner_alloc(ALLOC_ALIGN_BYTES) {
                let u32ptr = ptr as *mut u32;

                assert!(!v.contains(&u32ptr));

                v.push(u32ptr);
                unsafe { *u32ptr = index }

                index += 1;
            } else {
                break;
            }
        }

        for (index, u32ptr) in v.iter().enumerate() {
            unsafe {
                assert!(**u32ptr == index as u32);
            }
        }

        index as usize
    }

    #[test]
    fn bump_block_empty_block() {
        let mut b = BumpBlock::new().unwrap();

        let count = loop_check_allocate(&mut b);
        let expect = BLOCK_CAPACITY / ALLOC_ALIGN_BYTES;

        println!("expect={}, count={}", expect, count);
        assert!(count == expect);
    }

    #[test]
    fn bump_block_half_block() {
        // This block has an available hole as the second half of the block
        let mut b = BumpBlock::new().unwrap();

        for i in 0..(LINE_COUNT / 2) {
            b.meta.mark_line(i);
        }
        let occupied_bytes = (LINE_COUNT / 2) * LINE_SIZE;

        b.limit = b.cursor; // block is recycled

        let count = loop_check_allocate(&mut b);
        let expect = (BLOCK_CAPACITY - LINE_SIZE - occupied_bytes) / ALLOC_ALIGN_BYTES;

        println!("expect={}, count={}", expect, count);
        assert!(count == expect);
    }

    #[test]
    fn bump_block_conservatively_marked_block() {
        // This block has every other line marked, so the alternate lines are conservatively
        // marked. Nothing should be allocated in this block.

        let mut b = BumpBlock::new().unwrap();

        for i in 0..LINE_COUNT {
            if i % 2 == 0 {
                b.meta.mark_line(i);
            }
        }

        b.limit = b.cursor; // block is recycled

        let count = loop_check_allocate(&mut b);

        println!("count={}", count);
        assert!(count == 0);
    }

    #[test]
    fn block_meta_find_next_hole() {
        // A set of marked lines with a couple holes.
        // The first hole should be seen as conservatively marked.
        // The second hole should be the one selected.
        let block = Block::new(BLOCK_SIZE).unwrap();
        let mut meta = BlockMeta::new(block.as_ptr());

        meta.mark_line(0);
        meta.mark_line(1);
        meta.mark_line(2);
        meta.mark_line(4);
        meta.mark_line(10);

        // line 5 should be conservatively marked
        let expect = Some((10 * LINE_SIZE, 6 * LINE_SIZE));

        let got = meta.find_next_available_hole(10 * LINE_SIZE, LINE_SIZE);

        println!("test_find_next_hole got {:?} expected {:?}", got, expect);

        assert!(got == expect);
    }
    #[test]
    fn block_meta_find_next_hole_at_line_zero() {
        // Should find the hole starting at the beginning of the block
        let block = Block::new(BLOCK_SIZE).unwrap();
        let mut meta = BlockMeta::new(block.as_ptr());

        meta.mark_line(3);
        meta.mark_line(4);
        meta.mark_line(5);

        let expect = Some((3 * LINE_SIZE, 0));

        let got = meta.find_next_available_hole(3 * LINE_SIZE, LINE_SIZE);

        println!(
            "test_find_next_hole_at_line_zero got {:?} expected {:?}",
            got, expect
        );

        assert!(got == expect);
    }

    #[test]
    fn block_meta_find_next_hole_at_block_end() {
        // The first half of the block is marked.
        // The second half of the block should be identified as a hole.
        let block = Block::new(BLOCK_SIZE).unwrap();
        let mut meta = BlockMeta::new(block.as_ptr());

        let halfway = LINE_COUNT / 2;

        for i in halfway..LINE_COUNT {
            meta.mark_line(i);
        }

        // because halfway line should be conservatively marked
        let expect = Some((halfway * LINE_SIZE, 0));

        let got = meta.find_next_available_hole(BLOCK_CAPACITY, LINE_SIZE);

        println!(
            "test_find_next_hole_at_block_end got {:?} expected {:?}",
            got, expect
        );

        assert!(got == expect);
    }

    #[test]
    fn block_meta_find_hole_all_conservatively_marked() {
        // Every other line is marked.
        // No hole should be found.
        let block = Block::new(BLOCK_SIZE).unwrap();
        let mut meta = BlockMeta::new(block.as_ptr());

        for i in 0..LINE_COUNT {
            if i % 2 == 0 {
                // there is no stable step function for range
                meta.mark_line(i);
            }
        }

        let got = meta.find_next_available_hole(BLOCK_CAPACITY, LINE_SIZE);

        println!(
            "test_find_hole_all_conservatively_marked got {:?} expected None",
            got
        );

        assert!(got == None);
    }

    #[test]
    fn block_meta_find_entire_block() {
        // No marked lines. Entire block is available.
        let block = Block::new(BLOCK_SIZE).unwrap();
        let meta = BlockMeta::new(block.as_ptr());

        let expect = Some((BLOCK_CAPACITY, 0));
        let got = meta.find_next_available_hole(BLOCK_CAPACITY, LINE_SIZE);

        println!("test_find_entire_block got {:?} expected {:?}", got, expect);

        assert!(got == expect);
    }
}
