//! Wait-free Single-Producer Single-Consumer (SPSC) ring buffer for hard real-time audio.
//! Cache-line aligned (64B) atomic cursors with local shadow indices to prevent false sharing.
//! Zero heap allocation after `RingBuffer::new(capacity)`.

use std::cell::UnsafeCell;
use std::fmt;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[repr(align(64))]
struct CachePaddedAtomicUsize(AtomicUsize);

struct SharedRing<T> {
    buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,
    capacity: usize,
    write_idx: CachePaddedAtomicUsize,
    read_idx: CachePaddedAtomicUsize,
    abandoned: AtomicBool,
}

unsafe impl<T: Send> Send for SharedRing<T> {}
unsafe impl<T: Send> Sync for SharedRing<T> {}

impl<T> Drop for SharedRing<T> {
    fn drop(&mut self) {
        let mut r = *self.read_idx.0.get_mut();
        let w = *self.write_idx.0.get_mut();
        let cap2 = 2 * self.capacity;
        while r != w {
            let slot = if r < self.capacity {
                r
            } else {
                r - self.capacity
            };
            unsafe {
                (*self.buffer[slot].get()).assume_init_drop();
            }
            r += 1;
            if r == cap2 {
                r = 0;
            }
        }
    }
}

/// Error returned when pushing into a full ring buffer.
#[derive(Debug, PartialEq, Eq)]
pub enum PushError<T> {
    Full(T),
}

impl<T> fmt::Display for PushError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ring buffer is full")
    }
}

impl<T: fmt::Debug> std::error::Error for PushError<T> {}

/// Error returned when popping from an empty ring buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PopError {
    Empty,
}

impl fmt::Display for PopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ring buffer is empty")
    }
}

impl std::error::Error for PopError {}

/// Factory for creating a `(Producer<T>, Consumer<T>)` wait-free SPSC ring buffer pair.
pub struct RingBuffer<T>(std::marker::PhantomData<T>);

impl<T> RingBuffer<T> {
    /// Allocates a new wait-free SPSC ring buffer with exact `capacity` slots.
    pub fn new(capacity: usize) -> (Producer<T>, Consumer<T>) {
        let cap = capacity.max(1);
        let mut vec = Vec::with_capacity(cap);
        for _ in 0..cap {
            vec.push(UnsafeCell::new(MaybeUninit::uninit()));
        }
        let shared = Arc::new(SharedRing {
            buffer: vec.into_boxed_slice(),
            capacity: cap,
            write_idx: CachePaddedAtomicUsize(AtomicUsize::new(0)),
            read_idx: CachePaddedAtomicUsize(AtomicUsize::new(0)),
            abandoned: AtomicBool::new(false),
        });
        let producer = Producer {
            shared: Arc::clone(&shared),
            write_idx: 0,
            cached_read_idx: 0,
        };
        let consumer = Consumer {
            shared,
            read_idx: 0,
            cached_write_idx: 0,
        };
        (producer, consumer)
    }
}

/// Single-Producer handle of the wait-free ring buffer.
pub struct Producer<T> {
    shared: Arc<SharedRing<T>>,
    write_idx: usize,
    cached_read_idx: usize,
}

unsafe impl<T: Send> Send for Producer<T> {}

impl<T> Producer<T> {
    #[inline]
    pub fn capacity(&self) -> usize {
        self.shared.capacity
    }

    #[inline]
    fn used_slots(write_idx: usize, read_idx: usize, cap: usize) -> usize {
        if write_idx >= read_idx {
            write_idx - read_idx
        } else {
            2 * cap - read_idx + write_idx
        }
    }

    /// Number of free slots currently available for writing.
    #[inline]
    pub fn slots(&self) -> usize {
        let cap = self.shared.capacity;
        let r = self.shared.read_idx.0.load(Ordering::Acquire);
        cap - Self::used_slots(self.write_idx, r, cap)
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.slots() == 0
    }

    #[inline]
    pub fn is_abandoned(&self) -> bool {
        self.shared.abandoned.load(Ordering::Acquire)
    }

    /// Wait-free push of a single item (Zero Heap Allocation).
    #[inline]
    pub fn push(&mut self, value: T) -> Result<(), PushError<T>> {
        let cap = self.shared.capacity;
        if Self::used_slots(self.write_idx, self.cached_read_idx, cap) == cap {
            self.cached_read_idx = self.shared.read_idx.0.load(Ordering::Acquire);
            if Self::used_slots(self.write_idx, self.cached_read_idx, cap) == cap {
                return Err(PushError::Full(value));
            }
        }
        let slot = if self.write_idx < cap {
            self.write_idx
        } else {
            self.write_idx - cap
        };
        unsafe {
            (*self.shared.buffer[slot].get()).write(value);
        }
        let mut next = self.write_idx + 1;
        if next == 2 * cap {
            next = 0;
        }
        self.write_idx = next;
        self.shared.write_idx.0.store(next, Ordering::Release);
        Ok(())
    }
}

impl<T> Drop for Producer<T> {
    fn drop(&mut self) {
        self.shared.abandoned.store(true, Ordering::Release);
    }
}

/// Single-Consumer handle of the wait-free ring buffer.
pub struct Consumer<T> {
    shared: Arc<SharedRing<T>>,
    read_idx: usize,
    cached_write_idx: usize,
}

unsafe impl<T: Send> Send for Consumer<T> {}

impl<T> Consumer<T> {
    #[inline]
    pub fn capacity(&self) -> usize {
        self.shared.capacity
    }

    /// Number of items currently ready to be popped.
    #[inline]
    pub fn slots(&self) -> usize {
        let cap = self.shared.capacity;
        let w = self.shared.write_idx.0.load(Ordering::Acquire);
        Producer::<T>::used_slots(w, self.read_idx, cap)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slots() == 0
    }

    #[inline]
    pub fn is_abandoned(&self) -> bool {
        self.shared.abandoned.load(Ordering::Acquire)
    }

    /// Wait-free pop of a single item (Zero Heap Allocation).
    #[inline]
    pub fn pop(&mut self) -> Result<T, PopError> {
        let cap = self.shared.capacity;
        if self.read_idx == self.cached_write_idx {
            self.cached_write_idx = self.shared.write_idx.0.load(Ordering::Acquire);
            if self.read_idx == self.cached_write_idx {
                return Err(PopError::Empty);
            }
        }
        let slot = if self.read_idx < cap {
            self.read_idx
        } else {
            self.read_idx - cap
        };
        let value = unsafe { (*self.shared.buffer[slot].get()).assume_init_read() };
        let mut next = self.read_idx + 1;
        if next == 2 * cap {
            next = 0;
        }
        self.read_idx = next;
        self.shared.read_idx.0.store(next, Ordering::Release);
        Ok(value)
    }
}

impl<T> Drop for Consumer<T> {
    fn drop(&mut self) {
        self.shared.abandoned.store(true, Ordering::Release);
    }
}
