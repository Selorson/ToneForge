//! Wait-free Triple Buffer (`TripleBuffer<T>`) for sharing large real-time frames (e.g., 128 KB
//! `VisualTelemetryFrame`) between the Real-Time Audio Thread and a 120 FPS UI / WGPU Thread.
//!
//! Uses 3 heap-boxed buffers (`[UnsafeCell<Box<T>>; 3]`) and a single cache-padded `AtomicU8`
//! holding `(back_buffer_idx | DIRTY_BIT)` so `publish()` and `update()` are single `AtomicU8::swap`
//! instructions with zero heap allocation and zero stack-overflow risk for large `T`.

use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

const BACK_INDEX_MASK: u8 = 0b0000_0011;
const DIRTY_BIT: u8 = 0b0000_0100;

#[repr(align(64))]
struct SharedState<T> {
    buffers: [UnsafeCell<Box<T>>; 3],
    back_info: AtomicU8,
}

unsafe impl<T: Send> Send for SharedState<T> {}
unsafe impl<T: Send> Sync for SharedState<T> {}

/// Wait-free Triple Buffer factory.
pub struct TripleBuffer<T> {
    input: Input<T>,
    output: Output<T>,
}

impl<T: Clone + Send> TripleBuffer<T> {
    /// Creates a new triple buffer initialized with clones of `initial` on the heap.
    pub fn new(initial: &T) -> (Input<T>, Output<T>) {
        Self::from_boxes([
            Box::new(initial.clone()),
            Box::new(initial.clone()),
            Box::new(initial.clone()),
        ])
    }
}

impl<T: Send> TripleBuffer<T> {
    /// Creates a new triple buffer from 3 pre-allocated `Box<T>` instances (avoids stack copies for 128 KB structs).
    pub fn from_boxes(boxes: [Box<T>; 3]) -> (Input<T>, Output<T>) {
        let [b0, b1, b2] = boxes;
        let shared = Arc::new(SharedState {
            buffers: [UnsafeCell::new(b0), UnsafeCell::new(b1), UnsafeCell::new(b2)],
            back_info: AtomicU8::new(1), // 0 = input, 1 = shared back, 2 = output
        });
        let input = Input {
            shared: Arc::clone(&shared),
            input_idx: 0,
        };
        let output = Output {
            shared,
            output_idx: 2,
        };
        (input, output)
    }

    pub fn split(self) -> (Input<T>, Output<T>) {
        (self.input, self.output)
    }
}

impl<T: Default + Send> Default for TripleBuffer<T> {
    fn default() -> Self {
        let (input, output) = Self::from_boxes([
            Box::new(T::default()),
            Box::new(T::default()),
            Box::new(T::default()),
        ]);
        Self { input, output }
    }
}

/// Producer handle (`Input<T>`) owned by the Real-Time Audio Thread.
pub struct Input<T> {
    shared: Arc<SharedState<T>>,
    input_idx: u8,
}

unsafe impl<T: Send> Send for Input<T> {}

impl<T: Send> Input<T> {
    /// Direct mutable access to the current input buffer in place (Zero Copy, Zero Allocation).
    #[inline]
    pub fn input_buffer(&mut self) -> &mut T {
        unsafe { &mut *(*self.shared.buffers[self.input_idx as usize].get()) }
    }

    /// Alias for `input_buffer(&mut self)` for ergonomic in-place frame construction.
    #[inline]
    pub fn input_buffer_mut(&mut self) -> &mut T {
        self.input_buffer()
    }

    /// Publishes the current input buffer to the shared back-buffer slot using a single wait-free `swap`.
    #[inline]
    pub fn publish(&mut self) {
        let new_back = self.input_idx | DIRTY_BIT;
        let prev_back = self.shared.back_info.swap(new_back, Ordering::AcqRel);
        self.input_idx = prev_back & BACK_INDEX_MASK;
    }

    /// Writes `value` into the input buffer and immediately publishes it.
    #[inline]
    pub fn write(&mut self, value: T) {
        *self.input_buffer() = value;
        self.publish();
    }

    /// Returns `true` if the consumer has not yet read the last published buffer.
    #[inline]
    pub fn consumed(&self) -> bool {
        (self.shared.back_info.load(Ordering::Relaxed) & DIRTY_BIT) == 0
    }
}

/// Consumer handle (`Output<T>`) owned by the 120 FPS UI / WGPU Thread.
pub struct Output<T> {
    shared: Arc<SharedState<T>>,
    output_idx: u8,
}

unsafe impl<T: Send> Send for Output<T> {}

impl<T: Send> Output<T> {
    /// Returns `true` if a newer buffer has been published since the last `update()` / `read()`.
    #[inline]
    pub fn updated(&self) -> bool {
        (self.shared.back_info.load(Ordering::Relaxed) & DIRTY_BIT) != 0
    }

    /// Swaps in the latest published back-buffer if dirty. Returns `true` if a new frame was acquired.
    #[inline]
    pub fn update(&mut self) -> bool {
        let current = self.shared.back_info.load(Ordering::Relaxed);
        if (current & DIRTY_BIT) != 0 {
            let prev = self.shared.back_info.swap(self.output_idx, Ordering::AcqRel);
            self.output_idx = prev & BACK_INDEX_MASK;
            true
        } else {
            false
        }
    }

    /// Immutable reference to the current output buffer without fetching a new frame.
    #[inline]
    pub fn output_buffer(&self) -> &T {
        unsafe { &*(*self.shared.buffers[self.output_idx as usize].get()) }
    }

    /// Updates from the shared back-buffer if dirty and returns a reference to the latest frame.
    #[inline]
    pub fn read(&mut self) -> &T {
        self.update();
        self.output_buffer()
    }

    /// Alias for `output_buffer()`.
    #[inline]
    pub fn peek_output_buffer(&self) -> &T {
        self.output_buffer()
    }
}
