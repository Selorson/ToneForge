//! Windows MMCSS `"Pro Audio"` Thread Priority Elevation (`mmcss_pro_audio.rs`).
//!
//! Reconstructs FL Studio 26's real-time audio worker elevation (`0x0044b6d0` in `FLEngine_x64.dll`):
//! - Calls `AvSetMmThreadCharacteristicsW("Pro Audio\0")` + `AvSetMmThreadPriority(..., AVRT_PRIORITY_CRITICAL)`
//!   on thread startup.
//! - Calls `AvRevertMmThreadCharacteristics` automatically on drop.

/// Windows MMCSS thread priority levels (`AVRT_PRIORITY`).
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmcssThreadPriority {
    Low = -1,
    Normal = 0,
    High = 1,
    Critical = 2,
}

/// RAII guard for Windows Multimedia Class Scheduler Service (`MMCSS`) `"Pro Audio"` priority.
#[derive(Debug)]
pub struct ProAudioThreadGuard {
    #[cfg(target_os = "windows")]
    task_handle: isize,
    task_index: u32,
    priority: MmcssThreadPriority,
}

impl ProAudioThreadGuard {
    /// Elevates the calling thread to MMCSS `"Pro Audio"` with `AVRT_PRIORITY_CRITICAL` (`2`).
    #[inline]
    pub fn enter() -> Self {
        Self::elevate_current_thread()
    }

    /// Elevates the calling thread to MMCSS `"Pro Audio"` with `AVRT_PRIORITY_CRITICAL` (`2`).
    pub fn elevate_current_thread() -> Self {
        Self::elevate_with_priority(MmcssThreadPriority::Critical)
    }

    /// Elevates the calling thread to MMCSS `"Pro Audio"` with the requested `MmcssThreadPriority`.
    pub fn elevate_with_priority(priority: MmcssThreadPriority) -> Self {
        #[cfg(target_os = "windows")]
        {
            #[link(name = "avrt")]
            unsafe extern "system" {
                fn AvSetMmThreadCharacteristicsW(
                    task_name: *const u16,
                    task_index: *mut u32,
                ) -> isize;
                fn AvSetMmThreadPriority(avrt_handle: isize, priority: i32) -> i32;
            }

            // UTF-16LE null-terminated "Pro Audio\0"
            const PRO_AUDIO_UTF16: [u16; 10] = [
                b'P' as u16,
                b'r' as u16,
                b'o' as u16,
                b' ' as u16,
                b'A' as u16,
                b'u' as u16,
                b'd' as u16,
                b'i' as u16,
                b'o' as u16,
                0,
            ];
            let mut task_index: u32 = 0;
            let handle =
                unsafe { AvSetMmThreadCharacteristicsW(PRO_AUDIO_UTF16.as_ptr(), &mut task_index) };
            if handle != 0 {
                unsafe {
                    let _ = AvSetMmThreadPriority(handle, priority as i32);
                }
            }
            Self {
                task_handle: handle,
                task_index,
                priority,
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            Self {
                task_index: 0,
                priority,
            }
        }
    }

    /// Returns `true` if the OS granted an active MMCSS `"Pro Audio"` handle.
    #[inline]
    pub fn is_elevated(&self) -> bool {
        #[cfg(target_os = "windows")]
        {
            self.task_handle != 0
        }
        #[cfg(not(target_os = "windows"))]
        {
            false
        }
    }

    /// Returns the MMCSS task index assigned by Windows.
    #[inline]
    pub fn task_index(&self) -> u32 {
        self.task_index
    }

    /// Returns the configured MMCSS thread priority.
    #[inline]
    pub fn priority(&self) -> MmcssThreadPriority {
        self.priority
    }
}

impl Drop for ProAudioThreadGuard {
    fn drop(&mut self) {
        #[cfg(target_os = "windows")]
        {
            #[link(name = "avrt")]
            unsafe extern "system" {
                fn AvRevertMmThreadCharacteristics(avrt_handle: isize) -> i32;
            }
            if self.task_handle != 0 {
                unsafe {
                    let _ = AvRevertMmThreadCharacteristics(self.task_handle);
                }
                self.task_handle = 0;
            }
        }
    }
}
