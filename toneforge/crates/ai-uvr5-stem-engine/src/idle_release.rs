//! Nhả các bộ tách stem ONNX GPU khi rảnh (đợt tối ưu tài nguyên 2026-11-14).
//!
//! Đo trên RTX 4080: MỘT lần tách stem file 30 s để lại 11,1 GB VRAM + 14 GB RAM, rảnh 5 phút vẫn
//! y nguyên — 4 bộ tách sống trong biến toàn cục suốt đời app. Nay mỗi bộ nằm trong một `IdleSlot`:
//! dùng xong thì `touch()`, rảnh `IDLE_RELEASE` thì luồng canh đưa ô về "chưa nạp" (giải phóng phiên ONNX
//! và vùng nhớ CUDA), lần dùng sau tự nạp lại.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

pub const IDLE_RELEASE: Duration = Duration::from_secs(600);
const CHECK_EVERY: Duration = Duration::from_secs(30);

/// Thứ gì giữ phiên ONNX thì cài cái này để luồng canh buông được. `false` = đang bận, để lần kiểm sau.
pub trait Release: Sync {
    fn release(&self) -> bool;
}

static LAST_USED: Mutex<Option<Instant>> = Mutex::new(None);
static WATCHER: OnceLock<()> = OnceLock::new();
static SLOTS: Mutex<Vec<&'static dyn Release>> = Mutex::new(Vec::new());

pub fn register(slot: &'static dyn Release) {
    SLOTS.lock().unwrap_or_else(|e| e.into_inner()).push(slot);
}

/// Đánh dấu vừa dùng xong một bộ tách; lần đầu gọi thì bật luồng canh.
pub fn touch() {
    *LAST_USED.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
    WATCHER.get_or_init(|| {
        let _ = std::thread::Builder::new().name("stem-idle-release".into()).spawn(|| loop {
            std::thread::sleep(CHECK_EVERY);
            let mut last = LAST_USED.lock().unwrap_or_else(|e| e.into_inner());
            if should_release(*last, Instant::now(), IDLE_RELEASE) && release_all() {
                *last = None;
            }
        });
    });
}

fn should_release(last: Option<Instant>, now: Instant, idle: Duration) -> bool {
    matches!(last, Some(t) if now.duration_since(t) >= idle)
}

fn release_all() -> bool {
    let slots = SLOTS.lock().unwrap_or_else(|e| e.into_inner());
    slots.iter().fold(true, |all, s| s.release() && all)
}

enum SlotState<T> {
    Empty,
    Loaded(T),
    Failed,
}

/// Ô chứa một bộ tách: nạp lười, nhả được khi rảnh, tự nạp lại lần dùng sau.
/// Nạp hỏng (thiếu DLL/model) thì nhớ là hỏng như hành vi cũ, không thử lại mỗi lần.
pub struct IdleSlot<T: Send + 'static> {
    state: Mutex<SlotState<T>>,
    registered: AtomicBool,
}

impl<T: Send + 'static> IdleSlot<T> {
    pub const fn new() -> Self {
        Self { state: Mutex::new(SlotState::Empty), registered: AtomicBool::new(false) }
    }

    /// Chạy `f` trên bộ tách, nạp bằng `load` nếu ô đang trống. `None` nếu nạp hỏng.
    pub fn with<R>(&'static self, load: impl FnOnce() -> Option<T>, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        if !self.registered.swap(true, Ordering::AcqRel) {
            register(self);
        }
        let out = {
            let mut st = self.state.lock().ok()?;
            if matches!(*st, SlotState::Empty) {
                *st = match load() {
                    Some(t) => SlotState::Loaded(t),
                    None => SlotState::Failed,
                };
            }
            match &mut *st {
                SlotState::Loaded(t) => Some(f(t)),
                _ => None,
            }
        };
        touch();
        out
    }
}

impl<T: Send + 'static> Release for IdleSlot<T> {
    fn release(&self) -> bool {
        match self.state.try_lock() {
            Ok(mut st) => {
                if matches!(*st, SlotState::Loaded(_)) {
                    *st = SlotState::Empty;
                }
                true
            }
            Err(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn chi_nha_khi_da_ranh_du_lau() {
        let t0 = Instant::now();
        let idle = Duration::from_secs(600);
        assert!(!should_release(None, t0, idle));
        assert!(!should_release(Some(t0), t0 + Duration::from_secs(599), idle));
        assert!(should_release(Some(t0), t0 + Duration::from_secs(600), idle));
    }

    #[test]
    fn nha_xong_thi_lan_sau_nap_lai() {
        static SLOT: IdleSlot<u32> = IdleSlot::new();
        static LOADS: AtomicUsize = AtomicUsize::new(0);
        let load = || {
            LOADS.fetch_add(1, Ordering::SeqCst);
            Some(7)
        };
        assert_eq!(SLOT.with(load, |v| *v), Some(7));
        assert_eq!(SLOT.with(load, |v| *v), Some(7));
        assert_eq!(LOADS.load(Ordering::SeqCst), 1, "đang nạp thì không nạp lại");

        assert!(SLOT.release());
        assert_eq!(SLOT.with(load, |v| *v), Some(7));
        assert_eq!(LOADS.load(Ordering::SeqCst), 2, "đã nhả thì lần dùng sau nạp lại");
    }

    #[test]
    fn nap_hong_thi_nho_la_hong() {
        static SLOT: IdleSlot<u32> = IdleSlot::new();
        static LOADS: AtomicUsize = AtomicUsize::new(0);
        let load = || {
            LOADS.fetch_add(1, Ordering::SeqCst);
            None
        };
        assert_eq!(SLOT.with(load, |v| *v), None);
        assert_eq!(SLOT.with(load, |v| *v), None);
        assert_eq!(LOADS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn dang_ban_thi_khong_nha() {
        static SLOT: IdleSlot<u32> = IdleSlot::new();
        SLOT.with(|| Some(1), |_| ());
        let _busy = SLOT.state.lock().unwrap();
        assert!(!SLOT.release());
    }
}
