//! `hwnd_window`: Win32 Native `HWND` Floating Plugin Editor Window (`RegisterClassExW`,
//! `CreateWindowExW`, `AdjustWindowRectEx`, non-blocking `PeekMessageW` message pump, and clean
//! teardown) for hosting Steinberg VST3 `IPlugView` and CLAP 1.2 `clap_plugin_gui` editors.

use std::ffi::c_void;
use std::sync::OnceLock;
use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{COLOR_WINDOW, UpdateWindow};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::WindowsAndMessaging::{
    AdjustWindowRectEx, CS_HREDRAW, CS_VREDRAW, CW_USEDEFAULT, CreateWindowExW, DefWindowProcW,
    DestroyWindow, DispatchMessageW, IDC_ARROW, IsWindow, LoadCursorW, MSG, PM_REMOVE,
    PeekMessageW, RegisterClassExW, SW_HIDE, SW_SHOW, SWP_NOMOVE, SWP_NOZORDER, SetWindowPos,
    ShowWindow, TranslateMessage, WM_CLOSE, WNDCLASSEXW, WS_CAPTION, WS_MINIMIZEBOX, WS_OVERLAPPED,
    WS_POPUP, WS_SYSMENU, WS_THICKFRAME,
};

const HOST_WINDOW_CLASS_NAME: &[u16] = &[
    b'T' as u16,
    b'o' as u16,
    b'n' as u16,
    b'e' as u16,
    b'F' as u16,
    b'o' as u16,
    b'r' as u16,
    b'g' as u16,
    b'e' as u16,
    b'P' as u16,
    b'l' as u16,
    b'u' as u16,
    b'g' as u16,
    b'H' as u16,
    b'w' as u16,
    b'n' as u16,
    b'd' as u16,
    0,
];

unsafe extern "system" fn plugin_host_wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    if msg == WM_CLOSE {
        // Hide rather than unexpectedly freeing plugin view resources before detach;
        // host polls visibility or closes cleanly via `close_editor`.
        unsafe {
            ShowWindow(hwnd, SW_HIDE);
        }
        return 0;
    }
    unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
}

fn ensure_host_window_class_registered() -> Result<(), String> {
    static REGISTERED: OnceLock<Result<(), String>> = OnceLock::new();
    REGISTERED
        .get_or_init(|| unsafe {
            let hinstance = GetModuleHandleW(std::ptr::null());
            let wc = WNDCLASSEXW {
                cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
                style: CS_HREDRAW | CS_VREDRAW,
                lpfnWndProc: Some(plugin_host_wnd_proc),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: hinstance,
                hIcon: std::ptr::null_mut(),
                hCursor: LoadCursorW(std::ptr::null_mut(), IDC_ARROW),
                hbrBackground: (COLOR_WINDOW + 1) as *mut c_void,
                lpszMenuName: std::ptr::null(),
                lpszClassName: HOST_WINDOW_CLASS_NAME.as_ptr(),
                hIconSm: std::ptr::null_mut(),
            };
            let atom = RegisterClassExW(&wc);
            if atom == 0 {
                Err("RegisterClassExW failed for ToneForgePlugHwnd".to_string())
            } else {
                Ok(())
            }
        })
        .clone()
}

/// Native Win32 top-level floating container window for VST3 (`IPlugView`) and CLAP (`clap.gui`).
pub struct NativePluginWindow {
    hwnd: HWND,
    client_width: u32,
    client_height: u32,
}

// SAFETY: Win32 window handles can be stored inside `ClapHostedPluginNode` / `Vst3HostedPluginNode`
// (which implement `Send` for `UnifiedDspNode`), while GUI operations are invoked only from the host UI thread.
unsafe impl Send for NativePluginWindow {}

impl NativePluginWindow {
    /// Creates a Win32 container window sized so its client area matches `(client_width, client_height)`.
    /// When `visible` is `false`, the window stays hidden (ideal for headless automated GUI attach tests).
    pub fn open(
        title: &str,
        client_width: u32,
        client_height: u32,
        owner_hwnd: Option<isize>,
        resizable: bool,
        visible: bool,
    ) -> Result<Self, String> {
        ensure_host_window_class_registered()?;
        let w = client_width.clamp(160, 3840) as i32;
        let h = client_height.clamp(96, 2160) as i32;

        let mut style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
        if resizable {
            style |= WS_THICKFRAME;
        }
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: w,
            bottom: h,
        };
        unsafe {
            let _ = AdjustWindowRectEx(&mut rect, style, 0, 0);
        }
        let outer_w = (rect.right - rect.left).max(w);
        let outer_h = (rect.bottom - rect.top).max(h);

        let mut wide_title: Vec<u16> = title.encode_utf16().collect();
        wide_title.push(0);

        let parent_ptr = owner_hwnd
            .filter(|&v| v != 0)
            .map(|v| v as HWND)
            .unwrap_or(std::ptr::null_mut());

        let hwnd = unsafe {
            let hinstance = GetModuleHandleW(std::ptr::null());
            let created = CreateWindowExW(
                0,
                HOST_WINDOW_CLASS_NAME.as_ptr(),
                wide_title.as_ptr(),
                style,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                outer_w,
                outer_h,
                parent_ptr,
                std::ptr::null_mut(),
                hinstance,
                std::ptr::null(),
            );
            if created.is_null() {
                // Fallback popup window in restricted non-interactive desktop stations
                CreateWindowExW(
                    0,
                    HOST_WINDOW_CLASS_NAME.as_ptr(),
                    wide_title.as_ptr(),
                    WS_POPUP,
                    0,
                    0,
                    w,
                    h,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    hinstance,
                    std::ptr::null(),
                )
            } else {
                created
            }
        };

        if hwnd.is_null() {
            return Err("CreateWindowExW returned NULL HWND for plugin container".to_string());
        }

        if visible {
            unsafe {
                ShowWindow(hwnd, SW_SHOW);
                let _ = UpdateWindow(hwnd);
            }
        }

        Ok(Self {
            hwnd,
            client_width: w as u32,
            client_height: h as u32,
        })
    }

    #[inline]
    pub fn hwnd_raw(&self) -> *mut c_void {
        self.hwnd
    }

    #[inline]
    pub fn hwnd_isize(&self) -> isize {
        self.hwnd as isize
    }

    #[inline]
    pub fn client_size(&self) -> (u32, u32) {
        (self.client_width, self.client_height)
    }

    pub fn is_valid(&self) -> bool {
        if self.hwnd.is_null() {
            return false;
        }
        unsafe { IsWindow(self.hwnd) != 0 }
    }

    pub fn show(&self) {
        if self.is_valid() {
            unsafe {
                ShowWindow(self.hwnd, SW_SHOW);
                let _ = UpdateWindow(self.hwnd);
            }
        }
    }

    pub fn hide(&self) {
        if self.is_valid() {
            unsafe {
                ShowWindow(self.hwnd, SW_HIDE);
            }
        }
    }

    /// Resizes the outer frame so the inner client area matches `(new_client_w, new_client_h)`.
    pub fn resize_client(&mut self, new_client_w: u32, new_client_h: u32, resizable: bool) {
        if !self.is_valid() {
            return;
        }
        let w = new_client_w.clamp(160, 3840) as i32;
        let h = new_client_h.clamp(96, 2160) as i32;
        let mut style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
        if resizable {
            style |= WS_THICKFRAME;
        }
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: w,
            bottom: h,
        };
        unsafe {
            let _ = AdjustWindowRectEx(&mut rect, style, 0, 0);
            let _ = SetWindowPos(
                self.hwnd,
                std::ptr::null_mut(),
                0,
                0,
                (rect.right - rect.left).max(w),
                (rect.bottom - rect.top).max(h),
                SWP_NOMOVE | SWP_NOZORDER,
            );
        }
        self.client_width = w as u32;
        self.client_height = h as u32;
    }

    /// Non-blocking Win32 message pump (`PeekMessageW` + `TranslateMessage` + `DispatchMessageW`).
    pub fn pump_messages(&self) -> usize {
        let mut processed = 0usize;
        unsafe {
            let mut msg: MSG = std::mem::zeroed();
            while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
                processed += 1;
                if processed >= 256 {
                    break;
                }
            }
        }
        processed
    }

    pub fn destroy_now(&mut self) {
        if !self.hwnd.is_null() {
            unsafe {
                if IsWindow(self.hwnd) != 0 {
                    let _ = DestroyWindow(self.hwnd);
                }
            }
            self.hwnd = std::ptr::null_mut();
        }
    }
}

impl Drop for NativePluginWindow {
    fn drop(&mut self) {
        self.destroy_now();
    }
}
