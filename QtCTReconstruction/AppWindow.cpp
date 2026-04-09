#include "AppWindow.h"

#include "Utils.h"

#include <algorithm>
#include <cmath>

namespace {

void freeBitmap(Gdiplus::Bitmap*& ptr) {
    delete ptr;
    ptr = nullptr;
}

} // namespace

AppWindow::AppWindow(HINSTANCE instance, const int width, const int height)
    : m_instance(instance), m_width(width), m_height(height) {
    Gdiplus::GdiplusStartupInput gdiplusStartupInput;
    Gdiplus::GdiplusStartup(&m_gdiplusToken, &gdiplusStartupInput, nullptr);
}

AppWindow::~AppWindow() {
    freeBitmap(m_original);
    freeBitmap(m_recon);
    freeBitmap(m_diff);
    freeBitmap(m_sino);

    if (m_gdiplusToken != 0) {
        Gdiplus::GdiplusShutdown(m_gdiplusToken);
    }
}

bool AppWindow::create(const std::wstring& title) {
    INITCOMMONCONTROLSEX icc{};
    icc.dwSize = sizeof(icc);
    icc.dwICC = ICC_BAR_CLASSES;
    InitCommonControlsEx(&icc);

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(WNDCLASSEXW);
    wc.lpfnWndProc = AppWindow::WndProc;
    wc.hInstance = m_instance;
    wc.lpszClassName = L"CTReconstructionWindow";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);

    RegisterClassExW(&wc);

    m_hwnd = CreateWindowExW(
        0,
        wc.lpszClassName,
        title.c_str(),
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        m_width,
        m_height,
        nullptr,
        nullptr,
        m_instance,
        this);

    return m_hwnd != nullptr;
}

void AppWindow::show(const int cmd_show) {
    ShowWindow(m_hwnd, cmd_show);
    UpdateWindow(m_hwnd);
}

void AppWindow::updateDataset(const std::vector<ct::Slice>& originals,
                              const std::vector<ct::Slice>& reconstructions,
                              const std::vector<ct::Slice>& differences,
                              const std::vector<ct::Sinogram>& sinograms,
                              const size_t initial_z) {
    m_originalSlices = originals;
    m_reconstructionSlices = reconstructions;
    m_differenceSlices = differences;
    m_sinograms = sinograms;

    if (m_originalSlices.empty()) {
        return;
    }

    const size_t max_z = m_originalSlices.size() - 1;
    m_currentZ = (initial_z > max_z) ? max_z : initial_z;

    if (m_slider != nullptr) {
        SendMessageW(m_slider, TBM_SETRANGEMIN, TRUE, 0);
        SendMessageW(m_slider, TBM_SETRANGEMAX, TRUE, static_cast<LPARAM>(max_z));
        SendMessageW(m_slider, TBM_SETPOS, TRUE, static_cast<LPARAM>(m_currentZ));
        SendMessageW(m_slider, TBM_SETTICFREQ, 1, 0);
    }

    renderCurrentSlice();
    if (m_hwnd != nullptr) {
        InvalidateRect(m_hwnd, nullptr, TRUE);
    }
}

int AppWindow::messageLoop() {
    MSG msg{};
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return static_cast<int>(msg.wParam);
}

LRESULT CALLBACK AppWindow::WndProc(HWND hwnd, const UINT msg, const WPARAM wparam, const LPARAM lparam) {
    AppWindow* self = nullptr;
    if (msg == WM_NCCREATE) {
        const auto* cs = reinterpret_cast<CREATESTRUCTW*>(lparam);
        self = reinterpret_cast<AppWindow*>(cs->lpCreateParams);
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
        self->m_hwnd = hwnd;
    } else {
        self = reinterpret_cast<AppWindow*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    }

    return self ? self->handleMessage(msg, wparam, lparam) : DefWindowProcW(hwnd, msg, wparam, lparam);
}

LRESULT AppWindow::handleMessage(const UINT msg, const WPARAM wparam, const LPARAM lparam) {
    switch (msg) {
    case WM_CREATE: {
        m_slider = CreateWindowExW(
            0,
            TRACKBAR_CLASSW,
            L"",
            WS_CHILD | WS_VISIBLE | TBS_AUTOTICKS | TBS_HORZ,
            10,
            10,
            300,
            30,
            m_hwnd,
            nullptr,
            m_instance,
            nullptr);
        return 0;
    }
    case WM_SIZE: {
        const int w = LOWORD(lparam);
        const int h = HIWORD(lparam);
        layoutControls(w, h);
        InvalidateRect(m_hwnd, nullptr, TRUE);
        return 0;
    }
    case WM_HSCROLL: {
        if (reinterpret_cast<HWND>(lparam) == m_slider && !m_originalSlices.empty()) {
            const LRESULT pos = SendMessageW(m_slider, TBM_GETPOS, 0, 0);
            const size_t z = static_cast<size_t>(pos);
            if (z != m_currentZ && z < m_originalSlices.size()) {
                m_currentZ = z;
                renderCurrentSlice();
                InvalidateRect(m_hwnd, nullptr, TRUE);
            }
            return 0;
        }
        break;
    }
    case WM_PAINT: {
        PAINTSTRUCT ps{};
        HDC hdc = BeginPaint(m_hwnd, &ps);
        drawPanels(hdc);
        EndPaint(m_hwnd, &ps);
        return 0;
    }
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    default:
        return DefWindowProcW(m_hwnd, msg, wparam, lparam);
    }
    return DefWindowProcW(m_hwnd, msg, wparam, lparam);
}

void AppWindow::drawPanels(HDC hdc) {
    RECT client{};
    GetClientRect(m_hwnd, &client);
    const int slider_h = 56;
    const int panel_w = (client.right - client.left) / 2;
    const int panel_h = (client.bottom - client.top - slider_h) / 2;

    Gdiplus::Graphics g(hdc);
    g.SetInterpolationMode(Gdiplus::InterpolationModeNearestNeighbor);
    g.SetPixelOffsetMode(Gdiplus::PixelOffsetModeHalf);

    auto drawKeepAspect = [&g](Gdiplus::Bitmap* bmp, const int rx, const int ry, const int rw, const int rh) {
        if (bmp == nullptr || rw <= 0 || rh <= 0) {
            return;
        }
        const int iw = static_cast<int>(bmp->GetWidth());
        const int ih = static_cast<int>(bmp->GetHeight());
        if (iw <= 0 || ih <= 0) {
            return;
        }

        const float sx = static_cast<float>(rw) / static_cast<float>(iw);
        const float sy = static_cast<float>(rh) / static_cast<float>(ih);
        const float s = (sx < sy) ? sx : sy;

        const int dw = static_cast<int>(static_cast<float>(iw) * s);
        const int dh = static_cast<int>(static_cast<float>(ih) * s);
        const int dx = rx + (rw - dw) / 2;
        const int dy = ry + (rh - dh) / 2;
        g.DrawImage(bmp, dx, dy, dw, dh);
    };

    if (m_original != nullptr) {
        drawKeepAspect(m_original, 0, 0, panel_w, panel_h);
    }
    if (m_sino != nullptr) {
        g.DrawImage(m_sino, panel_w, 0, panel_w, panel_h);
    }
    if (m_recon != nullptr) {
        drawKeepAspect(m_recon, 0, panel_h, panel_w, panel_h);
    }
    if (m_diff != nullptr) {
        drawKeepAspect(m_diff, panel_w, panel_h, panel_w, panel_h);
    }

    SetBkMode(hdc, TRANSPARENT);
    TextOutW(hdc, 8, 8, L"Original", 8);
    TextOutW(hdc, panel_w + 8, 8, L"Sinogram", 8);
    TextOutW(hdc, 8, panel_h + 8, L"Reconstruction", 14);
    TextOutW(hdc, panel_w + 8, panel_h + 8, L"Difference", 10);

    const std::wstring z_label = L"Z: " + std::to_wstring(m_currentZ);
    TextOutW(hdc, 12, client.bottom - 48, z_label.c_str(), static_cast<int>(z_label.size()));
}

void AppWindow::layoutControls(const int client_width, const int client_height) {
    if (m_slider == nullptr) {
        return;
    }
    MoveWindow(m_slider, 100, client_height - 52, client_width - 120, 34, TRUE);
}

void AppWindow::renderCurrentSlice() {
    if (m_originalSlices.empty() || m_reconstructionSlices.size() != m_originalSlices.size() ||
        m_differenceSlices.size() != m_originalSlices.size() || m_sinograms.size() != m_originalSlices.size() ||
        m_currentZ >= m_originalSlices.size()) {
        return;
    }

    recreateBitmaps(
        m_originalSlices[m_currentZ],
        m_reconstructionSlices[m_currentZ],
        m_differenceSlices[m_currentZ],
        m_sinograms[m_currentZ]);
}

void AppWindow::recreateBitmaps(const ct::Slice& original,
                                const ct::Slice& reconstructed,
                                const ct::Slice& difference,
                                const ct::Sinogram& sinogram) {
    freeBitmap(m_original);
    freeBitmap(m_recon);
    freeBitmap(m_diff);
    freeBitmap(m_sino);

    m_original = sliceToBitmap(original, false);
    m_recon = sliceToBitmap(reconstructed, false);
    m_diff = sliceToBitmap(difference, true);
    m_sino = sinogramToBitmap(sinogram);
}

Gdiplus::Bitmap* AppWindow::sliceToBitmap(const ct::Slice& slice, const bool difference_map) {
    const int width = static_cast<int>(slice.width);
    const int height = static_cast<int>(slice.height);
    auto* bmp = new Gdiplus::Bitmap(width, height, PixelFormat32bppARGB);

    float min_v = slice.empty() ? 0.0f : slice[0][0];
    float max_v = min_v;
    for (const auto& row : slice) {
        for (size_t x = 0; x < slice.width; ++x) {
            const float v = row[x];
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }
    }
    if (std::abs(max_v - min_v) < 1e-6f) {
        max_v = min_v + 1.0f;
    }

    const float span = max_v - min_v;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float v = slice[static_cast<size_t>(height - 1 - y)][static_cast<size_t>(x)];

            float n;
            if (difference_map) {
                // Модуль разницы в фиксированном диапазоне 0—50 HU
                n = ct::utils::clamp(std::abs(v) / 50.0f, 0.0f, 1.0f);
            } else {
                // Обычная нормализация
                n = ct::utils::clamp((v - min_v) / span, 0.0f, 1.0f);
            }

            const BYTE c = static_cast<BYTE>(n * 255.0f);
            bmp->SetPixel(x, y, Gdiplus::Color(255, c, c, c));
        }
    }
    return bmp;
}



Gdiplus::Bitmap* AppWindow::sinogramToBitmap(const ct::Sinogram& sinogram) {
    if (sinogram.data.empty()) {
        return nullptr;
    }

    const int width = static_cast<int>(sinogram.data.width);
    const int height = static_cast<int>(sinogram.data.height);
    auto* bmp = new Gdiplus::Bitmap(width, height, PixelFormat32bppARGB);

    float min_v = sinogram.data[0][0];
    float max_v = sinogram.data[0][0];
    for (const auto& row : sinogram.data) {
        for (size_t x = 0; x < sinogram.data.width; ++x) {
            const float v = row[x];
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }
    }
    if (std::abs(max_v - min_v) < 1e-6f) {
        max_v = min_v + 1.0f;
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float v = sinogram.data[static_cast<size_t>(height - 1 - y)][static_cast<size_t>(x)];
            const float n = ct::utils::clamp((v - min_v) / (max_v - min_v), 0.0f, 1.0f);
            const BYTE c = static_cast<BYTE>(n * 255.0f);
            bmp->SetPixel(x, y, Gdiplus::Color(255, c, c, c));
        }
    }
    return bmp;
}
