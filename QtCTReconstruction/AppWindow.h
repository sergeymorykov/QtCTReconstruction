#pragma once

#include "CTTypes.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <commctrl.h>
#include <gdiplus.h>

#include <string>
#include <vector>

class AppWindow {
public:
    AppWindow(HINSTANCE instance, int width, int height);
    ~AppWindow();

    bool create(const std::wstring& title);
    void show(int cmd_show);
    void updateDataset(const std::vector<ct::Slice>& originals,
                       const std::vector<ct::Slice>& reconstructions,
                       const std::vector<ct::Slice>& differences,
                       const std::vector<ct::Sinogram>& sinograms,
                       size_t initial_z);
    int messageLoop();

private:
    static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam);
    LRESULT handleMessage(UINT msg, WPARAM wparam, LPARAM lparam);
    void drawPanels(HDC hdc);
    void renderCurrentSlice();
    void layoutControls(int client_width, int client_height);
    void recreateBitmaps(const ct::Slice& original, const ct::Slice& reconstructed, const ct::Slice& difference, const ct::Sinogram& sinogram);

    static Gdiplus::Bitmap* sliceToBitmap(const ct::Slice& slice, bool difference_map);
    static Gdiplus::Bitmap* sinogramToBitmap(const ct::Sinogram& sinogram);

    HINSTANCE m_instance = nullptr;
    HWND m_hwnd = nullptr;
    HWND m_slider = nullptr;
    int m_width = 0;
    int m_height = 0;

    ULONG_PTR m_gdiplusToken = 0;

    Gdiplus::Bitmap* m_original = nullptr;
    Gdiplus::Bitmap* m_recon = nullptr;
    Gdiplus::Bitmap* m_diff = nullptr;
    Gdiplus::Bitmap* m_sino = nullptr;

    std::vector<ct::Slice> m_originalSlices;
    std::vector<ct::Slice> m_reconstructionSlices;
    std::vector<ct::Slice> m_differenceSlices;
    std::vector<ct::Sinogram> m_sinograms;
    size_t m_currentZ = 0;
};
