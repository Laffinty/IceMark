#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "gui_chart.hpp"
#include <windows.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

static const int WIN_WIDTH = 1200;
static const int WIN_HEIGHT = 800;

struct DashboardData {
    std::vector<ChartSeries> series;
};

static DashboardData g_dashboard;

static std::wstring to_wstring(const std::string& s) {
    return std::wstring(s.begin(), s.end());
}

static void draw_text(HDC hdc, int x, int y, const std::string& text) {
    std::wstring w = to_wstring(text);
    TextOutW(hdc, x, y, w.c_str(), (int)w.length());
}

static void draw_chart(HDC hdc, const RECT& area,
                       const std::vector<double>& x_vals,
                       const std::vector<double>& y_vals,
                       const std::string& title,
                       const std::string& x_label,
                       const std::string& y_label) {
    if (x_vals.empty() || y_vals.empty() || x_vals.size() != y_vals.size()) return;

    int left   = area.left + 55;
    int right  = area.right - 15;
    int top    = area.top + 30;
    int bottom = area.bottom - 35;

    // Title
    SetTextColor(hdc, RGB(40, 40, 40));
    draw_text(hdc, left, area.top + 5, title);

    // Axes
    HPEN axis_pen = CreatePen(PS_SOLID, 1, RGB(180, 180, 180));
    HPEN old_pen = (HPEN)SelectObject(hdc, axis_pen);
    MoveToEx(hdc, left, top, NULL);
    LineTo(hdc, left, bottom);
    LineTo(hdc, right, bottom);
    SelectObject(hdc, old_pen);
    DeleteObject(axis_pen);

    double min_y = *std::min_element(y_vals.begin(), y_vals.end());
    double max_y = *std::max_element(y_vals.begin(), y_vals.end());
    if (max_y - min_y < 1e-6) max_y = min_y + 1.0;

    // Y labels
    SetTextColor(hdc, RGB(100, 100, 100));
    SetBkMode(hdc, TRANSPARENT);
    for (int i = 0; i <= 4; ++i) {
        double val = min_y + (max_y - min_y) * (4 - i) / 4.0;
        int y = bottom - (int)((val - min_y) / (max_y - min_y) * (bottom - top));
        std::string s = std::to_string((int)val);
        draw_text(hdc, left - 50, y - 6, s);
    }
    // Y axis name (rotated roughly by drawing vertically)
    draw_text(hdc, area.left + 2, top + 5, y_label);

    // X labels
    size_t nx = x_vals.size();
    size_t step = nx <= 8 ? 1 : nx / 8;
    for (size_t i = 0; i < nx; i += step) {
        int x = left + (int)((double)i / (nx - 1) * (right - left));
        std::string s = std::to_string((int)x_vals[i]);
        draw_text(hdc, x - 8, bottom + 6, s);
    }
    draw_text(hdc, (left + right) / 2 - 20, area.bottom - 15, x_label);

    // Polyline
    HPEN line_pen = CreatePen(PS_SOLID, 2, RGB(66, 133, 244));
    old_pen = (HPEN)SelectObject(hdc, line_pen);
    for (size_t i = 1; i < nx; ++i) {
        int x1 = left + (int)((double)(i - 1) / (nx - 1) * (right - left));
        int y1 = bottom - (int)((y_vals[i - 1] - min_y) / (max_y - min_y) * (bottom - top));
        int x2 = left + (int)((double)i / (nx - 1) * (right - left));
        int y2 = bottom - (int)((y_vals[i] - min_y) / (max_y - min_y) * (bottom - top));
        MoveToEx(hdc, x1, y1, NULL);
        LineTo(hdc, x2, y2);
    }
    SelectObject(hdc, old_pen);
    DeleteObject(line_pen);

    // Dots
    HBRUSH dot_brush = CreateSolidBrush(RGB(66, 133, 244));
    for (size_t i = 0; i < nx; ++i) {
        int x = left + (int)((double)i / (nx - 1) * (right - left));
        int y = bottom - (int)((y_vals[i] - min_y) / (max_y - min_y) * (bottom - top));
        Ellipse(hdc, x - 3, y - 3, x + 3, y + 3);
    }
    DeleteObject(dot_brush);
}

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
        RECT client;
        GetClientRect(hwnd, &client);

        FillRect(hdc, &client, (HBRUSH)GetStockObject(WHITE_BRUSH));

        int cols = 2;
        int rows = 2;
        int cell_w = (client.right - client.left) / cols;
        int cell_h = (client.bottom - client.top) / rows;

        for (size_t i = 0; i < g_dashboard.series.size() && i < 4; ++i) {
            RECT area;
            area.left   = (int)(i % cols) * cell_w + 10;
            area.top    = (int)(i / cols) * cell_h + 10;
            area.right  = area.left + cell_w - 20;
            area.bottom = area.top + cell_h - 20;
            const auto& s = g_dashboard.series[i];
            draw_chart(hdc, area, s.x_values, s.y_values, s.title, s.x_label, s.y_label);
        }

        EndPaint(hwnd, &ps);
        return 0;
    }
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

void show_results_dashboard(const std::vector<ChartSeries>& charts) {
    g_dashboard.series = charts;

    HINSTANCE hInstance = GetModuleHandle(NULL);
    WNDCLASSEXW wcex = {};
    wcex.cbSize = sizeof(WNDCLASSEXW);
    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = WndProc;
    wcex.hInstance = hInstance;
    wcex.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wcex.lpszClassName = L"IceMarkDashboard";
    wcex.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
    RegisterClassExW(&wcex);

    HWND hwnd = CreateWindowExW(
        0, L"IceMarkDashboard", L"IceMark Benchmark Results",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        WIN_WIDTH, WIN_HEIGHT,
        NULL, NULL, hInstance, NULL);

    if (!hwnd) return;

    ShowWindow(hwnd, SW_SHOWNORMAL);
    UpdateWindow(hwnd);

    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}
