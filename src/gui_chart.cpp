#include "gui_chart.hpp"
#include <windows.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

static const int WIN_WIDTH = 800;
static const int WIN_HEIGHT = 600;
static const int PADDING_LEFT = 70;
static const int PADDING_RIGHT = 30;
static const int PADDING_TOP = 50;
static const int PADDING_BOTTOM = 50;

struct ChartData {
    std::string title;
    std::string x_label;
    std::string y_label;
    std::vector<double> x_values;
    std::vector<double> y_values;
};

static ChartData g_chart;

static double get_text_width(HDC hdc, const wchar_t* text, int len) {
    SIZE sz;
    GetTextExtentPoint32W(hdc, text, len, &sz);
    return (double)sz.cx;
}

static double get_text_height(HDC hdc, const wchar_t* text, int len) {
    SIZE sz;
    GetTextExtentPoint32W(hdc, text, len, &sz);
    return (double)sz.cy;
}

static void draw_axis(HDC hdc, RECT& client, double max_y, double min_y, double max_x) {
    HPEN axis_pen = CreatePen(PS_SOLID, 1, RGB(200, 200, 200));
    HPEN old_pen = (HPEN)SelectObject(hdc, axis_pen);

    int chart_left = PADDING_LEFT;
    int chart_right = client.right - PADDING_RIGHT;
    int chart_top = PADDING_TOP;
    int chart_bottom = client.bottom - PADDING_BOTTOM;

    MoveToEx(hdc, chart_left, chart_top, NULL);
    LineTo(hdc, chart_left, chart_bottom);
    LineTo(hdc, chart_right, chart_bottom);

    SelectObject(hdc, old_pen);
    DeleteObject(axis_pen);
}

static void draw_labels(HDC hdc, RECT& client, double max_y, double min_y, double max_x) {
    int chart_left = PADDING_LEFT;
    int chart_right = client.right - PADDING_RIGHT;
    int chart_top = PADDING_TOP;
    int chart_bottom = client.bottom - PADDING_BOTTOM;

    HFONT hfont = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
    HFONT old_font = (HFONT)SelectObject(hdc, hfont);
    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, RGB(100, 100, 100));

    int num_y_labels = 5;
    double y_range = max_y - min_y;
    wchar_t buf[64];
    for (int i = 0; i <= num_y_labels; ++i) {
        double val = min_y + y_range * (num_y_labels - i) / (double)num_y_labels;
        int y_pos = chart_bottom - (int)((val - min_y) / y_range * (chart_bottom - chart_top));
        swprintf(buf, 64, L"%.1f", val);
        TextOutW(hdc, 5, y_pos - 5, buf, (int)wcslen(buf));
    }

    int num_x_labels = (std::min)((int)g_chart.x_values.size(), 8);
    if (num_x_labels < 2) num_x_labels = 2;
    for (int i = 0; i < num_x_labels; ++i) {
        int idx = i * (int)(g_chart.x_values.size() / num_x_labels);
        if (idx >= (int)g_chart.x_values.size()) idx = (int)g_chart.x_values.size() - 1;
        double val = g_chart.x_values[idx];
        int x_pos = chart_left + (int)(idx * 1.0 / (g_chart.x_values.size() - 1) * (chart_right - chart_left));
        swprintf(buf, 64, L"%d", (int)val);
        TextOutW(hdc, x_pos - 10, chart_bottom + 8, buf, (int)wcslen(buf));
    }

    SelectObject(hdc, old_font);
}

static void draw_title(HDC hdc, RECT& client, const std::string& title) {
    HFONT hfont = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
    HFONT old_font = (HFONT)SelectObject(hdc, hfont);
    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, RGB(60, 60, 60));

    std::wstring wtitle = L"IceMark - " + std::wstring(title.begin(), title.end());
    TextOutW(hdc, 10, 10, wtitle.c_str(), (int)wtitle.length());

    std::wstring wylabel = std::wstring(g_chart.y_label.begin(), g_chart.y_label.end());
    TextOutW(hdc, 10, client.bottom / 2, wylabel.c_str(), (int)wylabel.length());

    std::wstring wxlabel = std::wstring(g_chart.x_label.begin(), g_chart.x_label.end());
    int xlen = (int)wcslen(wxlabel.c_str());
    int xpos = (client.right - client.left) / 2 - (int)get_text_width(hdc, wxlabel.c_str(), xlen) / 2;
    TextOutW(hdc, xpos, client.bottom - 25, wxlabel.c_str(), xlen);

    SelectObject(hdc, old_font);
}

static void draw_polyline(HDC hdc, RECT& client, double max_y, double min_y, double max_x) {
    if (g_chart.x_values.size() < 2) return;

    int chart_left = PADDING_LEFT;
    int chart_right = client.right - PADDING_RIGHT;
    int chart_top = PADDING_TOP;
    int chart_bottom = client.bottom - PADDING_BOTTOM;

    HPEN line_pen = CreatePen(PS_SOLID, 2, RGB(66, 133, 244));
    HPEN old_pen = (HPEN)SelectObject(hdc, line_pen);

    std::vector<POINT> points(g_chart.x_values.size());
    double y_range = max_y - min_y;
    if (y_range < 0.001) y_range = 1.0;

    for (size_t i = 0; i < g_chart.x_values.size(); ++i) {
        double x_ratio = (double)i / (g_chart.x_values.size() - 1);
        double y_ratio = (g_chart.y_values[i] - min_y) / y_range;
        y_ratio = 1.0 - y_ratio;
        points[i].x = (int)(chart_left + x_ratio * (chart_right - chart_left));
        points[i].y = (int)(chart_top + y_ratio * (chart_bottom - chart_top));
    }

    Polyline(hdc, points.data(), (int)points.size());

    HBRUSH dot_brush = CreateSolidBrush(RGB(66, 133, 244));
    for (size_t i = 0; i < points.size(); ++i) {
        Ellipse(hdc, points[i].x - 3, points[i].y - 3, points[i].x + 3, points[i].y + 3);
    }
    DeleteObject(dot_brush);

    SelectObject(hdc, old_pen);
    DeleteObject(line_pen);
}

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
        RECT client;
        GetClientRect(hwnd, &client);

        double max_y = *std::max_element(g_chart.y_values.begin(), g_chart.y_values.end());
        double min_y = *std::min_element(g_chart.y_values.begin(), g_chart.y_values.end());
        double max_x = *std::max_element(g_chart.x_values.begin(), g_chart.x_values.end());

        draw_title(hdc, client, g_chart.title);
        draw_axis(hdc, client, max_y, min_y, max_x);
        draw_labels(hdc, client, max_y, min_y, max_x);
        draw_polyline(hdc, client, max_y, min_y, max_x);

        EndPaint(hwnd, &ps);
        return 0;
    }
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

void show_chart_window(const std::string& title,
                       const std::string& x_label,
                       const std::string& y_label,
                       const std::vector<double>& x_values,
                       const std::vector<double>& y_values) {
    g_chart.title = title;
    g_chart.x_label = x_label;
    g_chart.y_label = y_label;
    g_chart.x_values = x_values;
    g_chart.y_values = y_values;

    HINSTANCE hInstance = GetModuleHandle(NULL);

    WNDCLASSEXW wcex;
    wcex.cbSize = sizeof(WNDCLASSEXW);
    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = WndProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = hInstance;
    wcex.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wcex.lpszMenuName = NULL;
    wcex.lpszClassName = L"IceMarkChart";
    wcex.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
    RegisterClassExW(&wcex);

    HWND hwnd = CreateWindowExW(
        0,
        L"IceMarkChart",
        L"IceMark Chart",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        WIN_WIDTH, WIN_HEIGHT,
        NULL, NULL, hInstance, NULL
    );

    if (!hwnd) return;

    ShowWindow(hwnd, SW_SHOWNORMAL);
    UpdateWindow(hwnd);

    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}