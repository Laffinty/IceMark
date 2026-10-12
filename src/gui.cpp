#include "gui.hpp"
#include "scoring.hpp"

#include <windows.h>
#include <commctrl.h>
#include <commdlg.h>
#include <windowsx.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <thread>

// commctrl.h and windowsx.h each define an ANSI alias that, with UNICODE
// defined, captures the explicit W-suffixed names below and redirects them to
// the 4-parameter ANSI overloads.
#undef SendDlgItemMessage
#undef CreateWindowEx

namespace icemark {
namespace {

constexpr wchar_t kClassName[] = L"IceMarkV2Window";
constexpr UINT WM_APP_PROGRESS = WM_APP + 1;
constexpr UINT WM_APP_FINISHED = WM_APP + 2;

enum : int {
    IDC_PROFILE = 1001,
    IDC_REPEATS,
    IDC_THREADS,
    IDC_INT8,
    IDC_START,
    IDC_STOP,
    IDC_EXPORT,
    IDC_PROGRESS,
    IDC_LBL_PROFILE,
    IDC_LBL_REPEATS,
    IDC_LBL_THREADS,
};

const COLORREF kBg = RGB(246, 247, 249);
const COLORREF kPanel = RGB(255, 255, 255);
const COLORREF kBorder = RGB(214, 218, 224);
const COLORREF kInk = RGB(28, 32, 38);
const COLORREF kMuted = RGB(120, 128, 140);
const COLORREF kAccent = RGB(37, 99, 235);
const COLORREF kGood = RGB(22, 143, 92);
const COLORREF kWarn = RGB(196, 132, 10);
const COLORREF kBad = RGB(196, 48, 48);

struct AppState {
    RunOptions opt;

    std::thread worker;
    Harness* harness = nullptr;
    std::atomic<bool> running{false};

    std::mutex mtx;
    std::string stage = "就绪";
    double frac = 0.0;
    double elapsed = 0.0;

    bool have_result = false;
    Result result;
    Score score;
};

AppState g_app;
HWND g_hwnd = nullptr;
HFONT g_font = nullptr;
HFONT g_font_bold = nullptr;
HFONT g_font_big = nullptr;
HFONT g_font_title = nullptr;
int g_dpi = 96;

int px(int v) { return MulDiv(v, g_dpi, 96); }

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w((size_t)n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), &w[0], n);
    return w;
}

std::wstring fmtw(const wchar_t* spec, ...) {
    wchar_t buf[256];
    va_list ap;
    va_start(ap, spec);
    vswprintf(buf, 256, spec, ap);
    va_end(ap);
    return buf;
}

void make_fonts() {
    if (g_font) { DeleteObject(g_font); DeleteObject(g_font_bold);
                 DeleteObject(g_font_big); DeleteObject(g_font_title); }
    LOGFONT lf{};
    lf.lfHeight = -px(13);
    lf.lfQuality = CLEARTYPE_QUALITY;
    lf.lfWeight = FW_NORMAL;
    wcscpy_s(lf.lfFaceName, L"Microsoft YaHei UI");
    g_font = CreateFontIndirectW(&lf);

    lf.lfWeight = FW_SEMIBOLD;
    g_font_bold = CreateFontIndirectW(&lf);

    lf.lfHeight = -px(17);
    g_font_title = CreateFontIndirectW(&lf);

    lf.lfHeight = -px(52);
    lf.lfWeight = FW_BOLD;
    g_font_big = CreateFontIndirectW(&lf);
}

// --- drawing helpers -------------------------------------------------------

void panel(HDC hdc, const RECT& r) {
    HPEN p = CreatePen(PS_SOLID, 1, kBorder);
    HGDIOBJ op = SelectObject(hdc, p);
    HGDIOBJ ob = SelectObject(hdc, GetStockObject(WHITE_BRUSH));
    Rectangle(hdc, r.left, r.top, r.right, r.bottom);
    SelectObject(hdc, ob);
    SelectObject(hdc, op);
    DeleteObject(p);
}

void text(HDC hdc, HFONT f, COLORREF col, int x, int y, const std::wstring& s) {
    HFONT of = (HFONT)SelectObject(hdc, f);
    SetTextColor(hdc, col);
    SetBkMode(hdc, TRANSPARENT);
    TextOutW(hdc, x, y, s.c_str(), (int)s.size());
    SelectObject(hdc, of);
}

int text_w(HDC hdc, HFONT f, const std::wstring& s) {
    HFONT of = (HFONT)SelectObject(hdc, f);
    SIZE sz{};
    GetTextExtentPoint32W(hdc, s.c_str(), (int)s.size(), &sz);
    SelectObject(hdc, of);
    return sz.cx;
}

// Scores 0..100 on a 0..100 track. Returns the score for colouring.
double bar(HDC hdc, const RECT& track, double score, COLORREF fill) {
    RECT r = track;
    HGDIOBJ ob = SelectObject(hdc, GetStockObject(LTGRAY_BRUSH));
    HPEN p = CreatePen(PS_NULL, 0, 0);
    HGDIOBJ op = SelectObject(hdc, p);
    RoundRect(hdc, r.left, r.top, r.right, r.bottom, r.bottom - r.top, r.bottom - r.top);
    const double f = std::max(0.0, std::min(100.0, score)) / 100.0;
    const int w = (int)((r.right - r.left) * f);
    if (w > 2) {
        RECT f2{r.left, r.top, r.left + w, r.bottom};
        HBRUSH br = CreateSolidBrush(fill);
        HGDIOBJ obr = SelectObject(hdc, br);
        RoundRect(hdc, f2.left, f2.top, f2.right, f2.bottom, f2.bottom - f2.top,
                  f2.bottom - f2.top);
        SelectObject(hdc, obr);
        DeleteObject(br);
    }
    SelectObject(hdc, op);
    SelectObject(hdc, ob);
    DeleteObject(p);
    return f;
}

void nice_step(double range) {
    // unused placeholder kept out of the drawing path
    (void)range;
}

// Linear plot with gridlines, labels and error bars.
void chart(HDC hdc, const RECT& area, const Series& s, bool log_x) {
    const int left = area.left + px(58);
    const int right = area.right - px(14);
    const int top = area.top + px(28);
    const int bottom = area.bottom - px(30);
    if (right <= left || bottom <= top || s.x.empty() || s.x.size() != s.y.size()) return;

    text(hdc, g_font_bold, kInk, area.left + px(12), area.top + px(6), widen(s.label));

    double lo = s.y[0], hi = s.y[0];
    for (double v : s.y) { lo = std::min(lo, v); hi = std::max(hi, v); }
    double pad = (hi - lo) * 0.15;
    if (pad <= 0.0) pad = std::max(hi * 0.1, 1e-9);
    lo -= pad;
    hi += pad;

    const double x0 = s.x.front();
    const double x1 = s.x.back();
    auto tx = [&](double v) {
        if (!log_x) return (v - x0) / std::max(1e-12, x1 - x0);
        return (std::log2(std::max(1.0, v)) - std::log2(std::max(1.0, x0))) /
               std::max(1e-12, std::log2(std::max(1.0, x1)) - std::log2(std::max(1.0, x0)));
    };

    HPEN grid = CreatePen(PS_SOLID, 1, kBorder);
    HGDIOBJ op = SelectObject(hdc, grid);
    for (int i = 0; i <= 4; ++i) {
        const double v = lo + (hi - lo) * i / 4.0;
        const int y = bottom - (int)((v - lo) / (hi - lo) * (bottom - top));
        MoveToEx(hdc, left, y, nullptr);
        LineTo(hdc, right, y);
        std::wstring lb = (hi - lo) < 10.0 ? fmtw(L"%.2f", v) : fmtw(L"%.0f", v);
        text(hdc, g_font, kMuted, left - px(52) - text_w(hdc, g_font, lb), y - px(7), lb);
    }
    SelectObject(hdc, op);
    DeleteObject(grid);

    for (size_t i = 0; i < s.x.size(); ++i) {
        const int x = left + (int)(tx(s.x[i]) * (right - left));
        std::wstring lb = log_x ? fmtw(L"%g", (double)s.x[i]) : fmtw(L"%g", s.x[i]);
        text(hdc, g_font, kMuted, x - text_w(hdc, g_font, lb) / 2, bottom + px(7), lb);
    }
    text(hdc, g_font, kMuted, area.left + px(12), bottom + px(7) - px(0), widen(s.x_label));

    HPEN axis = CreatePen(PS_SOLID, 1, RGB(190, 195, 202));
    op = SelectObject(hdc, axis);
    MoveToEx(hdc, left, top, nullptr);
    LineTo(hdc, left, bottom);
    LineTo(hdc, right, bottom);
    SelectObject(hdc, op);
    DeleteObject(axis);

    // error bars first so the curve sits on top
    HPEN ep = CreatePen(PS_SOLID, 1, RGB(190, 199, 235));
    op = SelectObject(hdc, ep);
    for (size_t i = 0; i < s.y.size(); ++i) {
        if (i >= s.err.size() || s.err[i] <= 0.0) continue;
        const int x = left + (int)(tx(s.x[i]) * (right - left));
        const double cy = bottom - (s.y[i] - lo) / (hi - lo) * (bottom - top);
        const double dy = s.err[i] * s.y[i] / (hi - lo) * (bottom - top);
        MoveToEx(hdc, x, (int)(cy - dy), nullptr);
        LineTo(hdc, x, (int)(cy + dy));
    }
    SelectObject(hdc, op);
    DeleteObject(ep);

    HPEN lp = CreatePen(PS_SOLID, 2, kAccent);
    op = SelectObject(hdc, lp);
    for (size_t i = 1; i < s.y.size(); ++i) {
        MoveToEx(hdc, left + (int)(tx(s.x[i - 1]) * (right - left)),
                 bottom - (int)((s.y[i - 1] - lo) / (hi - lo) * (bottom - top)), nullptr);
        LineTo(hdc, left + (int)(tx(s.x[i]) * (right - left)),
               bottom - (int)((s.y[i] - lo) / (hi - lo) * (bottom - top)));
    }
    SelectObject(hdc, op);
    DeleteObject(lp);

    HBRUSH br = CreateSolidBrush(kAccent);
    for (size_t i = 0; i < s.y.size(); ++i) {
        const int x = left + (int)(tx(s.x[i]) * (right - left));
        const int y = bottom - (int)((s.y[i] - lo) / (hi - lo) * (bottom - top));
        HGDIOBJ obr = SelectObject(hdc, br);
        HPEN np = CreatePen(PS_SOLID, 1, RGB(255, 255, 255));
        HGDIOBJ onp = SelectObject(hdc, np);
        Ellipse(hdc, x - px(4), y - px(4), x + px(4), y + px(4));
        SelectObject(hdc, onp);
        SelectObject(hdc, obr);
        DeleteObject(np);
    }
    DeleteObject(br);
}

// --- layout ---------------------------------------------------------------

RECT g_rect_profile, g_rect_repeats, g_rect_threads, g_rect_int8;
RECT g_rect_start, g_rect_stop, g_rect_export, g_rect_progress;
RECT g_rect_info, g_rect_scores, g_rect_charts, g_rect_status, g_rect_runpanel;

void layout(HWND hwnd) {
    RECT c{};
    GetClientRect(hwnd, &c);
    const int W = c.right - c.left;
    const int H = c.bottom - c.top;
    const int pad = px(14);
    const int info_h = px(70);
    const int row_h = px(26);
    const int status_h = px(26);

    int y = pad;
    g_rect_info = {pad, y, W - pad, y + info_h};
    y = g_rect_info.bottom + px(10);

    // label + control pairs, left to right
    const int lbl_w = px(34);
    const int gap = px(14);
    int x = pad;
    auto pair = [&](int lbl_id, RECT& out) {
        HWND lbl = GetDlgItem(hwnd, lbl_id);
        if (lbl) {
            SetWindowPos(lbl, nullptr, x, y + px(5), lbl_w, row_h,
                         SWP_NOZORDER | SWP_NOACTIVATE);
        }
        x += lbl_w + px(4);
        out = {x, y, x + px(104), y + row_h};
        x += px(104) + gap;
    };
    pair(IDC_LBL_PROFILE, g_rect_profile);
    pair(IDC_LBL_REPEATS, g_rect_repeats);
    pair(IDC_LBL_THREADS, g_rect_threads);
    g_rect_int8 = {x, y, x + px(90), y + row_h};
    x = g_rect_int8.right;

    g_rect_export = {W - pad - px(96), y, W - pad, y + row_h};
    g_rect_stop = {g_rect_export.left - px(76), y, g_rect_export.left - px(8), y + row_h};
    g_rect_start = {g_rect_stop.left - px(84), y, g_rect_stop.left - px(8), y + row_h};
    y += row_h + px(10);

    g_rect_runpanel = {pad, y, W - pad, y + px(40)};
    g_rect_progress = {g_rect_runpanel.left + px(210), g_rect_runpanel.top + px(10),
                       g_rect_runpanel.right - px(12), g_rect_runpanel.bottom - px(10)};
    y = g_rect_runpanel.bottom + px(10);

    const int bottom = H - pad - status_h;
    const int left_w = px(390);
    g_rect_scores = {pad, y, pad + left_w, bottom};
    g_rect_charts = {g_rect_scores.right + px(12), y, W - pad, bottom};
    g_rect_status = {pad, bottom + px(8), W - pad, bottom + px(8) + status_h};

    auto place = [&](int id, const RECT& r) {
        if (HWND h = GetDlgItem(hwnd, id)) {
            SetWindowPos(h, nullptr, r.left, r.top, r.right - r.left, r.bottom - r.top,
                         SWP_NOZORDER | SWP_NOACTIVATE);
        }
    };
    place(IDC_PROFILE, g_rect_profile);
    place(IDC_REPEATS, g_rect_repeats);
    place(IDC_THREADS, g_rect_threads);
    place(IDC_INT8, g_rect_int8);
    place(IDC_START, g_rect_start);
    place(IDC_STOP, g_rect_stop);
    place(IDC_EXPORT, g_rect_export);
    place(IDC_PROGRESS, g_rect_progress);

    InvalidateRect(hwnd, nullptr, FALSE);
}

COLORREF score_color(double v) {
    if (v >= 70.0) return kGood;
    if (v >= 45.0) return kWarn;
    return kBad;
}

void paint(HDC hdc, HWND hwnd) {
    RECT c{};
    GetClientRect(hwnd, &c);
    HBRUSH bg = CreateSolidBrush(kBg);
    FillRect(hdc, &c, bg);
    DeleteObject(bg);

    Result res;
    Score sc;
    bool have = false;
    {
        std::lock_guard<std::mutex> lk(g_app.mtx);
        if (g_app.have_result) { res = g_app.result; sc = g_app.score; have = true; }
    }

    // --- machine info ---
    panel(hdc, g_rect_info);
    if (have) {
        const int ty = g_rect_info.top + px(10);
        std::wstring name = res.cpu.brand.empty() ? L"未知 CPU" : widen(res.cpu.brand);
        text(hdc, g_font_title, kInk, g_rect_info.left + px(14), ty, name);
        std::wstring line2 = fmtw(L"%u 物理核 / %u 逻辑核 · %s", res.cpu.physical_cores,
                                  res.cpu.logical_cores,
                                  res.cpu.has_ht ? L"SMT" : L"无 SMT");
        text(hdc, g_font, kMuted, g_rect_info.left + px(14), ty + px(26), line2);
        std::wstring line3 = widen(simd_capabilities_string(res.cpu)) + L"  ·  内核 " +
                             widen(res.dot_kind);
        text(hdc, g_font, kMuted, g_rect_info.left + px(14), ty + px(44), line3);
    } else {
        text(hdc, g_font_title, kInk, g_rect_info.left + px(14), g_rect_info.top + px(10),
             L"IceMark 2.0  ·  CPU 算力测评");
        text(hdc, g_font, kMuted, g_rect_info.left + px(14), g_rect_info.top + px(38),
             L"基于 AVX2/FMA 合成推理内核的算力测评，测量前会先做可信度自检。");
    }
    {
        CPUInfo c0 = detect_cpu();
        MemoryInfo m0 = detect_memory();
        std::wstring right =
            fmtw(L"内存 %.1f GB 总 / %.1f GB 可用", m0.total_bytes / 1073741824.0,
                 m0.avail_bytes / 1073741824.0);
        text(hdc, g_font, kMuted, g_rect_info.right - px(14) - text_w(hdc, g_font, right),
             g_rect_info.top + px(38), right);
        std::wstring sig = widen(simd_capabilities_string(c0));
        text(hdc, g_font, kMuted, g_rect_info.right - px(14) - text_w(hdc, g_font, sig),
             g_rect_info.top + px(14), sig);
    }

    // --- progress row ---
    panel(hdc, g_rect_runpanel);
    {
        bool running = g_app.running.load();
        std::wstring msg;
        {
            std::lock_guard<std::mutex> lk(g_app.mtx);
            msg = running ? widen(g_app.stage) : L"就绪";
        }
        if (running) {
            text(hdc, g_font_bold, kAccent, g_rect_runpanel.left + px(12),
                 g_rect_runpanel.top + px(11), msg);
        } else {
            text(hdc, g_font_bold, kMuted, g_rect_runpanel.left + px(12),
                 g_rect_runpanel.top + px(11), L"就绪");
            text(hdc, g_font, kMuted, g_rect_runpanel.left + px(48),
                 g_rect_runpanel.top + px(12), L"点击“开始测评”运行，约需 30–60 秒");
        }
    }

    // --- score panel ---
    panel(hdc, g_rect_scores);
    const int sx = g_rect_scores.left + px(14);
    int sy = g_rect_scores.top + px(12);
    text(hdc, g_font_bold, kInk, sx, sy, L"综合评分");

    if (!have) {
        text(hdc, g_font, kMuted, sx, sy + px(30), L"尚未测量。");
    } else if (!sc.valid) {
        sy += px(30);
        text(hdc, g_font, kBad, sx, sy, L"测量不可信，已拒绝出分");
        sy += px(24);
        for (const auto& why : sc.verdict.reasons) {
            const std::wstring wwhy = widen(why);
            HFONT of = (HFONT)SelectObject(hdc, g_font);
            RECT wrap{sx, sy, g_rect_scores.right - px(12), sy + px(18)};
            DrawTextW(hdc, wwhy.c_str(), (int)wwhy.size(), &wrap, DT_WORDBREAK | DT_NOPREFIX);
            SelectObject(hdc, of);
            sy += px(22);
        }
    } else {
        sy += px(24);
        std::wstring tot = fmtw(L"%.1f", sc.total);
        text(hdc, g_font_big, kInk, sx, sy, tot);
        const int tw = text_w(hdc, g_font_big, tot);
        text(hdc, g_font, kMuted, sx + tw + px(8), sy + px(30), L"/ 100");

        const char* grade = sc.verdict.grade;
        COLORREF gc = std::string(grade) == "可信" ? kGood
                     : (std::string(grade) == "参考" ? kWarn : kBad);
        std::wstring badge = widen(grade) + L" · 置信度 " +
                             fmtw(L"%.0f%%", sc.verdict.confidence * 100.0);
        text(hdc, g_font_bold, gc, sx, sy + px(60), badge);

        sy += px(88);
        for (const auto& p : sc.parts) {
            text(hdc, g_font, kInk, sx, sy, widen(p.name));
            std::wstring vs = fmtw(L"%.0f", p.score);
            text(hdc, g_font_bold, score_color(p.score),
                 g_rect_scores.right - px(14) - px(28), sy, vs);
            RECT track{sx, sy + px(17), g_rect_scores.right - px(46), sy + px(23)};
            bar(hdc, track, p.score, score_color(p.score));
            text(hdc, g_font, kMuted, sx, sy + px(27), widen(p.detail));
            sy += px(50);
        }
    }

    // --- charts ---
    panel(hdc, g_rect_charts);
    if (!have) {
        text(hdc, g_font, kMuted, g_rect_charts.left + px(14), g_rect_charts.top + px(14),
             L"测量完成后显示曲线。");
    } else {
        const int gap = px(8);
        const int cw = (g_rect_charts.right - g_rect_charts.left - px(28) - gap) / 2;
        const int ch = (g_rect_charts.bottom - g_rect_charts.top - px(20) - gap) / 2;
        RECT a{g_rect_charts.left + px(10), g_rect_charts.top + px(6),
               g_rect_charts.left + px(10) + cw, g_rect_charts.top + px(6) + ch};
        RECT b{a.right + gap, a.top, a.right + gap + cw, a.bottom};
        RECT c3{a.left, a.bottom + gap, a.left + cw, a.bottom + gap + ch};
        RECT d{b.left, b.bottom + gap, b.right, b.bottom + gap + ch};
        chart(hdc, a, res.prefill_m, false);
        chart(hdc, b, res.thread_scaling, false);
        chart(hdc, c3, res.ctx_decay, true);
        chart(hdc, d, res.quant, true);
    }

    // --- status bar ---
    RECT st = g_rect_status;
    HBRUSH br = CreateSolidBrush(RGB(236, 239, 243));
    FillRect(hdc, &st, br);
    DeleteObject(br);

    std::wstring msg;
    COLORREF mc = kInk;
    {
        std::lock_guard<std::mutex> lk(g_app.mtx);
        if (g_app.running) {
            msg = widen(g_app.stage);
        } else if (have) {
            msg = res.check.pass ? L"自检通过" : L"自检不通过";
            mc = res.check.pass ? kGood : kBad;
            msg += L"   带宽 ";
            msg += fmtw(L"%.1f/%.1f GB/s (%.2fx)", res.check.kernel_gb_s,
                        res.check.stream_gb_s, res.check.bandwidth_ratio);
            msg += L"   NaN/Inf " + std::to_wstring((unsigned long long)res.check.bad_values);
            msg += L"   噪声 " + fmtw(L"%.1f%%", res.check.worst_noise * 100.0);
            msg += L"   耗时 " + fmtw(L"%.1fs", res.total_seconds);
        } else {
            msg = L"就绪";
        }
    }
    text(hdc, g_font_bold, mc, st.left + px(10), st.top + px(5), msg);
}

// --- worker ---------------------------------------------------------------

void start_run() {
    if (g_app.running.load()) return;

    RunOptions o = g_app.opt;
    o.profile = (Profile)SendDlgItemMessageW(g_hwnd, IDC_PROFILE, CB_GETCURSEL, 0, 0);
    LRESULT rr = SendDlgItemMessageW(g_hwnd, IDC_REPEATS, CB_GETCURSEL, 0, 0);
    if (rr >= 0) o.repeats = (uint32_t)(3 + 2 * rr);
    LRESULT tr = SendDlgItemMessageW(g_hwnd, IDC_THREADS, CB_GETCURSEL, 0, 0);
    o.max_threads = (tr == 0) ? 0u : (uint32_t)SendDlgItemMessageW(g_hwnd, IDC_THREADS,
                                                                  CB_GETITEMDATA, (WPARAM)tr, 0);
    o.with_int8 = (SendDlgItemMessageW(g_hwnd, IDC_INT8, BM_GETCHECK, 0, 0) == BST_CHECKED);
    if (o.repeats < 3) o.repeats = 3;
    g_app.opt = o;

    {
        std::lock_guard<std::mutex> lk(g_app.mtx);
        g_app.stage = "启动中…";
        g_app.frac = 0.0;
        g_app.have_result = false;
    }
    g_app.running.store(true);
    EnableWindow(GetDlgItem(g_hwnd, IDC_START), FALSE);
    EnableWindow(GetDlgItem(g_hwnd, IDC_EXPORT), FALSE);
    EnableWindow(GetDlgItem(g_hwnd, IDC_STOP), TRUE);
    InvalidateRect(g_hwnd, nullptr, FALSE);

    g_app.worker = std::thread([]() {
        Result r;
        Score s;
        try {
            auto* h = new Harness(g_app.opt);
            g_app.harness = h;
            h->on_progress([](const std::string& stage, double frac) {
                {
                    std::lock_guard<std::mutex> lk(g_app.mtx);
                    g_app.stage = stage;
                    g_app.frac = frac;
                }
                PostMessageW(g_hwnd, WM_APP_PROGRESS, 0, 0);
            });
            r = h->run();
            delete h;
            g_app.harness = nullptr;
            s = compute_score(r);
        } catch (...) {
            r.opt = g_app.opt;
            r.error = "测量线程异常终止。";
            s = compute_score(r);
        }
        {
            std::lock_guard<std::mutex> lk(g_app.mtx);
            g_app.result = r;
            g_app.score = s;
            g_app.have_result = true;
            g_app.stage = r.cancelled ? "已取消" : (r.error.empty() ? "完成" : "失败");
            g_app.frac = 1.0;
        }
        PostMessageW(g_hwnd, WM_APP_FINISHED, 0, 0);
    });
}

void stop_run() {
    if (!g_app.running.load()) return;
    EnableWindow(GetDlgItem(g_hwnd, IDC_STOP), FALSE);
    if (g_app.harness) {
        g_app.harness->request_cancel();
        std::lock_guard<std::mutex> lk(g_app.mtx);
        g_app.stage = "正在取消…";
    }
}

void export_csv() {
    Result res;
    {
        std::lock_guard<std::mutex> lk(g_app.mtx);
        if (!g_app.have_result) return;
        res = g_app.result;
    }

    wchar_t file[MAX_PATH] = L"icemark-report.csv";
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = g_hwnd;
    ofn.lpstrFilter = L"CSV 文件\0*.csv\0所有文件\0*.*\0\0";
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = L"导出测评结果";
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST;
    if (!GetSaveFileNameW(&ofn)) return;

    FILE* fp = nullptr;
    if (_wfopen_s(&fp, file, L"w") != 0 || !fp) {
        MessageBoxW(g_hwnd, L"无法写入文件。", L"IceMark", MB_ICONERROR);
        return;
    }
    const unsigned char bom[3] = {0xEF, 0xBB, 0xBF};
    fwrite(bom, 1, 3, fp);
    fprintf(fp, "section,key,value\n");
    fprintf(fp, "machine,cpu,%s\n", res.cpu.brand.c_str());
    fprintf(fp, "machine,physical_cores,%u\n", res.cpu.physical_cores);
    fprintf(fp, "machine,logical_cores,%u\n", res.cpu.logical_cores);
    fprintf(fp, "machine,simd,%s\n", simd_capabilities_string(res.cpu).c_str());
    fprintf(fp, "machine,kernel,%s\n", res.dot_kind);
    fprintf(fp, "check,stream_gb_s,%.3f\n", res.check.stream_gb_s);
    fprintf(fp, "check,kernel_gb_s,%.3f\n", res.check.kernel_gb_s);
    fprintf(fp, "check,bandwidth_ratio,%.3f\n", res.check.bandwidth_ratio);
    fprintf(fp, "check,nan_inf,%llu\n", (unsigned long long)res.check.bad_values);
    fprintf(fp, "check,worst_noise,%.4f\n", res.check.worst_noise);
    fprintf(fp, "check,pass,%d\n", res.check.pass ? 1 : 0);

    auto dump = [&](const char* name, const Series& s) {
        for (size_t i = 0; i < s.x.size(); ++i) {
            fprintf(fp, "%s,%g,%.4f\n", name, s.x[i], s.y[i]);
        }
    };
    dump("prefill_gflops_by_M", res.prefill_m);
    dump("prefill_tokens_per_s_by_threads", res.thread_scaling);
    dump("decode_ms_per_token_by_ctx", res.ctx_decay);
    dump("int8_speedup_by_ctx", res.quant);
    fclose(fp);
}

// --- window ----------------------------------------------------------------

void fill_threads_combo() {
    HWND h = GetDlgItem(g_hwnd, IDC_THREADS);
    SendMessageW(h, CB_RESETCONTENT, 0, 0);
    SendMessageW(h, CB_ADDSTRING, 0, (LPARAM)L"自动（全部逻辑核）");
    SendMessageW(h, CB_SETITEMDATA, 0, 0);
    const CPUInfo cpu = detect_cpu();
    for (uint32_t n = 1; n <= cpu.logical_cores; n *= 2) {
        wchar_t buf[32];
        swprintf(buf, 32, L"%u 线程", n);
        const LRESULT idx = SendMessageW(h, CB_ADDSTRING, 0, (LPARAM)buf);
        SendMessageW(h, CB_SETITEMDATA, (WPARAM)idx, (LPARAM)n);
    }
    SendMessageW(h, CB_SETCURSEL, 0, 0);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_CREATE: {
        g_hwnd = hwnd;
        auto label = [&](int id, const wchar_t* t) {
            CreateWindowExW(0, L"STATIC", t, WS_CHILD | WS_VISIBLE | SS_LEFT, 0, 0, 0, 0, hwnd,
                            (HMENU)(INT_PTR)id, GetModuleHandleW(nullptr), nullptr);
        };
        label(IDC_LBL_PROFILE, L"模型");
        label(IDC_LBL_REPEATS, L"重复");
        label(IDC_LBL_THREADS, L"线程");

        HWND c1 = CreateWindowExW(0, L"COMBOBOX", nullptr,
                                  WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_VSCROLL, 0, 0, 0,
                                  200, hwnd, (HMENU)(INT_PTR)IDC_PROFILE, GetModuleHandleW(nullptr), nullptr);
        SendMessageW(c1, CB_ADDSTRING, 0, (LPARAM)L"快速");
        SendMessageW(c1, CB_ADDSTRING, 0, (LPARAM)L"标准");
        SendMessageW(c1, CB_ADDSTRING, 0, (LPARAM)L"深度");
        SendMessageW(c1, CB_SETCURSEL, (WPARAM)g_app.opt.profile, 0);

        HWND c2 = CreateWindowExW(0, L"COMBOBOX", nullptr,
                                  WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_VSCROLL, 0, 0, 0,
                                  200, hwnd, (HMENU)(INT_PTR)IDC_REPEATS, GetModuleHandleW(nullptr), nullptr);
        for (int i = 0; i < 4; ++i) {
            wchar_t b[16];
            swprintf(b, 16, L"%d 次", 3 + 2 * i);
            SendMessageW(c2, CB_ADDSTRING, 0, (LPARAM)b);
        }
        SendMessageW(c2, CB_SETCURSEL, 1, 0);
        for (int i = 0; i < 4; ++i) {
            if ((uint32_t)(3 + 2 * i) >= g_app.opt.repeats) {
                SendMessageW(c2, CB_SETCURSEL, i, 0);
                break;
            }
        }

        CreateWindowExW(0, L"COMBOBOX", nullptr,
                        WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_VSCROLL, 0, 0, 0,
                        200, hwnd, (HMENU)(INT_PTR)IDC_THREADS, GetModuleHandleW(nullptr), nullptr);
        fill_threads_combo();

        CreateWindowExW(0, L"BUTTON", L"INT8 对比", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 0,
                        0, 0, 0, hwnd, (HMENU)(INT_PTR)IDC_INT8, GetModuleHandleW(nullptr), nullptr);
        SendDlgItemMessageW(hwnd, IDC_INT8, BM_SETCHECK,
                        g_app.opt.with_int8 ? BST_CHECKED : BST_UNCHECKED, 0);

        CreateWindowExW(0, L"BUTTON", L"开始测评", WS_CHILD | WS_VISIBLE | WS_TABSTOP, 0, 0, 0,
                        0, hwnd, (HMENU)(INT_PTR)IDC_START, GetModuleHandleW(nullptr), nullptr);
        CreateWindowExW(0, L"BUTTON", L"停止", WS_CHILD | WS_VISIBLE | WS_TABSTOP, 0, 0, 0, 0,
                        hwnd, (HMENU)(INT_PTR)IDC_STOP, GetModuleHandleW(nullptr), nullptr);
        CreateWindowExW(0, L"BUTTON", L"导出 CSV", WS_CHILD | WS_VISIBLE | WS_TABSTOP, 0, 0,
                        0, 0, hwnd, (HMENU)(INT_PTR)IDC_EXPORT, GetModuleHandleW(nullptr), nullptr);
        EnableWindow(GetDlgItem(hwnd, IDC_STOP), FALSE);

        CreateWindowExW(0, PROGRESS_CLASSW, nullptr,
                        WS_CHILD | WS_VISIBLE | PBS_SMOOTH, 0, 0, 0, 20, hwnd,
                        (HMENU)(INT_PTR)IDC_PROGRESS, GetModuleHandleW(nullptr), nullptr);
        SendDlgItemMessageW(hwnd, IDC_PROGRESS, PBM_SETRANGE32, 0, 1000);
        SendDlgItemMessageW(hwnd, IDC_PROGRESS, PBM_SETPOS, 0, 0);

        if (g_font) {
            SendDlgItemMessageW(hwnd, IDC_START, WM_SETFONT, (WPARAM)g_font, TRUE);
            SendDlgItemMessageW(hwnd, IDC_STOP, WM_SETFONT, (WPARAM)g_font, TRUE);
            SendDlgItemMessageW(hwnd, IDC_EXPORT, WM_SETFONT, (WPARAM)g_font, TRUE);
            SendDlgItemMessageW(hwnd, IDC_INT8, WM_SETFONT, (WPARAM)g_font, TRUE);
            SendDlgItemMessageW(hwnd, IDC_PROFILE, WM_SETFONT, (WPARAM)g_font, TRUE);
            SendDlgItemMessageW(hwnd, IDC_REPEATS, WM_SETFONT, (WPARAM)g_font, TRUE);
            SendDlgItemMessageW(hwnd, IDC_THREADS, WM_SETFONT, (WPARAM)g_font, TRUE);
            SendDlgItemMessageW(hwnd, IDC_LBL_PROFILE, WM_SETFONT, (WPARAM)g_font, TRUE);
            SendDlgItemMessageW(hwnd, IDC_LBL_REPEATS, WM_SETFONT, (WPARAM)g_font, TRUE);
            SendDlgItemMessageW(hwnd, IDC_LBL_THREADS, WM_SETFONT, (WPARAM)g_font, TRUE);
        }
        layout(hwnd);
        return 0;
    }
    case WM_DPICHANGED: {
        g_dpi = HIWORD(wp);
        make_fonts();
        layout(hwnd);
        return 0;
    }
    case WM_SIZE:
        layout(hwnd);
        return 0;
    case WM_COMMAND:
        switch (LOWORD(wp)) {
        case IDC_START: start_run(); return 0;
        case IDC_STOP: stop_run(); return 0;
        case IDC_EXPORT: export_csv(); return 0;
        default: break;
        }
        break;
    case WM_APP_PROGRESS: {
        int v = 0;
        std::string stage;
        {
            std::lock_guard<std::mutex> lk(g_app.mtx);
            v = (int)(g_app.frac * 1000.0);
            stage = g_app.stage;
        }
        SendDlgItemMessageW(hwnd, IDC_PROGRESS, PBM_SETPOS, (WPARAM)v, 0);
        InvalidateRect(hwnd, nullptr, FALSE);
        return 0;
    }
    case WM_APP_FINISHED: {
        if (g_app.worker.joinable()) g_app.worker.join();
        g_app.running.store(false);
        EnableWindow(GetDlgItem(hwnd, IDC_START), TRUE);
        EnableWindow(GetDlgItem(hwnd, IDC_STOP), FALSE);
        EnableWindow(GetDlgItem(hwnd, IDC_EXPORT), TRUE);
        SendDlgItemMessageW(hwnd, IDC_PROGRESS, PBM_SETPOS, 1000, 0);
        InvalidateRect(hwnd, nullptr, FALSE);
        return 0;
    }
    case WM_ERASEBKGND:
        return 1;
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
        paint(hdc, hwnd);
        EndPaint(hwnd, &ps);
        return 0;
    }
    case WM_CLOSE:
        if (g_app.running.load()) {
            if (MessageBoxW(hwnd, L"测评正在运行，确定要取消并退出吗？", L"IceMark",
                            MB_ICONQUESTION | MB_YESNO) != IDYES)
                return 0;
            stop_run();
        }
        if (g_app.worker.joinable()) g_app.worker.join();
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

} // namespace

int run_gui(const RunOptions& initial) {
    // Per-monitor DPI awareness must be requested before any window exists,
    // otherwise the first frame is rendered at the wrong scale and never
    // re-laid out.
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    SetProcessDPIAware();
    {
        HDC dc = GetDC(nullptr);
        g_dpi = GetDeviceCaps(dc, LOGPIXELSX);
        ReleaseDC(nullptr, dc);
    }
    if (g_dpi <= 0) g_dpi = 96;
    make_fonts();

    g_app.opt = initial;
    g_app.opt.repeats = std::max(3u, initial.repeats);

    INITCOMMONCONTROLSEX icc{sizeof(icc), ICC_PROGRESS_CLASS};
    InitCommonControlsEx(&icc);

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
    wc.lpszClassName = kClassName;
    wc.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
    RegisterClassExW(&wc);

    HWND hwnd = CreateWindowExW(0, kClassName, L"IceMark 2.0  ·  CPU 算力测评",
                               WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, px(1180),
                               px(820), nullptr, nullptr, wc.hInstance, nullptr);
    if (!hwnd) return 1;

    ShowWindow(hwnd, SW_SHOWMAXIMIZED);
    UpdateWindow(hwnd);

    MSG msg;
    while (true) {
        // GetMessageW returns 0 for WM_QUIT and -1 on error; `> 0` folds both
        // into "stop", which would swallow a hard error silently.
        const BOOL r = GetMessageW(&msg, nullptr, 0, 0);
        if (r <= 0) break;
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    if (g_font) DeleteObject(g_font);
    if (g_font_bold) DeleteObject(g_font_bold);
    if (g_font_big) DeleteObject(g_font_big);
    if (g_font_title) DeleteObject(g_font_title);
    return 0;
}

} // namespace icemark