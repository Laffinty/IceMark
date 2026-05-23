#include "renderer.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

std::string Renderer::build_chart_row(const std::string& title,
                                       double max_y, double min_y,
                                       const std::vector<double>& x_values,
                                       const std::vector<double>& y_values,
                                       uint32_t row_idx, uint32_t total_rows) {
    double range = max_y - min_y;
    if (range < 0.0001) range = 1.0;

    double threshold = max_y - (range * row_idx / (double)total_rows);
    double prev_threshold = max_y - (range * (row_idx + 1) / (double)total_rows);

    std::ostringstream oss;

    if (row_idx == total_rows - 1) {
        oss << std::setw(12) << std::right << std::fixed << std::setprecision(0) << max_y << " |";
    } else if (row_idx == 0) {
        oss << std::setw(12) << std::right << std::fixed << std::setprecision(0) << min_y << " |";
    } else {
        oss << std::string(13, ' ') << "|";
    }

    for (size_t i = 0; i < x_values.size(); ++i) {
        double y = y_values[i];
        double x_norm = (double)i / std::max((size_t)1, x_values.size() - 1);
        double expected_y = max_y - x_norm * range;

        if (y >= threshold && y > prev_threshold) {
            oss << "*";
        } else if (std::abs(y - expected_y) < range * 0.15 && row_idx == total_rows / 2) {
            oss << "-";
        } else {
            oss << " ";
        }
    }

    return oss.str();
}

void Renderer::render_line_chart(const std::string& title,
                                  const std::vector<double>& x_values,
                                  const std::vector<double>& y_values,
                                  uint32_t width,
                                  uint32_t height) {
    if (x_values.empty() || y_values.empty()) return;

    double max_y = *std::max_element(y_values.begin(), y_values.end());
    double min_y = *std::min_element(y_values.begin(), y_values.end());

    std::ostringstream oss;
    oss << "\n  " << title << "\n";

    for (int row = (int)height - 1; row >= 0; --row) {
        std::string line = build_chart_row(title, max_y, min_y, x_values, y_values, row, height);
        oss << line << "\n";
    }

    std::ostringstream x_axis;
    x_axis << "             +";
    uint32_t x_axis_width = width - 14;
    for (uint32_t i = 0; i < x_axis_width; ++i) {
        if (i % (x_axis_width / 5) == 0 && i / (x_axis_width / 5) < (int)x_values.size()) {
            x_axis << (i % 2 == 0 ? "+" : "-");
        } else {
            x_axis << "-";
        }
    }
    oss << x_axis.str() << "\n";

    std::ostringstream labels;
    labels << "             ";
    uint32_t label_step = std::max(uint32_t(1), uint32_t(x_values.size() - 1) / 4);
    for (uint32_t i = 0; i < x_values.size(); i += label_step) {
        labels << " " << std::setw(4) << (int)x_values[i];
    }
    oss << labels.str() << "\n";

    std::cout << oss.str();
}

void Renderer::render_decode_chart(const std::string& title,
                                    const std::vector<double>& x_values,
                                    const std::vector<double>& y_values,
                                    uint32_t width,
                                    uint32_t height) {
    double max_y = *std::max_element(y_values.begin(), y_values.end());
    double min_y = *std::min_element(y_values.begin(), y_values.end());

    if (min_y > 0.0 && max_y / min_y < 1.1) {
        min_y = 0.0;
    }

    render_line_chart(title, x_values, y_values, width, height);
}