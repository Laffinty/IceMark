#pragma once
#include <string>
#include <vector>

class Renderer {
public:
    static void render_line_chart(const std::string& title,
                                   const std::vector<double>& x_values,
                                   const std::vector<double>& y_values,
                                   uint32_t width = 60,
                                   uint32_t height = 20);

    static void render_decode_chart(const std::string& title,
                                    const std::vector<double>& x_values,
                                    const std::vector<double>& y_values,
                                    uint32_t width = 60,
                                    uint32_t height = 20);

private:
    static std::string build_chart_row(const std::string& title,
                                        double max_y, double min_y,
                                        const std::vector<double>& x_values,
                                        const std::vector<double>& y_values,
                                        uint32_t row_idx, uint32_t total_rows);
};