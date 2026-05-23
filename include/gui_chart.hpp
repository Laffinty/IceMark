#pragma once
#include <string>
#include <vector>

void show_chart_window(const std::string& title,
                       const std::string& x_label,
                       const std::string& y_label,
                       const std::vector<double>& x_values,
                       const std::vector<double>& y_values);