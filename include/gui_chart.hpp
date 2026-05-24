#pragma once
#include <string>
#include <vector>

struct ChartSeries {
    std::string title;
    std::string x_label;
    std::string y_label;
    std::vector<double> x_values;
    std::vector<double> y_values;
};

void show_results_dashboard(const std::vector<ChartSeries>& charts);
