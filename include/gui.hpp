#pragma once
#include "harness.hpp"

namespace icemark {

// Runs the interactive Win32 dashboard. Blocks until the user closes it.
int run_gui(const RunOptions& initial);

} // namespace icemark