#pragma once

#include <cstddef>

// append model path(s) to python path (NB this is nonportable functionality)
void append_model_paths(const char* paths[], size_t n);

// main entry point
int run(int rank, int size, bool indep);
