#pragma once

// append model path to python path (NB this is nonportable functionality)
void append_model_path(const char* path);

// main entry point
int run(int rank, int size);
