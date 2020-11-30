#pragma once

#include <string>

namespace no {

const char* module_version();

// module-level attributes that should *not be modified directly* - call the env_init function. But note:
// - python functions are exposed for modifying verbose and checked
// - halt (if true) is reset in no::Model::run
namespace env {

extern bool verbose;
extern bool checked;
extern bool halt;
extern int rank;
extern int size;
extern int64_t uniqueIndex;

struct Context { enum Value { CPP, PY, SIZE }; };

extern std::string logPrefix[Context::SIZE];

}

}



