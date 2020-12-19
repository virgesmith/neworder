#pragma once

#include <string>
#include <atomic>

namespace no {

const char* module_version();

// module-level attributes that should *not be modified directly* - call the env_init function. But note:
// - python functions are exposed for modifying verbose and checked
// - halt (if true) is reset in no::Model::run
namespace env {

extern std::atomic_bool verbose;
extern std::atomic_bool checked;
extern std::atomic_bool halt;
extern std::atomic_int rank;
extern std::atomic_int size;
extern std::atomic_int64_t uniqueIndex;

struct Context { enum Value { CPP, PY, SIZE }; };

// strings are not trivially copyable so can't be atomic
extern std::string logPrefix[Context::SIZE];

}

}



