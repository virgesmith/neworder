#pragma once

#include "NewOrder.h"

#include <pybind11/embed.h>

#include <vector>
#include <map>

namespace no {

enum class CommandType { Eval, Exec };

// Define a piece of python code to be exec/eval-u(a)ted on calling operator()
// Perhaps better named LazyEval?
class NEWORDER_EXPORT Runtime final
{
public:

  // globals is ignored
  Runtime(const py::object& _globals, const py::object& locals);

  ~Runtime() = default;

  py::object operator()(const std::tuple<std::string, CommandType>& cmd) const;

private:

  py::object m_globals;
  py::object m_locals;
};

typedef std::tuple<std::string, CommandType> Command;

typedef std::vector<Command> CommandList;
typedef std::map<std::string, Command> CommandDict;

const char* module_name();

const char* module_version();

std::string python_version();

// interactive shell mk2 - uses the code module
void shell(/*const py::object& local*/);

NEWORDER_EXPORT void set_timeline(const py::tuple& spec);

} // namespace no



