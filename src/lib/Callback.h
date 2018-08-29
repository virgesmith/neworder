#pragma once

#include "python.h"

#include <vector>
#include <map>

namespace neworder {

// Define a piece of python code to be exec/eval-u(a)ted on calling operator()
// Perhaps better named LazyEval?
class Callback
{
public:
  // Construct using one of these two variants 
  static Callback exec(const std::string& code);
  static Callback eval(const std::string& code);
  
  ~Callback() = default;

  py::object operator()() const;

  bool is_exec() const { return m_exec; }

  const std::string code() const
  {
    return (m_exec ? "exec(\"" : "eval(\"") + m_code + "\")";
  }

private:
  // construct using one of the static functions
  Callback(const std::string& code, bool exec);

  bool m_exec;
  std::string m_code;
  py::object m_globals;
  py::object m_locals;
};

typedef std::map<std::string, Callback> CallbackTable;

const char* module_name();

const char* module_version();

std::string python_version();

// interactive shell mk2 - uses the code module
void shell(/*const py::object& local*/);

// msg is forcibly coerced to a string
void log(const py::object& msg);

void import_module();

// TODO perhaps better to copy to np.array?
template <class T>
py::list vector_to_py_list(const std::vector<T>& v) 
{
  py::list list;
  for (auto it = v.begin(); it != v.end(); ++it) 
  {
    list.append(*it);
  }
  return list;
}

// TODO perhaps better to copy to np.array?
template <class T>
std::vector<T> py_list_to_vector(const py::list& l) 
{
  std::vector<T> v(py::len(l));
  for (int i = 0; i < py::len(l); ++i) 
  {
    v[i] = py::extract<T>(l[i])();
  }
  return v;
}

template<typename T>
std::string vector_to_string(const std::vector<T>& v)
{
  if (v.empty()) 
    return "[]";

  std::ostringstream str;
  str << "[" << v[0];
  for (size_t i = 1; i < v.size(); ++i)
  {
    str << ", " << v[i];
  }
  str << "]";
  return str.str();
}

// Specialisation for strings - explicitly quotes each element
template<>
inline std::string vector_to_string(const std::vector<std::string>& v)
{
  if (v.empty()) 
    return "[]";

  std::ostringstream str;
  str << "['" << v[0];
  for (size_t i = 1; i < v.size(); ++i)
  {
    str << "', '" << v[i];
  }
  str << "']";
  return str.str();
}

} // namespace neworder

template<typename T>
std::vector<T> operator+(const std::vector<T>& v, T y)
{
  std::vector<T> r(v);
  for (auto& x: r)
    x += y;
  return r;
}

template<typename T>
std::vector<T> operator+(T x, const std::vector<T>& v)
{
  return v + x;
}

template<typename T>
std::vector<T> operator*(const std::vector<T>& v, T y)
{
  std::vector<T> r(v);
  for (auto& x: r)
    x *= y;
  return r;
}

template<typename T>
std::vector<T> operator*(T x, const std::vector<T>& v)
{
  return v * x;
}


