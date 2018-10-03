#pragma once

#include <vector>
#include <cstddef>

// fwd decl
namespace boost { namespace python { class tuple; } }
namespace py = boost::python;

namespace neworder {

class Timeline final
{
public:

  // Default "null" timeline is just one step of arbitrary size
  Timeline();

  Timeline(double begin, double end, int n);

  Timeline(const py/*boost::python*/::tuple& spec);

  ~Timeline() = default;

  Timeline(const Timeline&) = delete;
  Timeline& operator=(const Timeline&) = default;

  double time() const;
  size_t index() const;

  double dt() const;
  void step();

  bool is_checkpoint() const;

  const std::vector<size_t>& checkpoints() const;

  bool end() const;

  // returns a floating point number that compares less than any other number
  static double distant_past();

  // returns a floating point number that compares greater than any other number
  static double far_future();

private:
  std::vector<size_t> m_checkpoints;
  size_t m_steps; // total no. of steps

  double m_begin; 
  double m_dt; // timestep

  size_t m_index; // index of current time
  double m_time; // current time
};

}