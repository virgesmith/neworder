#pragma once

#include "Log.h"
#include "NewOrder.h"

#include <cstdint>
#include <type_traits>

namespace no {

template <typename T> constexpr const char* type_description() {
  if constexpr (std::is_same_v<T, int32_t>)
    return "an int32";
  else if constexpr (std::is_same_v<T, int64_t>)
    return "an int64";
  else
    static_assert(sizeof(T) == 0, "add a description for this type");
}

// nb::cast reports failure as a bare std::bad_cast (so RuntimeError), so raise a TypeError naming the value and the
// expected type. describe() returns what the value is, and is only called on failure.
template <typename T, std::invocable F> T checked_cast(nb::handle value, F&& describe) {
  T result{};
  if (!nb::try_cast(value, result))
    throw no::type_error("%% must be %%, got %%"s % describe() % type_description<T>() % nb::repr(value));
  return result;
}

template <typename T> T checked_cast(nb::handle value, const char* what) {
  return checked_cast<T>(value, [what]() { return std::string(what); });
}

} // namespace no

// helper class for accessing the python attributes of a C++ class with python bindings
template<typename T>
class PyAccessor final
{
public:
  typedef T cpp_type;
  explicit PyAccessor(cpp_type& object) : ref(nb::find(object)) { }

  nb::object get(const char* name) const
  {
    return ref.attr(name);
  }

  template<typename U>
  U get_as(const char* name) const
  {
    return no::checked_cast<U>(ref.attr(name),
                               [this, name]() { return std::to_string(nb::type_name(ref.type())) + "." + name; });
  }

  bool get_truthy(const char* name) const
  {
    return no::truthy(get(name));
  }

private:
  nb::object ref;
};
