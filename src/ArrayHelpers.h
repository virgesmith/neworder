
#pragma once

#include "NewOrder.h"

#include <nanobind/ndarray.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace no {

template <size_t D> using Index_t = std::array<Py_ssize_t, D>;

// A writable numpy array. Create new ones with allocate: wrapping memory allocated in C++ is several times slower,
// which is significant for the small arrays returned to case-based models, and leaves the result not owning its data.
template <typename T> using NpArray = nb::ndarray<nb::numpy, T, nb::c_contig, nb::device::cpu>;

namespace numpy {

// Fetches the numpy functions and types used below. Must be called once, on module initialisation
void init();

enum class DType { FLOAT64, INT64, BOOL };

template <typename T> constexpr DType dtype_of() {
  if constexpr (std::is_same_v<T, double>)
    return DType::FLOAT64;
  else if constexpr (std::is_same_v<T, int64_t>)
    return DType::INT64;
  else {
    static_assert(std::is_same_v<T, bool>, "unsupported array type");
    return DType::BOOL;
  }
}

bool is_ndarray(nb::handle obj) noexcept;

// np.empty(shape, dtype)
nb::object empty(std::span<const Py_ssize_t> shape, DType dtype);

// np.asarray(obj, dtype=dtype, order="C")
nb::object asarray(nb::handle obj, DType dtype);

// np.array(obj, dtype=dtype, order="C", copy=True)
nb::object copy(nb::handle obj, DType dtype);

} // namespace numpy

// A read-only, C-contiguous array argument. Anything numpy can convert to T can be passed from python, like
// numpy.typing.ArrayLike, and nb::ndarray alone doesn't do this: it only understands objects that already expose their
// memory, which excludes e.g. lists and pandas.Series.
template <typename T> class ArrayLike {
public:
  using View = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;

  ArrayLike() = default;
  explicit ArrayLike(View view) : m_view(std::move(view)) {}

  const T* data() const { return m_view.data(); }
  size_t size() const { return m_view.size(); }
  size_t ndim() const { return m_view.ndim(); }
  size_t shape(size_t i) const { return m_view.shape(i); }

private:
  View m_view;
};

// Converts a python object that isn't a function argument, e.g. a dict value. Raises if numpy can't convert it
template <typename T> ArrayLike<T> as_array(nb::handle obj) {
  return ArrayLike<T>(nb::cast<typename ArrayLike<T>::View>(numpy::asarray(obj, numpy::dtype_of<T>())));
}

template <typename T> const T* cbegin(const ArrayLike<T>& a) { return a.data(); }

template <typename T> const T* cend(const ArrayLike<T>& a) { return a.data() + a.size(); }

// Allocates an uninitialised array. Raises ValueError for a negative size
template <typename T> NpArray<T> allocate(std::span<const Py_ssize_t> shape) {
  return nb::cast<NpArray<T>>(numpy::empty(shape, numpy::dtype_of<T>()));
}

template <typename T> NpArray<T> allocate(Py_ssize_t size) { return allocate<T>(std::span<const Py_ssize_t>(&size, 1)); }

// Create an array, initialising with a function
// e.g. "ones" is make_array<double>(n, [](){ return 1.0; })
template <typename T, size_t D = 1> NpArray<T> make_array(const Index_t<D>& n, const std::function<T()>& f) {
  NpArray<T> a = allocate<T>(std::span<const Py_ssize_t>(n));
  std::generate(a.data(), a.data() + a.size(), f);
  return a;
}

// "nullary"/scalar unary op implemented as make_array
template <typename R, typename A> NpArray<R> unary_op(const ArrayLike<A>& arg, const std::function<R(A)>& f) {
  std::vector<Py_ssize_t> shape(arg.ndim());
  for (size_t i = 0; i < arg.ndim(); ++i)
    shape[i] = static_cast<Py_ssize_t>(arg.shape(i));

  NpArray<R> result = allocate<R>(shape);

  const A* p = arg.data();
  R* r = result.data();
  for (size_t i = 0; i < arg.size(); ++i) {
    r[i] = f(p[i]);
  }
  return result;
}

} // namespace no

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <typename T> struct type_caster<no::ArrayLike<T>> {
  NB_TYPE_CASTER(no::ArrayLike<T>, const_name("numpy.typing.ArrayLike"))

  // Dispatch tries every overload without conversions, then again with them. So restricting the first pass to real
  // ndarrays is what sends e.g. a numpy.float64, which also exposes its memory, to an overload taking a double
  bool from_python(handle src, uint32_t flags, cleanup_list* cleanup) noexcept {
    using View = typename no::ArrayLike<T>::View;
    const uint32_t no_convert = flags & ~static_cast<uint32_t>(cast_flags::convert);
    make_caster<View> view;

    if (!(flags & static_cast<uint32_t>(cast_flags::convert))) {
      if (!no::numpy::is_ndarray(src) || !view.from_python(src, no_convert, cleanup))
        return false;
    } else {
      try {
        // the view holds a reference to the converted array
        object converted = no::numpy::asarray(src, no::numpy::dtype_of<T>());
        if (!view.from_python(converted, no_convert, cleanup))
          return false;
      } catch (...) {
        return false;
      }
    }
    value = no::ArrayLike<T>(std::move(view.value));
    return true;
  }

  static handle from_cpp(const no::ArrayLike<T>&, rv_policy, cleanup_list*) noexcept { return handle(); }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
