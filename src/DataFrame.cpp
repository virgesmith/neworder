
#include "DataFrame.h"

#include "ArrayHelpers.h"
#include "Log.h"
#include "Model.h"
#include "MonteCarlo.h"

#include "NewOrder.h"

// uniqueness is - but reproduciblity isn't - guaranteed when multiple threads are used
py::array_t<int64_t> no::df::unique_index(size_t n) {
  int64_t s = no::env::size.load(std::memory_order_relaxed);
  int64_t i = no::env::uniqueIndex.fetch_add(n * s, std::memory_order_acq_rel);

  auto a = no::make_array<int64_t>({static_cast<py::ssize_t>(n)}, [&]() {
    int64_t ret{i};
    i += s;
    return ret;
  });
  return a;
}

// matrix is a transition matrix. Its row order must correspond to series.cat.categories order
py::object no::df::transition(no::Model& model,
                               py::array_t<double, py::array::c_style | py::array::forcecast> matrix_arg,
                               py::object& series) {
  // matrix is read-only, so it's safe to just force a contiguous copy if the caller's array isn't already one -
  // see the ArrayHelpers no::begin/no::cbegin/no::at helpers used below, which assume a contiguous, unit-stride,
  // default-ExtraFlags array_t.
  py::array_t<double> matrix = matrix_arg;

  py::object pandas = py::module_::import("pandas");
  if (!py::isinstance(series.attr("dtype"), pandas.attr("CategoricalDtype"))) {
    throw py::type_error(
        "series does not have a pandas 'category' dtype; convert it first, e.g. "
        "series = series.astype('category')");
  }
  py::object cat_accessor = series.attr("cat");
  py::ssize_t m = static_cast<py::ssize_t>(py::len(cat_accessor.attr("categories")));

  // check matrix is 2d, square & its size matches the number of categories
  if (matrix.ndim() != 2)
    throw py::value_error("cumulative transition matrix dimension is %%"s % matrix.ndim());
  if (matrix.shape(0) != matrix.shape(1))
    throw py::value_error("cumulative transition matrix shape is not square: %% by %%"s % matrix.shape(0) %
                          matrix.shape(1));
  if (m != matrix.shape(0))
    throw py::value_error("cumulative transition matrix size (%%) is not same as the number of categories (%%)"s %
                          matrix.shape(0) % m);

  // IMPORTANT NOTES:
  // - whilst numpy is row-major, pandas stores column-major, i.e. the columns are contiguous memory
  // - transposing (square matrices at least) in python doesn't change the memory layout, it just changes the view
  // - the code below assumes the transition matrix has a row major memory layout (i.e. row sums to unity not cols)

  // construct checked cumulative probabilities for each state to randomly interpolate
  std::vector<std::vector<double>> cumprobs(m);
  for (int i = 0; i < m; ++i) {
    cumprobs[i] = no::cumulative(no::cbegin(matrix) + (i * m), m);
  }

  // Codes are already indices 0..m-1 (or -1 for NaN/missing), so no value <-> index lookup is required. We
  // operate on a local (possibly copied) int64 buffer rather than relying on `.cat.codes` being an aliased,
  // in-place-writable view of the Categorical's internal storage - that isn't part of pandas' public contract,
  // and codes may be stored as int8/16/32/64 depending on the number of categories - then write the result back
  // explicitly via pd.Categorical.from_codes.
  py::object codes_obj = cat_accessor.attr("codes");
  py::array_t<int64_t, py::array::c_style | py::array::forcecast> codes_arg = codes_obj;
  // no::begin/no::at only accept the default-ExtraFlags array_t (see the matrix rebind above) - this is just a
  // flags reinterpretation of the same (already contiguous, per c_style above) buffer, not a copy.
  py::array_t<int64_t> codes = codes_arg;

  py::ssize_t n = codes.size();
  py::array_t<double> rpy = model.mc().ustream(n);

  double* r = no::begin(rpy);
  int64_t* pcodes = no::begin<int64_t>(codes);

  for (py::ssize_t i = 0; i < n; ++i) {
    int64_t j = pcodes[i];
    // codes are -1 for NaN/missing categories - leave any such rows untouched
    if (j < 0 || j >= m)
      continue;
    py::ssize_t k = no::interp(cumprobs[j], r[i]);
    pcodes[i] = k;
  }

  py::object from_codes = pandas.attr("Categorical").attr("from_codes");
  return from_codes(codes, cat_accessor.attr("categories"), cat_accessor.attr("ordered"));
}

template <typename T> void dump(const T* p, py::ssize_t n) {
  for (py::ssize_t i = 0; i < n; ++i, ++p) {
    no::log("%%"s % *p);
    // no::at<std::string>(arr, Index_t<1>{i}) += 1;
  }
}

// example of directly modifying a DF testing different dtypes
void no::df::testfunc(no::Model& model, py::object& df, const std::string& colname) {
  // .values? pd.Series -> np.array?
  py::array arr = df.attr(colname.c_str()); //.request();

  // no::log(arr.dtype());
  py::buffer_info buf = arr.request();

  py::ssize_t n = buf.shape[0];

  if (arr.dtype().is(py::dtype::of<int64_t>())) {
    dump(static_cast<int64_t*>(buf.ptr), n);
  } else if (arr.dtype().is(py::dtype::of<double>())) {
    dump(static_cast<double*>(buf.ptr), n);
  } else if (arr.dtype().is(py::dtype::of<bool>())) {
    dump(static_cast<bool*>(buf.ptr), n);
  }
  // else if (arr.dtype() == "object")
  // {
  //   py::str* p = static_cast<py::str*>(buf.ptr);
  // }
  // else if (arr.dtype() == py::object)
  // {
  //   py::object* p = static_cast<py::object*>(buf.ptr);
  //   for (py::ssize_t i = 0; i < n; ++i, ++p)
  //   {
  //     no::log(*p);
  //   }
  // }
  else {
    throw py::type_error("unsupported dtype '%%' in column '%%'"s % /*arr.dtype().cast<std::string>() %*/ colname);
  }
}

// TODO implement - see liam2-demo07
// void no::df::linked_change(py::object& df, const std::string& cat, const std::string& link_cat)
// {
//   // .values? pd.Series -> np.array?
//   py::array arr0 = df.attr(cat.c_str()); // this is a reference
//   // .values? pd.Series -> np.array?
//   py::array arr1 = df.attr(link_cat.c_str()); // this is a reference

// for ()
//   // {

//   // }
// }
