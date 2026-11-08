
#include "DataFrame.h"

#include "ArrayHelpers.h"
#include "Log.h"
#include "Model.h"
#include "MonteCarlo.h"

#include "NewOrder.h"

namespace {

bool is_categorical(const nb::object& pandas, const nb::object& series) {
  nb::object dtype = series.attr("dtype");
  nb::object categorical_dtype = pandas.attr("CategoricalDtype");
  int result = PyObject_IsInstance(dtype.ptr(), categorical_dtype.ptr());
  if (result < 0)
    throw nb::python_error();
  return result != 0;
}

} // namespace

// uniqueness is - but reproduciblity isn't - guaranteed when multiple threads are used
no::NpArray<int64_t> no::df::unique_index(size_t n) {
  int64_t s = no::env::size.load(std::memory_order_relaxed);
  int64_t i = no::env::uniqueIndex.fetch_add(n * s, std::memory_order_acq_rel);

  auto a = no::make_array<int64_t>({static_cast<Py_ssize_t>(n)}, [&]() {
    int64_t ret{i};
    i += s;
    return ret;
  });
  return a;
}

// matrix is a transition matrix. Its row order must correspond to series.cat.categories order
nb::object no::df::transition(no::MonteCarlo& mc, const ArrayLike<double>& matrix, const nb::object& series) {
  nb::object pandas = nb::module_::import_("pandas");
  if (!is_categorical(pandas, series)) {
    throw nb::type_error("series does not have a pandas 'category' dtype; convert it first, e.g. "
                         "series = series.astype('category')");
  }
  nb::object cat_accessor = series.attr("cat");
  Py_ssize_t m = static_cast<Py_ssize_t>(nb::len(nb::object(cat_accessor.attr("categories"))));

  // check matrix is 2d, square & its size matches the number of categories
  if (matrix.ndim() != 2)
    throw no::value_error("cumulative transition matrix dimension is %%"s % matrix.ndim());
  if (matrix.shape(0) != matrix.shape(1))
    throw no::value_error("cumulative transition matrix shape is not square: %% by %%"s % matrix.shape(0) %
                          matrix.shape(1));
  if (static_cast<size_t>(m) != matrix.shape(0))
    throw no::value_error("cumulative transition matrix size (%%) is not same as the number of categories (%%)"s %
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
  // operate on a local int64 copy rather than relying on `.cat.codes` being an aliased, in-place-writable view of
  // the Categorical's internal storage - that isn't part of pandas' public contract, and codes may be stored as
  // int8/16/32/64 depending on the number of categories - then write the result back explicitly via
  // pd.Categorical.from_codes.
  nb::object codes_obj = no::numpy::copy(nb::object(cat_accessor.attr("codes")), no::numpy::DType::INT64);
  no::NpArray<int64_t> codes = nb::cast<no::NpArray<int64_t>>(codes_obj);

  Py_ssize_t n = static_cast<Py_ssize_t>(codes.size());
  no::NpArray<double> rpy = mc.ustream(n);

  const double* r = rpy.data();
  int64_t* pcodes = codes.data();

  for (Py_ssize_t i = 0; i < n; ++i) {
    int64_t j = pcodes[i];
    // codes are -1 for NaN/missing categories - leave any such rows untouched
    if (j < 0 || j >= m)
      continue;
    Py_ssize_t k = no::interp(cumprobs[j], r[i]);
    pcodes[i] = k;
  }

  nb::object from_codes = pandas.attr("Categorical").attr("from_codes");
  return from_codes(codes_obj, cat_accessor.attr("categories"), cat_accessor.attr("ordered"));
}

// matrices maps each of group's category labels to the (square) transition matrix to apply to rows in that
// group; group and series must be the same length and row-aligned. Unlike transition()'s matrix argument,
// matrices are looked up by label rather than positionally by group.cat.codes, so the dict need not list its
// entries in categories order (and may contain unused extra keys).
nb::object no::df::transition_conditional(no::MonteCarlo& mc, const nb::dict& matrices, const nb::object& group,
                                          const nb::object& series) {
  nb::object pandas = nb::module_::import_("pandas");
  if (!is_categorical(pandas, series))
    throw nb::type_error("series does not have a pandas 'category' dtype; convert it first, e.g. "
                         "series = series.astype('category')");
  if (!is_categorical(pandas, group))
    throw nb::type_error("group does not have a pandas 'category' dtype; convert it first, e.g. "
                         "group = group.astype('category')");

  nb::object cat_accessor = series.attr("cat");
  Py_ssize_t m = static_cast<Py_ssize_t>(nb::len(nb::object(cat_accessor.attr("categories"))));

  nb::object group_cat_accessor = group.attr("cat");
  nb::object group_categories = group_cat_accessor.attr("categories");
  Py_ssize_t n_groups = static_cast<Py_ssize_t>(nb::len(group_categories));

  // per-group cumulative probabilities, indexed positionally by group code (group.cat.categories order) even
  // though matrices itself is keyed by label - this mirrors the cumprobs[from-state] table built in transition()
  std::vector<std::vector<std::vector<double>>> cumprobs(n_groups);
  for (Py_ssize_t g = 0; g < n_groups; ++g) {
    nb::object label = group_categories[nb::int_(g)];
    if (!matrices.contains(label))
      throw no::value_error("no transition matrix supplied for group '%%'"s % label);

    no::ArrayLike<double> matrix = no::as_array<double>(nb::object(matrices[label]));

    if (matrix.ndim() != 2)
      throw no::value_error("transition matrix for group '%%' dimension is %%"s % label % matrix.ndim());
    if (matrix.shape(0) != matrix.shape(1))
      throw no::value_error("transition matrix for group '%%' shape is not square: %% by %%"s % label %
                            matrix.shape(0) % matrix.shape(1));
    if (static_cast<size_t>(m) != matrix.shape(0))
      throw no::value_error(
          "transition matrix for group '%%' size (%%) is not same as the number of categories (%%)"s % label %
          matrix.shape(0) % m);

    cumprobs[g].resize(m);
    for (Py_ssize_t i = 0; i < m; ++i) {
      cumprobs[g][i] = no::cumulative(no::cbegin(matrix) + (i * m), m);
    }
  }

  nb::object codes_obj = no::numpy::copy(nb::object(cat_accessor.attr("codes")), no::numpy::DType::INT64);
  no::NpArray<int64_t> codes = nb::cast<no::NpArray<int64_t>>(codes_obj);

  no::ArrayLike<int64_t> group_codes = no::as_array<int64_t>(nb::object(group_cat_accessor.attr("codes")));

  Py_ssize_t n = static_cast<Py_ssize_t>(codes.size());
  if (group_codes.size() != static_cast<size_t>(n))
    throw no::value_error("group (%%) and series (%%) must have the same length"s % group_codes.size() % n);

  no::NpArray<double> rpy = mc.ustream(n);

  const double* r = rpy.data();
  int64_t* pcodes = codes.data();
  const int64_t* pgroup = group_codes.data();

  for (Py_ssize_t i = 0; i < n; ++i) {
    int64_t g = pgroup[i];
    // codes are -1 for NaN/missing categories - leave any such rows untouched
    if (g < 0 || g >= n_groups)
      continue;
    int64_t j = pcodes[i];
    if (j < 0 || j >= m)
      continue;
    Py_ssize_t k = no::interp(cumprobs[g][j], r[i]);
    pcodes[i] = k;
  }

  nb::object from_codes = pandas.attr("Categorical").attr("from_codes");
  return from_codes(codes_obj, cat_accessor.attr("categories"), cat_accessor.attr("ordered"));
}
