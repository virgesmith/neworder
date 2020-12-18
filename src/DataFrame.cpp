
#include "DataFrame.h"

#include "Model.h"
#include "MonteCarlo.h"
#include "Timer.h"
#include "Log.h"
#include "ArrayHelpers.h"

#include "NewOrder.h"

#include <map>


py::array_t<int64_t> no::df::unique_index(size_t n)
{
  int64_t i = no::env::uniqueIndex.load(std::memory_order_acquire);
  int64_t s = no::env::size.load(std::memory_order_relaxed);
  auto a = no::make_array<int64_t>({static_cast<py::ssize_t>(n)}, [&]() { int64_t ret = i; i += s; return ret; });
  no::env::uniqueIndex.store(i, std::memory_order_release);
  return a;
}


// TODO non-integer categories?
// TODO different output column?
// categories are all possible category labels. Order corresponds to row/col in matrix
// matrix is a transition matrix
void no::df::transition(no::Model& model, py::array_t<int64_t> categories, py::array_t<double> matrix, py::object &df, const std::string& colname)
{
  // Extract column from DF as np.array
  py::array col_untyped = df.attr(colname.c_str());

  // check col is int64
  if (!col_untyped.dtype().is(py::dtype::of<int64_t>()))
  {
    throw py::type_error("dataframe transitions can only be performed on columns containing int64 values");
  }
  py::array_t<int64_t> col = col_untyped;

  py::ssize_t m = categories.size();

  // check matrix is 2d, square & categories len = matrix len
  if (matrix.ndim() != 2)
    throw py::type_error("cumulative transition matrix dimension is %%"s % matrix.ndim());
  if (matrix.shape(0) != matrix.shape(1))
    throw py::type_error("cumulative transition matrix shape is not square: %% by %%"s % matrix.shape(0) % matrix.shape(1));
  if (m != matrix.shape(0))
    throw py::value_error("cumulative transition matrix size (%%) is not same as length of categories (%%)"s % matrix.shape(0) % m);

  // IMPORTANT NOTES:
  // - whilst numpy is row-major, pandas stores column-major, i.e. the columns are contiguous memory
  // - transposing (square matrices at least) in python doesn't change the memory layout, it just changes the view
  // - the code below assumes the transition matrix has a row major memory layout (i.e. row sums to unity not cols)

  // construct checked cumulative probabilities for each state to randomly interpolate
  std::vector<std::vector<double>> cumprobs(m);
  for (int i = 0; i < m; ++i)
  {
    cumprobs[i] = no::cumulative(no::cbegin(matrix) + (i * m), m);
    // // point to beginning of row
    // double* p = no::begin(matrix) + (i * m);
    // if (p[0] < 0.0 || p[0] > 1.0)
    //   throw py::value_error("invalid transition probability %% at (%%, 0)"s % p[0] % i);
    // cumprobs[i][0] = p[0];
    // for (int j = 1; j < m; ++j)
    // {
    //   if (p[j] < 0.0 || p[j] > 1.0)
    //     throw py::value_error("invalid transition probability %% at (%%, %%)"s % p[0] % i % j);
    //   cumprobs[i][j] = cumprobs[i][j-1] + p[j];
    // }
    // // check probabilities sum to unity within tolerance
    // if (fabs(cumprobs[i][m-1] - 1.0) > std::numeric_limits<double>::epsilon())
    //   throw py::value_error("probabilities don't sum to unity (%%) in transition matrix row %%"s % cumprobs[i][m-1] % i);
  }

  // reverse catgory lookup
  std::map<int64_t, int64_t> lookup;
  for (py::ssize_t i = 0; i < m; ++i)
  {
    lookup[no::at<int64_t>(categories, Index_t<1>{i})] = i;
  }

  py::ssize_t n = col.size();

  Timer t;
  // define a base model to init the MC engine
  py::array_t<double> rpy = model.mc().ustream(n);

  // possible unsafe access?
  double* r = no::begin(rpy);
  int64_t* pcat = no::begin<int64_t>(categories);

  for (py::ssize_t i = 0; i < n; ++i)
  {
    // look up the index, ignoring values that haven't been explicitly set in categories (like -1)
    auto it = lookup.find(no::at<int64_t>(col, Index_t<1>{i}));
    if (it == lookup.end())
      continue;
    int64_t j = it->second;
    py::ssize_t k = no::interp(cumprobs[j], r[i]/*no::at(r, Index_t<1>{i})*/);
    //no::log("interp %%:%% -> %%"s % j % r[i] % k);
    no::at<int64_t>(col, Index_t<1>{i}) = pcat[k]; //no::at<int64_t>(categories, Index_t<1>{k});
  }
  //no::log("transition %% elapsed: %%"s % n % t.elapsed_s());
}


template<typename T>
void dump(const T* p, py::ssize_t n)
{
  for (py::ssize_t i = 0; i < n; ++i, ++p)
  {
    no::log("%%"s % *p);
    //no::at<std::string>(arr, Index_t<1>{i}) += 1;
  }
}


// example of directly modifying a DF testing different dtypes
void no::df::testfunc(no::Model& model, py::object& df, const std::string& colname)
{
  // .values? pd.Series -> np.array?
  py::array arr = df.attr(colname.c_str()); //.request();

  //no::log(arr.dtype());
  py::buffer_info buf = arr.request();

  py::ssize_t n = buf.shape[0];


  if (arr.dtype().is(py::dtype::of<int64_t>()))
  {
    dump(static_cast<int64_t*>(buf.ptr), n);
  }
  else if (arr.dtype().is(py::dtype::of<double>()))
  {
    dump(static_cast<double*>(buf.ptr), n);
  }
  else if (arr.dtype().is(py::dtype::of<bool>()))
  {
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
  else
  {
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

