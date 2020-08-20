
#include "DataFrame.h"

#include "Model.h"
#include "MonteCarlo.h"
#include "Timer.h"
#include "MPIComms.h"

#include "NewOrder.h"

#include <map>

size_t interp(const std::vector<double>& cumprob, double x)
{
  // TODO would bisection search here be better (even for small arrays)?
  size_t lbound = 0;
  for (; lbound < cumprob.size() - 1; ++lbound)
  {
    if (cumprob[lbound] > x)
      break;
  }
  return lbound;
}


// TODO one-off-
// categories are all possible category labels. Order corresponds to row/col in matrix
// matrix is a transition matrix
void no::df::transition(no::Model& model, py::array categories, py::array matrix, py::object &df, const std::string& colname)
{
  // Extract column from DF as np.array
  py::array col = df.attr(colname.c_str());

  py::ssize_t m = categories.size();

  // check matrix is 2d, square & categories len = matrix len
  if (matrix.ndim() != 2)
    throw std::runtime_error("cumulative transition matrix dimension is %%"_s % matrix.ndim());
  if (matrix.shape(0) != matrix.shape(1))
    throw std::runtime_error("cumulative transition matrix shape is not square: %% by %%"_s % matrix.shape(0) % matrix.shape(1));
  if (m != matrix.shape(0))
    throw std::runtime_error("cumulative transition matrix size (%%) is not same as length of categories (%%)"_s % matrix.shape(0) % m);

  // construct checked cumulative probabilities for each state to randomly interpolate
  // IMPORTANT NOTE: whilst numpy is row-major, pandas stores column-major, i.e. the columns are contiguous memory
  // (transposing in python doesnt change the memory layout, it just changes the view)
  // the code below assumes the transition matrix is column major (i.e. col sums to unity not rows)
  // but produces a *row-major* cumulative probability matrix

  std::vector<std::vector<double>> cumprobs(m, std::vector<double>(m));
  for (int i = 0; i < m; ++i)
  {
    // point to beginning of row
    double* p = no::begin<double>(matrix) + i;
    if (p[0] < 0.0 || p[0] > 1.0) 
      // % is misinterpreted. NB to find code run cppcheck --xml 2> cppcheck,xml to find codes 
      // cppcheck-suppress zerodiv 
      throw std::runtime_error("invalid transition probability %% at (%%,%%)"_s % p[0] % i % 0);
    cumprobs[i][0] = p[0];
    for (int j = 1; j < m; ++j)
    {
      if (p[j] < 0.0 || p[j] > 1.0) 
      // cppcheck-suppress zerodiv 
        throw std::runtime_error("invalid transition probability %% at (%%,%%)"_s % p[0] % i % 0);
      cumprobs[i][j] = cumprobs[i][j-1] + p[j*m];
    }
    // check probabilities sum to unity within tolerance
    if (fabs(cumprobs[i][m-1] - 1.0) > std::numeric_limits<double>::epsilon())
      throw std::runtime_error("probabilities don't sum to unity (%%) in transition matrix row %%"_s % cumprobs[i][m-1] % i);
  }

  // reverse catgory lookup
  std::map<int64_t, int> lookup;
  for (py::ssize_t i = 0; i < m; ++i)
  {
    lookup[no::at<int64_t>(categories, Index_t<1>{i})] = (int64_t)i;
  }

  // no::log("row %% %% %% %%..."_s % p[0] % p[1] % p[2] % p[3]);
  // no::log("col %% %% %% %%..."_s % p[0] % p[m] % p[2*m] % p[3*m]);

  // std::mt19937& prng = no::getenv().prng();

  py::ssize_t n = col.size();

  Timer t;
  // define a base model to init the MC engine
  py::array_t<double> r = model.mc().ustream(n);
  // if bottleneck, can access through
  // auto p = r.unchecked<1>();
  // x = p[i];

  for (py::ssize_t i = 0; i < n; ++i)
  {
    // look up the index, ignoring values that havent been explicitly set in categories (like -1)
    auto it = lookup.find(no::at<int64_t>(col, Index_t<1>{i}));
    if (it == lookup.end())
      continue;
    int64_t j = it->second;
    py::ssize_t k = interp(cumprobs[j], no::at<double>(r, {i}));
    //no::log("interp %%:%% -> %%"_s % j % r[i] % k);
    no::at<int64_t>(col, {i}) = no::at<int64_t>(categories, Index_t<1>{k});
  }
  no::log("transition %% elapsed: %%"_s % n % t.elapsed_s());
}

// example of directly modifying a DF?
void no::df::directmod(no::Model& model, py::object& df, const std::string& colname)
{
  // .values? pd.Series -> np.array?
  py::array arr = df.attr(colname.c_str());
  py::ssize_t n = arr.size();

  for (py::ssize_t i = 0; i < n; ++i)
  {
    no::at<int64_t>(arr, Index_t<1>{i}) += 1;
  }
}


// void no::df::linked_change(py::object& df, const std::string& cat, const std::string& link_cat)
// {
//   // .values? pd.Series -> np.array?
//   py::array arr0 = df.attr(cat.c_str()); // this is a reference 
//   // .values? pd.Series -> np.array?
//   py::array arr1 = df.attr(link_cat.c_str()); // this is a reference 

//   throw std::runtime_error("ongoing dev (liam2-demo07?)");
//   // for ()
//   // {

//   // }
// }

// // append two DFs? pointless to call c++ that just calls python
// py::object no::df::append(const py::object& df1, const py::object& df2)
// {
//   py::dict kwargs;
//   kwargs["ignore_index"] = true;
//   py::object result = df1.attr("append")(df2, kwargs);
//   return result;
// }
