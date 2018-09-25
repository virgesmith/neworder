
#include "DataFrame.h"

#include "Timer.h"
#include "MPIComms.h"

#include "python.h"

void neworder::df::transition(np::ndarray categories, np::ndarray matrix, np::ndarray& col)
{
  // checks:
  // categories len = matrix len
  // matrix is 2d and square

  if (matrix.get_nd() != 2)
    throw std::runtime_error("cumulative transition matrix dimension is %%"_s % matrix.get_nd());
  if (matrix.get_shape()[0] != matrix.get_shape()[1])
    throw std::runtime_error("cumulative transition matrix shape is not square: %% by %%"_s % matrix.get_shape()[0] % matrix.get_shape()[1]);
  if (pycpp::size(categories) != (size_t)matrix.get_shape()[0])
    throw std::runtime_error("cumulative transition matrix size (%%) is not same as length of categories (%%)"_s % matrix.get_shape()[0] % pycpp::size(categories));



  Timer t;
  std::mt19937& prng = pycpp::getenv().prng();
  std::uniform_real_distribution<> dist(0.0, 1.0);

  size_t n = pycpp::size(col);

  std::vector<double> r(n);
  std::generate(r.begin(), r.end(), [&](){ return dist(prng);}); 

  (void)categories;
  (void)matrix;

  for (size_t i = 0; i < n; ++i)
  {
    pycpp::at<int64_t>(col, i) += (int64_t)i;
  }
  neworder::log("transition %% elapsed: %%"_s % n % t.elapsed_s());
}

// directly modify DF?
void neworder::df::directmod(py::object& df, const std::string& colname)
{
  py::object col = df.attr(colname.c_str());
  // .values? pd.Series -> np.array?
  np::ndarray arr = np::from_object(col);
  size_t n = pycpp::size(arr);

  for (size_t i = 0; i < n; ++i)
  {
    pycpp::at<int64_t>(arr, i) += 1;
  }
}

// append two DFs?
py::object neworder::df::append(const py::object& df1, const py::object& df2)
{
  py::dict kwargs;
  kwargs["ignore_index"] = true;
  py::object result = df1.attr("append")(df2, kwargs);
  return result;
}



