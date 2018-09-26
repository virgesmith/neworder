
#include "DataFrame.h"

#include "Timer.h"
#include "MPIComms.h"

#include "python.h"

#include <map>

size_t interp(const std::vector<double>& cumprob, double x)
{
  size_t lbound = 0;
  for (; lbound < cumprob.size() - 1; ++lbound)
  {
    if (cumprob[lbound] > x)
      break;
  }
  return lbound;
}

// categories are all possible categorty labels. Order corresponds to row/col in matrix
// matrix is a transition matrix
void neworder::df::transition(np::ndarray categories, np::ndarray matrix, py::object &df, const std::string& colname)
{
  // Extract column from DF as np.array
  np::ndarray col = np::from_object(df.attr(colname.c_str()));

  int m = pycpp::size(categories);

  // check matrix is 2d, square & categories len = matrix len
  if (matrix.get_nd() != 2)
    throw std::runtime_error("cumulative transition matrix dimension is %%"_s % matrix.get_nd());
  if (matrix.get_shape()[0] != matrix.get_shape()[1])
    throw std::runtime_error("cumulative transition matrix shape is not square: %% by %%"_s % matrix.get_shape()[0] % matrix.get_shape()[1]);
  if (m != matrix.get_shape()[0])
    throw std::runtime_error("cumulative transition matrix size (%%) is not same as length of categories (%%)"_s % matrix.get_shape()[0] % m);

  // construct checked cumulative probabilities for each state to randomly interpolate
  std::vector<std::vector<double>> cumprobs(m, std::vector<double>(m));
  for (int i = 0; i < m; ++i)
  {
    // point to beginning of row
    double* p = pycpp::begin<double>(matrix) + i * m;
    if (p[0] < 0.0 || p[0] > 1.0) 
      throw std::runtime_error("invalid transition probability %% at (%%,%%)"_s % p[0] % i % 0);
    cumprobs[i][0] = p[0];
    for (int j = 1; j < m; ++j)
    {
      if (p[j] < 0.0 || p[j] > 1.0) 
        throw std::runtime_error("invalid transition probability %% at (%%,%%)"_s % p[0] % i % 0);
      cumprobs[i][j] = cumprobs[i][j-1] + p[j];
    }
    if (cumprobs[i][m-1] != 1.0)
      throw std::runtime_error("probabilities don't sum to unity in transition matrix row %%"_s % i);
  }

  // TODO check each row is monotonic 0->1

  // catgory lookup
  std::map<int64_t, int> lookup;
  for (int i = 0; i < m; ++i)
  {
    lookup[pycpp::at<int64_t>(categories, i)] = i;
    //neworder::log("%%->%%"_s % pycpp::at<int64_t>(categories, i) % lookup[pycpp::at<int64_t>(categories, i)]);
  }

  // neworder::log("row %% %% %% %%..."_s % p[0] % p[1] % p[2] % p[3]);
  // neworder::log("col %% %% %% %%..."_s % p[0] % p[m] % p[2*m] % p[3*m]);

  std::mt19937& prng = pycpp::getenv().prng();
  std::uniform_real_distribution<> dist(0.0, 1.0);

  size_t n = pycpp::size(col);

  Timer t;
  std::vector<double> r(n);
  std::generate(r.begin(), r.end(), [&](){ return dist(prng);}); 

  for (size_t i = 0; i < n; ++i)
  {
    // look up the index...
    int64_t j = lookup[pycpp::at<int64_t>(col, i)];
    int64_t k = interp(cumprobs[j], r[i]);
    //neworder::log("interp %%:%% -> %%"_s % j % r[i] % k);
    pycpp::at<int64_t>(col, i) = pycpp::at<int64_t>(categories, k);
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
