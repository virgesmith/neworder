
#include "DataFrame.h"

#include "MPIComms.h"

#include "python.h"

void neworder::df::transition(np::ndarray& col)
{
  // std::mt19937& prng = pycpp::getenv().prng();
  // std::uniform_real_distribution<> dist(0.0, 1.0);

  size_t n = pycpp::size(col);

  // std::vector<double> r(n);
  // std::generate(r.begin(), r.end(), [](){ return dist(prng);}); 

  for (size_t i = 0; i < n; ++i)
  {
    pycpp::at<int64_t>(col, i) += (int64_t)i;
  }
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



