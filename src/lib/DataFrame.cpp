
#include "DataFrame.h"

#include "MPISendReceive.h"

void neworder::df::transition(np::ndarray& col)
{
  // std::mt19937& prng = pycpp::Environment::get().prng();
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
    pycpp::at<int64_t>(arr, i) += (int64_t)i;
  }
}

#include "Log.h"

// append two DFs?
py::object neworder::df::append(const py::object& df1, const py::object& df2)
{
  // TODO kwargs ignore_index
  py::dict kwargs;
  kwargs["ignore_index"] = true;
  py::object result = df1.attr("append")(df2, kwargs);
  return result;
}

// to the next rank 
void neworder::df::send(const py::object& o, int rank)
{
  py::object pickle = py::import("pickle");
  py::object serialised = pickle.attr("dumps")(o);

  // something along the lines of (see https://docs.python.org/3.6/c-api/buffer.html)
  void* p = (Py_buffer*)serialised.ptr(); 
  neworder::log("sending 0 (*%%) to 1"_s % (size_t)p);
  neworder::mpi::send(0, 1);
}

py::object neworder::df::receive(int rank)
{
  // py::object pickle = py::import("pickle");
  // py::object o = pickle.attr("dumps")(o);
  int x;
  neworder::mpi::receive(x, 0);
  neworder::log("got %% from 0"_s % x);
  return py::object();
}

void neworder::df::send_csv(const py::object& df, int rank)
{
  py::object io = py::import("io");
  py::object buf = io.attr("StringIO")();
  // py::dict kwargs;
  // kwargs["index"] = py::object(false);
  
  df.attr("to_csv")(buf); //, kwargs); //TODO why not working 
  std::string csvbuf = py::extract<std::string>(buf.attr("getvalue")())();
  neworder::mpi::send(csvbuf, rank);
}

py::object neworder::df::receive_csv(int rank)
{
  std::string buf;
  neworder::mpi::receive(buf, rank);

  py::object io = py::import("io");
  py::object pd = py::import("pandas");
  py::object pybuf = io.attr("StringIO")(buf);
  py::object df = pd.attr("read_csv")(pybuf);
  return df;
}


