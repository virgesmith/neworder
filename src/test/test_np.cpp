
// test numpy and no::nparray functions
#include "test.h"

#include "Inspect.h"
#include "Module.h"
#include "Environment.h"
#include "NPArray.h"

#include "NewOrder.h"
#include "ArrayHelpers.h"

#include <vector>
#include <string>
#include <memory>


void test_np()
{
  no::log("numpy test");

  py::object module = py::module::import("neworder");
  py::object root = py::module::import("__main__");
  no::Runtime runtime("neworder");

  {
    // create an array and expose to python...
    py::array a = no::zeros<double>({2,3,5});
    root.attr("a") = a;
    CHECK(a.ndim() == 3);
    CHECK(a.size() == 30);
    CHECK(a.shape()[0] == 2);
    CHECK(a.shape()[1] == 3);
    CHECK(a.shape()[2] == 5);
    CHECK(a.strides()[0] == 3*5 * sizeof(double));
    CHECK(a.strides()[1] == 5 * sizeof(double));
    CHECK(a.strides()[2] == 1 * sizeof(double));
  } 
  // create an array and expose to python...
  py::array a = no::zeros<double>({3,4});
  root.attr("a") = a;
  CHECK(a.ndim() == 2);
  CHECK(a.size() == 12);
  CHECK(a.shape()[0] == 3);
  CHECK(a.shape()[1] == 4);
  CHECK(a.strides()[0] == 4 * sizeof(double));
  CHECK(a.strides()[1] == 1 * sizeof(double));
  
  for (ssize_t i = 0; i < a.size(); ++i) 
  {
    no::Command cmd = {("a[%%,%%]"_s % (i/4) % (i%4)), no::CommandType::Eval};
    CHECK(runtime(cmd).cast<double>() == 0.0);
  }

  // python modifies array
  runtime({"a[1,1]=6.25", no::CommandType::Exec});  

  // check C++ sees its been modified
  {
    // cant have comma in macro
    double v = no::at<double, 2>(a, {1, 1}); 
    CHECK(v == 6.25);
  }
  //modify the array in C++ using "iterators"
  size_t i = 0;
  for (double* p = no::begin<double>(a); p != no::end<double>(a); ++i, ++p)
  { 
    *p = (double)i / 10;
  }
  CHECK(runtime({"a[0,0] == 0.0", no::CommandType::Eval}).cast<bool>());  
  CHECK(runtime({"a[0,1] == 0.1", no::CommandType::Eval}).cast<bool>());  
  CHECK(runtime({"a[1,1] == 0.5", no::CommandType::Eval}).cast<bool>());  
  CHECK(runtime({"a[2,1] == 0.9", no::CommandType::Eval}).cast<bool>());  
  CHECK(runtime({"a[2,2] == 1.0", no::CommandType::Eval}).cast<bool>());  
  CHECK(runtime({"a[2,3] == 1.1", no::CommandType::Eval}).cast<bool>());  

  // modifying using index
  for (size_t i = 0; i < a.shape()[0]; ++i)
  {
    for (size_t j = 0; j < a.shape()[1]; ++j)
    no::at<double, 2>(a, {i,j}) = (double)(i*10+j);
  }
  CHECK(runtime({"a[0,0] == 0.0", no::CommandType::Eval}).cast<bool>());  
  CHECK(runtime({"a[0,1] == 1.0", no::CommandType::Eval}).cast<bool>());  
  CHECK(runtime({"a[1,1] == 11.0", no::CommandType::Eval}).cast<bool>());  
  CHECK(runtime({"a[2,1] == 21.0", no::CommandType::Eval}).cast<bool>());  
  CHECK(runtime({"a[2,2] == 22.0", no::CommandType::Eval}).cast<bool>());  
  CHECK(runtime({"a[2,3] == 23.0", no::CommandType::Eval}).cast<bool>());  

  // load a DF and try to extract/modify...
  runtime({"import pandas as pd;import neworder;neworder.df=pd.read_csv('../../tests/df.csv')", no::CommandType::Exec});
  py::object df = module.attr("df");
  py::array cols = df.attr("columns").attr("values");
  // // check unchanged (no pybind11 support for string arrays)
  // CHECK(runtime({"df.columns.values[0] == 'PID'", no::CommandType::Eval}).cast<bool>());
  // CHECK(runtime({"df.columns.values[1] == 'Area'", no::CommandType::Eval}).cast<bool>());

  // TODO how to read/write string arrays...

  // Can't modify DF values directly as 2d-array (it copies), need to select individual columns
  py::array v = df.attr("PID" );
  CHECK(no::at<int64_t>(v, {0}) == 0);
  CHECK(no::at<int64_t>(v, {(size_t)v.size()-1}) == v.size()-1);
  // increment each value
  for (int64_t* p = no::begin<int64_t>(v); p != no::end<int64_t>(v); ++p) *p += 1;
  // check python sees update
  // CHECK(runtime({"df.PID[0] == 1", no::CommandType::Eval}).cast<bool>());
  // CHECK(runtime({"df.PID[len(df)-1] == len(df)", no::CommandType::Eval}).cast<bool>());
  // v[0] = "MOVED!";
  // check changed
  // CHECK(no::Callback::eval("df.Changed[0] == 'MOVED!'")());
  // // check unchanged
  // CHECK(no::Callback::eval("df.Changed[1] == 'E02000001'")());

  //no::Callback::exec("import pandas as pd;import neworder;neworder.log(neworder.df.head())")();

  py::array in = no::zeros<double>({9});
  // UnaryArrayFunc f(1.0, 2.75);
  // py::array out2 = g(in, out);
  py::array out = no::unary_op<double, double>(in, [](double x) { return 1.0 * x + 2.75; });
  CHECK(no::at<double>(out, {0}) == 2.75);

  // BinaryArrayFunc g(3.125, 1.0);
  // py::array out2 = g(in, out);
  py::array out2 = no::binary_op<double, double, double>(in, out, [](double x, double y) { return 3.125 * (x + y) + 1.0; });
  
  CHECK(no::at<double>(out2, {0}) == 2.75 * 3.125 + 1.0);

  // Test vector-scalar operations

  auto dot = [](const py::array& x, const py::array& y) { 
    py::array products = no::binary_op<double, double, double>(x, y, [](double a, double b){ return a * b; });
    double result = 0.0;
    for (double* p = no::begin<double>(products); p != no::end<double>(products); ++p)
    {
      result += *p;
    }
    return result;
  };

  py::array v1 = no::make_array<double>({10}, [](){ return 2.0; });
  py::array v2 = no::make_array<double>({10}, [](){ return 0.5; });
  CHECK(dot(v1, v2) == 10.0);

  // // now see if it works with vector * scalar
  // py::object scalar(1.0);
  // CHECK(DotFunc(v1, scalar) == 20.0);

  // py::array n1 = no::nparray::isnever(v1);
  // CHECK(!n1[0]);

  py::array x = no::zeros<double>({1});
  py::array p = no::logistic(x, 0.0, 1.0);
  CHECK(no::at<double>(p, {0}) == 0.5);

  py::array x2 = no::logit(p);
  CHECK(no::at<double>(x2, {0}) == 0.0);

}