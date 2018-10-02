
// test boost.numpy
#include "test.h"

#include "Inspect.h"
#include "Module.h"
#include "Environment.h"

#include "python.h"
#include "numpy.h"

#include <vector>
#include <string>
#include <memory>


void test_np()
{
  neworder::log("boost.Python.numpy test");

  //neworder::Environment& env = neworder::getenv();

  py::object module = py::import("neworder");

  // create an array and expose to python...
  py::tuple shape = py::make_tuple(3, 3);
  np::dtype dtype = np::dtype::get_builtin<double>();
  np::ndarray a = np::zeros(shape, dtype);
  module.attr("a") = a;
  CHECK(a.get_nd() == 2);
  CHECK(pycpp::size(a) == 9);

  // python modifies array
  neworder::Callback::exec("a[1,1]=6.25")();  

  // check C++ sees its been modified
  CHECK(pycpp::at<double>(a,4) == 6.25);

  // modify the array in C++ using "iterators"
  size_t i = 0;
  for (double* p = pycpp::begin<double>(a); p != pycpp::end<double>(a); ++i, ++p)
  { 
    *p = (double)i / 10;
  }
  CHECK(neworder::Callback::eval("a[0,0] == 0.0")());  
  CHECK(neworder::Callback::eval("a[0,1] == 0.1")());  
  CHECK(neworder::Callback::eval("a[1,1] == 0.4")());  
  CHECK(neworder::Callback::eval("a[2,1] == 0.7")());  
  CHECK(neworder::Callback::eval("a[2,2] == 0.8")());  

  // modifying usibg index
  for (size_t i = 0; i < pycpp::size(a); ++i)
  {
    pycpp::at<double>(a, i) = (double)i / 100;
  }
  CHECK(neworder::Callback::eval("a[0,0] == 0.00")());  
  CHECK(neworder::Callback::eval("a[0,1] == 0.01")());  
  CHECK(neworder::Callback::eval("a[1,1] == 0.04")());  
  CHECK(neworder::Callback::eval("a[2,1] == 0.07")());  
  CHECK(neworder::Callback::eval("a[2,2] == 0.08")());  

  // load a DF and try to extract/modify...
  neworder::Callback::exec("import pandas as pd;import neworder;neworder.df=pd.read_csv('../../tests/df.csv')")();
  py::object df = module.attr("df");
  np::ndarray c = np::from_object(df.attr("columns").attr("values"));
  c[1] = "Changed";
  // check unchanged
  CHECK(neworder::Callback::eval("df.columns.values[0] == 'PID'")());
  // check changed
  CHECK(neworder::Callback::eval("df.columns.values[1] == 'Changed'")());

  // Can't modify DF values directly as 2d-array (it copies), need to select individual columns
  np::ndarray v = np::from_object(df.attr("Changed"));
  v[0] = "MOVED!";
  // check changed
  CHECK(neworder::Callback::eval("df.Changed[0] == 'MOVED!'")());
  // check unchanged
  CHECK(neworder::Callback::eval("df.Changed[1] == 'E02000001'")());

  //neworder::Callback::exec("import pandas as pd;import neworder;neworder.log(neworder.df.head())")();

  struct UnaryArrayFunc : pycpp::UnaryArrayOp<double, double>
  {
    UnaryArrayFunc(double m, double c) : m_m(m), m_c(c) { }
    
    double operator()(double x) { return m_m * x + m_c; }

    // workaround: above function hides base-class implementations of operator() 
    // see https://stackoverflow.com/questions/1628768/why-does-an-overridden-function-in-the-derived-class-hide-other-overloads-of-the/1629074#1629074
    using pycpp::UnaryArrayOp<double, double>::operator();

  private:
    double m_m;
    double m_c;
  };

  struct BinaryArrayFunc : pycpp::BinaryArrayOp<double, double, double>
  {

    BinaryArrayFunc(double m, double c) : m_m(m), m_c(c) { }
    
    double operator()(double x, double y) { return m_m * (x + y) + m_c; }

    // workaround: above function hides base-class implementations of operator() 
    // see https://stackoverflow.com/questions/1628768/why-does-an-overridden-function-in-the-derived-class-hide-other-overloads-of-the/1629074#1629074
    using pycpp::BinaryArrayOp<double, double, double>::operator();

  private:
    double m_m;
    double m_c;
  };

  np::ndarray in = pycpp::zero_1d_array<double>(9);
  UnaryArrayFunc f(1.0, 2.75);
  np::ndarray out = f(in);
  CHECK(pycpp::at<double>(out, 0) == 2.75);

  BinaryArrayFunc g(3.125, 1.0);
  np::ndarray out2 = g(in, out);
  
  CHECK(pycpp::at<double>(out2, 0) == 2.75 * 3.125 + 1.0);

  // Test vector-scalar operations
  // Inner product - rather than having operator() for syntactic sugar I'm using the constructor for this purpose
  // and providing an explicit operator double to produce the result. This avoids the extra pair of brackets (FWIW)
  struct DotFunc : pycpp::BinaryArrayOp<double, double, double>
  {
    typedef pycpp::BinaryArrayOp<double, double, double> super;
    
    double operator()(double x, double y) { return x * y; }

    // no workaround: above function hides base-class implementations of operator() 
    // We actually want to provide our own implementation - that returns a scalar rather than a vector 
    // So the base implementation remains hidden, we use it in our override to calculate the products,
    // which are then summed. 

    DotFunc(const py::object& arg1, const py::object& arg2) : m_result(0.0)
    {
      np::ndarray products = super::operator()(arg1, arg2);

      for (double* p = pycpp::begin<double>(products); p != pycpp::end<double>(products); ++p)
      {
        m_result += *p;
      }
    }

    operator double()
    {
      return m_result;
    }

  private:
    double m_result;
  };

  np::ndarray v1 = pycpp::make_array<double>(10, [](){ return 2.0; });
  np::ndarray v2 = pycpp::make_array<double>(10, [](){ return 0.5; });
  CHECK(DotFunc(v1, v2) == 10.0);

  // now see if it works with vector * scalar
  py::object scalar(1.0);
  CHECK(DotFunc(v1, scalar) == 20.0);

}