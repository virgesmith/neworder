// Deprecated

// test4 - boost.numpy
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

  pycpp::Environment& env = pycpp::Environment::get();

  py::object module = py::import("neworder");

  // create an array and expose to python...
  py::tuple shape = py::make_tuple(3, 3);
  np::dtype dtype = np::dtype::get_builtin<double>();
  np::ndarray a = np::zeros(shape, dtype);
  module.attr("a") = a;

  // TODO proper test stuff

  neworder::Callback::exec("import neworder;neworder.log(neworder.a);a[1,1]=3.14")();  

  // check its been modified
  neworder::log(pycpp::as_string(a));

  // modify it again
  // yuck
  double* p = reinterpret_cast<double*>(a.get_data());

  int dim = a.get_nd();
  // assumes dim >=1 
  int s = a.shape(0);
  for (int i = 1; i < dim; ++i)
    s *= a.shape(i);
  for (int i = 0; i < s; ++i)
    p[i] = (double)i / 10;

  neworder::Callback::exec("import neworder;neworder.log(neworder.a)")();  

  // load a DF and try to extract/modify...
  neworder::Callback::exec("import pandas as pd;import neworder;neworder.df=pd.read_csv('../../tests/df.csv')")();

  py::object df = module.attr("df");
  np::ndarray c = np::from_object(df.attr("columns").attr("values"));
  neworder::log(pycpp::as_string(c));
  c[1] = "Changed";

  // Can't modify DF values directly as 2d-array (it copies), need to select individual columns
  np::ndarray v = np::from_object(df.attr("Changed"));
  neworder::log(pycpp::as_string(v));
  v[0] = "MOVED!";
  neworder::log(pycpp::as_string(v));
  neworder::Callback::exec("import pandas as pd;import neworder;neworder.log(neworder.df.head())")();


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
  UnaryArrayFunc f(1.0, 2.718);
  np::ndarray out = f(in);
  neworder::log(out);

  BinaryArrayFunc g(3.141, 1.0);
  np::ndarray out2 = g(in, out);
  
  neworder::log(out2);

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
  np::ndarray v2 = pycpp::make_array<double>(10, [](){ return 0.4; });
  neworder::log(v1);
  neworder::log(v2);
  neworder::log("Dot prod: " + std::to_string(DotFunc(v1, v2)));

  // now see if it works with vector * scalar
  py::object scalar(1.0);
  neworder::log("scalar * vector sum: " + std::to_string(DotFunc(v1, scalar)));

}