
// test numpy and no::nparray functions
#include "test.h"

#include "Inspect.h"
#include "Module.h"
#include "Environment.h"
#include "NPArray.h"

#include "NewOrder.h"
#include "numpy.h"

#include <vector>
#include <string>
#include <memory>


void test_np()
{
  no::log("numpy test");

  //no::Environment& env = no::getenv();

  py::object module = py::module::import("neworder");

  // create an array and expose to python...
  np::array a = np::zeros<double>({3,3});
  module.attr("a") = a;
  CHECK(a.ndim() == 2);
  CHECK(pycpp::size(a) == 9);

  // python modifies array
  no::Callback::exec("a[1,1]=6.25")();  

  // check C++ sees its been modified
  CHECK(pycpp::at<double>(a,4) == 6.25);

  // modify the array in C++ using "iterators"
  size_t i = 0;
  for (double* p = pycpp::begin<double>(a); p != pycpp::end<double>(a); ++i, ++p)
  { 
    *p = (double)i / 10;
  }
  CHECK(no::Callback::eval("a[0,0] == 0.0")());  
  CHECK(no::Callback::eval("a[0,1] == 0.1")());  
  CHECK(no::Callback::eval("a[1,1] == 0.4")());  
  CHECK(no::Callback::eval("a[2,1] == 0.7")());  
  CHECK(no::Callback::eval("a[2,2] == 0.8")());  

  // modifying usibg index
  for (size_t i = 0; i < pycpp::size(a); ++i)
  {
    pycpp::at<double>(a, i) = (double)i / 100;
  }
  CHECK(no::Callback::eval("a[0,0] == 0.00")());  
  CHECK(no::Callback::eval("a[0,1] == 0.01")());  
  CHECK(no::Callback::eval("a[1,1] == 0.04")());  
  CHECK(no::Callback::eval("a[2,1] == 0.07")());  
  CHECK(no::Callback::eval("a[2,2] == 0.08")());  

  // load a DF and try to extract/modify...
  no::Callback::exec("import pandas as pd;import neworder;neworder.df=pd.read_csv('../../tests/df.csv')")();
  py::object df = module.attr("df");
  np::array c = df.attr("columns").attr("values");
  pycpp::at<const char*>(c, 1) = "Changed";
  // check unchanged
  CHECK(no::Callback::eval("df.columns.values[0] == 'PID'")());
  // check changed
  //CHECK(no::Callback::eval("df.columns.values[1] == 'Changed'")());

  // Can't modify DF values directly as 2d-array (it copies), need to select individual columns
  // np::array v = df.attr("Changed");
  // v[0] = "MOVED!";
  // check changed
  // CHECK(no::Callback::eval("df.Changed[0] == 'MOVED!'")());
  // // check unchanged
  // CHECK(no::Callback::eval("df.Changed[1] == 'E02000001'")());

  //no::Callback::exec("import pandas as pd;import neworder;neworder.log(neworder.df.head())")();

  // struct UnaryArrayFunc : pycpp::UnaryArrayOp<double, double>
  // {
  //   UnaryArrayFunc(double m, double c) : m_m(m), m_c(c) { }
    
  //   double operator()(double x) { return m_m * x + m_c; }

  //   // workaround: above function hides base-class implementations of operator() 
  //   // see https://stackoverflow.com/questions/1628768/why-does-an-overridden-function-in-the-derived-class-hide-other-overloads-of-the/1629074#1629074
  //   using pycpp::UnaryArrayOp<double, double>::operator();

  // private:
  //   double m_m;
  //   double m_c;
  // };

  // struct BinaryArrayFunc : pycpp::BinaryArrayOp<double, double, double>
  // {

  //   BinaryArrayFunc(double m, double c) : m_m(m), m_c(c) { }
    
  //   double operator()(double x, double y) { return m_m * (x + y) + m_c; }

  //   // workaround: above function hides base-class implementations of operator() 
  //   // see https://stackoverflow.com/questions/1628768/why-does-an-overridden-function-in-the-derived-class-hide-other-overloads-of-the/1629074#1629074
  //   using pycpp::BinaryArrayOp<double, double, double>::operator();

  // private:
  //   double m_m;
  //   double m_c;
  // };

  np::array in = pycpp::zero_1d_array<double>(9);
  // UnaryArrayFunc f(1.0, 2.75);
  np::array out = pycpp::unary_op<double, double>(in, [](double x) { return 1.0 * x + 2.75; });
  CHECK(pycpp::at<double>(out, 0) == 2.75);

  // BinaryArrayFunc g(3.125, 1.0);
  // np::array out2 = g(in, out);
  np::array out2 = pycpp::binary_op<double, double, double>(in, out, [](double x, double y) { return 3.125 * (x + y) + 1.0; });
  
  CHECK(pycpp::at<double>(out2, 0) == 2.75 * 3.125 + 1.0);

  // Test vector-scalar operations
  // Inner product - rather than having operator() for syntactic sugar I'm using the constructor for this purpose
  // and providing an explicit operator double to produce the result. This avoids the extra pair of brackets (FWIW)
  // struct DotFunc : pycpp::BinaryArrayOp<double, double, double>
  // {
  //   typedef pycpp::BinaryArrayOp<double, double, double> super;
    
  //   double operator()(double x, double y) { return x * y; }

  //   // no workaround: above function hides base-class implementations of operator() 
  //   // We actually want to provide our own implementation - that returns a scalar rather than a vector 
  //   // So the base implementation remains hidden, we use it in our override to calculate the products,
  //   // which are then summed. 

  //   DotFunc(const py::object& arg1, const py::object& arg2) : m_result(0.0)
  //   {
  //     np::array products = super::operator()(arg1, arg2);

  //     for (double* p = pycpp::begin<double>(products); p != pycpp::end<double>(products); ++p)
  //     {
  //       m_result += *p;
  //     }
  //   }

  //   operator double()
  //   {
  //     return m_result;
  //   }

  // private:
  //   double m_result;
  // };

  auto dot = [](const np::array& x, const np::array& y) { 
    np::array products = pycpp::binary_op<double, double, double>(x, y, [](double a, double b){ return a * b; });
    double result = 0.0;
    for (double* p = pycpp::begin<double>(products); p != pycpp::end<double>(products); ++p)
    {
      result += *p;
    }
    return result;
  };

  np::array v1 = pycpp::make_array<double>(10, [](){ return 2.0; });
  np::array v2 = pycpp::make_array<double>(10, [](){ return 0.5; });
  CHECK(dot(v1, v2) == 10.0);

  // // now see if it works with vector * scalar
  // py::object scalar(1.0);
  // CHECK(DotFunc(v1, scalar) == 20.0);

  // np::array n1 = no::nparray::isnever(v1);
  // CHECK(!n1[0]);

}