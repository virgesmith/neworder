
// test neworder embedded module

#include "test.h"

#include "MonteCarlo.h"

#include <vector>
#include <string>

#include "NewOrder.h"
#include "NPArray.h"

void test_mc()
{
  no::MonteCarlo mc(0, 1, 19937); 
  CHECK(mc.seed() == 19937);
  CHECK(mc.indep());
  py::array a = mc.ustream(5);
  no::log(a);
  CHECK(fabs(no::at<double>(a,{0}) - 0.33778882725164294) < 1e-8);
  CHECK(fabs(no::at<double>(a,{1}) - 0.04767065867781639) < 1e-8);
  CHECK(fabs(no::at<double>(a,{2}) - 0.8131122114136815) < 1e-8);
  CHECK(fabs(no::at<double>(a,{3}) - 0.24954832065850496) < 1e-8);
  CHECK(fabs(no::at<double>(a,{4}) - 0.3385562978219241) < 1e-8);

  mc.reset();
  py::array h = mc.hazard(0.5, 1000000);
  CHECK(no::sum<int>(h) == 500151)
}