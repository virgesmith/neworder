
// test neworder embedded module

#include "test.h"

#include "MonteCarlo.h"
#include "Environment.h"

#include <vector>
#include <string>

#include "NewOrder.h"
#include "NPArray.h"

void test_mc()
{
  // skip this if in parallel mode
  if (no::getenv().size() != 1)
    return;
    
  no::MonteCarlo mc; 
  CHECK(mc.seed() == 19937);
  CHECK(no::getenv().indep());
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

  // mc.reset();

  // auto a = mc.first_arrival({0.1, 0.2, 0.3}, 1.0, 6, 0.0);
  // CHECK(a.size() == 6);
  // CHECK(no::at<double>(a, {0}) == 3.6177811673165667);
  // CHECK(no::at<double>(a, {1}) == 0.6896205251312125);
  // CHECK(no::at<double>(a, {2}) == 3.610216282947799);
  // CHECK(no::at<double>(a, {3}) == 7.883336832344425);
  // CHECK(no::at<double>(a, {4}) == 6.461894711350323);
  // CHECK(no::at<double>(a, {5}) == 2.8566436418145944);
 

}