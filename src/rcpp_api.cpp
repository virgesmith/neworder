
#include "mc.h"

#include <Rcpp.h>
using namespace Rcpp;


//' TODO documentation
//'
//' @return a List .
//' @examples
//' #TODO
//' @export
// [[Rcpp::export]]
NumericVector urand01_vector(size_t length)
{
  // TODO probably too many copies going on here
  const std::vector<double> r = urand01(length);

  return NumericVector(r.begin(), r.end());
}

//' TODO documentation
//'
//' @return a List .
//' @examples
//' #TODO
//' @export
// [[Rcpp::export]]
IntegerVector select_at_random(DataFrame input, std::string& drivingColumn, NumericVector probs)
{
  size_t n = input.rows();
  
  // TODO really need a map from col values to probs (if non-contiguous, start at !=0)
  // TODO check drivingColumn and probs match - unique()?
  // TODO multiple drivingColumns/n-d probs
  
  const std::vector<double>& rand = urand01(n);
  const std::vector<int>& driver = input[drivingColumn];
  std::vector<int> r(n);
  
  for (size_t i = 0; i < n; ++i)
  {
    r[i] = (rand[i] < probs[driver[i]]);
  }
  return IntegerVector(r.begin(), r.end());
}

//' TODO documentation
//'
//' @return a List .
//' @examples
//' #TODO
//' @export
// [[Rcpp::export]]
bool age_column(DataFrame input, const std::string& colName)
{
  // odd limitation that DF is by ref by default, yet modifying cols requires copy and overwrite
  size_t n = input.rows();
  // this copies column, seems no easy way round this
  IntegerVector m = input[colName];
  for (size_t i = 0; i < n; ++i)
  {
    ++m[i];
  }
  // if something goes wrong, return false so we know original DF has not been modified
  // put modified column back into original DF
  input[colName] = m;
  return true;
}
