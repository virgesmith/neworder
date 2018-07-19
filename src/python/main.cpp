#include "Environment.h"

#include <vector>
#include <string>


int test1(int, const char*[]);
int test2(const std::string& modulename, const std::string& objectname, const std::vector<std::string>& methodnames);
int test3(const std::string& modulename, const std::string& objectname, const std::string& membername, const std::string& methodname);

int main() 
{
  pycpp::Environment env;

  // argv[0] would be name of binary
  const char* args[] = { "test1", "op", "mul", "2", "3" };
  test1(sizeof(args)/sizeof(args[0]), args);

  const char* args1[] = { "test1", "pop", "func"};
  test1(sizeof(args1)/sizeof(args1[0]), args1);

  test2("pop", "population", {"size", "die", "size", "birth", "birth", "size"});

  test3("pop", "population", "array", "print");
}