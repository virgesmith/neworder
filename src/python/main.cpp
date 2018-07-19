
#include <vector>
#include <string>


int test1(int, const char*[]);
int test2(const std::string& modulename, const std::string& objectname, const std::vector<std::string>& methodname);

int main() 
{
  // argv[0] would be name of binary
  const char* args[] = { "test1", "op", "mul", "2", "3" };

  test1(sizeof(args)/sizeof(args[0]), args);

  test2("pop", "population", {"size", "die", "size"});

}