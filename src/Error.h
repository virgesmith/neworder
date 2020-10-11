#include <stdexcept>

#include <string>

namespace no {

// error that gets translated to a python NotImplementedError
class NotImplementedError : public std::exception
{
public:
  explicit NotImplementedError(const char* msg): m_msg(msg) { }
  explicit NotImplementedError(const std::string& msg): m_msg(msg) { }
  virtual ~NotImplementedError() noexcept { }

  virtual const char* what() const noexcept;
  
private:
  std::string m_msg;
};

// map error types defined here to python exceptions
void exception_translator(std::exception_ptr p);

}