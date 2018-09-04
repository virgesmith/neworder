
#pragma once

#include "python.h"


// enum struct LogLevel : char { DEBUG = 'D', INFO = 'I', WARN = 'W', ERROR = 'E', ASSERT = 'A', FATAL = 'F' };

// struct LogInfo
// {
// 	LogInfo(LogLevel l, const std::string& msg) : threadId(std::this_thread::get_id()), level(l), message(msg)
// 	{
// 	}

//   //LogInfo(const LogInfo&) = delete;
//   LogInfo& operator=(const LogInfo&) = delete;

// 	std::thread::id threadId;
// 	LogLevel level;
// 	// timestamped when received, not on construction
// 	mutable std::chrono::microseconds timestamp;
// 	std::string message;
// };


// // Termination version, need no arg case (as opposed to 1 arg) to deal with no-substitution cases
// inline void substitute(std::string& str)
// {
// }

// template<typename T, typename... Args>
// void substitute(std::string& str, T&& value, Args&&... args)
// {
//   size_t s = str.find("%%");
//   if (s != std::string::npos) 
//   {
//     str.replace(s, 2, std::to_string(value)); 
//     substitute(str, std::forward<Args>(args)...); 
//   }
// }

// template<typename... Args>
// void substitute(std::string& str, const char* value, Args&&... args)
// {
//   size_t s = str.find("%%");
//   if (s != std::string::npos) 
//   {
//     str.replace(s, 2, value); 
//     substitute(str, std::forward<Args>(args)...); 
//   }
// }

// template<typename... Args>
// void substitute(std::string& str, const std::string& value, Args&&... args)
// {
//   size_t s = str.find("%%");
//   if (s != std::string::npos)
//   {
//     str.replace(s, 2, value);
//     substitute(str, std::forward<Args>(args)...);
//   }
// }

// // non-const string version
// template<typename... Args>
// void substitute(std::string& str, std::string& value, Args&&... args)
// {
//   size_t s = str.find("%%");
//   if (s != std::string::npos) 
//   {
//     str.replace(s, 2, value); 
//     substitute(str, std::forward<Args>(args)...); 
//   }
// }

// /*template<size_t N, typename... Args>
// void substitute(std::string& str, char(& value)[N], Args&&... args)
// {
//   size_t s = str.find("%%");
//   if (s != std::string::npos)
//   {
//     str.replace(s, 2, &value[0], N);
//     substitute(str, std::forward<Args>(args)...);
//   }
// }*/

// template<typename... Args>
// std::string format(const char* const msg, Args&&... args)
// {
//   std::string s(msg);
//   substitute(s, std::forward<Args>(args)...);
//   return s;
// }

// C++14 implements the ""s -> std::string, use this for C++11
std::string operator ""_s(const char* p, size_t s);

// need an rvalue ref as might/will be a temporary
template<typename T> 
std::string operator%(std::string&& str, T value)
{
  size_t s = str.find("%%");
  if (s != std::string::npos)
  {
    str.replace(s, 2, std::to_string(value)); 
  }
  return std::move(str);
}

// specialise for const char*
template<> 
std::string operator%(std::string&& str, const char* value);

// non-template for string (specialisation isn't matched for some reason)
std::string operator%(std::string&& str, const std::string& value);

namespace neworder {

// msg is forcibly coerced to a string
void log(const py::object& msg);
void log(const std::string& msg);

}
