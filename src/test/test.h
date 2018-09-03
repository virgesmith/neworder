
#pragma once

#include "Log.h"
#include "Global.h"
#include "Module.h"

struct Test
{
  Test() : t(0), f(0) { }
  size_t t; 
  size_t f;
};

#define LOG_INFO(x) neworder::log(x)
#define LOG_WARNING(x) neworder::log(x)
#define LOG_ERROR(x) neworder::log(x)
 
// TODO see if tmp string can be avoided...

#define CHECK(cond) \
  ++Global::instance<Test>().t; \
  if (!(cond)) \
  { \
    ++Global::instance<Test>().f; \
    std::string tmp = format("FAIL %%:%% %%", __FILE__, __LINE__); \
    LOG_ERROR(format(tmp.c_str(), #cond)); \
  } \
  else \
  { \
    std::string tmp = format("PASS %%:%% %%", __FILE__, __LINE__); \
    LOG_INFO(format(tmp.c_str(), #cond)); \
  }

#define CHECK_THROWS(expr, except) \
	{ \
		++Global::instance<Test>().t; \
		bool caught = false; \
		try \
		{ \
			expr; \
		} \
		catch(except& e) \
		{ \
			caught = true; \
		} \
		catch(...) \
		{ \
		} \
	  Global::instance<Test>().f += caught ? 0 : 1; \
	  if (caught) \
    { \
      std::string tmp = format("PASS %%:%% %% throws %%", __FILE__, __LINE__); \
	    LOG_INFO(format(tmp.c_str(), #expr, #except)); \
    } \
    else \
    { \
      std::string tmp = format("FAIL %%:%% %% throws %%", __FILE__, __LINE__); \
      LOG_INFO(format(tmp.c_str(), #expr, #except)); \
    } \
	}

#define REPORT() \
  LOG_INFO(format("Tests run: %%", Global::instance<Test>().t)); \
  if (Global::instance<Test>().f) \
  { \
    LOG_WARNING(format("%% FAILURES", Global::instance<Test>().f)); \
  } \
  else \
  { \
    LOG_INFO("SUCCESS"); \
  }

// Use this to notify env that tests failed, e.g. to stop make continuing to install target
#define RETURN() \
  return (Global::instance<Test>().f != 0);

