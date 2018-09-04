
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

#define CHECK(cond) \
  ++Global::instance<Test>().t; \
  if (!(cond)) \
  { \
    ++Global::instance<Test>().f; \
    LOG_ERROR("FAIL %%:%% %%"_s % __FILE__ % __LINE__ % #cond); \
  } \
  else \
  { \
    LOG_INFO("PASS %%:%% %%"_s % __FILE__ % __LINE__ % #cond); \
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
      LOG_INFO("PASS %%:%% %% throws %%"_s % __FILE__ % __LINE__ % #expr % #except); \
    } \
    else \
    { \
      LOG_ERROR("FAIL %%:%% %% throws %%"_s % __FILE__ % __LINE__ % #expr % #except); \
    } \
	}

#define REPORT() \
  LOG_INFO("Tests run: %%"_s % Global::instance<Test>().t); \
  if (Global::instance<Test>().f) \
  { \
    LOG_WARNING("%% FAILURES"_s % Global::instance<Test>().f); \
  } \
  else \
  { \
    LOG_INFO("SUCCESS"); \
  }

// Use this to notify env that tests failed, e.g. to stop make continuing to install target
#define RETURN() \
  return (Global::instance<Test>().f != 0);

