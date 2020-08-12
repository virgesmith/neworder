
#pragma once

#include "Global.h"
#include "Log.h"
#include <condition_variable>
#include <mutex>
#include <queue>
#include <atomic>
#include <string>
#include <thread>

class ConcurrentLogger
{
public:

	~ConcurrentLogger();
	
  void push(LogInfo&& item);
	
	void setFilter(const std::string& filter);

private:

  void push_impl(LogInfo&& item);

	ConcurrentLogger();
	
	ConcurrentLogger(const ConcurrentLogger&) = delete;
	ConcurrentLogger& operator=(const ConcurrentLogger&) = delete;
		  
  void pop();
  
	bool m_finished;
  std::queue<LogInfo> m_queue;
	std::mutex m_mutex;
  std::condition_variable m_cond;
	std::chrono::system_clock::time_point m_startTime;
	std::string m_filter;
	std::thread m_dispatcher;
		
	friend ConcurrentLogger& Global::instance<ConcurrentLogger>();
};

#define LOG_FILTER(x) Global::instance<ConcurrentLogger>().setFilter(x)

#define LOG_DEBUG(x) Global::instance<ConcurrentLogger>().push(LogInfo(LogLevel::DEBUG,x))
#define LOG_INFO(x) Global::instance<ConcurrentLogger>().push(LogInfo(LogLevel::INFO,x))
#define LOG_WARNING(x) Global::instance<ConcurrentLogger>().push(LogInfo(LogLevel::WARN,x))
#define LOG_ERROR(x) Global::instance<ConcurrentLogger>().push(LogInfo(LogLevel::ERROR,x))
