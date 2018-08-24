
// TODO work out how to create a shared C++-only lib (with e.g. humanleague)
// even if it has to be header-only (which probably isnt a bad thing)

#pragma once

#include <memory>
#include <mutex>

// Threadsafe global singleton access
namespace Global
{
  template<typename T>
  T& instance()
  {
    static thread_local std::unique_ptr<T> instance;
    static thread_local std::once_flag init;
    std::call_once(init, [](){ instance.reset(new T); });
    return *instance;
  }
}
