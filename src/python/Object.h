
#pragma once

// TODO merge this back into humanleague at some point

#include <string>
#include <vector>
#include <map>
#include <cstddef>

typedef struct _object PyObject;

// To understand the memory management, read:
// https://docs.python.org/3/extending/extending.html#ownership-rules
// Basically:
// All pycpp::Objects constructed with a Py_... "constructor" function have their refcount already set
// Copy, move and assigment requires modification of the refcount
// Destruction always decrements the refcount
// All functions that accept or return a PyObject* directly must explicitly increment the ref count
// Returning values to python should be done via release() (which does a Py_INCREF to avoid the object being deleted on exit)
// Note difference between list and dict mem management above

namespace pycpp {

  // not directly instantiated, use derived types
  class Object
  {
  public:
  
    PyObject* operator&() const ;
    bool operator!() const;
    
    // Inc ref count and return pointer - use to safely return object to python (only)
    PyObject* release();

    // Return the ref count (useful for memory debugging)
    int refcount() const;
    
  protected:
  
    Object(const Object& obj);
    
    Object(Object&& obj);
    
    Object& operator=(const Object& obj);
  
    explicit Object(PyObject* obj);

    virtual ~Object();

    PyObject* m_obj; 
  };

  // map C++ scalar type(s) to Object derived type
  template<typename T> struct PyType;
  
  class Bool : public Object 
  {
  public:
    explicit Bool(bool x);
    
    explicit Bool(PyObject* p);
    
    operator bool() const;
  };

  template<> struct PyType<bool> { typedef Bool Type; };

  class Int : public Object 
  {
  public:
    explicit Int(int x);
    explicit Int(uint32_t x);
    explicit Int(int64_t x);
    explicit Int(size_t x);
    
    explicit Int(PyObject* p);

//    Int(const Int& o);
    
    operator int() const;
    // NB python doesnt have native unsigned type
    operator uint32_t() const;
    operator int64_t() const;
    operator size_t() const;
  };
  
  // define C++ types that map to Int
  template<> struct PyType<int> { typedef Int Type; };
  template<> struct PyType<uint32_t> { typedef Int Type; };
  template<> struct PyType<int64_t> { typedef Int Type; };
  template<> struct PyType<size_t> { typedef Int Type; };

  class Double : public Object 
  {
  public:
    explicit Double(double x);

    explicit Double(PyObject* p);
    
    operator double() const;
  };
  
  template<> struct PyType<double> { typedef Double Type; };

  class String : public Object
  {
  public: 
    explicit String(const char* s);
    explicit String(const std::string& s);
    explicit String(PyObject* p);

    operator const char*() const; 
    operator std::string() const; 
  };
  
  template<> struct PyType<const char*> { typedef String Type; };
  template<> struct PyType<std::string> { typedef String Type; };

  class List : public Object
  {
  public:
    // uninitialised list with given length
    explicit List(size_t length = 0);
    
    explicit List(PyObject* list);
    
    // Construct from vector of scalar types that correspond to a pycpp type
    template<typename T>
    explicit List(const std::vector<T>& v) : List(v.size())
    {
      for (size_t i = 0; i < v.size(); ++i)
      {
        set(i, typename PyType<T>::Type(v[i]));
      }
    }
    
    // return value given we don't know the actual type??
    PyObject* operator[](size_t i) const;
    
    // set (move semantics)
    void set(int index, Object&& obj);
    
    // append (move semantics)
    void push_back(Object&& obj);

    // return value given we don't know the actual type??
    //PyObject* get(int index) const;
    
    template<typename T/*, typename P = typename Type<T>::PyType*/>
    std::vector<T> toVector() const
    {
      const size_t n = size();
      std::vector<T> v(n);
      for (size_t i = 0; i < n; ++i)
      {
        v[i] = typename PyType<T>::Type(this->operator[](i));
      }
      return v;
    }
    
    // appears to be no clear mechanism for lists
    //void clear() const;

    int size() const;
  };

  class Tuple : public Object
  {
  public:
    // uninitialised list with given length
    explicit Tuple(int length);
    
    explicit Tuple(PyObject* tuple);

    int size() const;

    // return value given we don't know the actual type??
    PyObject* operator[](size_t i) const;
    
    // set (move semantics)
    void set(int index, Object&& obj);
    
    // append (move semantics)
    //void push_back(Object&& obj);
  };
  
  class Dict : public Object
  {
  public:
    // empty dict
    Dict();
    
    explicit Dict(PyObject* dict);
    
    // Construct from map of string to scalar types that correspond to a pycpp type
    template<typename T>
    explicit Dict(const std::map<std::string, T>& m) : Dict()
    {
      for (auto it = m.cbegin(); it != m.cend(); ++it)
      {
        insert(it->first.c_str(), typename PyType<T>::Type(it->second));
      }
    }

    // return value given we don't know the actual type??
    PyObject* operator[](const char*) const;
    
    // set (shallow copy semantics)
    //void insert(const char*, const Object& obj);
    
    // set (move semantics)
    void insert(const char*, Object&& obj);
    
    // TODO find...
    
    void clear() const;

    int size() const;
  };
}
