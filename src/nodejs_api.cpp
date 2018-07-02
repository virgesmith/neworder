
#include "Simulation.h"

#include <node.h>
#include <v8.h>

#include <iostream>


void eval(const v8::FunctionCallbackInfo<v8::Value>& args)
{
  v8::Isolate* isolate = Global::instance<Simulation>().isolate(); // v8::Isolate::GetCurrent();

  try
  {
    //v8::Local<v8::Context>& context = Global::instance<Simulation>().context(); 
    v8::Local<v8::Context> context = v8::Context::New(isolate);

    v8::String::Utf8Value source(isolate, args[0]->ToString());
    //std::cout << "in: " << *source << std::endl;

    // Compile the source code.
    v8::Local<v8::Script> script = v8::Script::Compile(context, args[0]->ToString()).ToLocalChecked();
    // Run the script to get the result.
    v8::Local<v8::Value> result = script->Run(context).ToLocalChecked();
    (void)result;

    args.GetReturnValue().Set(result);
    //v8::String::Utf8Value response(isolate, result);

    //std::cout << "out: " << *result << std::endl;
    //response_str = *response; 
  }
  catch(const std::exception& e)
  {
    args.GetReturnValue().Set(v8::String::NewFromUtf8(isolate, e.what()));
  }
  catch(...)
  {
    args.GetReturnValue().Set(v8::String::NewFromUtf8(isolate, "unhandled exception"));
  }
  //args.GetReturnValue().Set(v8::String::NewFromUtf8(isolate, response_str.c_str()));
}

