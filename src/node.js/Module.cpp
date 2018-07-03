
#include "Module.h"

#include <node.h>
#include <v8.h>

void init(v8::Handle<v8::Object> target)
{
  //NODE_SET_METHOD(target, "init", init);
  NODE_SET_METHOD(target, "eval", eval);
}

NODE_MODULE(neworder, init) 


