'use strict'
// test.js
var fs = require("fs");

var neworder_api = require("../build/Release/neworder.node");

var result = neworder_api.eval("'hello' + 'world'");
console.log(result);

var result = neworder_api.eval("1 + 2");
console.log(result);

var result = neworder_api.eval("3 + 'world'");
console.log(result);

var result = neworder_api.eval("function x(y) { return y+1; } x(41)");
console.log(result);

// var defs = fs.readFileSync("./tests/initialise.js", "utf8") 
// var result = neworder_api.eval(defs);
// console.log("defs eval:" + result);
// FATAL ERROR: v8::ToLocalChecked Empty MaybeLocal.
//var result = neworder_api.eval('require("./definitions.js")');

// //console.log(defs);
// var init = `
// var x = new Person(0, "E09000001", 49, 1, 5);
// console.log(x);
// //x.inc_age(1);
// x.age = x.age + 1;
// console.log(x);
// 5+5;`

// var result = neworder_api.eval(init);
// console.log("init eval:" + result);

// var result = neworder_api.eval("x.age = x.age + 1;");
// var result = neworder_api.eval("x.age = x.age + 1; console.log(x);");


// var init = fs.readFileSync("./tests/initialise.js", "utf8") 
// //console.log(init);
// var result = neworder_api.eval(init);
// console.log("init eval:" + result);

