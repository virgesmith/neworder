'use strict'


var fs = require('fs');
var csv = require('csv-parser');

var defs = require("./definitions.js")

// var x = new defs.Person(0, "E09000001", 49, 1, 5);
// console.log(x);
// //x.inc_age(1);
// x.age = x.age + 1;
// console.log(x);


var starting_pop_data = "./tests/ssm_E09000001_MSOA11_ppp_2011.csv"

var population = []
fs.createReadStream(starting_pop_data)
  .pipe(csv())
  .on('data', function(data) {
    //PID,Area,DC1117EW_C_SEX,DC1117EW_C_AGE,DC2101EW_C_ETHPUK11
    population.push(new defs.Person(data.PID, data.Area, data.DC1117EW_C_AGE, data.DC1117EW_C_SEX, data.DC2101EW_C_ETHPUK11))
    console.log(population.length)
   });
   
// for (var p in population) {
//   console.log(p.age);
// }




