'use strict'

module.exports = {
  "Entity": Entity,
  "Person": Person
}

// entity definitions

function Entity(id) {
  this.id = id;
  this.alive = true;
}

// Entity.prototype.getid = function() {
//   return this.id;
// }

// Person constructor
function Person(id, location, age, gender, ethnicity) {
  this.id = id; // chain entity???
  this.location = location;
  this.age = age;
  this.gender = gender;
  this.ethnicity = ethnicity;
}

Person.prototype = Object.create(new Entity())
//Person.prototype.inc_age = function(dt) { this.age = this.age + dt; }
