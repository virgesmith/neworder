## Hello World

This simplest example is illustrates of the structure required, and how it fits together. All the code is extensively commented. The code can be run like so

```bash
python examples/hello_world/model.py
```
which will output something like

```text
[py 0/1] Hello neworder_user
```
or if you change the `verbose` argument, 

```
[no 0/1] neworder 1.0.0/module python 3.8.2 (default, Jul 16 2020, 14:00:26)  [GCC 9.3.0] env={indep:1, verbose:1}
[no 0/1] model init: mc={indep:1, seed:19937}
[no 0/1] starting model run. start time=0.000000, timestep=0.000000, checkpoint(s) at [1]
[no 0/1] t=0.000000(0) calling: <__main__.HelloWorld object at 0x7f642ca367c0>.modify(0)
[no 0/1] defaulted to no-op Model::modify()
[no 0/1] t=0.000000(1) calling <__main__.HelloWorld object at 0x7f642ca367c0>.transition()
[no 0/1] t=0.000000(1) calling <__main__.HelloWorld object at 0x7f642ca367c0>.check()
[no 0/1] t=0.000000(1) calling <__main__.HelloWorld object at 0x7f642ca367c0>.checkpoint()
[py 0/1] Hello neworder_user
[no 0/1] SUCCESS exec time=0.000279s

```
this output is explained below.

The API reference can be found [here](./reference.md)

### Input

The `neworder` framework expects an instance of a `Model` class, which in turn requires a `Timeline` object.

Not all models require an explicit discrete timeline (like this one), so a method is provided to construct a dummy timeline (which is a single step of length zero). 

The model class provided is a base class from which the user should subclass, implemention the following class methods:

- a constructor that initialises the base class with a timeline.
- optionally, a `modify` method which is used to 
- a `step` method that governs the evolution of the model
- optionally, a `check` method which is executed after each step
- a `checkpoint` method which is run at certain timesteps, and always on the final step.

The model is defined and initialised in [model.py](examples/hello_world/model.py). 

### Output

The log output from *neworder* is prefixed with a source identifier in square brackets, containing the following information for debugging purposes:

- Source of message: `no` if logged from the framework itself, `py` if logged from python code (via the `neworder.log()` function). The former are only displayed in verbose mode.

- the process id ('rank' in MPI parlance) and the total number of processes ('size' in MPI parlance) - in serial mode these default to 0/1.


```bash
$ python examples/hello_world/model.py 
[no 0/1] neworder 1.0.0/module python 3.8.2 (default, Jul 16 2020, 14:00:26)  [GCC 9.3.0] env={indep:1, verbose:1}
[no 0/1] model init: mc={indep:1, seed:19937}
[no 0/1] starting model run. start time=0.000000, timestep=0.000000, checkpoint(s) at [1]
[no 0/1] t=0.000000(0) calling: <__main__.HelloWorld object at 0x7ffa1b20a810>.modify(0)
[no 0/1] defaulted to no-op Model::modify()
[no 0/1] t=0.000000(1) calling <__main__.HelloWorld object at 0x7ffa1b20a810>.step()
[no 0/1] t=0.000000(1) calling <__main__.HelloWorld object at 0x7ffa1b20a810>.check()
[no 0/1] t=0.000000(1) calling <__main__.HelloWorld object at 0x7ffa1b20a810>.checkpoint()
[py 0/1] Hello az
[no 0/1] SUCCESS exec time=0.000289s
```

### Understanding the workflow and the output

All models must have a timeline over which they run. For cases where an explicit timeline isn't necessary, such as this one, we use a *null* timeline, which is just a single instantaneous transition.

By default the start time and timestep is zero, and there is a single timestep. This example doesn't require a timeline.

The environment initialises, indicating the random seed and the python version used:
```
[no 0/1] env: seed=19937 python 3.6.6 (default, Sep 12 2018, 18:26:19)  [GCC 8.0.1 20180414 (experimental) [trunk revision 259383]]
[no 0/1] starting microsimulation...
```
As no timeline has been specified, we just have single timestep and a single checkpoint (the end). The model is initialised:
```
[no 0/1] t=0.000000 initialise: greeter
```
...an object is constructed and assigned to the variable `greeter`. In [config.py](examples/hello_world/config.py), from the module `greet`, construct an object of type `Greet`, passing no parameters:
```
neworder.initialisations = {
  "greeter": { "module": "greet", "class_": "Greet", "parameters": [] }
}
```
The time loop now increments, and the transitions are processed:
```
[no 0/1] t=1.000000 transition: who
```
The transition named 'who' simply executes the `get_name()` method of the `greeter` object. (If you look in [greet.py](examples/hello_world/greet.py) you will see that the method uses an operating system call to get the username.)
```
neworder.transitions = {
  "who": "greeter.get_name()"
}
```
Optionally, checks can be implemented to run after each timestep, to check the state of the microsimulation. In [config.py](examples/hello_world/config.py), we have defined:

```json
neworder.do_checks = True
neworder.checks = {
  "eval": "True",
}
```
and thus see the corresponding
```
[no 0/1] t=1.000000 check: eval
```
in the output. The check must evaluate to a boolean, and if `False` the model will stop. In this example the dummy check simply evaluates `True` (which is of course `True`).

We have now reached the end of the timeline and the checkpoint code - call the () method (i.e. `__call__`) of the greeter object
```
neworder.checkpoints = {
  "say_hello" : "greeter()",
}
```
...which says hello:
```
[no 0/1] t=1.000000 checkpoint: say_hello
[py 0/1] Hello neworder_user
```

Finally the framework indicates the model ran successfully:
```
[no 0/1] SUCCESS
```

The 'model' configuration is here: [examples/hello_world/config.py](examples/hello_world/config.py). This file refers to a second file in which the "model" is defined, see [examples/hello_world/greet.py](examples/hello_world/greet.py)

