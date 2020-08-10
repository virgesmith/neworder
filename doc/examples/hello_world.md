## Hello World

This example is a simple illustration of the structure required, and how it fits together. All the code is extensively commented, and can servce as a skeleton for new project. 

### Input 

All models require the following:


The model is defined and initialised in [config.py](examples/hello_world/config.py). The neworder runtime will automatically execute it.

### Output

The log output from *neworder* is prefixed with a source identifier in square brackets, containing the following information for debugging purposes:
- Source of message: `no` if logged from the framework itself, `py` if logged from python code (via the `neworder.log()` function).
- the process id ('rank' in MPI parlance) and the total number of processes ('size' in MPI parlance) - in serial mode these default to 0/1.


```bash
$ ./run_example.sh hello_world
[no 0/1] neworder 0.0.0 env: mc=(indep:1, seed:19937) python 3.8.2 (default, Jul 16 2020, 14:00:26)  [GCC 9.3.0]
[no 0/1] registered transition who: neworder.model.set_name()
[no 0/1] registered check eval: True
[no 0/1] registered checkpoint say_hello: neworder.model()
[no 0/1] starting microsimulation. start time=0.000000, timestep=0.000000, checkpoint(s) at [1]
[no 0/1] t=0.000000(1) transition: who 
[no 0/1] t=0.000000(1) check: eval 
[no 0/1] t=0.000000(1) checkpoint: say_hello
[py 0/1] Hello neworder_user
[no 0/1] SUCCESS exec time=0.000497s
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

