# Hello World

This simplest example illustrates the structure required, how it fits together and how it's executed by the framework. The annotated python code can be found here: [examples/hello_world/model.py](examples/hello_world/model.py). 

To run the model:

```bash
python examples/hello_world/model.py
```

## Input

The `neworder` framework expects an instance of a `Model` class, which in turn requires a `Timeline` object.

This model doesn't require an explicit discrete timeline, so for models of this type a method is provided to construct an empty timeline (which is a single step of length zero). 

In more complex examples, the timeline would normally refer to absolute time, or for "case-based" models (to use MODGEN parlance), the age of a cohort. In some simple model configurations, each individual's history can be constructed in a single pass, and in this type of situation a null timeline is appropriate.

`neworder` provides a base Model class from which the user should subclass, implementing the following class methods:

- a constructor that initialises the base class with a timeline
- optionally, a `modify` method which is used to change inputs or behaviour across parallel processes,
- a `step` method that governs the evolution of the model
- optionally, a `check` method which is executed after each step and must return a boolean. A result of `False` will halt the model.
- a `checkpoint` method which is run at certain timesteps, and always on the final step.

This (somewhat contrived) example initialised a model with a username, which is initially unknown. The single step of the model queries the OS for the user's name, and the checkpoint greets the user.

Firstly we create our model class, subclassing `neworder.Model`:

```python
import neworder

class HelloWorld(neworder.Model):

  def __init__(self):
    super().__init__(neworder.Timeline.null())
    self.name = None
...
```

the `modify` method is not relevant in this single-process example so it not implemented. Here's the `step` method:

```python
...
  def step(self):
    self.name = os.getlogin()
...
```

and the `check` method simply confirms that the username was changed by the step method:

```python
...
  def check(self):
    return self.name is not None
...
```

Note that the this method must return a boolean, if not `True` the neworder runtime will assume an error and stop execution.

Finally, the `checkpoint` methods prints the greeting:

```python
...
  def checkpoint(self):
    neworder.log("Hello %s" % self.name)
...
```

using the `neworder.log` function is preferred to plain `print` statements as they add useful context for debugging purposes. The API reference can be found [here](./reference.md)

### Execution

The model is run by first constructing an instance of our model

```python
hello_world = HelloWorld()
```

then invoking it like so

```python
ok = neworder.run(hello_world)
```

which returns a boolean, `True` for success.

### Output

The model will output something like

```text
[py 0/1] Hello neworder_user
```

or, if you change the `verbose` initialisation argument to `True`, 

```text
[no 0/1] neworder 1.0.0/module python 3.8.2 (default, Jul 16 2020, 14:00:26)  [GCC 9.3.0]
[no 0/1] model init: mc={seed:19937}
[no 0/1] starting model run. start time=0.000000, timestep=0.000000, checkpoint(s) at [1]
[no 0/1] t=0.000000(0) HelloWorld.modify(0)
[no 0/1] defaulted to no-op Model::modify()
[no 0/1] t=0.000000(1) HelloWorld.step()
[no 0/1] t=0.000000(1) HelloWorld.check() [ok]
[no 0/1] t=0.000000(1) HelloWorld.checkpoint()
[py 0/1] Hello neworder_user
[no 0/1] SUCCESS exec time=0.000442s
```

this output is explained line-by-line below.

The log output is prefixed with a source identifier in square brackets, containing the following information for debugging purposes:

- Source of message: `no` if logged from the framework itself, `py` if logged from python code (via the `neworder.log()` function). The former are only displayed in verbose mode.

- the process id ('rank' in MPI parlance) and the total number of processes ('size' in MPI parlance) - in serial mode these default to 0/1.

### Understanding the workflow and the output

When using `Timeline.null()` the start time, end time and timestep are all zero, and there is a single step, and a single checkpoint at step 1.  

First we get some information about the environment, and confirmation of the initialisation parameters:

```text
[no 0/1] neworder 1.0.0/module python 3.8.2 (default, Jul 16 2020, 14:00:26)  [GCC 9.3.0]
[no 0/1] model init: mc={seed:19937}
[no 0/1] starting model run. start time=0.000000, timestep=0.000000, checkpoint(s) at [1]
```

then the model starts to run, firstly applying any per-process modifications

```text
[no 0/1] t=0.000000(0) HelloWorld.modify(0)
[no 0/1] defaulted to no-op Model::modify()
```

and in this case we have none, as `modify` hasn't been implemented in the subclass. Now the `step` method is called:

```text
[no 0/1] t=0.000000(1) HelloWorld.step()
```

followed by the `check` method:

```text
[no 0/1] t=0.000000(1) HelloWorld.check() [ok]
```

which succeeded (try making it fail and then running the model, and also removing the method entirely). Then the single checkpoint is reached:

```text
[no 0/1] t=0.000000(1) HelloWorld.checkpoint()
```

which prints the result:

```text
[py 0/1] Hello neworder_user
```

and finally the model reports its status and execution time:

```text
[no 0/1] SUCCESS exec time=0.000273s
```
