# ![logo](img/favicon-32.png) neworder


!!! warning "Zensical migration"
    [Zensical](https://zensical.org) is not yet as feature complete as mkdocs. See specifically
    [feature parity](https://zensical.org/compatibility/features/) and
    [plugins](https://zensical.org/compatibility/plugins/).

    - [X] macros
    - [ ] video


![Population pyramid](examples/img/pyramid.gif)

{{ include_snippet("./README.md", "readme", show_filename=False)}}

### Examples

Download the examples zipfile/archive can from the [releases](https://github.com/virgesmith/neworder/releases) page,
or pull the [docker image](https://hub.docker.com/r/virgesmith/neworder).

The docker image should be run interactively. Some of the examples require permission to connect to the host's
graphical display, e.g.

```bash
docker pull virgesmith/neworder
xhost +local:
docker run --net=host -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -it virgesmith/neworder
```

NB The above works on ubuntu but may require modification on other OSs.

Then in the container, e.g.

```bash
python examples/mortality/model.py
```

### Developer

See [Contributing](./developer.md) for installation steps.
