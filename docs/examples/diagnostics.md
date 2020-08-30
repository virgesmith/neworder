
# Diagnostics

This isn't really an example, it just outputs useful diagnostic information to track down bugs/problems, and opens a debug shell so that the neworder environment can be inspected. Below we use neworder interactively to sample 5 stopping times based on a 10% hazard rate:

<pre>
[no 0/1] env: seed=19937 python 3.6.7 (default, Oct 22 2018, 11:32:17)  [GCC 8.2.0]
[py 0/1] MODULE=neworder0.0.0
[py 0/1] PYTHON=3.6.7 (default, Oct 22 2018, 11:32:17)  [GCC 8.2.0]
[py 0/1] Loaded libs:
[py 0/1]   linux-vdso.so.1 (0x00007ffdb5f63000)
[py 0/1]   libpython3.6m.so.1.0 => /usr/lib/x86_64-linux-gnu/libpython3.6m.so.1.0 (0x00007fb595232000)
[py 0/1]   libneworder.so => src/lib/libneworder.so (0x00007fb594fee000)
[py 0/1]   libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007fb594c65000)
[py 0/1]   libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007fb594a4d000)
[py 0/1]   libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fb59465c000)
[py 0/1]   libexpat.so.1 => /lib/x86_64-linux-gnu/libexpat.so.1 (0x00007fb59442a000)
[py 0/1]   libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007fb59420d000)
[py 0/1]   libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007fb593fee000)
[py 0/1]   libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007fb593dea000)
[py 0/1]   libutil.so.1 => /lib/x86_64-linux-gnu/libutil.so.1 (0x00007fb593be7000)
[py 0/1]   libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007fb593849000)
[py 0/1]   /lib64/ld-linux-x86-64.so.2 (0x00007fb595afc000)
[py 0/1] PYTHONPATH=examples/diagnostics:examples/shared
[no 0/1] starting microsimulation. timestep=0.000000, checkpoint(s) at [1]
[no 0/1] t=0.000000(1) checkpoint: shell
[starting neworder debug shell]
>>> import neworder
>>> neworder.stopping(0.1, 5)
array([30.43439191, 13.88102712,  1.69985666, 13.28639123,  1.75969325])
>>> <b><font color="red">ctrl-D</font></b>
[exiting neworder debug shell]
[no 0/1] SUCCESS exec time=22.416254s
</pre>

See [examples/diagnostics/config.py](examples/diagnostics/config.py)
