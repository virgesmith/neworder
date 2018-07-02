{
  'targets': [
    {
      'target_name': 'neworder',
      'cflags_cc': [ '-g -O2 -Wall -Werror -std=c++11' ],
      'cflags_cc!': [ '-fno-rtti', '-fno-exceptions' ],
      'sources': [ 'src/nodejs_api.cpp',
                   'src/Simulation.cpp',
                   'src/Module.cpp'],
      'include_dirs': ['../..'],
    },
  ],
}

