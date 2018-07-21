{
  'targets': [
    {
      'target_name': 'neworder',
      'cflags_cc': [ '-g -O2 -Wall -Werror -std=c++11' ],
      'cflags_cc!': [ '-fno-rtti', '-fno-exceptions' ],
      'sources': [ 'src/node.js/nodejs_api.cpp',
                   'src/node.js/Simulation.cpp',
                   'src/node.js/Module.cpp'],
      'include_dirs': ['src/include'],
    },
  ],
}

