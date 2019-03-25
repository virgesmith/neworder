#pragma once

// Version.h

#define STR2(x) #x
#define STR(x) STR2(x)

#define NEWORDER_VERSION_MAJOR 0
#define NEWORDER_VERSION_MINOR 0
#define NEWORDER_VERSION_PATCH 0

#define NEWORDER_VERSION_STRING STR(NEWORDER_VERSION_MAJOR) "." \
                                STR(NEWORDER_VERSION_MINOR) "." \
                                STR(NEWORDER_VERSION_PATCH)  
