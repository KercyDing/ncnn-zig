// Windows GNU: ncnn's platform.h includes <process.h>, while its simpleomp
// runtime calls pthread_once. Provide that single primitive with the native
// Windows API instead of pulling in winpthreads.

#include <windows.h>

typedef INIT_ONCE pthread_once_t;
#define PTHREAD_ONCE_INIT INIT_ONCE_STATIC_INIT

static BOOL CALLBACK ncnn_zig_once_thunk(PINIT_ONCE once, PVOID param, PVOID* context) {
    (void)once;
    (void)context;
    ((void (*)(void))param)();
    return TRUE;
}

static inline int pthread_once(pthread_once_t* once, void (*fn)(void)) {
    return InitOnceExecuteOnce(once, ncnn_zig_once_thunk, (PVOID)fn, NULL) ? 0 : -1;
}

// Build the upstream translation unit with the shim above in scope.
#include "simpleomp.cpp"
