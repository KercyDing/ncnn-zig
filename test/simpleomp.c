// Correct loop results also hold for a serial fallback, so verify that the
// simpleomp runtime actually dispatches work to more than one OS thread.

#include <stdint.h>
#include <stdio.h>

int omp_get_max_threads(void);

#if defined(_WIN32)
#include <windows.h>
#define THREAD_ID ((uintptr_t)GetCurrentThreadId())
#else
#include <pthread.h>
#define THREAD_ID ((uintptr_t)pthread_self())
#endif

int main(void) {
    enum { N = 64 };

    if (omp_get_max_threads() < 2) {
        printf("simpleomp: single core, skipped\n");
        return 0;
    }

    uintptr_t ids[N] = {0};
    int values[N] = {0};

#pragma omp parallel for num_threads(4)
    for (int i = 0; i < N; i++) {
        ids[i] = THREAD_ID;
        values[i] = i + 1;
    }

    int distinct = 0;
    int sum = 0;
    for (int i = 0; i < N; i++) {
        sum += values[i];
        int seen = 0;
        for (int j = 0; j < i; j++) {
            if (ids[j] == ids[i]) seen = 1;
        }
        if (!seen) distinct++;
    }

    printf("simpleomp: %d threads, sum=%d\n", distinct, sum);
    return distinct >= 2 && sum == N * (N + 1) / 2 ? 0 : 1;
}
