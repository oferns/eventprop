#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

int main(void) {
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    
    // Test with n_post = 5 (Tiny)
    int n_post = 5;
    double* post_times = calloc(21 * n_post, sizeof(double));
    
    printf("Running with n_post=%d (should be sequential)...\n", n_post);
    #ifdef _OPENMP
    if (n_post >= 40)
        printf("Would use OpenMP\n");
    else
        printf("Skipping OpenMP (n_post < 40)\n");
    #endif
    
    for (int target_nrn = 0; target_nrn < n_post; target_nrn++) {
        int tid = omp_get_thread_num();
        printf("Thread %d processing neuron %d\n", tid, target_nrn);
        int offset = target_nrn * 21;
        for (int i = 0; i < 20; i++) {
            post_times[offset + i] = (double)i / 20.0;
        }
    }
    
    free(post_times);
    return 0;
}
