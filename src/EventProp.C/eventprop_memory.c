/**
 * @file eventprop_memory.c
 * @brief Public output-freeing functions.
 */
#include "eventprop.h"
#include <stdlib.h>

void eventprop_free_spike_output(const SpikeOutput output)
{
    free(output.times);
    free(output.sources);
    free(output.currents);
    free(output.errors);
}

void eventprop_free_sums_output(const SumsOutput output)
{
    free(output.sum0);
    free(output.sum1);
}

void eventprop_free_maxima_output(const MaximaOutput output)
{
    free(output.times);
    free(output.values);
    free(output.errors);
}

void eventprop_free_backward_output(const BackwardOutput output){
    free(output.grad);
    free(output.spike_errors);
}
