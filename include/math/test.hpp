
#pragma once

// Functions for profiling respective gpu operations
void profileVectorAdd(int *a, int * b, int *c, int TESTS, int N, int *ms);
void profileVectorInAdd(int *a, int * b, int TESTS, int N, int *ms);
void profileVectorSub(int *a, int * b, int *c, int TESTS, int N, int *ms);
void profileVectorInSub(int *a, int * b, int TESTS, int N, int *ms);

/** Debug outputs */
void debugOutputs();

/** Helper function to get integer arguments */
void getArg(int argc, char *argv[], int index, int *dest);

/** Helper function for filling arrays */
void populate(int *array_a, int *array_b, int n);

/** Helper function for validating arrays against each other */
void validate(int *array_a, int *array_b, int n);
