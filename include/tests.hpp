
#pragma once

// Functions for testing respective gpu operations
void testVectorAdd(int *a, int * b, int *c, int TESTS, int WARPS, int N, int *ms);
void testVectorInAdd(int *a, int * b, int TESTS, int WARPS, int N, int *ms);
void testVectorSub(int *a, int * b, int *c, int TESTS, int WARPS, int N, int *ms);
void testVectorInSub(int *a, int * b, int TESTS, int WARPS, int N, int *ms);

/** Make sure functions return correct output */
void testOutputs(int WARPS);

/** Helper function for testing array math */
void populate (int *array_a, int *array_b, int size);

/** Helper function for validating array math */
void validate(int *cpu, int *gpu, int n);
