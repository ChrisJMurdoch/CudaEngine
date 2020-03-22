
#pragma once

/** Helper function to get arguments */
void getArg(int argc, char *argv[], int index, int *dest);

/** Helper function for testing array math */
void populate (int *array_a, int *array_b, int size);

/** Helper function for validating array math */
void validate(int *cpu, int *gpu, int n);
