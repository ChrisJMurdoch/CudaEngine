
#pragma once

void gpu_add(int *a, int *b, int *dest, int n, int warps=4);
void cpu_add(int *a, int *b, int *dest, int n);
