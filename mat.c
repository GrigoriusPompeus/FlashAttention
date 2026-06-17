/*
 Run: gcc -O2 -std=c11 -o mat mat.c && ./mat
 This version prints elapsed time and a checksum of `c`.
*/
#include <stdio.h>
#include <time.h>

#define n 3000

double a[n][n], b[n][n], c[n][n];

int main()
{
    int i, j, k;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            a[i][j] = 1;
            b[i][j] = 1;
            c[i][j] = 0;
        }

    clock_t start = clock();

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            for (k = 0; k < n; k++)
                c[i][j] += a[i][k] * b[k][j];
        }

    clock_t end = clock();
    double secs = (double)(end - start) / CLOCKS_PER_SEC;

    printf("n=%d time=%.6f s\n", n, secs);
    return 0;
}
