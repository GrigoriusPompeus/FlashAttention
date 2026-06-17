/*
 Run: gcc -O2 -std=c11 -o mat1_test mat1_test.c && ./mat1_test
*/
#include <stdio.h>
#include <time.h>

#define n 200

double a[n][n], b[n][n], c[n][n];

int main()
{
    int i, j, k;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            a[i][j] = 1.0;
            b[i][j] = 1.0;
            c[i][j] = 0.0;
        }

    clock_t start = clock();

    for (j = 0; j < n; j++)
        for (k = 0; k < n; k++)
            for (i = 0; i < n; i++)
                c[i][j] += a[i][k] * b[k][j];

    clock_t end = clock();
    double secs = (double)(end - start) / CLOCKS_PER_SEC;

    double sum = 0.0;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            sum += c[i][j];

    printf("n=%d time=%.6f s checksum=%.0f\n", n, secs, sum);
    return 0;
}
