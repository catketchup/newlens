/* File foo.c */
void m(int n, double *x, double *y) {
  int i;
  for (i=0;i<n;i++) {
    y[i] = x[i] + i;
  }
}
