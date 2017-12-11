
/*******************************************************************************
QL algorithm with implicit shifts, to determine the eigenvalues and eigenvectors
of a real, symmetric, tridiagonal matrix, or of a real, symmetric matrix
previously reduced by tred2 sec. 11.2. On input, d[1..n] contains the diagonal
elements of the tridiagonal matrix. On output, it returns the eigenvalues. The
vector e[1..n] inputs the subdiagonal elements of the tridiagonal matrix, with
e[1] arbitrary. On output e is destroyed. When finding only the eigenvalues,
several lines may be omitted, as noted in the comments. If the eigenvectors of
a tridiagonal matrix are desired, the matrix z[1..n][1..n] is input as the
identity matrix. If the eigenvectors of a matrix that has been reduced by tred2
are required, then z is input as the matrix output by tred2. In either case,
the kth column of z returns the normalized eigenvector corresponding to d[k].
*******************************************************************************/
void tqli(double d[], double e[], int n, double **z);