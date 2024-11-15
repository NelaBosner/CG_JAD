int cg_jad_oblique_var_org( int curve, int linsear, int conj, int m, int n, double tol, double **A, int *ldA, double *X, int ldX );

/* Matrices are stored columnwise as 1D fields. A is set of matrices, where A[p] is the p-th input matrix, and ldA is field of leading dimensions for each A[p]. */
/*
curve : choice of the curve for the line search
    0 - geodesic,
    1 - retraction
linsear : choice of the line search algorithm
    0 - Nelder-Mead method,
    1 - Armijo's backtracking
conj : choice of conjugacy formula
    0 - exact conjugation,
    1 - Polak-Riebiere formula,
    2 - Fletcher-Reeves formula
m : number of input matrices
n : dimension of input matrices
tol: tolerance on the function drop
A : set of input matrices
ldA : leading dimensions of A
X : initial solution approximation on the manifold
ldX : leading dimension of X
*/
