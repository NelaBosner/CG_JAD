// Test file for Riemannian conjugate gradient method on the oblique manifold for joint approximate diagonalization.
// This file implements the scalability test described in the "Numerical experiments" section of the paper
// "Parallel implementations of Riemannian conjugate gradient methods for joint approximate diagonalization"
//
// Author: Nela Bosner
//
#include <mkl.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include "cg_jad_oblique_var_org.h"
#include "cg_jad_oblique_var_par.h"

int max_num_cores = 1;

int main()
{
	int m[] = {10, 20, 50, 100, 200, 1000}; // Choices for the number of input matrices
	int n[] = {10, 100, 1000, 2000}; // Choices for the matrix dimension

    // .. Optimal number of BLAS threads for functions multiple_gemms() and traces_by_ddots(), and for parameters m and n in order given by first two outer loops and arrays m[] 
    and n[].
    int nt_mkl_gemm_n[] = {1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1}; // Optimal numbers of BLAS threads for multiple_gemms() with trans1='n'.
    int nt_mkl_gemm_t[] = {1, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1}; // Optimal numbers of BLAS threads for multiple_gemms() with trans1='t'.
    int nt_mkl_dots_n[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1}; // Optimal numbers of BLAS threads for traces_by_ddots() with trans='n'.
    int nt_mkl_dots_t[] = {1, 1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 1, 1, 1}; // Optimal numbers of BLAS threads for traces_by_ddots() with trans='t'.
	int *num_mkl_threads[4];
    num_mkl_threads[0] = nt_mkl_gemm_n;
    num_mkl_threads[1] = nt_mkl_gemm_t;
    num_mkl_threads[2] = nt_mkl_dots_n;
    num_mkl_threads[3] = nt_mkl_dots_t;

	double tol = 1e-5;

	int i, im, inmt_d, inmt_g, in, info, j, n2, nmklt_d, nmklt_g, nt_d, nt_g, p;
    int curve, linesear, conj;
    int IDIST = 3;
	int iter_s, iter_p;
    int ISEED[] = { 4000, 3000, 2000, 1001 };
    int *ldA;
    double ZERO = 0, ONE = 1;
	double baze, col_normt_s, t_p, tol_X0 = 1e-3;
    double *Qe, *tau, *W, *X, *Xw, **A;
	struct timeval tb, te, time;

	FILE *fp;

	fp = fopen("test_cg_jad_oblique_var_results_nt1.txt","w");


	for ( im = 0; im < 6; im++ ) // Loop for choosing the number of input matrices. Uncomment for the first test round.
	{
		A = (double**) malloc(m[im]*sizeof(double*));
		ldA = (int*) malloc(m[im]*sizeof(int));

		for ( in = 0; in < 4; in++ ) // Loop for choosing the matrix dimension. Uncomment for the first test round.
		{
			n2 = n[in] * n[in];
			Qe = (double*) malloc(n2*sizeof(double));
    		X = (double*) malloc(n2*sizeof(double));
            Xw = (double*) malloc(n2*sizeof(double));
    		W = (double*) malloc(n2*sizeof(double));
    		tau = (double*) malloc(n[in]*sizeof(double));

			for ( p = 0; p < m[im]; p++ )
    		{
        		A[p] = (double*) malloc(n2*sizeof(double));
        		ldA[p] = n[in];
    		}

            // .. Generating the input matrices.
			dlarnv( &IDIST, ISEED, &n2, Qe ); // Uncomment for the first test round.
			for ( i = 0; i < n[in]; i++ ) // Uncomment for the first test round.
    		{ // Uncomment for the first test round.
        		col_norm = 1 / dnrm2( &n[in], &Qe[i*n[in]], &incx ); // Uncomment for the first test round.
        		dscal( &n[in], &col_norm, Qe+i*n[in], &incx ); // Uncomment for the first test round.
    		} // Uncomment for the first test round.
            dgetrf( &n[in], &n[in], Qe, &n[in], ipiv, &info ); // Uncomment for the first test round.
            dgetri( &n[in], Qe, &n[in], ipiv, W, &n2, &info ); // Uncomment for the first test round.

			baze = 10; //1e-3;
    		for ( p = 0; p < m[im]; p++ )
			{
				dlaset( "A", &n[in], &n[in], &ZERO, &ZERO, A[p], &ldA[p] );
        		for ( i = 0; i < n[in]; i++ )
					A[p][i+i*ldA[p]] =  pow(-1,i) * ((i+1) + p*baze);//pow(-1,i) * ((i+1) + 10*p);
				dgemm( "N", "N", &n[in], &n[in], &n[in], &ONE, Qe, &n[in], A[p], &ldA[p], &ZERO, W, &n[in] ); // Uncomment for the first test round.
				dgemm( "N", "T", &n[in], &n[in], &n[in], &ONE, W, &n[in], Qe, &n[in], &ZERO, A[p], &ldA[p] ); // Uncomment for the first test round.
			}

			//-------------------------------------------------------------------------------------------

            // .. Generating the initial solution approximation.
			dlaset( "A", &n[in], &n[in], &ZERO, &ONE, Qe, &n[in] );
            dlarnv( &IDIST, ISEED, &n2, X);
			//for ( i = 0; i < n[in]*n[in]; i++ )
			//{
				//X[i] = Qe[i] + tol_X0 * X[i];
			//}
			for ( i = 0; i < n[in]; i++ )
    		{
        		col_norm = 1 / dnrm2( &n[in], &X[i*n[in]], &incx );
        		dscal( &n[in], &col_norm, X+i*n[in], &incx );	
    		}
            dlacpy( "A", &n[in], &n[in], X, &n[in], Xw, &n[in] );

            //-------------------------------------------------------------------------------------------
			
            //for ( curve = 0; curve < 2; curve++ ) // Loop for choosing the curve for line search.
            {
                curve = 0; //geodesic
                //for ( linesear = 0; linesear < 2; linesear++ ) // Loop for choosing the line search algorithm.
                {
                    linesear = 1; //Armijo's backtracking
                    //for ( conj = 0; conj < 3; conj++ ) // Loop for choosing the conjugacy formula.
                    {
                        conj = 0; //exact conjugacy
                        // .. Execute and measure the time of the original form of the algorithm.
						//dlacpy( "A", &n[in], &n[in], Xw, &n[in], X, &n[in] );
                        //gettimeofday( &tb, NULL );
                        //iter_s = cg_jad_oblique_var_org( curve, linesear, conj, m[im], n[in], tol, A, ldA, X, n[in]);
                        //gettimeofday( &te, NULL );
                        //timersub( &te, &tb, &time );
                        //t_s = ((1000.*(double)(time.tv_sec) + (double)(time.tv_usec) * 0.001)) / 1000.;

                        //-------------------------------------------------------------------------------------------

                        //for ( inmt_g = 0; inmt_g < 2; inmt_g++ ) // Loop for choosing the optimal number of BLAS threads for multiple_gemms(). Uncomment for the first test 
                        round.
                        {
                            //nmklt_g = num_mkl_threads[inmt_g][in+im*4];
                            nmklt_g = 1;
                            nt_g = max_num_cores / nmklt_g - 1;
                            //for ( inmt_d = 0; inmt_d < 2; inmt_d++ ) // Loop for choosing the optimal number of BLAS threads for traces_by_ddots(). Uncomment for the first 
                            test round.
                            {
                                //nmklt_d = num_mkl_threads[2+inmt_d][in+im*4];
                                nmklt_d = 1;
                                nt_d = max_num_cores / nmklt_d - 1;

                                // .. Execute and measure the time of the parallel form of the algorithm.
                                dlacpy( "A", &n[in], &n[in], Xw, &n[in], X, &n[in] );

                                gettimeofday( &tb, NULL );
                                iter_p = cg_jad_oblique_var_par( curve, linesear, conj, m[im], n[in], nmklt_d, nt_d, nmklt_g, nt_g, tol, A, ldA, X, n[in] );
                                gettimeofday( &te, NULL );
                                timersub( &te, &tb, &time );
                                t_p = ((1000.*(double)(time.tv_sec) + (double)(time.tv_usec) * 0.001)) / 1000.;

                                // .. Store the timing results.
                                fprintf(fp, "%d %d %d %d %d %d %d %d %d %d %d %d %g\n", m[im], n[in], curve, linesear, conj, inmt_g, nmklt_g, nt_g, inmt_d, nmklt_d, nt_d, iter_p, t_p );
                                fflush(fp);
                            }
                        }
                    }
				}
			}

			for ( p = 0; p < m[im]; p++ )
			{
				free(A[p]);
			}
			free(Qe);
			free(X);
            free(Xw);
			free(W);
			free(tau);

		}

		free(A);
		free(ldA);

	}

	fclose(fp);

}
