#include <mkl.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include "cg_jad_oblique_var_org.h"
#include "cg_jad_oblique_var_par.h"

int max_num_cores = 24;

int main()
{
	int m[] = {10, 20, 50, 100, 200, 1000}; // Choices for the number of input matrices
	int n[] = {10, 100, 1000, 2000}; // Choices for the matrix dimension

    // .. Optimal number of BLAS threads for functions multiple_gemms() and traces_by_ddots(), and for parameters m and n in order given by first two outer loops and arrays m[] and n[].
    int nt_mkl_gemm_n[] = {8, 8, 12, 8, 8, 8, 6, 6, 8, 8, 3, 6, 8, 8, 12, 2, 8, 3, 6, 3, 24, 6, 2, 1}; // Optimal numbers of BLAS threads for multiple_gemms() with trans1='n'.
    int nt_mkl_gemm_t[] = {6, 4, 12, 24, 8, 6, 24, 12, 8, 6, 8, 4, 8, 4, 6, 4, 8, 3, 12, 2, 1, 2, 8, 1}; // Optimal numbers of BLAS threads for multiple_gemms() with trans1='t'.
    int nt_mkl_dots_n[] = {12, 2, 1, 1, 3, 2, 1, 1, 8, 2, 1, 1, 4, 2, 1, 1, 12, 1, 1, 1, 1, 1, 1, 1}; // Optimal numbers of BLAS threads for traces_by_ddots() with trans='n'.
    int nt_mkl_dots_t[] = {8, 3, 4, 2, 3, 4, 3, 12, 3, 8, 6, 1, 6, 8, 3, 2, 4, 3, 8, 1, 8, 3, 12, 3}; // Optimal numbers of BLAS threads for traces_by_ddots() with trans='t'.
    int inmt_g_opt[] = {0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}; // Optimal choices of parameter inmt_g in the second test round, for m=1000 and n=100.
    int inmt_d_opt[] = {1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1}; // Optimal choices of parameter inmt_d in the second test round, for m=1000 and n=100.
	int *num_mkl_threads[4];
    num_mkl_threads[0] = nt_mkl_gemm_n;
    num_mkl_threads[1] = nt_mkl_gemm_t;
    num_mkl_threads[2] = nt_mkl_dots_n;
    num_mkl_threads[3] = nt_mkl_dots_t;

	double tol = 1e-5;

	int i, im, inmt_d, inmt_g, in, info, j, n2, nmklt_d, nmklt_g, nt_d, nt_g, p;
    int curve, linesear, conj, incx=1;
    int IDIST = 3;
	int iter_s, iter_p;
    int ISEED[] = { 4000, 3000, 2000, 1001 };
    int *ldA;
    double ZERO = 0, ONE = 1;
	double baze, col_norm, t_s, t_p, tol_X0 = 1e-3;
    double *Qe, *tau, *W, *X, *Xw, **A;
	struct timeval tb, te, time;

	FILE *fp;

	fp = fopen("test_cg_jad_oblique_var_results.txt","w");


	//for ( im = 0; im < 6; im++ ) // Loop for choosing the number of input matrices. Uncomment for the first test round.
	{
		im = 5; // Choice for the second test round. Comment for the first test round.

		A = (double**) malloc(m[im]*sizeof(double*));
		ldA = (int*) malloc(m[im]*sizeof(int));

		//for ( in = 0; in < 4; in++ ) // Loop for choosing the matrix dimension. Uncomment for the first test round.
		{
			in = 1; // Choice for the second test round. Comment for the first test round.

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
			//dlarnv( &IDIST, ISEED, &n2, Qe ); // Uncomment for the first test round.
			//for ( i = 0; i < n[in]; i++ ) // Uncomment for the first test round.
    		//{ // Uncomment for the first test round.
        		//col_norm = 1 / dnrm2( &n[in], &Qe[i*n[in]], &incx ); // Uncomment for the first test round.
        		//dscal( &n[in], &col_norm, Qe+i*n[in], &incx ); // Uncomment for the first test round.
    		//} // Uncomment for the first test round.
            //dgetrf( &n[in], &n[in], Qe, &n[in], ipiv, &info ); // Uncomment for the first test round.
            //dgetri( &n[in], Qe, &n[in], ipiv, W, &n2, &info ); // Uncomment for the first test round.

			baze = 1e-3;
    		for ( p = 0; p < m[im]; p++ )
			{
				dlaset( "A", &n[in], &n[in], &ZERO, &ZERO, A[p], &ldA[p] );
        		for ( i = 0; i < n[in]; i++ )
					A[p][i+i*ldA[p]] = pow(-1,i) * ((i+1) + p*baze);
				//dgemm( "T", "N", &n[in], &n[in], &n[in], &ONE, Qe, &n[in], A[p], &ldA[p], &ZERO, W, &n[in] ); // Uncomment for the first test round.
				//dgemm( "N", "N", &n[in], &n[in], &n[in], &ONE, W, &n[in], Qe, &n[in], &ZERO, A[p], &ldA[p] ); // Uncomment for the first test round.
			}

			//-------------------------------------------------------------------------------------------

            // .. Generating the initial solution approximation.
			dlaset( "A", &n[in], &n[in], &ZERO, &ONE, Qe, &n[in] );
            dlarnv( &IDIST, ISEED, &n2, X);
			for ( i = 0; i < n[in]*n[in]; i++ )
			{
				X[i] = Qe[i] + tol_X0 * X[i];
			}
			for ( i = 0; i < n[in]; i++ )
    		{
        		col_norm = 1 / dnrm2( &n[in], &X[i*n[in]], &incx );
        		dscal( &n[in], &col_norm, X+i*n[in], &incx );	
    		}
            dlacpy( "A", &n[in], &n[in], X, &n[in], Xw, &n[in] );
            
            //-------------------------------------------------------------------------------------------

            for ( curve = 0; curve < 2; curve++ ) // Loop for choosing the curve for line search.
            {
                for ( linesear = 0; linesear < 2; linesear++ ) // Loop for choosing the line search algorithm.
                {
                    for ( conj = 0; conj < 3; conj++ ) // Loop for choosing the conjugacy formula.
                    {
                        // .. Execute and measure the time of the original form of the algorithm.
						dlacpy( "A", &n[in], &n[in], Xw, &n[in], X, &n[in] );
                        gettimeofday( &tb, NULL );
                        iter_s = cg_jad_oblique_var_org( curve, linesear, conj, m[im], n[in], tol, A, ldA, X, n[in]);
                        gettimeofday( &te, NULL );
                        timersub( &te, &tb, &time );
                        t_s = ((1000.*(double)(time.tv_sec) + (double)(time.tv_usec) * 0.001)) / 1000.;

                        //-------------------------------------------------------------------------------------------

                        //for ( inmt_g = 0; inmt_g < 2; inmt_g++ ) // Loop for choosing the optimal number of BLAS threads for multiple_gemms(). Uncomment for the first test round.
                        {
                            inmt_g = inmt_g_opt[curve*6+linesear*3+conj]; // Comment for the first test round.
                            nmklt_g = num_mkl_threads[inmt_g][in+im*4];
                            nt_g = max_num_cores / nmklt_g - 1;
                            //for ( inmt_d = 0; inmt_d < 2; inmt_d++ ) // Loop for choosing the optimal number of BLAS threads for traces_by_ddots(). Uncomment for the first test round.
                            {
                                inmt_d = inmt_d_opt[curve*6+linesear*3+conj]; // Comment for the first test round.
                                nmklt_d = num_mkl_threads[2+inmt_d][in+im*4];
                                nt_d = max_num_cores / nmklt_d - 1;

                                // .. Execute and measure the time of the parallel form of the algorithm.
                                dlacpy( "A", &n[in], &n[in], Xw, &n[in], X, &n[in] );

                                gettimeofday( &tb, NULL );
                                iter_p = cg_jad_oblique_var_par( curve, linesear, conj, m[im], n[in], nmklt_d, nt_d, nmklt_g, nt_g, tol, A, ldA, X, n[in] );
                                gettimeofday( &te, NULL );
                                timersub( &te, &tb, &time );
                                t_p = ((1000.*(double)(time.tv_sec) + (double)(time.tv_usec) * 0.001)) / 1000.;

                                // .. Store the timing results.
                                fprintf(fp, "%d %d %d %d %d %d %d %d %d %d %d %d %g %d %g\n", m[im], n[in], curve, linesear, conj, inmt_g, nmklt_g, nt_g, inmt_d, nmklt_d, nt_d, iter_s, t_s, iter_p, t_p );
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
