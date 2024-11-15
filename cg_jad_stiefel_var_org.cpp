// Original implementation of Riemannian conjugate gradient method on the Stiefel manifold for joint approximate diagonalization.
// This file contains the main function implementing the method, and other auxiliary functions.
//
// The main function call has the form:
// int cg_jad_stiefel_var_org( int curve, int linsear, int conj, int m, int n, double tol, double **A, int *ldA, double *X, int ldX );
// where
// curve : choice of the curve for the line search
//    0 - geodesic, 
//    1 - retraction
// linsear : choice of the line search algorithm
//    0 - Nelder-Mead method, 
//    1 - Armijo's backtracking
// conj : choice of conjugacy formula
//    0 - exact conjugacy, 
//    1 - Polak-Riebiere formula, 
//    2 - Fletcher-Reeves formula
// m : number of input matrices
// n : dimension of input matrices
// tol: tolerance on the function drop
// A : set of input matrices
// ldA : leading dimensions of A
// X : initial solution approximation on the manifold
// ldX : leading dimension of X
//
// Matrices are stored columnwise as 1D fields. 
// A is set of matrices, where A[p] is the p-th input matrix, and ldA is field of leading dimensions for each A[p].
//
// Author: Nela Bosner
//
#include <mkl.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>

static double trace( int n, double *A, int ldA )
{
    int i;
    double tr = 0;

    for ( i = 0; i < n; i++ )
        tr += A[i+i*ldA];

    return tr;
}

static void Off( int n, double *A, int ldA )
{
    int i;

    for ( i = 0; i < n; i++ )
        A[i+i*ldA] = 0.0;
}

static void lin_comb_of_2matrices( int n, double alpha, double *A, int ldA, double beta, double *B, int ldB, double *R, int ldR )
{
    int i, j;

    for( i = 0; i < n; i++ )
    {
        for( j = 0; j < n; j++ )
        {
            R[i+j*ldR] = alpha * A[i+j*ldA] + beta * B[i+j*ldB];
        }
    }
}

static void lin_comb_trans( int n, double alpha, double beta, double *A, int ldA, double *R, int ldR )
{
    int i, j;

    for( i = 0; i < n; i++ )
    {
        for( j = 0; j < n; j++ )
        {
            R[i+j*ldR] = alpha * A[i+j*ldA] + beta * A[j+i*ldA];
        }
    }
}

static void scaling_of_matrix( int n, double alpha, double *A, int ldA, double *R, int ldR )
{
    int i, j;

    for( i = 0; i < n; i++ )
    {
        for( j = 0; j < n; j++ )
        {
            R[i+j*ldR] = alpha * A[i+j*ldA];
        }
    }
}

static double Fn( int m, int n, double **A, int *ldA, double *X, int ldX )
{
    int p;
    double f=0.0, dZERO=0.0, dONE=1.0;
    double *B, *W;

    B = (double *) malloc(n*n*sizeof(double));
    W = (double *) malloc(n*n*sizeof(double));

    for ( p = 0; p < m; p++ )
    {
        dgemm( "N", "N", &n, &n, &n, &dONE, A[p], &ldA[p], X, &ldX, &dZERO, W, &n );
        dgemm( "T", "N", &n, &n, &n, &dONE, X, &ldX, W, &n, &dZERO, B, &n );
        Off( n, B, n );
        f += pow(dlange( "F", &n, &n, B, &n, W ),2);
    }
    f *= 0.5;

    free(B);
    free(W);
    return f;
}

static void Fn_X( int m, int n, double **A, int *ldA,double *X, int ldX, double *FX, int ldFX )
{
    int p;
    double beta, dZERO=0.0, dONE=1.0, dTWO=2.0;
    double *AX, *B;

    AX = (double *) malloc(n*n*sizeof(double));
    B = (double *) malloc(n*n*sizeof(double));

    dlaset( "A", &n, &n, &dZERO, &dZERO, FX, &ldFX );

    for ( p = 0; p < m; p++ )
    {
        dgemm( "N", "N", &n, &n, &n, &dONE, A[p], &ldA[p], X, &ldX, &dZERO, AX, &n );
        dgemm( "T", "N", &n, &n, &n, &dONE, X, &ldX, AX, &n, &dZERO, B, &n );
        Off( n, B, n );
        if ( p == 0 )
            beta = 0.0;
        else
            beta = 1.0;
        dgemm( "N", "N", &n, &n, &n, &dTWO, AX, &n, B, &n, &beta, FX, &n );
    }

    free(AX);
    free(B);
}

static double Fn_XX( int m, int n, double **A, int *ldA, double *X, int ldX, double *Xi1, int ldXi1, double *Xi2, int ldXi2 )
{
    int p;
    double f=0.0, dZERO=0.0, dONE=1.0;
    double *AX, *W, *W1, *W2;

    AX = (double *) malloc(n*n*sizeof(double));
    W = (double *) malloc(n*n*sizeof(double));
    W1 = (double *) malloc(n*n*sizeof(double));
    W2 = (double *) malloc(n*n*sizeof(double));

    for ( p = 0; p < m; p++ )
    {
        dgemm( "N", "N", &n, &n, &n, &dONE, A[p], &ldA[p], X, &ldX, &dZERO, AX, &n );
        dgemm( "T", "N", &n, &n, &n, &dONE, Xi1, &ldXi1, AX, &n, &dZERO, W1, &n );
        dgemm( "T", "N", &n, &n, &n, &dONE, Xi2, &ldXi2, AX, &n, &dZERO, W2, &n );
        Off( n, W2, n );
        dgemm( "N", "T", &n, &n, &n, &dONE, W1, &n, W2, &n, &dZERO, W, &n );
        f += trace( n, W, n );
        dgemm( "T", "T", &n, &n, &n, &dONE, W1, &n, W2, &n, &dZERO, W, &n );
        f += trace( n, W, n );
        dgemm( "T", "N", &n, &n, &n, &dONE, X, &ldX, AX, &n, &dZERO, W2, &n );
		Off( n, W2, n );
        dgemm( "N", "N", &n, &n, &n, &dONE, A[p], &ldA[p], Xi2, &ldXi2, &dZERO, AX, &n );
        dgemm( "T", "N", &n, &n, &n, &dONE, Xi1, &ldXi1, AX, &n, &dZERO, W1, &n );
        dgemm( "N", "N", &n, &n, &n, &dONE, W1, &n, W2, &n, &dZERO, W, &n );
        f += trace( n, W, n );
    }
    f *= 2;

    free(AX);
    free(W);
    free(W1);
    free(W2);
    return f;
}

static void ss_alg_exp( double t, int n, double *Ag, int ldAg, double *R, int ldR )
{
    double thetam[] = { 1.5e-2, 2.5e-1, 9.5e-1, 2.1e0 };
    double theta13=5.4e0;
    double dZERO=0.0, dONE=1.0;
    int info, *ipiv, i, j, s, scal = 0;
	int ldAg2, ldAg4, ldAg6, n1Ag;
	double id;
    double *Ag2, *Ag4, *Ag6, *Ag8, *P, *Q, *Up, *Vp;

    double b3[] = { 120, 60, 12, 1 };
    double b5[] = { 30240, 15120, 3360, 420, 30, 1 };
    double b7[] = { 17297280, 8648640,1995840, 277200, 25200,1512, 56, 1 };
    double b9[] = { 17643225600, 8821612800, 2075673600, 302702400, 30270240, 2162160, 110880, 3960, 90, 1 };
    double b13[] = { 64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800, 129060195264000, 10559470521600, 670442572800, 33522128640, 1323241920, 40840800, 
    960960, 16380, 182, 1 };

	ldAg2 = n;
	ldAg4 = n;
	ldAg6 = n;

	Ag2 = (double*) malloc(n*n*sizeof(double));
    Ag4 = (double*) malloc(n*n*sizeof(double));
    Ag6 = (double*) malloc(n*n*sizeof(double));
    P = (double*) malloc(n*n*sizeof(double*));
    Q = (double*) malloc(n*n*sizeof(double*));
    Up = (double*) malloc(n*n*sizeof(double*));
    Vp = (double*) malloc(n*n*sizeof(double*));

	ipiv = (int*) malloc(n*sizeof(int));

	dgemm( "N", "N", &n, &n, &n, &dONE, Ag, &n, Ag, &n, &dZERO, Ag2, &n );
	dgemm( "N", "N", &n, &n, &n, &dONE, Ag2, &n, Ag2, &n, &dZERO, Ag4, &n );
	dgemm( "N", "N", &n, &n, &n, &dONE, Ag4, &n, Ag2, &n, &dZERO, Ag6, &n );
	n1Ag = dlange( "1", &n, &n, Ag, &n, P );

    if ( t * n1Ag <= thetam[0] )
    {
        for( i = 0; i < n; i++)
        {
            for( j = 0; j < n; j++ )
            {
                if ( i == j )
                    id = 1;
                else
                    id = 0;
                Up[i+j*n] = pow(t,3) * b3[3] * Ag2[i+j*ldAg2] + t * b3[1] * id;
                Q[i+j*n] = pow(t,2) * b3[2] * Ag2[i+j*ldAg2] + b3[0] * id;
            }
        }
        dgemm( "N", "N", &n, &n, &n, &dONE, Ag, &ldAg, Up, &n, &dZERO, P, &n );
    }

    else if ( t * n1Ag <= thetam[1] )
    {
        for( i = 0; i < n; i++)
        {
            for( j = 0; j < n; j++ )
            {
                if ( i == j )
                    id = 1;
                else
                    id = 0;
                Up[i+j*n] = pow(t,5) * b5[5] * Ag4[i+j*ldAg4] + pow(t,3) * b5[3] * Ag2[i+j*ldAg2] + t * b5[1] * id;
                Q[i+j*n] = pow(t,4) * b5[4] * Ag4[i+j*ldAg4] + pow(t,2) * b5[2] * Ag2[i+j*ldAg2] + b5[0] * id;

            }
        }
        dgemm( "N", "N", &n, &n, &n, &dONE, Ag, &ldAg, Up, &n, &dZERO, P, &n );
    }

    else if ( t * n1Ag <= thetam[2] )
    {
        for( i = 0; i < n; i++)
        {
            for( j = 0; j < n; j++ )
            {
                if ( i == j )
                    id = 1;
                else
                    id = 0;
                Up[i+j*n] = pow(t,7) * b7[7] * Ag6[i+j*ldAg6] + pow(t,5) * b7[5] * Ag4[i+j*ldAg4] + pow(t,3) * b7[3] * Ag2[i+j*ldAg2] + t * b7[1] * id;
                Q[i+j*n] = pow(t,6) * b7[6] * Ag6[i+j*ldAg6] + pow(t,4) * b7[4] * Ag4[i+j*ldAg4] + pow(t,2) * b7[2] * Ag2[i+j*ldAg2] + b7[0] * id;
            }
        }
        dgemm( "N", "N", &n, &n, &n, &dONE, Ag, &ldAg, Up, &n, &dZERO, P, &n );
    }

    else if ( t * n1Ag <= thetam[3] )
    {
        Ag8 = (double *) malloc(n*n*sizeof(double));
        dgemm( "N", "N", &n, &n, &n, &dONE, Ag4, &ldAg4, Ag4, &ldAg4, &dZERO, Ag8, &n );

        for( i = 0; i < n; i++)
        {
            for( j = 0; j < n; j++ )
            {
                if ( i == j )
                    id = 1;
                else
                    id = 0;
                Up[i+j*n] = pow(t,9) * b9[9] * Ag8[i+j*n] + pow(t,7) * b9[7] * Ag6[i+j*ldAg6] + pow(t,5) * b9[5] * Ag4[i+j*ldAg4] + pow(t,3) * b9[3] * Ag2[i+j*ldAg2] + t * b9[1] * 
                id;
                Q[i+j*n] = pow(t,8) * b9[8] * Ag8[i+j*n] + pow(t,6) * b9[6] * Ag6[i+j*ldAg6] + pow(t,4) * b9[4] * Ag4[i+j*ldAg4] + pow(t,2) * b9[2] * Ag2[i+j*ldAg2] + b9[0] * id;
            }
        }
        dgemm( "N", "N", &n, &n, &n, &dONE, Ag, &ldAg, Up, &n, &dZERO, P, &n );
        free(Ag8);
    }

    else
    {
        s = ceil(log2(t*n1Ag/theta13));
        if ( s != 0 ) scal = 1;
        t = t / pow(2,s);
        
        for( i = 0; i < n; i++)
        {
            for( j = 0; j < n; j++ )
            {
                if ( i == j )
                    id = 1;
                else
                    id = 0;

                P[i+j*n] = pow(t,13) * b13[13] * Ag6[i+j*ldAg6] + pow(t,11) * b13[11] * Ag4[i+j*ldAg4] + pow(t,9) * b13[9] * Ag2[i+j*ldAg2];
                Up[i+j*n] = pow(t,7) * b13[7] * Ag6[i+j*ldAg6] + pow(t,5) * b13[5] * Ag4[i+j*ldAg4] + pow(t,3) * b13[3] * Ag2[i+j*ldAg2] + t * b13[1] * id;

                Vp[i+j*n] = pow(t,12) * b13[12] * Ag6[i+j*ldAg6] + pow(t,10) * b13[10] * Ag4[i+j*ldAg4] + pow(t,8) * b13[8] * Ag2[i+j*ldAg2];
                Q[i+j*n] = pow(t,6) * b13[6] * Ag6[i+j*ldAg6] + pow(t,4) * b13[4] * Ag4[i+j*ldAg4] + pow(t,2) * b13[2] * Ag2[i+j*ldAg2] + b13[0] * id;
            }
        }
        dgemm( "N", "N", &n, &n, &n, &dONE, Ag6, &ldAg6, P, &n, &dONE, Up, &n );
        dgemm( "N", "N", &n, &n, &n, &dONE, Ag, &ldAg, Up, &n, &dZERO, P, &n );
        dgemm( "N", "N", &n, &n, &n, &dONE, Ag6, &ldAg6, Vp, &n, &dONE, Q, &n );
    }

    for( i = 0; i < n; i++)
    {
        for( j = 0; j < n; j++ )
        {
            double u = P[i+j*n];
            double v = Q[i+j*n];

            P[i+j*n] = u + v;
            Q[i+j*n] = -u + v;
        }
    }

    dgesv( 	&n, &n, Q, &n, ipiv, P, &n, &info );
    dlacpy( "A", &n, &n, P, &n, R, &ldR );

    if ( scal )
    {
        for ( i = 1; i <= s; i++ )
        {
            dgemm( "N", "N", &n, &n, &n, &dONE, R, &ldR, R, &ldR, &dZERO, P, &n );
            dlacpy( "A", &n, &n, P, &n, R, &ldR );
        }

    }

	free(Ag2);
	free(Ag4);
	free(Ag6);
    free(P);
    free(Q);
    free(Up);
    free(Vp);
    free(ipiv);

}

static double Fn_geod( double t, int m, int n, double *Ag, int ldAg, double **A, int *ldA, double *X, int ldX, double *eAt, int ldeAt )
{
    int p;
    double y;
    double dZERO=0.0, dONE=1.0;
    double *C, *W1, *W2;

	C = (double*) malloc(n*n*sizeof(double));
    W1 = (double*) malloc(n*n*sizeof(double));
	W2 = (double*) malloc(n*n*sizeof(double));

    ss_alg_exp( t, n, Ag, ldAg, eAt, ldeAt );

    y = 0;

	for( p = 0; p < m; p++ )
    {
		dgemm( "N", "N", &n, &n, &n, &dONE, X, &ldX, eAt, &ldeAt, &dZERO, W1, &n );
        dgemm( "N", "N", &n, &n, &n, &dONE, A[p], &ldA[p], W1, &n, &dZERO, W2, &n );
        dgemm( "T", "N", &n, &n, &n, &dONE, W1, &n, W2, &n, &dZERO, C, &n ); 
        Off( n, C, n );
        y += pow(dlange( "F", &n, &n, C, &n, W1 ),2);
    }

    y *= 0.5;

	free(C);
    free(W1);
    free(W2);

    return y;
}

static double Fn_retr( double t, int m, int n, double **A, int *ldA, double *X, int ldX, double *H, int ldH, double *eAt, int ldeAt )
{
    int info, n2=n*n, p;
    double y;
    double dZERO=0.0, dONE=1.0;
    double *B, *tau, *W1, *W2;

    B = (double*) malloc(n*n*sizeof(double));
    W1 = (double*) malloc(n*n*sizeof(double));
    W2 = (double*) malloc(n*n*sizeof(double));
    tau = (double*) malloc(n*sizeof(double));

    lin_comb_of_2matrices( n, 1.0, X, ldX, t, H, ldH, eAt, ldeAt );

    dgeqrf( &n, &n, eAt, &ldeAt, tau, W2, &n2, &info	);
    dorgqr( &n, &n, &n, eAt, &ldeAt, tau, W2, &n2, &info );

    y = 0;

	for( p = 0; p < m; p++ )
    {
        dgemm( "N", "N", &n, &n, &n, &dONE, A[p], &ldA[p], eAt, &ldeAt, &dZERO, W2, &n );
        dgemm( "T", "N", &n, &n, &n, &dONE, eAt, &ldeAt, W2, &n, &dZERO, B, &n );
        Off( n, B, n );
        y += pow(dlange( "F", &n, &n, B, &n, W1 ),2);
    }

    y *= 0.5;

    free(B);
	free(W1);
    free(W2);
    free(tau);

    return y;
}

static double nelder_mead_curve( int curve, double tol, int m, int n, double fX, double *Ag, int ldAg, double **A, int *ldA, double *X, int ldX, double *H, int ldH, double *eAt, int ldeAt )
{
    int fun_eval, nt, shrink;
    double alpha, beta, c, dONE, dZERO, gamma, delta, F0, F1, Fc, Fe, Fr, pomd, *pomp, *Q0, *Q1, t0, t0_old, t1, t1_old, tc, te, tr, *W, *W1, *W2, *W3, *W4;
    
    W = (double*) malloc(n*n*sizeof(double));
    W1 = (double*) malloc(n*n*sizeof(double));
    W2 = (double*) malloc(n*n*sizeof(double));
    W3 = (double*) malloc(n*n*sizeof(double));
    W4 = (double*) malloc(n*n*sizeof(double));

	dZERO = 0.0;
	dONE = 1.0;

    alpha = 1.0;
    beta = 0.5;
    gamma = 2;
    delta = 0.5;

    t0 = 0.0;
    t1 = 1/sqrt(fX);
	t0_old = t0;
	t1_old = t1;
	
	F0 = fX;
    if ( curve == 0 )
    {
		dlaset( "A", &n, &n, &dZERO, &dONE, W, &n );
        F1 = Fn_geod( t1, m, n, Ag, ldAg, A, ldA, X, ldX, W1, n );
    }
    else
    {
		dlacpy( "A", &n, &n, X, &ldX, W, &n ); 
        F1 = Fn_retr( t1, m, n, A, ldA, X, ldX, H, ldH, W1, n );
    }
    Q0 = W;
    Q1 = W1;
	fun_eval = 1;

    while ( (fabs(F1-F0)/fabs(F0) > tol) && (fabs(F0) > tol) && (fabs(F1) > tol) && (fabs(t1-t0)>tol) && (fun_eval<=100) )
    {
		// .. Restart in case of negative parameters.
		if ((t0<0) || (t1<0))
		{
			t0 = t0_old;
			t1_old /= 10;
			t1 = t1_old;
			F0 = fX;
    		if ( curve == 0 )
   			{
        		dlaset( "A", &n, &n, &dZERO, &dONE, W, &n );
        		F1 = Fn_geod( t1, m, n, Ag, ldAg, A, ldA, X, ldX, W1, n );
    		}
    		else
    		{
        		dlacpy( "A", &n, &n, X, &ldX, W, &n ); 
        		F1 = Fn_retr( t1, m, n, A, ldA, X, ldX, H, ldH, W1, n );
    		}	
			fun_eval++;	
		}

        shrink = 0;
        if ( F1 < F0 ) 
        {
            pomd = t0;
            t0 = t1;
            t1 = pomd;
            pomd = F0;
            F0 = F1;
            F1 = pomd;
            pomp = Q0;
            Q0 = Q1;
            Q1 = pomp;
        }
        c = t0;

        tr = c + alpha * (c - t1);
        if ( curve == 0 )
        {
            Fr = Fn_geod( tr, m, n, Ag, ldAg, A, ldA, X, ldX, W2, n );
        }
        else
        {
            Fr = Fn_retr( tr, m, n, A, ldA, X, ldX, H, ldH, W2, n );
        }
		fun_eval++;

        // .. Reflect will never happen.

        // .. Expand
        if ( Fr < F0 ) 
        {
            te = c + gamma * (tr - c);
            if ( curve == 0 )
            {
                Fe = Fn_geod( te, m, n, Ag, ldAg, A, ldA, X, ldX, W3, n );
            }
            else
            {
                Fe = Fn_retr( te, m, n, A, ldA, X, ldX, H, ldH, W3, n );
            }
			fun_eval++;
            if ( Fe < Fr ) 
            {
                t1 = te;
                F1 = Fe;
                Q1 = W3;
            }
            else
            {
                t1 = tr;
                F1 = Fr;
                Q1 = W2;
            }
            continue;
        }

        // .. Contract
        if ( Fr >= F0 ) 
        {
            if ( Fr < F1 ) 
            {
                tc = c + beta * (tr - c);
                if ( curve == 0 )
                {
                    Fc = Fn_geod( tc, m, n, Ag, ldAg, A, ldA, X, ldX, W4, n );
                }
                else
                {
                    Fc = Fn_retr( tc, m, n, A, ldA, X, ldX, H, ldH, W4, n );
                }
				fun_eval++;
                if ( Fc < Fr ) 
                {
                    t1 = tc;
                    F1 = Fc;
                    Q1 = W4;
                }
                else
                {
                    shrink = 1;
                }
            }
            else
            {
                tc = c + beta * (t1 - c);
                if ( curve == 0 )
                {
                    Fc = Fn_geod( tc, m, n, Ag, ldAg, A, ldA, X, ldX, W4, n );
                }
                else
                {
                    Fc = Fn_retr( tc, m, n, A, ldA, X, ldX, H, ldH, W4, n );
                }
				fun_eval++;
                if ( Fc < F1 ) 
                {
                    t1 = tc;
                    F1 = Fc;
                    Q1 = W4;
                }
                else
                {
                    shrink = 1;
                }
            }
        }

        // .. Shrink
        if (shrink)
        {
            t1 = t0 + delta * (t1 - t0);
            if ( curve == 0 )
            {
                F1 = Fn_geod( t1, m, n, Ag, ldAg, A, ldA, X, ldX, Q1, n );;
            }
            else
            {
                F1 = Fn_retr( t1, m, n, A, ldA, X, ldX, H, ldH, Q1, n );
            }
		fun_eval++;
        }
    }

    if ( F0 <= F1 ) 
    {
        dlacpy( "A", &n, &n, Q0, &n, eAt, &n );
        free(W);
        free(W1);
        free(W2);
        free(W3);
        free(W4);
        return t0;
    }
    else
    {
        dlacpy( "A", &n, &n, Q1, &n, eAt, &n );
        free(W);
        free(W1);
        free(W2);
        free(W3);
        free(W4);
        return t1;
    }

}

double armijo_backtracking_curve( int curve, int m, int n, double F0, double *FX, int ldFX, double *H, int ldH, double *Ag, int ldAg, double **A, int *ldA, double *X, int ldX, double *eAt, int ldeAt )
{
    int func_eval, max_steps;
    double alpha, beta, dF_curveX, F1, nH, sigma, *W, *W1;
    double dZERO=0.0, dONE=1.0;
    
    W = (double*) malloc(n*n*sizeof(double));
    W1 = (double*) malloc(n*n*sizeof(double));

    // .. dF_curveX=trace(FX'*H);
    dgemm( "T", "N", &n, &n, &n, &dONE, FX, &ldFX, H, &ldH, &dZERO, W, &n );
    dF_curveX = trace( n, W, n );

    beta = 0.5;
    sigma = pow(2,-13);
    max_steps = 25;

    nH = dlange( "F", &n, &n, H, &n, W);
    alpha = 1.0 / nH;

    if ( curve == 0 )
    {
        F1 = Fn_geod( alpha, m, n, Ag, ldAg, A, ldA, X, ldX, W1, n );
    }
    else
    {
        F1 = Fn_retr( alpha, m, n, A, ldA, X, ldX, H, ldH, W1, n );
    }
    func_eval = 1;

    while ( F1 > F0 + sigma * alpha * dF_curveX )
    {
        alpha = beta * alpha;
        if ( curve == 0 )
        {
            F1 = Fn_geod( alpha, m, n, Ag, ldAg, A, ldA, X, ldX, W1, n );
        }
        else
        {
            F1 = Fn_retr( alpha, m, n, A, ldA, X, ldX, H, ldH, W1, n );
        }
        func_eval++;

        if ( func_eval > max_steps )
        {
            break;
        }

    }

    if ( F1 > F0 )
    {
        alpha = 0;
        F1 = F0;
		if ( curve == 0 )
    	{
			dlaset( "A", &n, &n, &dZERO, &dONE, W1, &n );
    	}
    	else
    	{
			dlacpy( "A", &n, &n, X, &ldX, W1, &n ); 
    	}
    }

    dlacpy( "A", &n, &n, W1, &n, eAt, &n );
    free(W);
    free(W1);
    return alpha;
}

int cg_jad_stiefel_var_org( int curve, int linsear, int conj, int m, int n, double tol, double **A, int *ldA, double *X, int ldX )
{
    int iter;
    double tol_linsear = n*dlamch( "E" );
    double dZERO=0.0, dONE=1.0, dHALF = 0.5, dmHALF = -0.5, dTWO = 2.0;
    double beta, cond, f0, fX, FXX1, FXX2, Hess1, Hess2, t;
    double *Ag, *eAt, *FX, *G, *G0, *H, *X0, *Xi, *W, *W1, *W2;

    Ag = (double*) malloc(n*n*sizeof(double));
    eAt = (double*) malloc(n*n*sizeof(double));
    FX = (double*) malloc(n*n*sizeof(double));
    G = (double*) malloc(n*n*sizeof(double));
    G0 = (double*) malloc(n*n*sizeof(double));
	H = (double*) malloc(n*n*sizeof(double));
    X0 = (double*) malloc(n*n*sizeof(double));
    Xi = (double*) malloc(n*n*sizeof(double));
    W = (double*) malloc(n*n*sizeof(double));
    W1 = (double*) malloc(n*n*sizeof(double));
    W2 = (double*) malloc(n*n*sizeof(double));

    Fn_X( m, n, A, ldA, X, ldX, FX, n );

    dgemm( "T", "N", &n, &n, &n, &dONE, FX, &n, X, &ldX, &dZERO, W1, &n );
    dlacpy( "A", &n, &n, FX, &n, G, &n );
    dgemm( "N", "N", &n, &n, &n, &dmHALF, X, &ldX, W1, &n, &dHALF, G, &n );

    scaling_of_matrix( n, -1, G, n, H, n );

    if ( curve == 0 )
    {
        dgemm( "T", "N", &n, &n, &n, &dONE, X, &ldX, H, &n, &dZERO, Ag, &n );
    }

    iter = 0;
	fX=Fn( m, n, A, ldA, X, ldX );
	f0 = fX;
    cond = ( fX > tol * f0 );

    while ( cond )
    {
        iter++;
        if ( linsear == 0 )
        {
            // .. Nelder-Mead
            t = nelder_mead_curve( curve, tol_linsear, m, n, fX, Ag, n, A, ldA, X, ldX, H, n, eAt, n );
		}
        else
        {
            // .. Armijo's backtracking
            t = armijo_backtracking_curve( curve, m, n, fX, FX, n, H, n, Ag, n, A, ldA, X, ldX, eAt, n );
        }

		// .. If t=0 switch to -gradient.
		if ( t == 0 )
		{
			scaling_of_matrix( n, -1, G, n, H, n );
			if ( linsear == 0 )
        	{
            	// .. Nelder-Mead
            	t = nelder_mead_curve( curve, tol_linsear, m, n, fX, Ag, n, A, ldA, X, ldX, H, n, eAt, n );
			}
        	else
        	{
            	// .. Armijo's backtracking
            	t = armijo_backtracking_curve( curve, m, n, fX, FX, n, H, n, Ag, n, A, ldA, X, ldX, eAt, n );
        	}

		}

        if ( curve == 0 )
        {
            if ( conj > 0 )
            {
                // .. Old approximation on manifold is in X0.
                dlacpy( "A", &n, &n, X, &ldX, X0, &n );
            }
            dgemm( "N", "N", &n, &n, &n, &dONE, X, &ldX, eAt, &n, &dZERO, W2, &n );
            dlacpy( "A", &n, &n, W2, &n, X, &ldX );
            dgemm( "N", "N", &n, &n, &n, &dONE, X, &n, Ag, &n, &dZERO, Xi, &n );
        }
        else
        {
            dlacpy( "A", &n, &n, eAt, &n, X, &n );
            // .. Computing Xi = X*(X'*H-H'*Q)/2;
            dgemm( "T", "N", &n, &n, &n, &dONE, X, &ldX, H, &n, &dZERO, Xi, &n );
            lin_comb_trans( n, 0.5, -0.5, Xi, n, W2, n );
            dgemm( "N", "N", &n, &n, &n, &dONE, X, &ldX, W2, &n, &dZERO, Xi, &n );
        }

        Fn_X( m, n, A, ldA, X, ldX, FX, n );

		fX=Fn( m, n, A, ldA, X, ldX );

        cond = ( fX > tol * f0 ) && (iter<1000); 

        if ( !cond )
		{
            break;
		}

        // .. FX'*X is now in W.
        dgemm( "T", "N", &n, &n, &n, &dONE, FX, &n, X, &ldX, &dZERO, W, &n ); 
        if ( conj > 0 )
		{
            // .. Old gradient on manifold is in G0.
            dlacpy( "A", &n, &n, G, &n, G0, &n );
        }
        dlacpy( "A", &n, &n, FX, &n, G, &n );
        dgemm( "N", "N", &n, &n, &n, &dmHALF, X, &ldX, W, &n, &dHALF, G, &n );

        if ( conj == 0 )
        {
            // .. Exact conjugacy
            lin_comb_trans( n, 1, 1, W, n, W1, n );
            dgemm( "N", "T", &n, &n, &n, &dONE, W1, &n, Xi, &n, &dZERO, W2, &n );
           
            FXX1 = Fn_XX( m, n, A, ldA, X, ldX, G, n, Xi, n );
            FXX2 = Fn_XX( m, n, A, ldA, X, ldX, Xi, n, Xi, n );
            
            dgemm( "N", "N", &n, &n, &n, &dONE, W2, &n, G, &n, &dZERO, W1, &n );
            Hess1 = FXX1 - trace( n, W1, n ) / 2;
            dgemm( "N", "N", &n, &n, &n, &dONE, W2, &n, Xi, &n, &dZERO, W1, &n );
            Hess2 = FXX2 - trace( n, W1, n ) / 2;

            beta = Hess1 / Hess2;
        }

        if ( conj == 1 )
        {
            // .. Polak-Ribiere
            if ( curve == 0 )
            {
                ss_alg_exp(t/2, n, Ag, n, W, n );
                dgemm( "N", "N", &n, &n, &n, &dONE, X0, &n, W, &n, &dZERO, W1, &n );
                dgemm( "N", "T", &n, &n, &n, &dONE, W1, &n, X0, &n, &dZERO, W2, &n );
                dgemm( "N", "N", &n, &n, &n, &dONE, W2, &n, G0, &n, &dZERO, W1, &n );
                dgemm( "N", "N", &n, &n, &n, &dONE, W1, &n, W, &n, &dZERO, W2, &n );
            }
            else
            {
                dgemm( "T", "N", &n, &n, &n, &dONE, X, &ldX, G0, &n, &dZERO, W, &n );
                lin_comb_trans( n, 0.5, -0.5, W, n, W1, n );
                dgemm( "N", "N", &n, &n, &n, &dONE, X, &ldX, W1, &n, &dZERO, W2, &n );
            }
            // .. Translated old gradient G0 is now in W2
            lin_comb_of_2matrices( n, 1, G, n, -1, W2, n, W1, n );

            dgemm( "T", "N", &n, &n, &n, &dONE, W1, &n, G, &n, &dZERO, W2, &n );
            Hess1 = trace( n, W2, n );
            dgemm( "T", "N", &n, &n, &n, &dONE, G0, &n, G0, &n, &dZERO, W2, &n );
            Hess2 = trace( n, W2, n );

            beta = Hess1 / Hess2;
        }

        if ( conj == 2 )
        {
            // .. Fletcher-Reeves
            dgemm( "T", "N", &n, &n, &n, &dONE, G, &n, G, &n, &dZERO, W2, &n );
            Hess1 = trace( n, W2, n );
            dgemm( "T", "N", &n, &n, &n, &dONE, G0, &n, G0, &n, &dZERO, W2, &n );
            Hess2 = trace( n, W2, n );

            beta = Hess1 / Hess2;
        }

        if ( iter % (n*(n-1)/2) == 0 )
		{
            scaling_of_matrix( n, -1, G, n, H, n );
		}
        else
		{
            lin_comb_of_2matrices( n, -1, G, n, beta, Xi, n, H, n );
		}

		if ( curve == 0 )
    	{
        	dgemm( "T", "N", &n, &n, &n, &dONE, X, &ldX, H, &n, &dZERO, Ag, &n );
		}
    }

    free(Ag);
    free(eAt);
    free(FX);
    free(G);
    free(G0);
    free(H);
    free(X0);
    free(Xi);
	free(W);
    free(W1);
    free(W2);

	return iter;

}
