// Original implementation of Riemannian conjugate gradient method on the oblique manifold for joint approximate diagonalization.
// This file contains the main function implementing the method, and other auxiliary functions.
//
// The main function call has the form:
// int cg_jad_oblique_var_org( int curve, int linsear, int conj, int m, int n, double tol, double **A, int *ldA, double *X, int ldX );
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

    for( j = 0; j < n; j++ )
    {
        for( i = 0; i < n; i++ )
        {
            R[i+j*ldR] = alpha * A[i+j*ldA] + beta * B[i+j*ldB];
        }
    }
}

static void scaling_of_matrix( int n, double alpha, double *A, int ldA, double *R, int ldR )
{
    int i, j;

    for( j = 0; j < n; j++ )
    {
        for( i = 0; i < n; i++ )
        {
            R[i+j*ldR] = alpha * A[i+j*ldA];
        }
    }
}

// .. Function for computing diagonal of a matrix product diag(A^T*B), stored as a vector.
static void diagonal_of_transpose_product( int n, double *A, int ldA, double *B, int ldB, double *C )
{
    int incx = 1;
    int j;

    for ( j = 0; j < n; j++ )
	{
        C[j] = ddot( &n, &A[j*ldA], &incx, &B[j*ldB], &incx );
    }
}

// .. Function for computing a product of a matrix A with a diagonal matrix C stored in a vector, R = E + alpha * A * C.
static void product_of_matrix_and_diagonal( int n, double *E, int ldE, double alpha, double *A, int ldA, double *C, double *R, int ldR )
{
    int i, j;

    for ( j = 0; j < n; j++ )
    {
        for ( i = 0; i < n; i++ )
        {
            R[i+j*ldR] = E[i+j*ldE] + alpha * A[i+j*ldA] * C[j];
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

static void new_geod_X( double t, int n, double *X, int ldX, double *H, int ldH )
{
    int i, incx = 1, j;
    double l;

    for ( int j = 0; j < n; j++ )
    {
        l = dnrm2( &n, &H[j*ldH], &incx );
        for ( i = 0; i < n; i++ )
        {
            X[i+j*ldX] = X[i+j*ldX] * cos(l*t) + H[i+j*ldH] * sin(l*t) / l;
        }
    }
}

static void geod( double t, int n, double *X, int ldX, double *H, int ldH, double *Geod, int ldGeod )
{
    int i, incx = 1, j;
    double l;

    for ( int j = 0; j < n; j++ )
    {
        l = dnrm2( &n, &H[j*ldH], &incx );
        for ( i = 0; i < n; i++ )
        {
            Geod[i+j*ldGeod] = X[i+j*ldX] * cos(l*t) + H[i+j*ldH] * sin(l*t) / l;
        }
    }
}

static void dgeod( double t, int n, double *X, int ldX, double *H, int ldH, double *dGeod, int lddGeod )
{
    int i, incx = 1, j;
    double l;

    for ( int j = 0; j < n; j++ )
    {
        l = dnrm2( &n, &H[j*ldH], &incx );
        for ( i = 0; i < n; i++ )
        {
            dGeod[i+j*lddGeod] = -X[i+j*ldX] * sin(l*t) * l + H[i+j*ldH] * cos(l*t);
        }
    }
}

static void new_retr_X( double t, int n, double *X, int ldX, double *H, int ldH, double *iN )
{
    int i, incx = 1, j;
    double *W;

	W = (double *) malloc(n*n*sizeof(double));
    
    lin_comb_of_2matrices( n, 1.0, X, ldX, t, H, ldH, W, n );
    dlacpy( "A", &n, &n, W, &n, X, &ldX );

    for ( int j = 0; j < n; j++ )
    {
        iN[j] = 1 / dnrm2( &n, &X[j*ldH], &incx );
        dscal( &n, &iN[j], X+j*n, &incx );	  
    }

	free(W);
}

static void retr( double t, int n, double *X, int ldX, double *H, int ldH, double *Retr, int ldRetr )
{
    int i, incx = 1, j;
    double col_norm;
    
    lin_comb_of_2matrices( n, 1.0, X, ldX, t, H, ldH, Retr, ldRetr );

    for ( int j = 0; j < n; j++ )
    {
        col_norm = 1 / dnrm2( &n, &Retr[j*ldH], &incx );
        dscal( &n, &col_norm, Retr+j*n, &incx );	  
    }
}

static double Fn_geod( double t, int m, int n, double **A, int *ldA, double *X, int ldX, double *H, int ldH )
{
    int p;
    double y;
    double dZERO=0.0, dONE=1.0;
	double *B, *Geod, *W;

    Geod = (double*) malloc(n*n*sizeof(double));

	B = (double*) malloc(n*n*sizeof(double));
    W = (double*) malloc(n*n*sizeof(double));

    geod( t, n, X, ldX, H, ldH, Geod, n );

    y = 0;

	for( p = 0; p < m; p++ )
    {
        dgemm( "N", "N", &n, &n, &n, &dONE, A[p], &ldA[p], Geod, &n, &dZERO, W, &n );
        dgemm( "T", "N", &n, &n, &n, &dONE, Geod, &n, W, &n, &dZERO, B, &n );
        Off( n, B, n );
        y += pow(dlange( "F", &n, &n, B, &n, W ),2);
    }

    y *= 0.5;

	free(B);
    free(W);
    free(Geod);

    return y;
}

static double Fn_retr( double t, int m, int n, double **A, int *ldA, double *X, int ldX, double *H, int ldH )
{
    int i, info, n2=n*n, p;
	int iONE=1;
    double col_norm, y;
    double dZERO=0.0, dONE=1.0;
    double *B,  *W1, *W2;

    B = (double*) malloc(n*n*sizeof(double));
    W1 = (double*) malloc(n*n*sizeof(double));
    W2 = (double*) malloc(n*n*sizeof(double));
    
    retr( t, n, X, ldX, H, ldH, W1, n );
    
    y = 0;

	for( p = 0; p < m; p++ )
    {
        dgemm( "N", "N", &n, &n, &n, &dONE, A[p], &ldA[p], W1, &n, &dZERO, W2, &n );
        dgemm( "T", "N", &n, &n, &n, &dONE, W1, &n, W2, &n, &dZERO, B, &n );
        Off( n, B, n );
        y += pow(dlange( "F", &n, &n, B, &n, W1 ),2);
    }

    y *= 0.5;

    free(B);
	free(W1);
    free(W2);

    return y;
}

static double nelder_mead_curve( int curve, double tol, int m, int n, double fX, double **A, int *ldA, double *X, int ldX, double *H, int ldH )
{
    int fun_eval, nt, shrink;
    double alpha, beta, c, dONE, dZERO, gamma, delta, F0, F1, Fc, Fe, Fr, pomd, t0, t0_old, t1, t1_old, tc, te, tr;

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
        F1 = Fn_geod( t1, m, n, A, ldA, X, ldX, H, ldH );
    }
    else
    {
        F1 = Fn_retr( t1, m, n, A, ldA, X, ldX, H, ldH );
    }
    fun_eval = 1;

    while ( (fabs(F1-F0)/fabs(F0) > tol) && (fabs(F0) > tol) && (fabs(F1) > tol) && (fabs(t1-t0)>tol) && (fun_eval<=100) )
    {
		// .. In case of negative argument; restart
		if ((t0<0) || (t1<0))
		{
			t0 = t0_old;
			t1_old /= 10;
			t1 = t1_old;
			F0 = fX;
    		if ( curve == 0 )
   			{
        		F1 = Fn_geod( t1, m, n, A, ldA, X, ldX, H, ldH );
    		}
    		else
    		{
        		F1 = Fn_retr( t1, m, n, A, ldA, X, ldX, H, ldH );
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
        }

        c = t0;

        tr = c + alpha * (c - t1);
        if ( curve == 0 )
        {
            Fr = Fn_geod( tr, m, n, A, ldA, X, ldX, H, ldH );
        }
        else
        {
            Fr = Fn_retr( tr, m, n, A, ldA, X, ldX, H, ldH );
        }
        fun_eval++;

        // .. Reflect will never happen.

        // .. Expand
        if ( Fr < F0 )
        {
            te = c + gamma * (tr - c);
            if ( curve == 0 )
            {
                Fe = Fn_geod( te, m, n, A, ldA, X, ldX, H, ldH );
            }
            else
            {
                Fe = Fn_retr( te, m, n, A, ldA, X, ldX, H, ldH );
            }
            fun_eval++;
            if ( Fe < Fr )
            {
                t1 = te;
                F1 = Fe;
            }
            else
            {
                t1 = tr;
                F1 = Fr;
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
                    Fc = Fn_geod( tc, m, n, A, ldA, X, ldX, H, ldH );
                }
                else
                {
                    Fc = Fn_retr( tc, m, n, A, ldA, X, ldX, H, ldH );
                }
                fun_eval++;
                if ( Fc < Fr )
                {
                    t1 = tc;
                    F1 = Fc;
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
                    Fc = Fn_geod( tc, m, n, A, ldA, X, ldX, H, ldH );
                }
                else
                {
                    Fc = Fn_retr( tc, m, n, A, ldA, X, ldX, H, ldH );
                }
                fun_eval++;
                if ( Fc < F1 )
                {
                    t1 = tc;
                    F1 = Fc;
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
                F1 = Fn_geod( t1, m, n, A, ldA, X, ldX, H, ldH );
            }
            else
            {
                F1 = Fn_retr( t1, m, n, A, ldA, X, ldX, H, ldH );
            }
            fun_eval++;
        }
    }

    if ( F0 <= F1 )
    {
        return t0;
    }
    else
    {
        return t1;
    }

}

double armijo_backtracking_curve( int curve, int m, int n, double F0, double *FX, int ldFX, double *H, int ldH, double **A, int *ldA, double *X, int ldX )
{
    int func_eval, max_steps;
    double alpha, beta, dF_curveX, F1, nH, sigma, *W;
    double dZERO=0.0, dONE=1.0;
    
    W = (double*) malloc(n*n*sizeof(double));

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
        F1 = Fn_geod( alpha, m, n, A, ldA, X, ldX, H, ldH );
    }
    else
    {
        F1 = Fn_retr( alpha, m, n, A, ldA, X, ldX, H, ldH );
    }
    func_eval = 1;

    while ( F1 > F0 + sigma * alpha * dF_curveX )
    {
        alpha = beta * alpha;
        if ( curve == 0 )
        {
            F1 = Fn_geod( alpha, m, n, A, ldA, X, ldX, H, ldH );
        }
        else
        {
            F1 = Fn_retr( alpha, m, n, A, ldA, X, ldX, H, ldH );
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
    }

    free(W);
    return alpha;
}

int cg_jad_oblique_var_org( int curve, int linsear, int conj, int m, int n, double tol, double **A, int *ldA, double *X, int ldX )
{
    int i, iter;
    int incx = 1;
    double tol_linsear = n*dlamch( "E" );
    double dZERO=0.0, dONE=1.0, dHALF = 0.5, dmHALF = -0.5, dTWO = 2.0;
    double beta, cond, f0, fX, FXX1, FXX2, Hess1, Hess2, res, ro, ro0, t;
    double *C1, *C2, *FX, *G, *G0, *H, *I, *X0, *Xi, *W, *W1;

    C1 = (double*) malloc(n*sizeof(double));
    C2 = (double*) malloc(n*sizeof(double));
    FX = (double*) malloc(n*n*sizeof(double));
    G = (double*) malloc(n*n*sizeof(double));
    G0 = (double*) malloc(n*n*sizeof(double));
	H = (double*) malloc(n*n*sizeof(double));
    I = (double*) malloc(n*sizeof(double));
    X0 = (double*) malloc(n*n*sizeof(double));
    Xi = (double*) malloc(n*n*sizeof(double));
    W = (double*) malloc(n*n*sizeof(double));
    W1 = (double*) malloc(n*n*sizeof(double));

    Fn_X( m, n, A, ldA, X, ldX, FX, n );

    diagonal_of_transpose_product( n, X, ldX, FX, n, I );
    product_of_matrix_and_diagonal( n, FX, n, -1, X, ldX, I, G, n );

    scaling_of_matrix( n, -1, G, n, H, n );

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
            t = nelder_mead_curve( curve, tol_linsear, m, n, fX, A, ldA, X, ldX, H, n );
		}
        else
        {
            // .. Armijo's backtracking
            t = armijo_backtracking_curve( curve, m, n, fX, FX, n, H, n, A, ldA, X, ldX );
        }

		// .. If t=0 switch to -gradient.
		if ( t == 0 )
		{
			scaling_of_matrix( n, -1, G, n, H, n );
			if ( linsear == 0 )
        	{
            	// .. Nelder-Mead
            	t = nelder_mead_curve( curve, tol_linsear, m, n, fX, A, ldA, X, ldX, H, n );
			}
        	else
        	{
            	// .. Armijo's backtracking
            	t = armijo_backtracking_curve( curve, m, n, fX, FX, n, H, n, A, ldA, X, ldX );
        	}

		}
       
        if ( curve == 0 )
        {
            if ( conj == 1 )
            {
                // .. Old approximation on manifold is in X0.
                dlacpy( "A", &n, &n, X, &ldX, X0, &n );
            }
            dgeod( t, n, X, n, H, n, Xi, n );
            new_geod_X( t, n, X, n, H, n );
        }
        else
        {
            // .. Diagonal matrix with inverses of column norms of X+t*H is in C1.
            new_retr_X( t, n, X, ldX, H, n, C1 );
            // .. Computing Xi = H-t*new_X*diag(H(:,i)'*H(:,i)/norm((X+t*H)(:,i))).
            diagonal_of_transpose_product( n, H, n, H, n, I );
            for ( i = 0; i < n; i++ )
            {
                I[i] = I[i] * C1[i];
            }
            product_of_matrix_and_diagonal( n, H, n, -t, X, n, I, Xi, n );
        }       

        Fn_X( m, n, A, ldA, X, ldX, FX, n );

		fX=Fn( m, n, A, ldA, X, ldX );

        cond = ( fX > tol * f0 ) && (iter<1000); 
        
        if ( !cond )
		{
            break;
		}

        // .. diag(X'*FX) is now in I.
        diagonal_of_transpose_product( n, X, ldX, FX, n, I );
        if ( conj > 0 )
		{
            // .. Old gradient on manifold is in G0.
            dlacpy( "A", &n, &n, G, &n, G0, &n );
        }
        product_of_matrix_and_diagonal( n, FX, n, -1, X, ldX, I, G, n );

        if ( conj == 0 )
        {
            // .. Exact conjugacy
            diagonal_of_transpose_product( n, G, n, Xi, n, C1 );
            diagonal_of_transpose_product( n, Xi, n, Xi, n, C2 );

            FXX1 = Fn_XX( m, n, A, ldA, X, ldX, G, n, Xi, n );
            FXX2 = Fn_XX( m, n, A, ldA, X, ldX, Xi, n, Xi, n );

            res = ddot( &n, I, &incx, C1, &incx );
            Hess1 = FXX1 - res;

            res = ddot( &n, I, &incx, C2, &incx );
            Hess2 = FXX2 - res;

            beta = Hess1 / Hess2;
        }
        
        if ( conj == 1 )
        {
            // .. Polak-Ribiere
            if ( curve == 0 )
            {
				diagonal_of_transpose_product( n, H, n, G0, n, I );
                for ( i = 0; i < n; i++)
                {
                    C1[i] = dnrm2( &n, &H[i*n], &incx );
                    C2[i] = sin(C1[i]*t)*I[i]/C1[i];
                }           
                product_of_matrix_and_diagonal( n, G0, n, -1, X0, n, C2, W, n );
                for ( i = 0; i < n; i++)
                {
                    C2[i] = (cos(C1[i]*t)-1)*I[i]/(C1[i]*C1[i]);
                }
                product_of_matrix_and_diagonal( n, W, n, 1, H, n, C2, W1, n );
            }
            else
            {
                // .. Computing Xi = G0-t*new_X*diag(H(:,i)'*G0(:,i)/norm((X+t*H)(:,i))).
                diagonal_of_transpose_product( n, H, n, G0, n, I );
				// .. Diagonal matrix with inverses of column norms of X+t*H is still in C1.
				for ( i = 0; i < n; i++ )
            	{
                	I[i] = I[i] * C1[i];
            	}
            	product_of_matrix_and_diagonal( n, G0, n, -t, X, n, I, W1, n );
            }
            // .. Translated old gradient G0 is now in W1
            lin_comb_of_2matrices( n, 1, G, n, -1, W1, n, W, n );

            dgemm( "T", "N", &n, &n, &n, &dONE, W, &n, G, &n, &dZERO, W1, &n );
            Hess1 = trace( n, W1, n );
            dgemm( "T", "N", &n, &n, &n, &dONE, G0, &n, G0, &n, &dZERO, W1, &n );
            Hess2 = trace( n, W1, n );

            beta = Hess1 / Hess2;
        }
        
        if ( conj == 2 )
        {
            // .. Fletcher-Reeves
            dgemm( "T", "N", &n, &n, &n, &dONE, G, &n, G, &n, &dZERO, W1, &n );
            Hess1 = trace( n, W1, n );
            dgemm( "T", "N", &n, &n, &n, &dONE, G0, &n, G0, &n, &dZERO, W1, &n );
            Hess2 = trace( n, W1, n );

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
    }

    free(C1);
    free(C2);
    free(FX);
    free(G);
    free(G0);
    free(H);
    free(I);
    free(X0);
    free(Xi);
    free(W);
    free(W1);

	return iter;

}
