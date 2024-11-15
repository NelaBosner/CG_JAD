// Parallel implementation of Riemannian conjugate gradient method on the Stiefel manifold for joint approximate diagonalization.
// This file contains the main function implementing the method, and other auxiliary functions.
//
// The main function call has the form:
// int cg_jad_stiefel_var_par( int curve, int linsear, int conj, int m, int n, int mklt_ddots, int t_ddots, int mklt_gemms, int t_gemms, double tol, double **A, int *ldA, double *X, int ldX );
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
// mklt_dots : number of MKL threads for computing dot products 
// t_ddots : number of threads calling MKL dot product routine
// mklt_gemms : number of MKL threads for computing matrix-matrix products
// t_gemms : number of threads calling MKL matrix-matrix product routine
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
#include <pthread.h>

static int m, n, ldX, ldAg, ldAg2, ldAg4, ldAg6;
static int *ipiv, *ldA, *ldAX, *ldB;
static double n1Ag;
static double dZERO=0.0, dONE=1.0, dmONE=-1.0, dHALF=0.5, dmHALF=-0.5, dTWO=2.0;
static double *loc_sum;
static double **A, *Ag, *Ag2, *Ag4, *Ag6, *Ag8, **AX, **B, **D, *eAt, *FX, *G, *G0, *H, *I, *P, *Q, *Up, *Vp, *W, *W1, *W2, *W3, *W4, **WW1, **WW2, *X, *X0, *Xi;
static double b3[] = { 120, 60, 12, 1 };
static double b5[] = { 30240, 15120, 3360, 420, 30, 1 };
static double b7[] = { 17297280, 8648640,1995840, 277200, 25200,1512, 56, 1 };
static double b9[] = { 17643225600, 8821612800, 2075673600, 302702400, 30270240, 2162160, 110880, 3960, 90, 1 };
static double b13[] = { 64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800, 129060195264000, 10559470521600, 670442572800, 33522128640, 1323241920, 40840800, 960960, 16380, 182, 1 };
static double thetam[] = { 1.5e-2, 2.5e-1, 9.5e-1, 2.1e0 };

// .. Parameters for MKL (BLAS).
static int num_mkl_threads_ddots;
static int nt_ddots;
static int num_mkl_threads_gemms;
static int nt_gemms;

// .. Number of threads.
static int max_num_threads = 23; // Number of physical procesors minus one. Master thread is not included. 

// .. Threads.
static pthread_t *threads;

static int stopped;

// .. Synchronization structure.
struct barrier_s
{
    pthread_mutex_t lock;
    pthread_cond_t cond;
    int limit;
    int count;
};
typedef struct barrier_s barrier_t;
static barrier_t *barrier;

// .. Functions on the synchronization structure.
static barrier_t *barrier_new(int limit)
{
    barrier_t *barrier = (barrier_t*) malloc(sizeof(barrier_t));
    pthread_mutex_init(&barrier->lock, 0);
    pthread_cond_init(&barrier->cond, 0);
    barrier->limit = limit;
    barrier->count = 0;
    return barrier;
}

static void barrier_delete(barrier_t *barrier)
{
    pthread_cond_destroy(&barrier->cond);
    pthread_mutex_destroy(&barrier->lock);
    free(barrier);
}


static void barrier_wait(barrier_t *barrier)
{
    pthread_mutex_lock(&barrier->lock);
    barrier->count += 1;
    if (barrier->count == barrier->limit)
    {
        barrier->count = 0;
        pthread_cond_broadcast(&barrier->cond);
    }
    else
    {
        pthread_cond_wait(&barrier->cond, &barrier->lock);
    }
    pthread_mutex_unlock(&barrier->lock);
}

// .. Working function for the thread function.
typedef void (*working_function_t)(int, int, void*);
static working_function_t function;

// .. Argument oF the thread function.
static void* argument;
struct thread_data
{
	int rank;
};
static struct thread_data **wd;

// .. Waiting thread function.
static void *waiting_thread_function(void *arg)
{
    struct thread_data *thread_data = (struct thread_data*) arg;

    const int rank = thread_data->rank;

    for (;;)
    {
        barrier_wait(barrier);

        if (stopped)
        {
            break;
        }
        else
        {
            function(max_num_threads,
                     rank,
                     argument);

            barrier_wait(barrier);
        }
    }

    return NULL;
}

static int imin( int a, int b )
{
    return (a<=b?a:b);
}

// .. Data structure and function for partitioning the work on n objects over nt threads.
typedef struct partition_limts_s
{
    int start;
    int end;
} partition_limits_t;

// .. The function is computing limiting indices for thread rank i.
static partition_limits_t partition( int n, int nt, int i )
{
    int small = n / nt;
    int large = small + 1;
    int num_large = n % nt;
    int this_size = (i < num_large ? large : small);
    partition_limits_t limits;
    limits.start = (i <= num_large ? i*large : num_large*large+(i-num_large)*small );
    limits.end = limits.start + this_size - 1;
    return limits;
}

//
// .. Master is working on the first block of objects, with rank = 0, hence total number of threads is equal to nt+1.
//

// .. Argument structure for computing off(dA[p]).
typedef struct off_context_s
{
    int nt;
	int m;
	int n;
    double **A;
} off_context_t;
static off_context_t *off_context;

// .. Working function for computing off(dA[p]).
static void off( int numthreads, int rank, void *arg ) // #objects_for_parallelization = m*n;
{
	// .. Extract context argument.
	off_context_t *context = (off_context_t*)arg;

	// .. Extract variables from the context struct.
    int nt = context->nt;
	int m = context->m;
    int n = context->n;
    double **A = context->A;

    int i, k, p;

    if ( rank < nt )
    {
        partition_limits_t limits = partition( m*n, nt+1, rank+1 );

        for ( k = limits.start; k <= limits.end; k++ )
        {
			p = k / n;
			i = k % n;
            A[p][i+i*n] = 0;
        }
    }
}

// .. Argument structure for removing diagonals of A[p].
typedef struct remove_diagonals_context_s
{
    int nt;
	int m;
	int n;
    double **A;
    double **D;
} remove_diagonals_context_t;
static remove_diagonals_context_t *remove_diagonals_context;

// .. Working function for removing diagonals of dA[p], and storing them as vectors.
static void remove_diagonals( int numthreads, int rank, void *arg ) // #objects_for_parallelization = m*n;
{
	// .. Extract context argument.
	remove_diagonals_context_t *context = (remove_diagonals_context_t*)arg;

	// .. Extract variables from the context struct.
    int nt = context->nt;
	int m = context->m;
    int n = context->n;
    double **A = context->A;
    double **D = context->D;

    int i, k, p;

    if ( rank < nt )
    {
        partition_limits_t limits = partition( m*n, nt+1, rank+1 );

        for ( k = limits.start; k <= limits.end; k++ )
        {
			p = k / n;
			i = k % n;
            D[p][i] = A[p][i+i*n];
            A[p][i+i*n] = 0;
        }
    }
}

// .. Working function for returning diagonals of dA[p].
static void return_diagonals( int numthreads, int rank, void *arg ) // #objects_for_parallelization = m*n;
{
	// .. Extract context argument.
	remove_diagonals_context_t *context = (remove_diagonals_context_t*)arg;

	// .. Extract variables from the context struct.
    int nt = context->nt;
	int m = context->m;
    int n = context->n;
    double **A = context->A;
    double **D = context->D;

    int i, k, p;

    if ( rank < nt )
    {
        partition_limits_t limits = partition( m*n, nt+1, rank+1 );

        for ( k = limits.start; k <= limits.end; k++ )
        {
			p = k / n;
			i = k % n;
            A[p][i+i*n] = D[p][i];
        }
    }
}

// .. Argument structure for scaling of a matrix.
typedef struct scaling_of_matrix_context_s
{
    int nt;
	int n;
    double alpha;
    double *A;
    int ldA;
    double *R;
    int ldR;
} scaling_of_matrix_context_t;
static scaling_of_matrix_context_t *scaling_of_matrix_context;

// .. Working function for scaling of a matrix.
static void scaling_of_matrix( int numthreads, int rank, void *arg ) // #objects_for_parallelization = n*n
{
	// .. Extract context argument.
	scaling_of_matrix_context_t *context = (scaling_of_matrix_context_t*)arg;

	// .. Extract variables from the context struct.
    int nt = context->nt;
    int n = context->n;
    double alpha = context->alpha;
    double *A = context->A;
    int ldA = context->ldA;
    double *R = context->R;
    int ldR = context->ldR;

    int i, j, l;

    if ( rank < nt )
    {
        partition_limits_t limits = partition( n*n, nt+1, rank+1 );

        for( l = limits.start; l <= limits.end; l++ )
        {
            i = l % n;
            j = l / n;
            R[i+j*ldR] = alpha * A[i+j*ldA];
        }
    }
}

// .. Argument structure for linear combination of two matrices.
typedef struct lin_comb_of_2matrices_context_s
{
    int nt;
	int n;
    double alpha;
    double *A;
    int ldA;
    double beta;
    double *B;
    int ldB;
    double *R;
    int ldR;
} lin_comb_of_2matrices_context_t;
static lin_comb_of_2matrices_context_t *lin_comb_of_2matrices_context;

// .. Working function for computing a linear combination of two matrices.
static void lin_comb_of_2matrices( int numthreads, int rank, void *arg ) // #objects_for_parallelization = n*n
{
	// .. Extract context argument.
	lin_comb_of_2matrices_context_t *context = (lin_comb_of_2matrices_context_t*)arg;

	// .. Extract variables from the context struct.
    int nt = context->nt;
    int n = context->n;
    double alpha = context->alpha;
    double *A = context->A;
    int ldA = context->ldA;
    double beta = context->beta;
    double *B = context->B;
    int ldB = context->ldB;
    double *R = context->R;
    int ldR = context->ldR;

    int i, j, l;

    if ( rank < nt )
    {
        partition_limits_t limits = partition( n*n, nt+1, rank+1 );

        for( l = limits.start; l <= limits.end; l++ )
        {
            i = l % n;
            j = l / n;
            R[i+j*ldR] = alpha * A[i+j*ldA] + beta * B[i+j*ldB];
        }
    }
}

// .. Argument structure for computing R=alpha*A+beta*A'.
typedef struct lin_comb_trans_context_s
{
    int nt;
	int n;
    double alpha;
    double beta;
    double *A;
    int ldA;
    double *R;
    int ldR;
} lin_comb_trans_context_t;
static lin_comb_trans_context_t *lin_comb_trans_context;

// .. Working function for computing R=alpha*A+beta*A'2.
static void lin_comb_trans( int numthreads, int rank, void *arg ) // #objects_for_parallelization = n*n
{
	// .. Extract context argument.
	lin_comb_trans_context_t *context = (lin_comb_trans_context_t*)arg;

	// .. Extract variables from the context struct.
    int nt = context->nt;
    int n = context->n;
    double alpha = context->alpha;
    double beta = context->beta;
    double *A = context->A;
    int ldA = context->ldA;
    double *R = context->R;
    int ldR = context->ldR;

    int i, j, l;

    if ( rank < nt )
    {
        partition_limits_t limits = partition( n*n, nt+1, rank+1 );

        for( l = limits.start; l <= limits.end; l++ )
        {
            i = l % n;
            j = l / n;
            R[i+j*ldR] = alpha * A[i+j*ldA] + beta * A[j+i*ldA];
        }
    }
}

// .. Argument structure for summing m matrices.
typedef struct sum_of_matrices_context_s
{
    int nt;
    int m;
	int n;
    double **A;
    int *ldA;
    double *S;
    int ldS;
} sum_of_matrices_context_t;
static sum_of_matrices_context_t *sum_of_matrices_context;

// .. Working function for summing m matrices dA[p].
static void sum_of_matrices( int numthreads, int rank, void *arg ) // #objects_for_parallelization = n*n
{
	// .. Extract context argument.
	sum_of_matrices_context_t *context = (sum_of_matrices_context_t*)arg;

	// .. Extract variables from the context struct.
    int nt = context->nt;
    int m = context->m;
    int n = context->n;
    double **A = context->A;
    int *ldA = context->ldA;
    double *S = context->S;
    int ldS = context->ldS;

    int i, j, l, p;

    if ( rank < nt )
    {
        partition_limits_t limits = partition( n*n, nt+1, rank+1 );

        for( l = limits.start; l <= limits.end; l++ )
        {
            i = l % n;
            j = l / n;
            S[i+j*ldS] = 0;

            for ( p = 0; p < m; p++ )
            {
                S[i+j*ldS] += A[p][i+j*ldA[p]];
            }
        }
    }
}

// .. Argument structure for computing sum_p trace(A[p]' * B[p]) or sum_p trace(A[p] * B[p])
typedef struct traces_by_ddots_context_s
{
	int nt;
    char trans;
	int m;
	int n;
    double **A;
    int *ldA;
    double **B;
    int *ldB;
	double *loc_sum;

} traces_by_ddots_context_t;
static traces_by_ddots_context_t *traces_by_ddots_context;

// .. Working function for computing sum_p trace(A[p]' * B[p]) or sum_p trace(A[p] * B[p])
static void traces_by_ddots( int numthreads, int rank, void *arg ) // #objects_for_parallelization = m (trans='T') or m*n (trans='N')
{
	// .. Extract context argument.
	traces_by_ddots_context_t *context = (traces_by_ddots_context_t*)arg;

	// .. Extract variables from the context struct.
	int nt = context->nt;
    char trans = context->trans;
    int m = context->m;
    int n = context->n;
    double **A = context->A;
    int *ldA = context->ldA;
    double **B = context->B;
    int *ldB = context->ldB;
	double *loc_sum = context->loc_sum;

	partition_limits_t limits;
	int incy = 1;
	int n2 = n * n;
	int i, k, p;
	double sum = 0;

	if ( rank < nt )
	{
        if ( trans == 'T' ) // #objects_for_parallelization = m -------> one matrix one ddot
        {
            limits = partition( m, nt+1, rank+1 );
            for( p = limits.start; p <= limits.end; p++ )
            {
                sum += ddot( &n2, A[p], &incy, B[p], &incy );
            }
        }
        else // #objects_for_parallelization = m*n ------> one matrix n ddots
        {
            limits = partition( m*n, nt+1, rank+1 );
            for ( k = limits.start; k <= limits.end; k++ )
            {
                p = k / n;
                i = k % n;
                sum += ddot( &n, &(A[p][i]), &ldA[p], &(B[p][i*ldB[p]]), &incy);
            }
        }

        loc_sum[rank+1] = sum;
	}
}

// .. Argument structure for multiple gemms of type (1) B[p] = scal * A[p] * X, (2) B[p] = scal * X * A[p], or (12) C[p] = scal * A[p] * B[p].
typedef struct multiple_gemms_context_s
{
    int nt;
    char trans1;
    char trans2;
    int multiple_arg;
	int m;
	int n;
    double scal;
    double **A;
    int *ldA;
    double **B;
    int *ldB;
    double *X;
    int ldX;
    double **C;
    int *ldC;
} multiple_gemms_context_t;
static multiple_gemms_context_t *multiple_gemms_context;

// .. Working function for multiple gemms of type (1) C[p] = scal * A[p] * X, (2) C[p] = scal * X * A[p], or (12) C[p] = scal * A[p] * B[p].
static void multiple_gemms( int numthreads, int rank, void *arg ) // #objects_for_parallelization = m
{
	// .. Extract context argument.
	multiple_gemms_context_t *context = (multiple_gemms_context_t*)arg;

	// .. Extract variables from the context struct.
    int nt = context->nt;
    char trans1 = context->trans1;
    char trans2 = context->trans2;
    int multiple_arg = context->multiple_arg;
    int m = context->m;
    int n = context->n;
    double scal = context->scal;
    double **A = context->A;
    int *ldA = context->ldA;
    double **B = context->B;
    int *ldB = context->ldB;
    double *X = context->X;
    int ldX = context->ldX;
    double **C = context->C;
    int *ldC = context->ldC;

    int p;

    if ( rank < nt )
    {
        partition_limits_t limits = partition( m, nt+1, rank+1 );

        for( p = limits.start; p <= limits.end; p++ )
        {
            if ( multiple_arg == 1 )
                dgemm( &trans1, &trans2, &n, &n, &n, &scal, A[p], &ldA[p], X, &ldX, &dZERO, C[p], &ldC[p] );
            else if ( multiple_arg == 2 )
                dgemm( &trans1, &trans2, &n, &n, &n, &scal, X, &ldX, A[p], &ldA[p], &dZERO, C[p], &ldC[p] );
            else
                dgemm( &trans1, &trans2, &n, &n, &n, &scal, A[p], &ldA[p], B[p], &ldB[p], &dZERO, C[p], &ldC[p] );
        }
    }
}

// .. Argument structure for computing parts of U and V for ss_alg_exp().
typedef struct parts_U_V_context_s
{
    int nt;
    double t;
    int n;
    double *b;
    double *Ag;
    double *Ag2;
    double *Ag4;
    double *Ag6;
    double *Ag8;
    double *U1;
    double *U2;
    double *V1;
    double *V2;
} parts_U_V_context_t;
static parts_U_V_context_t *parts_U_V_context;

// .. Working function for computing a factor of U3, and V3.
static void compute_dU3_fact_dV3( int numthreads, int rank, void *arg ) // #objects_for_parallelization = n*n
{
    // .. Extract context argument.
	parts_U_V_context_t *context = (parts_U_V_context_t*)arg;

	// .. Extract variables from the context struct.
    int nt = context->nt;
    double t = context->t;
    int n = context->n;
    double *b3 = context->b;
    double *Ag = context->Ag;
    double *Ag2 = context->Ag2;
    double *Ag4 = context->Ag4;
    double *Ag6 = context->Ag6;
    double *Ag8 = context->Ag8;
    double *U1 = context->U1;
    double *U2 = context->U2;
    double *V1 = context->V1;
    double *V2 = context->V2;

    int i, j, l;
    double id;

    if ( rank < nt )
    {
        partition_limits_t limits = partition( n*n, nt+1, rank+1 );

        for( l = limits.start; l <= limits.end; l++ )
        {
            i = l % n;
            j = l / n;
            if ( i == j )
                id = 1;
            else
                id = 0;
            U1[i+j*n] = pow(t,3) * b3[3] * Ag2[i+j*ldAg2] + t * b3[1] * id;
            V1[i+j*n] = pow(t,2) * b3[2] * Ag2[i+j*ldAg2] + b3[0] * id;
        }
    }
}

// .. Working function for computing a factor of U5, and V5.
static void compute_dU5_fact_dV5( int numthreads, int rank, void *arg ) // #objects_for_parallelization = n*n
{
    // .. Extract context argument.
	parts_U_V_context_t *context = (parts_U_V_context_t*)arg;

	// .. Extract variables from the context struct.
    int nt = context->nt;
    double t = context->t;
    int n = context->n;
    double *b5 = context->b;
    double *Ag = context->Ag;
    double *Ag2 = context->Ag2;
    double *Ag4 = context->Ag4;
    double *Ag6 = context->Ag6;
    double *Ag8 = context->Ag8;
    double *U1 = context->U1;
    double *U2 = context->U2;
    double *V1 = context->V1;
    double *V2 = context->V2;

    int i, j, l;
    double id;

    if ( rank < nt )
    {
        partition_limits_t limits = partition( n*n, nt+1, rank+1 );

        for( l = limits.start; l <= limits.end; l++ )
        {
            i = l % n;
            j = l / n;
            if ( i == j )
                id = 1;
            else
                id = 0;
            U1[i+j*n] = pow(t,5) * b5[5] * Ag4[i+j*ldAg4] + pow(t,3) * b5[3] * Ag2[i+j*ldAg2] + t * b5[1] * id;
            V1[i+j*n] = pow(t,4) * b5[4] * Ag4[i+j*ldAg4] + pow(t,2) * b5[2] * Ag2[i+j*ldAg2] + b5[0] * id;
        }
    }
}

// .. Working function for computing a factor of U7, and V7.
static void compute_dU7_fact_dV7( int numthreads, int rank, void *arg ) // #objects_for_parallelization = n*n
{
    // .. Extract context argument.
	parts_U_V_context_t *context = (parts_U_V_context_t*)arg;

	// .. Extract variables from the context struct.
    int nt = context->nt;
    double t = context->t;
    int n = context->n;
    double *b7 = context->b;
    double *Ag = context->Ag;
    double *Ag2 = context->Ag2;
    double *Ag4 = context->Ag4;
    double *Ag6 = context->Ag6;
    double *Ag8 = context->Ag8;
    double *U1 = context->U1;
    double *U2 = context->U2;
    double *V1 = context->V1;
    double *V2 = context->V2;

    int i, j, l;
    double id;

    if ( rank < nt )
    {
        partition_limits_t limits = partition( n*n, nt+1, rank+1 );

        for( l = limits.start; l <= limits.end; l++ )
        {
            i = l % n;
            j = l / n;
            if ( i == j )
                id = 1;
            else
                id = 0;
            U1[i+j*n] = pow(t,7) * b7[7] * Ag6[i+j*ldAg6] + pow(t,5) * b7[5] * Ag4[i+j*ldAg4] + pow(t,3) * b7[3] * Ag2[i+j*ldAg2] + t * b7[1] * id;
            V1[i+j*n] = pow(t,6) * b7[6] * Ag6[i+j*ldAg6] + pow(t,4) * b7[4] * Ag4[i+j*ldAg4] + pow(t,2) * b7[2] * Ag2[i+j*ldAg2] + b7[0] * id;
        }
    }
}

// .. Working function for computing a factor of U9, and V9.
static void compute_dU9_fact_dV9( int numthreads, int rank, void *arg ) // #objects_for_parallelization = n*n
{
    // .. Extract context argument.
	parts_U_V_context_t *context = (parts_U_V_context_t*)arg;

	// .. Extract variables from the context struct.
    int nt = context->nt;
    double t = context->t;
    int n = context->n;
    double *b9 = context->b;
    double *Ag = context->Ag;
    double *Ag2 = context->Ag2;
    double *Ag4 = context->Ag4;
    double *Ag6 = context->Ag6;
    double *Ag8 = context->Ag8;
    double *U1 = context->U1;
    double *U2 = context->U2;
    double *V1 = context->V1;
    double *V2 = context->V2;

    int i, j, l;
    double id;

    if ( rank < nt )
    {
        partition_limits_t limits = partition( n*n, nt+1, rank+1 );

        for( l = limits.start; l <= limits.end; l++ )
        {
            i = l % n;
            j = l / n;
            if ( i == j )
                id = 1;
            else
                id = 0;
            U1[i+j*n] = pow(t,9) * b9[9] * Ag8[i+j*n] + pow(t,7) * b9[7] * Ag6[i+j*ldAg6] + pow(t,5) * b9[5] * Ag4[i+j*ldAg4] + pow(t,3) * b9[3] * Ag2[i+j*ldAg2] + t * b9[1] * id;
            V1[i+j*n] = pow(t,8) * b9[8] * Ag8[i+j*n] + pow(t,6) * b9[6] * Ag6[i+j*ldAg6] + pow(t,4) * b9[4] * Ag4[i+j*ldAg4] + pow(t,2) * b9[2] * Ag2[i+j*ldAg2] + b9[0] * id;
        }
    }
}

// .. Working function for computing parts of U13 and parts of V13.
static void compute_dU13_fact_dV13( int numthreads, int rank, void *arg ) // #objects_for_parallelization = n*n
{
    // .. Extract context argument.
	parts_U_V_context_t *context = (parts_U_V_context_t*)arg;

	// .. Extract variables from the context struct.
    int nt = context->nt;
    double t = context->t;
    int n = context->n;
    double *b13 = context->b;
    double *Ag = context->Ag;
    double *Ag2 = context->Ag2;
    double *Ag4 = context->Ag4;
    double *Ag6 = context->Ag6;
    double *Ag8 = context->Ag8;
    double *U1 = context->U1;
    double *U2 = context->U2;
    double *V1 = context->V1;
    double *V2 = context->V2;

    int i, j, l;
    double id;

    if ( rank < nt )
    {
        partition_limits_t limits = partition( n*n, nt+1, rank+1 );

        for( l = limits.start; l <= limits.end; l++ )
        {
            i = l % n;
            j = l / n;
            if ( i == j )
                id = 1;
            else
                id = 0;
            U1[i+j*n] = pow(t,13) * b13[13] * Ag6[i+j*ldAg6] + pow(t,11) * b13[11] * Ag4[i+j*ldAg4] + pow(t,9) * b13[9] * Ag2[i+j*ldAg2];
            U2[i+j*n] = pow(t,7) * b13[7] * Ag6[i+j*ldAg6] + pow(t,5) * b13[5] * Ag4[i+j*ldAg4] + pow(t,3) * b13[3] * Ag2[i+j*ldAg2] + t * b13[1] * id;

            V1[i+j*n] = pow(t,12) * b13[12] * Ag6[i+j*ldAg6] + pow(t,10) * b13[10] * Ag4[i+j*ldAg4] + pow(t,8) * b13[8] * Ag2[i+j*ldAg2];
            V2[i+j*n] = pow(t,6) * b13[6] * Ag6[i+j*ldAg6] + pow(t,4) * b13[4] * Ag4[i+j*ldAg4] + pow(t,2) * b13[2] * Ag2[i+j*ldAg2] + b13[0] * id;
        }
    }
}

// .. Argument structure for computing P and Q for ss_alg_exp().
typedef struct compute_P_Q_context_s
{
    int nt;
    int n;
    double *P;
    double *Q;
} compute_P_Q_context_t;
static compute_P_Q_context_t *compute_P_Q_context;

// .. Working function for computing P and Q for ss_alg_exp().
static void compute_P_Q( int numthreads, int rank, void *arg ) // #objects_for_parallelization = n*n
{
    // .. Extract context argument.
	compute_P_Q_context_t *context = (compute_P_Q_context_t*)arg;

	// .. Extract variables from the context struct.
    int nt = context->nt;
    int n = context->n;
    double *P = context->P;
    double *Q = context->Q;

    int i, j, l;
    double u, v;

    if ( rank < nt )
    {
        partition_limits_t limits = partition( n*n, nt+1, rank+1 );

        for( l = limits.start; l <= limits.end; l++ )
        {
            i = l % n;
            j = l / n;
            u = P[i+j*n];
            v = Q[i+j*n];

            P[i+j*n] = u + v;
            Q[i+j*n] = -u + v;
        }
    }
}

//
// .. Finished with thread functions.
//

static void prepare_CPU( double **Am, int *ldAm, double *Xm, int ldXm )
{
    int i, p;

	stopped = 0;

    threads = (pthread_t*) malloc(sizeof(pthread_t) * max_num_threads);
    wd = (struct thread_data**) malloc(max_num_threads*sizeof(struct thread_data*));
	for ( p = 0; p < max_num_threads; p++ )
		wd[p] = (struct thread_data*) malloc(sizeof(struct thread_data));

	off_context = (off_context_t*) malloc(sizeof(off_context_t));
	remove_diagonals_context = (remove_diagonals_context_t*) malloc(sizeof(remove_diagonals_context_t));
	scaling_of_matrix_context = (scaling_of_matrix_context_t*) malloc(sizeof(scaling_of_matrix_context_t));
	lin_comb_of_2matrices_context = (lin_comb_of_2matrices_context_t*) malloc(sizeof(lin_comb_of_2matrices_context_t));
	lin_comb_trans_context = (lin_comb_trans_context_t*) malloc(sizeof(lin_comb_trans_context_t));
	sum_of_matrices_context = (sum_of_matrices_context_t*) malloc(sizeof(sum_of_matrices_context_t));
    traces_by_ddots_context = (traces_by_ddots_context_t*) malloc(sizeof(traces_by_ddots_context_t));
    multiple_gemms_context = (multiple_gemms_context_t*) malloc(sizeof(multiple_gemms_context_t));
	parts_U_V_context = (parts_U_V_context_t*) malloc(sizeof(parts_U_V_context_t));
	compute_P_Q_context = (compute_P_Q_context_t*) malloc(sizeof(compute_P_Q_context_t));

    A = (double**) malloc(m*sizeof(double*));
    ldA = (int*) malloc(m*sizeof(int));
    for( p = 0; p < m; p++ )
    {
        A[p] = Am[p];
        ldA[p] = ldAm[p];
    }
    X = Xm;
    ldX = ldXm;
    AX = (double**) malloc(m*sizeof(double*));
    ldAX = (int*) malloc(m*sizeof(int));
    B = (double**) malloc(m*sizeof(double*));
    ldB = (int*) malloc(m*sizeof(int));
    WW1 = (double**) malloc(m*sizeof(double*));
    WW2 = (double**) malloc(m*sizeof(double*));
    D = (double**) malloc(m*sizeof(double*));
    for( p = 0; p < m; p++ )
    {
        AX[p] = (double*) malloc(n*n*sizeof(double));
        ldAX[p] = n;
        B[p] = (double*) malloc(n*n*sizeof(double));
        ldB[p] = n;
        WW1[p] = (double*) malloc(n*n*sizeof(double));
        WW2[p] = (double*) malloc(n*n*sizeof(double));
        D[p] = (double*) malloc(n*sizeof(double));
    }

    Ag = (double*) malloc(n*n*sizeof(double));
    Ag2 = (double*) malloc(n*n*sizeof(double));
    Ag4 = (double*) malloc(n*n*sizeof(double));
    Ag6 = (double*) malloc(n*n*sizeof(double));
	ldAg = n;
	ldAg2 = n;
	ldAg4 = n;
	ldAg6 = n;
    eAt = (double*) malloc(n*n*sizeof(double));
    FX = (double*) malloc(n*n*sizeof(double));
    G = (double*) malloc(n*n*sizeof(double));
    G0 = (double*) malloc(n*n*sizeof(double));
    H = (double*) malloc(n*n*sizeof(double));
	I = (double*) malloc(n*n*sizeof(double));
    P = (double*) malloc(n*n*sizeof(double*));
    Q = (double*) malloc(n*n*sizeof(double*));
    Up = (double*) malloc(n*n*sizeof(double*));
    Vp = (double*) malloc(n*n*sizeof(double*));
    W = (double*) malloc(n*n*sizeof(double));
    W1 = (double*) malloc(n*n*sizeof(double));
    W2 = (double*) malloc(n*n*sizeof(double));
    W3 = (double*) malloc(n*n*sizeof(double));
    W4 = (double*) malloc(n*n*sizeof(double));
    X0 = (double*) malloc(n*n*sizeof(double));
    Xi = (double*) malloc(n*n*sizeof(double));
    ipiv = (int*) malloc(n*sizeof(int));

    loc_sum = (double*) malloc((max_num_threads+1)*sizeof(double));

    barrier = barrier_new(max_num_threads+1);

	// .. Activate the thread function.

	for ( i = 0; i < max_num_threads; ++i )
    {
        wd[i]->rank = i;
        pthread_create(&threads[i], 0, waiting_thread_function, (void*) wd[i]);
    }

    //.. mkl_num_threads are always set to max_num_threads+1
    mkl_set_num_threads(max_num_threads+1);
}

static void finalize_CPU()
{
	int i, p;

    stopped = 1;

    barrier_wait(barrier);

    for ( i = 0; i < max_num_threads; i++ )
    {
        pthread_join(threads[i], 0);
    }

    barrier_delete(barrier);

    free(threads);
	for ( p = 0; p < max_num_threads; p++ )
		free(wd[p]);
    free(wd);

    free(off_context);
	free(remove_diagonals_context);
	free(scaling_of_matrix_context);
	free(lin_comb_of_2matrices_context);
	free(lin_comb_trans_context);
	free(sum_of_matrices_context);
	free(traces_by_ddots_context);
    free(multiple_gemms_context);
	free(parts_U_V_context);
	free(compute_P_Q_context);

    for( p = 0; p < m; p++ )
    {
        free(AX[p]);
        free(B[p]);
        free(WW1[p]);
        free(WW2[p]);
        free(D[p]);
    }
	free(A);
	free(ldA);
    free(AX);
    free(ldAX);
    free(B);
    free(ldB);
    free(WW1);
    free(WW2);
    free(D);
    free(Ag);
    free(Ag2);
    free(Ag4);
    free(Ag6);
    free(eAt);
    free(FX);
    free(G);
    free(G0);
    free(H);
	free(I);
    free(P);
    free(Q);
    free(Up),
    free(Vp);
    free(W);
    free(W1);
    free(W2);
    free(W3);
    free(W4);
    free(X0);
    free(Xi);
    free(ipiv);
    free(loc_sum);

	mkl_set_num_threads(max_num_threads+1);
}

// .. Function for scaling of a matrix
static void scaling_of_matrix_fun( int n, double alpha, double *A, int ldA, double *R, int ldR )
{
    int nt;

    //------------------------------------------------------------------

    nt = imin(max_num_threads,n*n-1);

    // .. Create the context for scaling a matrix.
    scaling_of_matrix_context->nt = nt;
    scaling_of_matrix_context->n = n;
    scaling_of_matrix_context->alpha = alpha;
    scaling_of_matrix_context->A = A;
    scaling_of_matrix_context->ldA = ldA;
    scaling_of_matrix_context->R = R;
    scaling_of_matrix_context->ldR = ldR;

    // .. Call the working function for scaling a matrix.
    function = scaling_of_matrix;
    argument = scaling_of_matrix_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	scaling_of_matrix( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------
}

// .. Function for linear combination of two matrices.
static void lin_comb_of_2matrices_fun( int n, double alpha, double *A, int ldA, double beta, double *B, int ldB, double *R, int ldR )
{
    int nt;

    //------------------------------------------------------------------

    nt = imin(max_num_threads,n*n-1);

    // .. Create the context for scaling a matrix.
    lin_comb_of_2matrices_context->nt = nt;
    lin_comb_of_2matrices_context->n = n;
    lin_comb_of_2matrices_context->alpha = alpha;
    lin_comb_of_2matrices_context->A = A;
    lin_comb_of_2matrices_context->ldA = ldA;
    lin_comb_of_2matrices_context->beta = beta;
    lin_comb_of_2matrices_context->B = B;
    lin_comb_of_2matrices_context->ldB = ldB;
    lin_comb_of_2matrices_context->R = R;
    lin_comb_of_2matrices_context->ldR = ldR;

    // .. Call the working function for scaling a matrix.
    function = lin_comb_of_2matrices;
    argument = lin_comb_of_2matrices_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	lin_comb_of_2matrices( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------
}

// .. Function for computing alpha*A+beta*A'.
static void lin_comb_trans_fun( int n, double alpha, double beta, double *A, int ldA, double *R, int ldR )
{
    int nt;

    //------------------------------------------------------------------

    nt = imin(max_num_threads,n*n-1);

    // .. Create the context for scaling a matrix.
    lin_comb_trans_context->nt = nt;
    lin_comb_trans_context->n = n;
    lin_comb_trans_context->alpha = alpha;
    lin_comb_trans_context->beta = beta;
    lin_comb_trans_context->A = A;
    lin_comb_trans_context->ldA = ldA;
    lin_comb_trans_context->R = R;
    lin_comb_trans_context->ldR = ldR;

    // .. Call the working function for scaling a matrix.
    function = lin_comb_trans;
    argument = lin_comb_trans_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	lin_comb_trans( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------
}


// .. Function for computing trace of a matrix product
static double trace_of_product( char trans, double *A, int ldA, double *B, int ldB )
{
    int i, nt;
    double trace;

    //------------------------------------------------------------------

    if ( trans == 'T' )
    {
        nt = 0;
        mkl_set_num_threads(max_num_threads+1);
    }
    else
    {
        nt = imin(nt_ddots,n-1);
        mkl_set_num_threads(num_mkl_threads_ddots);
    }

    // .. Create the context for computing sum of traces of matrix products.
    traces_by_ddots_context->nt = nt;
    traces_by_ddots_context->trans = trans;
    traces_by_ddots_context->m = 1;
    traces_by_ddots_context->n = n;
    traces_by_ddots_context->A = &A;
    traces_by_ddots_context->ldA = &ldA;
    traces_by_ddots_context->B = &B;
    traces_by_ddots_context->ldB = &ldB;
    traces_by_ddots_context->loc_sum = loc_sum;

    // .. Call the working function for computing diagonal of a matrix product.
    function = traces_by_ddots;
	argument = traces_by_ddots_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	traces_by_ddots( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    trace = 0;
	for ( i = 0; i <= nt; i++ )
		trace += loc_sum[i];

    mkl_set_num_threads(max_num_threads+1);

    return trace;
}

static double F()
{
    int i, nt;
    double f = 0;
    
    //------------------------------------------------------------------
    
    nt = imin(nt_ddots,m-1);
    mkl_set_num_threads(num_mkl_threads_ddots);
    
    // .. Create the context for computing sum of traces of matrix products.
    traces_by_ddots_context->nt = nt;
    traces_by_ddots_context->trans = 'T';
    traces_by_ddots_context->m = m;
    traces_by_ddots_context->n = n;
    traces_by_ddots_context->A = B; // Diagonals are already removed from B[p]
    traces_by_ddots_context->ldA = ldB;
    traces_by_ddots_context->B = B;
    traces_by_ddots_context->ldB = ldB;
    traces_by_ddots_context->loc_sum = loc_sum;
    
    // .. Call the working function for computing sum of traces of matrix products.
    function = traces_by_ddots;
	argument = traces_by_ddots_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	traces_by_ddots( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------
    
	for ( i = 0; i <= nt; i++ )
		f += loc_sum[i];
    
    f *= 0.5;
    
    mkl_set_num_threads(max_num_threads+1);

    return f;
}

static void F_X()
{
    int i, nt;
	double beta;

    mkl_set_num_threads(num_mkl_threads_gemms);

    //------------------------------------------------------------------

    nt = imin(nt_gemms,m-1);

    // .. Create the context for multiple gemms.
    multiple_gemms_context->nt = nt;
    multiple_gemms_context->trans1 = 'N';
    multiple_gemms_context->trans2 = 'N';
    multiple_gemms_context->multiple_arg = 1;
	multiple_gemms_context->m = m;
	multiple_gemms_context->n = n;
    multiple_gemms_context->scal = dONE;
    multiple_gemms_context->A = A;
    multiple_gemms_context->ldA = ldA;
    multiple_gemms_context->B = A;
    multiple_gemms_context->ldB = ldA;
    multiple_gemms_context->X = X;
    multiple_gemms_context->ldX = ldX;
    multiple_gemms_context->C = AX;
    multiple_gemms_context->ldC = ldB;

    // .. Call the working function for multiple gemms.
    function = multiple_gemms;
	argument = multiple_gemms_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	multiple_gemms( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    nt = imin(nt_gemms,m-1);

    // .. Create the context for multiple gemms.
    multiple_gemms_context->nt = nt;
    multiple_gemms_context->trans1 = 'T';
    multiple_gemms_context->trans2 = 'N';
    multiple_gemms_context->multiple_arg = 2;
    multiple_gemms_context->m = m;
	multiple_gemms_context->n = n;
    multiple_gemms_context->scal = dONE;
    multiple_gemms_context->A = AX;
    multiple_gemms_context->ldA = ldB;
    multiple_gemms_context->B = AX;
    multiple_gemms_context->ldB = ldB;
    multiple_gemms_context->X = X;
    multiple_gemms_context->ldX = ldX;
    multiple_gemms_context->C = B;
    multiple_gemms_context->ldC = ldB;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	multiple_gemms( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    nt = imin(max_num_threads,m*n-1);

    // .. Create the context for removing diagonals.
    remove_diagonals_context->nt = nt;
    remove_diagonals_context->m = m;
	remove_diagonals_context->n = n;
    remove_diagonals_context->A = B;
    remove_diagonals_context->D = D;

    // .. Call the working function for removing diagonals.
    function = remove_diagonals;
	argument = remove_diagonals_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	remove_diagonals( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    nt = imin(nt_gemms,m-1);

    // .. Create the context for multiple gemms.
    multiple_gemms_context->nt = nt;
    multiple_gemms_context->trans1 = 'N';
    multiple_gemms_context->trans2 = 'N';
    multiple_gemms_context->multiple_arg = 12;
	multiple_gemms_context->m = m;
	multiple_gemms_context->n = n;
    multiple_gemms_context->scal = dTWO;
    multiple_gemms_context->A = AX;
    multiple_gemms_context->ldA = ldB;
    multiple_gemms_context->B = B;
    multiple_gemms_context->ldB = ldB;
    multiple_gemms_context->X = X;
    multiple_gemms_context->ldX = ldX;
    multiple_gemms_context->C = WW1;
    multiple_gemms_context->ldC = ldB;

    // .. Call the working function for multiple gemms.
    function = multiple_gemms;
	argument = multiple_gemms_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	multiple_gemms( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    nt = imin(max_num_threads,n*n-1);

    // .. Create the context for summing matrices.

    sum_of_matrices_context->nt = nt;
    sum_of_matrices_context->m = m;
	sum_of_matrices_context->n = n;
    sum_of_matrices_context->A = WW1;
    sum_of_matrices_context->ldA = ldB;
    sum_of_matrices_context->S = FX;
    sum_of_matrices_context->ldS = n;

    // .. Call the working function for summing matrices.
    function = sum_of_matrices;
	argument = sum_of_matrices_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	sum_of_matrices( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    mkl_set_num_threads(max_num_threads+1);
}

static double F_XX(double *Xi1, int ldXi1, double *Xi2, int ldXi2 )
{
    int i, nt;
	double f, sum;

	f = 0;

    //------------------------------------------------------------------

    nt = imin(nt_gemms,m-1);
    mkl_set_num_threads(num_mkl_threads_gemms);

    // .. Create the context for multiple gemms.
    multiple_gemms_context->nt = nt;
    multiple_gemms_context->trans1 = 'T';
    multiple_gemms_context->trans2 = 'N';
    multiple_gemms_context->multiple_arg = 1;
	multiple_gemms_context->m = m;
	multiple_gemms_context->n = n;
    multiple_gemms_context->scal = dONE;
    multiple_gemms_context->A = AX;
    multiple_gemms_context->ldA = ldB;
    multiple_gemms_context->B = B;
    multiple_gemms_context->ldB = ldB;
    multiple_gemms_context->X = Xi1;
    multiple_gemms_context->ldX = ldXi1;
    multiple_gemms_context->C = WW1;
    multiple_gemms_context->ldC = ldB;

    // .. Call the working function for multiple gemms.
    function = multiple_gemms;
	argument = multiple_gemms_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	multiple_gemms( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    nt = imin(nt_gemms,m-1);
    mkl_set_num_threads(num_mkl_threads_gemms);

    // .. Create the context for multiple gemms.
    multiple_gemms_context->nt = nt;
    multiple_gemms_context->trans1 = 'T';
    multiple_gemms_context->trans2 = 'N';
    multiple_gemms_context->multiple_arg = 1;
	multiple_gemms_context->m = m;
	multiple_gemms_context->n = n;
    multiple_gemms_context->scal = dONE;
    multiple_gemms_context->A = AX;
    multiple_gemms_context->ldA = ldB;
    multiple_gemms_context->B = B;
    multiple_gemms_context->ldB = ldB;
    multiple_gemms_context->X = Xi2;
    multiple_gemms_context->ldX = ldXi2;
    multiple_gemms_context->C = WW2;
    multiple_gemms_context->ldC = ldB;

    // .. Call the working function for multiple gemms.
    function = multiple_gemms;
	argument = multiple_gemms_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	multiple_gemms( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    // .. Removing diagonals of WW2.
    
    nt = imin(max_num_threads,m*n-1);
    
    // .. Create the context for removing diagonals.
    off_context->nt = nt;
    off_context->m = m;
	off_context->n = n;
    off_context->A = WW2;

    // .. Call the working function for removing diagonals.
    function = off;
	argument = off_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	off( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    nt = imin(nt_ddots,m*n-1);
    mkl_set_num_threads(num_mkl_threads_ddots);

    // .. Create the context for computing sum of traces of matrix products.
    traces_by_ddots_context->nt = nt;
    traces_by_ddots_context->trans = 'N';
    traces_by_ddots_context->m = m;
    traces_by_ddots_context->n = n;
    traces_by_ddots_context->A = WW1;
    traces_by_ddots_context->ldA = ldB;
    traces_by_ddots_context->B = WW2;
    traces_by_ddots_context->ldB = ldB;
    traces_by_ddots_context->loc_sum = loc_sum;

    // .. Call the working function for computing sum of traces of matrix products.
    function = traces_by_ddots;
	argument = traces_by_ddots_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	traces_by_ddots( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    sum = 0;
	for ( i = 0; i <= nt; i++ )
		sum += loc_sum[i];

    //------------------------------------------------------------------

    f += sum;

    //------------------------------------------------------------------

    nt = imin(nt_ddots,m-1);
    mkl_set_num_threads(num_mkl_threads_ddots);

    // .. Create the context for computing sum of traces of matrix products.
    traces_by_ddots_context->nt = nt;
    traces_by_ddots_context->trans = 'T';
    traces_by_ddots_context->m = m;
    traces_by_ddots_context->n = n;
    traces_by_ddots_context->A = WW1;
    traces_by_ddots_context->ldA = ldB;
    traces_by_ddots_context->B = WW2;
    traces_by_ddots_context->ldB = ldB;
    traces_by_ddots_context->loc_sum = loc_sum;

    // .. Call the working function for computing sum of traces of matrix products.
    function = traces_by_ddots;
	argument = traces_by_ddots_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	traces_by_ddots( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    sum = 0;
	for ( i = 0; i <= nt; i++ )
		sum += loc_sum[i];

    //------------------------------------------------------------------

    f += sum;

    //------------------------------------------------------------------

    nt = imin(nt_gemms,m-1);
    mkl_set_num_threads(num_mkl_threads_gemms);

    // .. Create the context for multiple gemms.
    multiple_gemms_context->nt = nt;
    multiple_gemms_context->trans1 = 'N';
    multiple_gemms_context->trans2 = 'N';
    multiple_gemms_context->multiple_arg = 1;
	multiple_gemms_context->m = m;
	multiple_gemms_context->n = n;
    multiple_gemms_context->scal = dONE;
    multiple_gemms_context->A = A;
    multiple_gemms_context->ldA = ldA;
    multiple_gemms_context->B = B;
    multiple_gemms_context->ldB = ldB;
    multiple_gemms_context->X = Xi2;
    multiple_gemms_context->ldX = ldXi2;
    multiple_gemms_context->C = WW2;
    multiple_gemms_context->ldC = ldB;

    // .. Call the working function for multiple gemms.
    function = multiple_gemms;
	argument = multiple_gemms_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	multiple_gemms( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    nt = imin(nt_gemms,m-1);
    mkl_set_num_threads(num_mkl_threads_gemms);

    // .. Create the context for multiple gemms.
    multiple_gemms_context->nt = nt;
    multiple_gemms_context->trans1 = 'T';
    multiple_gemms_context->trans2 = 'N';
    multiple_gemms_context->multiple_arg = 2;
	multiple_gemms_context->m = m;
	multiple_gemms_context->n = n;
    multiple_gemms_context->scal = dONE;
    multiple_gemms_context->A = WW2;
    multiple_gemms_context->ldA = ldB;
    multiple_gemms_context->B = B;
    multiple_gemms_context->ldB = ldB;
    multiple_gemms_context->X = Xi1;
    multiple_gemms_context->ldX = ldXi1;
    multiple_gemms_context->C = WW1;
    multiple_gemms_context->ldC = ldB;

    // .. Call the working function for multiple gemms.
    function = multiple_gemms;
	argument = multiple_gemms_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	multiple_gemms( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    nt = imin(nt_ddots,m-1);
    mkl_set_num_threads(num_mkl_threads_ddots);

    // .. Create the context for computing sum of traces of matrix products.
    traces_by_ddots_context->nt = nt;
    traces_by_ddots_context->trans = 'T'; // We can transpose, since B[p] are symmetric: tr(WW1[p]*B[p]) = tr(B[p]*WW1[p]) = tr(B[p]'*WW1[p]) = tr(WW1[p]'*B[p])
    traces_by_ddots_context->m = m;
    traces_by_ddots_context->n = n;
    traces_by_ddots_context->A = WW1;
    traces_by_ddots_context->ldA = ldB;
    traces_by_ddots_context->B = B;
    traces_by_ddots_context->ldB = ldB;
    traces_by_ddots_context->loc_sum = loc_sum;

    // .. Call the working function for computing sum of traces of matrix products.
    function = traces_by_ddots;
	argument = traces_by_ddots_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	traces_by_ddots( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    sum = 0;
	for ( i = 0; i <= nt; i++ )
		sum += loc_sum[i];

    //------------------------------------------------------------------

    f += sum;

    f *= 2;

    mkl_set_num_threads(max_num_threads+1);

    return f;
}

static void ss_alg_exp( double t, double *R, int ldR )
{
    double theta13=5.4e0;
    int info, i, nt, s, scal = 0;

    nt = imin(max_num_threads,n*n-1);

    // .. Create the context for computing parts of U and V.
    parts_U_V_context->nt = nt;
    parts_U_V_context->t = t;
    parts_U_V_context->n = n;
    parts_U_V_context->Ag = Ag;
    parts_U_V_context->Ag2 = Ag2;
    parts_U_V_context->Ag4 = Ag4;
    parts_U_V_context->Ag6 = Ag6;

    if ( t * n1Ag <= thetam[0] )
    {
        // .. The degree of Pade approximant for this t is 3.
        //------------------------------------------------------------------

        parts_U_V_context->b = b3;
        parts_U_V_context->U1 = Up;
        parts_U_V_context->V1 = Q;

        // .. Call the working function for computing parts of U and V.
        function = compute_dU3_fact_dV3;
        argument = parts_U_V_context;

        // .. Wake the threads.
        barrier_wait(barrier);

        // .. Master thread is also working on its part.
        compute_dU3_fact_dV3( max_num_threads, -1, argument );

        // .. Synchronize the threads.
        barrier_wait(barrier);

        //------------------------------------------------------------------

        dgemm( "N", "N", &n, &n, &n, &dONE, Ag, &ldAg, Up, &n, &dZERO, P, &n );
    }

    else if ( t * n1Ag <= thetam[1] )
    {
        // .. The degree of Pade approximant for this t is 5.
        //------------------------------------------------------------------

        parts_U_V_context->b = b5;
        parts_U_V_context->U1 = Up;
        parts_U_V_context->V1 = Q;

        // .. Call the working function for computing parts of U and V.
        function = compute_dU5_fact_dV5;
        argument = parts_U_V_context;

        // .. Wake the threads.
        barrier_wait(barrier);

        // .. Master thread is also working on its part.
        compute_dU5_fact_dV5( max_num_threads, -1, argument );

        // .. Synchronize the threads.
        barrier_wait(barrier);

        //------------------------------------------------------------------

        dgemm( "N", "N", &n, &n, &n, &dONE, Ag, &ldAg, Up, &n, &dZERO, P, &n );
    }

    else if ( t * n1Ag <= thetam[2] )
    {
        // .. The degree of Pade approximant for this t is 7.
        //------------------------------------------------------------------

        parts_U_V_context->b = b7;
        parts_U_V_context->U1 = Up;
        parts_U_V_context->V1 = Q;

        // .. Call the working function for computing parts of U and V.
        function = compute_dU7_fact_dV7;
        argument = parts_U_V_context;

        // .. Wake the threads.
        barrier_wait(barrier);

        // .. Master thread is also working on its part.
        compute_dU7_fact_dV7( max_num_threads, -1, argument );

        // .. Synchronize the threads.
        barrier_wait(barrier);

        //------------------------------------------------------------------

        dgemm( "N", "N", &n, &n, &n, &dONE, Ag, &ldAg, Up, &n, &dZERO, P, &n );
    }

    else if ( t * n1Ag <= thetam[3] )
    {
        // .. The degree of Pade approximant for this t is 9.
        
        Ag8 = (double *) malloc(n*n*sizeof(double));
        dgemm( "N", "N", &n, &n, &n, &dONE, Ag4, &ldAg4, Ag4, &ldAg4, &dZERO, Ag8, &n );

        //------------------------------------------------------------------

        parts_U_V_context->b = b9;
        parts_U_V_context->Ag8 = Ag8;
        parts_U_V_context->U1 = Up;
        parts_U_V_context->V1 = Q;

        // .. Call the working function for computing parts of U and V.
        function = compute_dU9_fact_dV9;
        argument = parts_U_V_context;

        // .. Wake the threads.
        barrier_wait(barrier);

        // .. Master thread is also working on its part.
        compute_dU9_fact_dV9( max_num_threads, -1, argument );

        // .. Synchronize the threads.
        barrier_wait(barrier);

        //------------------------------------------------------------------

        dgemm( "N", "N", &n, &n, &n, &dONE, Ag, &ldAg, Up, &n, &dZERO, P, &n );
        free(Ag8);
    }

    else
    {
        // .. The degree of Pade approximant for this t is 13.
        
        s = ceil(log2(t*n1Ag/theta13));
        if ( s != 0 ) scal = 1;
        t = t / pow(2,s);

        //------------------------------------------------------------------

		parts_U_V_context->t = t;
        parts_U_V_context->b = b13;
        parts_U_V_context->U1 = P;
        parts_U_V_context->U2 = Up;
        parts_U_V_context->V1 = Vp;
        parts_U_V_context->V2 = Q;

        // .. Call the working function for computing parts of U and V.
        function = compute_dU13_fact_dV13;
        argument = parts_U_V_context;

        // .. Wake the threads.
        barrier_wait(barrier);

        // .. Master thread is also working on its part.
        compute_dU13_fact_dV13( max_num_threads, -1, argument );

        // .. Synchronize the threads.
        barrier_wait(barrier);

        //------------------------------------------------------------------

        dgemm( "N", "N", &n, &n, &n, &dONE, Ag6, &ldAg6, P, &n, &dONE, Up, &n );
        dgemm( "N", "N", &n, &n, &n, &dONE, Ag, &ldAg, Up, &n, &dZERO, P, &n );
        dgemm( "N", "N", &n, &n, &n, &dONE, Ag6, &ldAg6, Vp, &n, &dONE, Q, &n );
    }


    //------------------------------------------------------------------

    // .. Create the context for computing P and Q.
    compute_P_Q_context->nt = nt;
    compute_P_Q_context->n = n;
    compute_P_Q_context->P = P;
    compute_P_Q_context->Q = Q;

    // .. Call the working function for computing parts of U and V.
    function = compute_P_Q;
    argument = compute_P_Q_context;

    // .. Wake the threads.
    barrier_wait(barrier);

    // .. Master thread is also working on its part.
    compute_P_Q( max_num_threads, -1, argument );

    // .. Synchronize the threads.
    barrier_wait(barrier);

    //------------------------------------------------------------------

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

}

// .. Function for computing F on the geodesic.
static double F_geod(double t, double *R)
{
    int i, nt;
    double y;

    ss_alg_exp(t, R, n );

    //
    // .. Diagonals are returned to B[p].
    //

    y = 0;

    //------------------------------------------------------------------

    nt = imin(nt_gemms,m-1);
    mkl_set_num_threads(num_mkl_threads_gemms);

    // .. Create the context for multiple gemms.
    multiple_gemms_context->nt = nt;
    multiple_gemms_context->trans1 = 'T';
    multiple_gemms_context->trans2 = 'N';
    multiple_gemms_context->multiple_arg = 2;
	multiple_gemms_context->m = m;
	multiple_gemms_context->n = n;
    multiple_gemms_context->scal = dONE;
    multiple_gemms_context->A = B;
    multiple_gemms_context->ldA = ldB;
    multiple_gemms_context->B = B;
    multiple_gemms_context->ldB = ldB;
    multiple_gemms_context->X = R;
    multiple_gemms_context->ldX = n;
    multiple_gemms_context->C = WW1;
    multiple_gemms_context->ldC = ldB;

    // .. Call the working function for multiple gemms.
    function = multiple_gemms;
	argument = multiple_gemms_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	multiple_gemms( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    nt = imin(nt_gemms,m-1);
    mkl_set_num_threads(num_mkl_threads_gemms);

    // .. Create the context for multiple gemms.
    multiple_gemms_context->nt = nt;
    multiple_gemms_context->trans1 = 'N';
    multiple_gemms_context->trans2 = 'N';
    multiple_gemms_context->multiple_arg = 1;
	multiple_gemms_context->m = m;
	multiple_gemms_context->n = n;
    multiple_gemms_context->scal = dONE;
    multiple_gemms_context->A = WW1;
    multiple_gemms_context->ldA = ldB;
    multiple_gemms_context->B = B;
    multiple_gemms_context->ldB = ldB;
    multiple_gemms_context->X = R;
    multiple_gemms_context->ldX = n;
    multiple_gemms_context->C = WW2;
    multiple_gemms_context->ldC = ldB;

    // .. Call the working function for multiple gemms.
    function = multiple_gemms;
	argument = multiple_gemms_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	multiple_gemms( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    // .. Removing diagonals of WW2. 
    
    nt = imin(max_num_threads,m*n-1);

    // .. Create the context for removing diagonals.
    off_context->nt = nt;
    off_context->m = m;
	off_context->n = n;
    off_context->A = WW2;

    // .. Call the working function for removing diagonals.
    function = off;
	argument = off_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	off( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    nt = imin(nt_ddots,m-1);
    mkl_set_num_threads(num_mkl_threads_ddots);

    // .. Create the context for computing sum of traces of matrix products.
    traces_by_ddots_context->nt = nt;
    traces_by_ddots_context->trans = 'T'; // WW2[p] is symmetric.
    traces_by_ddots_context->m = m;
    traces_by_ddots_context->n = n;
    traces_by_ddots_context->A = WW2;
    traces_by_ddots_context->ldA = ldB;
    traces_by_ddots_context->B = WW2;
    traces_by_ddots_context->ldB = ldB;
    traces_by_ddots_context->loc_sum = loc_sum;

    // .. Call the working function for computing sum of traces of matrix products.
    function = traces_by_ddots;
	argument = traces_by_ddots_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	traces_by_ddots( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

	for ( i = 0; i <= nt; i++ )
		y += loc_sum[i];

    //------------------------------------------------------------------

    y /= 2;

    mkl_set_num_threads(max_num_threads+1);

    return y;
}

// .. Function for computing F on the retraction.
static double F_retr(double t, double *R)
{
    int i, info, nt, n2=n*n;
    double y;
	double *LW, *tau;

	LW = (double*) malloc(n*n*sizeof(double));
	tau = (double*) malloc(n*sizeof(double));

    lin_comb_of_2matrices_fun( n, 1.0, X, ldX, t, H, n, R, n );

	mkl_set_num_threads(max_num_threads+1);

    dgeqrf( &n, &n, R, &n, tau, LW, &n2, &info	);
    dorgqr ( &n, &n, &n, R, &n, tau, LW, &n2, &info );

    y = 0;

    //------------------------------------------------------------------

    nt = imin(nt_gemms,m-1);
    mkl_set_num_threads(num_mkl_threads_gemms);

    // .. Create the context for multiple gemms.
    multiple_gemms_context->nt = nt;
    multiple_gemms_context->trans1 = 'T';
    multiple_gemms_context->trans2 = 'N';
    multiple_gemms_context->multiple_arg = 2;
	multiple_gemms_context->m = m;
	multiple_gemms_context->n = n;
    multiple_gemms_context->scal = dONE;
    multiple_gemms_context->A = A;
    multiple_gemms_context->ldA = ldA;
    multiple_gemms_context->B = B;
    multiple_gemms_context->ldB = ldB;
    multiple_gemms_context->X = R;
    multiple_gemms_context->ldX = n;
    multiple_gemms_context->C = WW1;
    multiple_gemms_context->ldC = ldB;

    // .. Call the working function for multiple gemms.
    function = multiple_gemms;
	argument = multiple_gemms_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	multiple_gemms( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    nt = imin(nt_gemms,m-1);
    mkl_set_num_threads(num_mkl_threads_gemms);

    // .. Create the context for multiple gemms.
    multiple_gemms_context->nt = nt;
    multiple_gemms_context->trans1 = 'N';
    multiple_gemms_context->trans2 = 'N';
    multiple_gemms_context->multiple_arg = 1;
	multiple_gemms_context->m = m;
	multiple_gemms_context->n = n;
    multiple_gemms_context->scal = dONE;
    multiple_gemms_context->A = WW1;
    multiple_gemms_context->ldA = ldB;
    multiple_gemms_context->B = B;
    multiple_gemms_context->ldB = ldB;
    multiple_gemms_context->X = R;
    multiple_gemms_context->ldX = n;
    multiple_gemms_context->C = WW2;
    multiple_gemms_context->ldC = ldB;

    // .. Call the working function for multiple gemms.
    function = multiple_gemms;
	argument = multiple_gemms_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	multiple_gemms( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    // .. Removing diagonals of WW2.
    
    nt = imin(max_num_threads,m*n-1);

    // .. Create the context for removing diagonals.
    off_context->nt = nt;
    off_context->m = m;
	off_context->n = n;
    off_context->A = WW2;

    // .. Call the working function for removing diagonals.
    function = off;
	argument = off_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	off( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    nt = imin(nt_ddots,m-1);
    mkl_set_num_threads(num_mkl_threads_ddots);

    // .. Create the context for computing sum of traces of matrix products.
    traces_by_ddots_context->nt = nt;
    traces_by_ddots_context->trans = 'T'; // WW2[p] is symmetric.
    traces_by_ddots_context->m = m;
    traces_by_ddots_context->n = n;
    traces_by_ddots_context->A = WW2;
    traces_by_ddots_context->ldA = ldB;
    traces_by_ddots_context->B = WW2;
    traces_by_ddots_context->ldB = ldB;
    traces_by_ddots_context->loc_sum = loc_sum;

    // .. Call the working function for computing sum of traces of matrix products.
    function = traces_by_ddots;
	argument = traces_by_ddots_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	traces_by_ddots( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

	for ( i = 0; i <= nt; i++ )
		y += loc_sum[i];

    //------------------------------------------------------------------

    y /= 2;

    mkl_set_num_threads(max_num_threads+1);

	free(LW);
	free(tau);

    return y;
}

double nelder_mead_curve( int curve, double tol, double fX )
{
    int fun_eval, nt, shrink;
    double alpha, beta, c, gamma, delta, F0, F1, Fc, Fe, Fr, pomd, *pomp, *Q0, *Q1, t0, t0_old, t1, t1_old, tc, te, tr;

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
        //------------------------------------------------------------------

        nt = imin(max_num_threads,m*n-1);

        // .. Create the context for returning (removing) diagonals.
        remove_diagonals_context->nt = nt;
        remove_diagonals_context->m = m;
        remove_diagonals_context->n = n;
        remove_diagonals_context->A = B;
        remove_diagonals_context->D = D;

        // .. Call the working function for returning diagonals.
        function = return_diagonals;
        argument = remove_diagonals_context;

        // .. Wake the threads.
        barrier_wait(barrier);

        // .. Master thread is also working on its part.
        return_diagonals( max_num_threads, -1, argument );

        // .. Synchronize the threads.
        barrier_wait(barrier);

        //------------------------------------------------------------------

		dlaset( "A", &n, &n, &dZERO, &dONE, W, &n );
        F1 = F_geod(t1, W1);
    }
    else
    {
		dlacpy( "A", &n, &n, X, &ldX, W, &n );
        F1 = F_retr(t1, W1);
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
        		F1 = F_geod(t1, W1);
    		}
    		else
   	 		{
				dlacpy( "A", &n, &n, X, &ldX, W, &n );
        		F1 = F_retr(t1, W1);
    		}
    		Q0 = W;
    		Q1 = W1;
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
            Fr = F_geod(tr, W2);
        }
        else
        {
            Fr = F_retr(tr, W2);
        }
		fun_eval++;

        // .. Reflect will never happen.

        // .. Expand
        if ( Fr < F0 ) 
        {
            te = c + gamma * (tr - c);
            if ( curve == 0 )
            {
                Fe = F_geod(te, W3);
            }
            else
            {
                Fe = F_retr(te, W3);
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
                    Fc = F_geod(tc, W4); 
                }
                else
                {
                    Fc = F_retr(tc, W4);
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
                    Fc = F_geod(tc, W4);
                }
                else
                {
                    Fc = F_retr(tc, W4);
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
                F1 = F_geod(t1, W1);
            }
            else
            {
                F1 = F_retr(t1, W1);
            }
			fun_eval++;
            Q1 = W1;
        }
    }

    if ( F0 <= F1 ) 
    {
        dlacpy( "A", &n, &n, Q0, &n, eAt, &n );
        return t0;
    }
    else
    {
        dlacpy( "A", &n, &n, Q1, &n, eAt, &n );
        return t1;
    }

}

double armijo_backtracking_curve( int curve, double F0 )
{
    int func_eval, max_steps, nt;
    double alpha, beta, dF_curveX, F1, nH, sigma;

    // .. dF_curveX=trace(FX'*H);
    dF_curveX = trace_of_product( 'T', FX, n, H, n );

    beta = 0.5;
    sigma = pow(2,-13);
    max_steps = 25;

    nH = dlange( "F", &n, &n, H, &n, W);
    alpha = 1.0 / nH;

    if ( curve == 0 )
    {
        //------------------------------------------------------------------

        // .. Return diagonals to B[p].
        
        nt = imin(max_num_threads,m*n-1);

        // .. Create the context for returning (removing) diagonals.
        remove_diagonals_context->nt = nt;
        remove_diagonals_context->m = m;
        remove_diagonals_context->n = n;
        remove_diagonals_context->A = B;
        remove_diagonals_context->D = D;

        // .. Call the working function for returning diagonals.
        function = return_diagonals;
        argument = remove_diagonals_context;

        // .. Wake the threads.
        barrier_wait(barrier);

        // .. Master thread is also working on its part.
        return_diagonals( max_num_threads, -1, argument );

        // .. Synchronize the threads.
        barrier_wait(barrier);

        //------------------------------------------------------------------
        F1 = F_geod(alpha, W1);
    }
    else
    {
        F1 = F_retr(alpha, W1);
    }
    func_eval = 1;

    while ( F1 > F0 + sigma * alpha * dF_curveX )
    {
        alpha = beta * alpha;
        if ( curve == 0 )
        {
            F1 = F_geod(alpha, W1);
        }
        else
        {
            F1 = F_retr(alpha, W1);
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
    return alpha;
}

int cg_jad_stiefel_var_par( int curve, int linsear, int conj, int mm, int nm, int mklt_ddots, int t_ddots, int mklt_gemms, int t_gemms, double tol, double **Am, int *ldAm, double *Xm, int ldXm )
{
    int i, iter, nt;
    double tol_linsear = nm*dlamch( "E" );
    double beta, cond, fX, f0, FXX1, FXX2, Hess1, Hess2, t;

    m = mm;
    n = nm;
	num_mkl_threads_ddots = mklt_ddots;
	nt_ddots = t_ddots;
	num_mkl_threads_gemms = mklt_gemms;
	nt_gemms = t_gemms;

    prepare_CPU( Am, ldAm, Xm, ldXm );

    F_X();

    dgemm( "T", "N", &n, &n, &n, &dONE, FX, &n, X, &ldX, &dZERO, W1, &n );
    dlacpy( "A", &n, &n, FX, &n, G, &n );
    dgemm( "N", "N", &n, &n, &n, &dmHALF, X, &ldX, W1, &n, &dHALF, G, &n );

    scaling_of_matrix_fun( n, -1, G, n, H, n );

    if ( curve == 0 )
    {
        dgemm( "T", "N", &n, &n, &n, &dONE, X, &ldX, H, &n, &dZERO, Ag, &n );
        dgemm( "N", "N", &n, &n, &n, &dONE, Ag, &n, Ag, &n, &dZERO, Ag2, &n );
        dgemm( "N", "N", &n, &n, &n, &dONE, Ag2, &n, Ag2, &n, &dZERO, Ag4, &n );
        dgemm( "N", "N", &n, &n, &n, &dONE, Ag4, &n, Ag2, &n, &dZERO, Ag6, &n );
        n1Ag = dlange( "1", &n, &n, Ag, &n, W1 );
    }

    iter = 0;
	fX=F();
	f0 = fX;
    cond = ( fX > tol * f0 );

   	while ( cond )
    {
        iter++;

        if ( linsear == 0 )
        {
            // .. Nelder-Mead
            t = nelder_mead_curve( curve, tol_linsear, fX );
		}
        else
        {
            // .. Armijo's backtracking
            t = armijo_backtracking_curve( curve, fX );
        }

		// .. If t=0 switch to -gradient.
		if ( t == 0 )
		{
			scaling_of_matrix_fun( n, -1, G, n, H, n );
			if ( linsear == 0 )
        	{
            	// .. Nelder-Mead
            	t = nelder_mead_curve( curve, tol_linsear, fX );
			}
        	else
        	{
            	// .. Armijo's backtracking
            	t = armijo_backtracking_curve( curve, fX );
        	}

		}

        if ( curve == 0 )
        {
            if ( conj > 0 )
            {
                // .. Old approximation on manifold is in X0.
                dlacpy( "A", &n, &n, X, &ldX, X0, &n );
            }
            dgemm( "N", "N", &n, &n, &n, &dONE, X, &ldX, eAt, &n, &dZERO, W1, &n );
            dlacpy( "A", &n, &n, W1, &n, X, &ldX );
            dgemm( "N", "N", &n, &n, &n, &dONE, X, &n, Ag, &n, &dZERO, Xi, &n );
        }
        else
        {
            dlacpy( "A", &n, &n, eAt, &n, X, &n );
            // .. Computing Xi = X*(X'*H-H'*Q)/2;
            dgemm( "T", "N", &n, &n, &n, &dONE, X, &ldX, H, &n, &dZERO, Xi, &n );
            lin_comb_trans_fun( n, 0.5, -0.5, Xi, n, W1, n );
            dgemm( "N", "N", &n, &n, &n, &dONE, X, &ldX, W1, &n, &dZERO, Xi, &n );
        }

        F_X();

		fX=F();

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
            lin_comb_trans_fun( n, 1, 1, W, n, W1, n );
            dgemm( "N", "T", &n, &n, &n, &dONE, W1, &n, Xi, &n, &dZERO, I, &n );

            FXX1 = F_XX( G, n, Xi, n );
            FXX2 = F_XX( Xi, n, Xi, n );

            Hess1 = FXX1 - trace_of_product( 'N', I, n, G, n ) / 2;
            Hess2 = FXX2 - trace_of_product( 'N', I, n, Xi, n ) / 2;

            beta = Hess1 / Hess2;
        }

        if ( conj == 1 )
        {
            // .. Polak-Ribiere
            if ( curve == 0 )
            {
                ss_alg_exp(t/2, W, n );
                dgemm( "N", "N", &n, &n, &n, &dONE, X0, &n, W, &n, &dZERO, W1, &n );
                dgemm( "N", "T", &n, &n, &n, &dONE, W1, &n, X0, &n, &dZERO, W2, &n );
                dgemm( "N", "N", &n, &n, &n, &dONE, W2, &n, G0, &n, &dZERO, W1, &n );
                dgemm( "N", "N", &n, &n, &n, &dONE, W1, &n, W, &n, &dZERO, W2, &n );
            }
            else
            {
                dgemm( "T", "N", &n, &n, &n, &dONE, X, &ldX, G0, &n, &dZERO, W, &n );
                lin_comb_trans_fun( n, 0.5, -0.5, W, n, W1, n );
                dgemm( "N", "N", &n, &n, &n, &dONE, X, &ldX, W1, &n, &dZERO, W2, &n );
            }
            // .. Translated old gradient G0 is now in W2
            lin_comb_of_2matrices_fun( n, 1, G, n, -1, W2, n, W1, n );

            Hess1 = trace_of_product( 'T', W1, n, G, n );
            Hess2 = trace_of_product( 'T', G0, n, G0, n );

            beta = Hess1 / Hess2;
        }

        if ( conj == 2 )
        {
            // .. Fletcher-Reeves
            Hess1 = trace_of_product( 'T', G, n, G, n );
            Hess2 = trace_of_product( 'T', G0, n, G0, n );

            beta = Hess1 / Hess2;
        }

        if ( iter % (n*(n-1)/2) == 0 )
		{
            scaling_of_matrix_fun( n, -1, G, n, H, n );
		}
        else
		{
            lin_comb_of_2matrices_fun( n, -1, G, n, beta, Xi, n, H, n );
		}

        if ( curve == 0 )
        {
            dgemm( "T", "N", &n, &n, &n, &dONE, X, &ldX, H, &n, &dZERO, Ag, &n );
            dgemm( "N", "N", &n, &n, &n, &dONE, Ag, &n, Ag, &n, &dZERO, Ag2, &n );
            dgemm( "N", "N", &n, &n, &n, &dONE, Ag2, &n, Ag2, &n, &dZERO, Ag4, &n );
            dgemm( "N", "N", &n, &n, &n, &dONE, Ag4, &n, Ag2, &n, &dZERO, Ag6, &n );
            n1Ag = dlange( "1", &n, &n, Ag, &n, W1 );
        }
    }

    finalize_CPU();

	return iter;

}
