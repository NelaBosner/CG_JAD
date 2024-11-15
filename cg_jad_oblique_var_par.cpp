// Parallel implementation of Riemannian conjugate gradient method on the oblique manifold for joint approximate diagonalization.
// This file contains the main function implementing the method, and other auxiliary functions.
//
// The main function call has the form:
// int cg_jad_oblique_var_par( int curve, int linsear, int conj, int m, int n, int mklt_ddots, int t_ddots, int mklt_gemms, int t_gemms, double tol, double **A, int *ldA, double *X, int ldX );
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

static int m, n, ldX;
static int *ldA, *ldB;
static double dZERO=0.0, dONE=1.0, dmONE=-1.0, dHALF=0.5, dmHALF=-0.5, dTWO=2.0;
static double *loc_sum;
static double **A, **AX, **B, **B1, **B2, *C1, *C2, *CSL[3], *Ct, **DD, **D[10], *FX, *G, *G0, *H, *I, *Lam, *St, *W1, *W2, **WW1, **WW2, *X, *X0, *Xi;

// .. Parameters for MKL.
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
// .. Master is working on the first block of objects, with rank = 0, hence number of threads is equal to nt+1.
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

// .. Working function for removing diagonals of A[p], and storing them as vectors.
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

// .. Argument structure for computing diagonal of a matrix product diag(A^T*B), stored as a vector.
typedef struct diagonal_of_transpose_product_context_s
{
    int nt;
	int n;
    double *A;
    int ldA;
    double *B;
    int ldB;
    double *C;
} diagonal_of_transpose_product_context_t;
static diagonal_of_transpose_product_context_t *diagonal_of_transpose_product_context;

// .. Working function for computing diagonal of a matrix product diag(A^T*B), stored as a vector.
static void diagonal_of_transpose_product( int numthreads, int rank, void *arg ) // #objects_for_parallelization = n
{
	// .. Extract context argument.
	diagonal_of_transpose_product_context_t *context = (diagonal_of_transpose_product_context_t*)arg;

	// .. Extract variables from the context struct.
    int nt = context->nt;
    int n = context->n;
    double *A = context->A;
    int ldA = context->ldA;
    double *B = context->B;
    int ldB = context->ldB;
    double *C = context->C;

    int incx = 1;
    int j;

    if ( rank < nt )
    {
        partition_limits_t limits = partition( n, nt+1, rank+1 );

        for( j = limits.start; j <= limits.end; j++ )
        {
            C[j] = ddot( &n, &A[j*ldA], &incx, &B[j*ldB], &incx );
        }
    }

}

// .. Argument structure for computing a product of a matrix A with diagonal matrix C stored in a vector, R = E + alpha * A * C.
typedef struct product_of_matrix_and_diagonal_context_s
{
    int nt;
	int n;
    double *E;
    int ldE;
    double alpha;
    double *A;
    int ldA;
    double *C;
    double *R;
    int ldR;
} product_of_matrix_and_diagonal_context_t;
static product_of_matrix_and_diagonal_context_t *product_of_matrix_and_diagonal_context;

// .. Working function for computing a product of a matrix A with diagonal matrix C stored in a vector, R = E + alpha * A * C.
static void product_of_matrix_and_diagonal( int numthreads, int rank, void *arg ) // #objects_for_parallelization = n*n
{
	// .. Extract context argument.
	product_of_matrix_and_diagonal_context_t *context = (product_of_matrix_and_diagonal_context_t*)arg;

	// .. Extract variables from the context struct.
    int nt = context->nt;
    int n = context->n;
    double *E = context->E;
    int ldE = context->ldE;
    double alpha = context->alpha;
    double *A = context->A;
    int ldA = context->ldA;
    double *C = context->C;
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
            R[i+j*ldR] = E[i+j*ldE] + alpha * A[i+j*ldA] * C[j];
        }
    }
}

// .. Argument structure for computing diagonal matrix Lambda stored as a vector Lam.
typedef struct compute_Lam_context_s
{
    int nt;
	int n;
    double *H;
    int ldH;
    double *Lam;
} compute_Lam_context_t;
static compute_Lam_context_t *compute_Lam_context;

// .. Working function for computing diagonal matrix Lambda stored as a vector Lam.
static void compute_Lam( int numthreads, int rank, void *arg ) // #objects_for_parallelization = n
{
	// .. Extract context argument.
	compute_Lam_context_t *context = (compute_Lam_context_t*)arg;

	// .. Extract variables from the context struct.
    int nt = context->nt;
    int n = context->n;
    double *H = context->H;
    int ldH = context->ldH;
    double *Lam = context->Lam;

    int incx = 1;
    int j;

    if ( rank < nt )
    {
        partition_limits_t limits = partition( n, nt+1, rank+1 );

        for( j = limits.start; j <= limits.end; j++ )
        {
            Lam[j] = dnrm2( &n, &H[j*ldH], &incx );
        }
    }

}

// .. Argument structure for computing diagonal matrices Ct, St, Ct^2, Ct*St*Lam^(-1), St^2*Lam^(-2) or (I+t^2*lam^2)^(-1).
typedef struct compute_diag_Ct_St_context_s
{
    int nt;
    double t;
	int n;
    int curve;
    double *Lam;
    double *Ct;
    double *St;
    double **CSL;
} compute_diag_Ct_St_context_t;
static compute_diag_Ct_St_context_t *compute_diag_Ct_St_context;

// .. Working function for computing diagonal matrices Ct^2, Ct*St*Lam^(-1), St^2*Lam^(-2) or (I+t^2*lam^2)^(-1).
static void compute_diag_CSL( int numthreads, int rank, void *arg ) // #objects_for_parallelization = n
{
	// .. Extract context argument.
	compute_diag_Ct_St_context_t *context = (compute_diag_Ct_St_context_t*)arg;

	// .. Extract variables from the context struct.
    int nt = context->nt;
    double t = context->t;
    int n = context->n;
    int curve = context->curve;
    double *Lam = context->Lam;
    double *Ct = context->Ct;
    double *St = context->St;
    double **CSL = context->CSL;

    int j;
    double aux1, aux2;

    if ( rank < nt )
    {
        partition_limits_t limits = partition( n, nt+1, rank+1 );

        for( j = limits.start; j <= limits.end; j++ )
        {
            if ( curve == 0 )
            {
				aux1 = cos( Lam[j] * t );
            	aux2 = sin( Lam[j] * t );
                CSL[0][j] = aux1 * aux1;
                CSL[1][j] = aux1 * aux2 / Lam[j];
                CSL[2][j] = aux2 * aux2 / (Lam[j] * Lam[j]);
            }
            else
            {
                aux1 = 1 + t * t * Lam[j] * Lam[j];
                CSL[0][j] = 1 / aux1;
            }
        }
    }

}

// .. Working function for computing diagonal matrices Ct, St or (I+t^2*lam^2)^(-1/2).
static void compute_diag_Ct_St( int numthreads, int rank, void *arg ) // #objects_for_parallelization = n
{
	// .. Extract context argument.
	compute_diag_Ct_St_context_t *context = (compute_diag_Ct_St_context_t*)arg;

	// .. Extract variables from the context struct.
    int nt = context->nt;
    double t = context->t;
    int n = context->n;
    int curve = context->curve;
    double *Lam = context->Lam;
    double *Ct = context->Ct;
    double *St = context->St;
    double **CSL = context->CSL;

    int j;
    double aux;

    if ( rank < nt )
    {
        partition_limits_t limits = partition( n, nt+1, rank+1 );

        for( j = limits.start; j <= limits.end; j++ )
        {
			if ( curve == 0 )
            {
            	Ct[j] = cos( Lam[j] * t );
            	St[j] = sin( Lam[j] * t );
			}
            else
            {
                aux = 1 + t * t * Lam[j] * Lam[j];
                Ct[j] = 1 / sqrt(aux);
            }
        }
    }

}

// .. Argument structure for computing point on and tangent to the geodesic or the retraction.
typedef struct curve_pt_context_s
{
    int nt;
	int n;
	int curve;
	double t;
    double *X;
    int ldX;
    double *H;
    int ldH;
    double *Ct;
    double *St;
    double *Lam;
    double *Xi;
    int ldXi;
} curve_pt_context_t;
static curve_pt_context_t *curve_pt_context;

// .. Working function for computing point on and tangent to the geodesic or the retraction.
static void curve_pt( int numthreads, int rank, void *arg ) // #objects_for_parallelization = n*n
{
	// .. Extract context argument.
	curve_pt_context_t *context = (curve_pt_context_t*)arg;

	// .. Extract variables from the context struct.
    int nt = context->nt;
    int n = context->n;
	int curve = context->curve;
	double t = context->t;
    double *X = context->X;
    int ldX = context->ldX;
    double *H = context->H;
    int ldH = context->ldH;
    double *Ct = context->Ct;
    double *St = context->St;
    double *Lam = context->Lam;
    double *Xi = context->Xi;
    int ldXi = context->ldXi;

    int i, j, l;

    if ( rank < nt )
    {
        partition_limits_t limits = partition( n*n, nt+1, rank+1 );

        for( l = limits.start; l <= limits.end; l++ )
        {
            i = l % n;
            j = l / n;
			if ( curve == 0 )
			{
            	Xi[i+j*ldXi] = -X[i+j*ldX] * St[j] * Lam[j] + H[i+j*ldH] * Ct[j];
            	X[i+j*ldX] = X[i+j*ldX] * Ct[j] + H[i+j*ldH] * St[j] / Lam[j];
			}
			else
			{
				X[i+j*ldX] = (X[i+j*ldX] + t * H[i+j*ldH]) * Ct[j];
				Xi[i+j*ldXi] = H[i+j*ldH] - t * X[i+j*ldX] * Lam[j] * Lam[j] * Ct[j];
			}
        }
    }
}

// .. Argument structure for computing vector transport of V along geodesic or retraction.
typedef struct vector_trans_context_s
{
    int nt;
	int n;
	int curve;
	double t;
    double *X;
    int ldX;
    double *H;
    int ldH;
	double *V;
	int ldV;
	double *M; // diag(H(:,i)'*V(:,i))
    double *Ct;
    double *St;
    double *Lam;
    double *pV;
    int ldpV;
} vector_trans_context_t;
static vector_trans_context_t *vector_trans_context;

// .. Working function for computing vector transport of V along geodesic or retraction.
static void vector_trans( int numthreads, int rank, void *arg ) // #objects_for_parallelization = n*n
{
	// .. Extract context argument.
	vector_trans_context_t *context = (vector_trans_context_t*)arg;

	// .. Extract variables from the context struct.
    int nt = context->nt;
    int n = context->n;
	int curve = context->curve;
	double t = context->t;
    double *X = context->X;
    int ldX = context->ldX;
    double *H = context->H;
    int ldH = context->ldH;
	double *V = context->V;
    int ldV = context->ldV;
	double *M = context->M;
    double *Ct = context->Ct;
    double *St = context->St;
    double *Lam = context->Lam;
    double *pV = context->pV;
    int ldpV = context->ldpV;

    int i, j, l;

    if ( rank < nt )
    {
        partition_limits_t limits = partition( n*n, nt+1, rank+1 );

        for( l = limits.start; l <= limits.end; l++ )
        {
            i = l % n;
            j = l / n;
			if ( curve == 0 )
			{
				pV[i+j*ldpV] = V[i+j*ldV] - X[i+j*ldX] * St[j] * M[j] / Lam[j] + H[i+j*ldH] * (Ct[j] - 1) * M[j] / (Lam[j] * Lam[j]);
			}
			else
			{
				pV[i+j*ldpV] = V[i+j*ldV] - t * X[i+j*ldX] * M[j] * Ct[j];
			}
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

// .. Working function for summing m matrices A[p].
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

// .. Argument structure for computing diagonal matrices D1[p],...,D10[p].
typedef struct compute_diag_D_context_s
{
	int nt;
	int m;
	int n;
    int curve;
    double **B;
    double **B1;
    double **B2;
    double **CSL;
    double ***D;
} compute_diag_D_context_t;
static compute_diag_D_context_t *compute_diag_D_context;

// .. Working function for computing diagonal matrices D1[p],...,D10[p].
static void compute_diag_D( int numthreads, int rank, void *arg ) // #objects_for_parallelization = 10*m*n
{
	// .. Extract context argument.
	compute_diag_D_context_t *context = (compute_diag_D_context_t*)arg;

	// .. Extract variables from the context struct.
	int nt = context->nt;
    int m = context->m;
    int n = context->n;
    int curve = context->curve;
    double **B = context->B;
    double **B1 = context->B1;
    double **B2 = context->B2;
    double **CSL = context->CSL;
    double ***D = context->D;

    int id, id2, j, k, l, p;
    double aux, res;

    if ( rank < nt )
    {
        partition_limits_t limits = partition( 10*m*n, nt+1, rank+1 );

        for( l = limits.start; l <= limits.end; l++ )
        {
            id = l / (m*n);
            id2 = l % (m*n);
            p = id2 / n;
            j = id2 % n;

            res = 0;
            for ( k = 0; k < n; k++)
            {
                if ( k != j )
                {
                    switch( id )
                    {
                        case 0: // D1
                        {
                            // .. The same for both curves.                          
                            res += B[p][j+k*n] * CSL[0][k] * B[p][k+j*n];
                            break;
                        }
                        case 1: // D2
                        {
                            // .. The same for both curves.
                            res += B[p][j+k*n] * CSL[0][k] * B1[p][k+j*n];
                            break;
                        }
                        case 2: // D3
                        {
                            if ( curve == 0 )
                                aux = CSL[1][k];
                            else
                                aux = CSL[0][k];
                            res += B[p][j+k*n] * aux * B1[p][j+k*n];
                            break;
                        }
                        case 3: // D4
                        {
                            if ( curve == 0 )
                                aux = CSL[1][k];
                            else
                                aux = CSL[0][k];
                            res += B[p][j+k*n] * aux * B2[p][k+j*n];
                            break;
                        }
                        case 4: // D5
                        {                  
                            if ( curve == 0 )
                                aux = CSL[1][k];
                            else
                                aux = CSL[0][k];
                            res += B1[p][j+k*n] * aux * B1[p][k+j*n];
                            break;
                        }
                        case 5: // D6
                        {
                            if ( curve == 0 )
                                aux = CSL[2][k];
                            else
                                aux = CSL[0][k];
                            res += B1[p][j+k*n] * aux * B1[p][j+k*n];
                            break;
                        }
                        case 6: // D7
                        {
                            // .. The same for both curves.
                            res += B1[p][k+j*n] * CSL[0][k] * B1[p][k+j*n];
                            break;
                        }
                        case 7: // D8
                        {
                            if ( curve == 0 )
                                aux = CSL[2][k];
                            else
                                aux = CSL[0][k];
                            res += B1[p][j+k*n] * aux * B2[p][k+j*n];
                            break;
                        }
                        case 8: // D9
                        {
                            if ( curve == 0 )
                                aux = CSL[1][k];
                            else
                                aux = CSL[0][k];
                            res += B1[p][k+j*n] * aux * B2[p][k+j*n];
                            break;
                        }
                        case 9: // D10
                        {
                            if ( curve == 0 )
                                aux = CSL[2][k];
                            else
                                aux = CSL[0][k];
                            res += B2[p][j+k*n] * aux * B2[p][k+j*n];
                            break;
                        }
                    }
                }
            }

            D[id][p][j] = res;
        }
    }
}

// .. Argument structure for computing sum of 10*m traces of diagonal matrices for F_curve.
typedef struct compute_sum_traces_for_F_curve_context_s
{
	int nt;
	int m;
	int n;
    int curve;
	double t;
    double **CSL;
    double ***D;
    double *loc_sum;
} compute_sum_traces_for_F_curve_context_t;
static compute_sum_traces_for_F_curve_context_t *compute_sum_traces_for_F_curve_context;

// .. Working function for computing sum of 10*m traces of diagonal matrices for F_curve (F_geod or F_retr).
static void compute_sum_traces_for_F_curve( int numthreads, int rank, void *arg ) // #objects_for_parallelization = 10*m
{
	// .. Extract context argument.
	compute_sum_traces_for_F_curve_context_t *context = (compute_sum_traces_for_F_curve_context_t*)arg;

	// .. Extract variables from the context struct.
	int nt = context->nt;
    int m = context->m;
    int n = context->n;
    int curve = context->curve;
	double t = context->t;
    double **CSL = context->CSL;
    double ***D = context->D;
    double *loc_sum = context->loc_sum;

    int incx = 1;
    int id, l, p;
    double sum = 0;

    if ( rank < nt )
    {
        partition_limits_t limits = partition( 10*m, nt+1, rank+1 );

        for( l = limits.start; l <= limits.end; l++ )
        {
            id = l / m;
            p = l % m;

            switch( id )
            {
                case 0:
                {
                    if ( curve == 0 )
                        sum += ddot( &n, D[0][p], &incx, CSL[0], &incx );
                    else
                        sum += ddot( &n, D[0][p], &incx, CSL[0], &incx );
					break;
                }
                case 1:
                {
                    if ( curve == 0 )
                        sum += 2 * ddot( &n, D[1][p], &incx, CSL[1], &incx );
                    else
                        sum += 2 * t * ddot( &n, D[1][p], &incx, CSL[0], &incx );
					break;
                }
                case 2:
                {
                    if ( curve == 0 )
                        sum += 2 * ddot( &n, D[2][p], &incx, CSL[0], &incx );
                    else
                        sum += 2 * t * ddot( &n, D[2][p], &incx, CSL[0], &incx );
					break;
                }
                case 3:
                {
                    if ( curve == 0 )
                        sum += 2 * ddot( &n, D[3][p], &incx, CSL[1], &incx );
                    else
                        sum += 2 * t * t * ddot( &n, D[3][p], &incx, CSL[0], &incx );
					break;
                }
                case 4:
                {
                    if ( curve == 0 )
                        sum += 2 * ddot( &n, D[4][p], &incx, CSL[1], &incx );
                    else
                        sum += 2 * t * t * ddot( &n, D[4][p], &incx, CSL[0], &incx );
					break;
                }
                case 5:
                {
                    if ( curve == 0 )
                        sum += ddot( &n, D[5][p], &incx, CSL[0], &incx );
                    else
                        sum += t * t * ddot( &n, D[5][p], &incx, CSL[0], &incx );
					break;
                }
                case 6:
                {
                    if ( curve == 0 )
                        sum += ddot( &n, D[6][p], &incx, CSL[2], &incx );
                    else
                        sum += t * t * ddot( &n, D[6][p], &incx, CSL[0], &incx );
					break;
                }
                case 7:
                {
                    if ( curve == 0 )
                        sum += 2 * ddot( &n, D[7][p], &incx, CSL[1], &incx );
                    else
                        sum += 2 * t * t * t * ddot( &n, D[7][p], &incx, CSL[0], &incx );
					break;
                }
                case 8:
                {
                    if ( curve == 0 )
                        sum += 2 * ddot( &n, D[8][p], &incx, CSL[2], &incx );
                    else
                        sum += 2 * t * t * t * ddot( &n, D[8][p], &incx, CSL[0], &incx );
					break;
                }
                case 9:
                {
                    if ( curve == 0 )
                        sum += ddot( &n, D[9][p], &incx, CSL[2], &incx );
                    else
                        sum += t * t * t * t * ddot( &n, D[9][p], &incx, CSL[0], &incx );
					break;
                }
            }
        }

        sum /= 2;
        loc_sum[rank+1] = sum;
    }
}

static void prepare_CPU( double **Am, int *ldAm, double *Xm, int ldXm )
{
    int i, p;

    threads = (pthread_t*) malloc(sizeof(pthread_t) * max_num_threads);
    wd = (struct thread_data**) malloc(max_num_threads*sizeof(struct thread_data*));
	for ( p = 0; p < max_num_threads; p++ )
		wd[p] = (struct thread_data*) malloc(sizeof(struct thread_data*));

    off_context = (off_context_t*) malloc(sizeof(off_context_t));
	remove_diagonals_context = (remove_diagonals_context_t*) malloc(sizeof(remove_diagonals_context_t));
	scaling_of_matrix_context = (scaling_of_matrix_context_t*) malloc(sizeof(scaling_of_matrix_context_t));
	lin_comb_of_2matrices_context = (lin_comb_of_2matrices_context_t*) malloc(sizeof(lin_comb_of_2matrices_context_t));
    diagonal_of_transpose_product_context = (diagonal_of_transpose_product_context_t*) malloc(sizeof(diagonal_of_transpose_product_context_t));
    product_of_matrix_and_diagonal_context = (product_of_matrix_and_diagonal_context_t*) malloc(sizeof(product_of_matrix_and_diagonal_context_t));
    curve_pt_context = (curve_pt_context_t*) malloc(sizeof(curve_pt_context_t));
	vector_trans_context = (vector_trans_context_t*) malloc(sizeof(vector_trans_context_t));
    compute_Lam_context = (compute_Lam_context_t*) malloc(sizeof(compute_Lam_context_t));
    compute_diag_Ct_St_context = (compute_diag_Ct_St_context_t*) malloc(sizeof(compute_diag_Ct_St_context_t));
	sum_of_matrices_context = (sum_of_matrices_context_t*) malloc(sizeof(sum_of_matrices_context_t));
    traces_by_ddots_context = (traces_by_ddots_context_t*) malloc(sizeof(traces_by_ddots_context_t));
    multiple_gemms_context = (multiple_gemms_context_t*) malloc(sizeof(multiple_gemms_context_t));
    compute_diag_D_context = (compute_diag_D_context_t*) malloc(sizeof(compute_diag_D_context_t));
    compute_sum_traces_for_F_curve_context = (compute_sum_traces_for_F_curve_context_t*) malloc(sizeof(compute_sum_traces_for_F_curve_context_t));



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
    B = (double**) malloc(m*sizeof(double*));
    ldB = (int*) malloc(m*sizeof(int));
    B1 = (double**) malloc(m*sizeof(double*));
    B2 = (double**) malloc(m*sizeof(double*));
    WW1 = (double**) malloc(m*sizeof(double*));
    WW2 = (double**) malloc(m*sizeof(double*));
	DD = (double**) malloc(m*sizeof(double*));
    for( p = 0; p < m; p++ )
    {
        AX[p] = (double*) malloc(n*n*sizeof(double));
        B[p] = (double*) malloc(n*n*sizeof(double));
        ldB[p] = n;
        B1[p] = (double*) malloc(n*n*sizeof(double));
        B2[p] = (double*) malloc(n*n*sizeof(double));
        WW1[p] = (double*) malloc(n*n*sizeof(double));
        WW2[p] = (double*) malloc(n*n*sizeof(double));
		DD[p] = (double*) malloc(n*sizeof(double));
    }

    for ( i = 0; i < 10; i++ )
    {
        D[i] = (double**) malloc(m*sizeof(double*));
        for ( p = 0; p < m; p++ )
        {
            D[i][p] = (double*) malloc(n*sizeof(double));
        }
    }
    for ( i = 0; i < 3; i++ )
    {
        CSL[i] = (double*) malloc(n*sizeof(double));
    }


    FX = (double*) malloc(n*n*sizeof(double));
    G = (double*) malloc(n*n*sizeof(double));
	G0 = (double*) malloc(n*n*sizeof(double));
    H = (double*) malloc(n*n*sizeof(double));
    W1 = (double*) malloc(n*n*sizeof(double));
    W2 = (double*) malloc(n*n*sizeof(double));
	X0 = (double*) malloc(n*n*sizeof(double));
    Xi = (double*) malloc(n*n*sizeof(double));

    C1 = (double*) malloc(n*sizeof(double));
    C2 = (double*) malloc(n*sizeof(double));
    Ct = (double*) malloc(n*sizeof(double));
    I = (double*) malloc(n*sizeof(double));
    Lam = (double*) malloc(n*sizeof(double));
    St = (double*) malloc(n*sizeof(double));

    loc_sum = (double*) malloc((max_num_threads+1)*sizeof(double));

    barrier = barrier_new(max_num_threads+1);

	// .. Activate the thread function.

    stopped = 0;
	for ( i = 0; i < max_num_threads; ++i )
    {
        wd[i]->rank = i;
        pthread_create(&threads[i], 0, waiting_thread_function, (void*) wd[i]);
    }
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
    free(diagonal_of_transpose_product_context);
    free(product_of_matrix_and_diagonal_context);
    free(curve_pt_context);
	free(vector_trans_context);
    free(compute_Lam_context);
    free(compute_diag_Ct_St_context);
	free(sum_of_matrices_context);
	free(traces_by_ddots_context);
    free(multiple_gemms_context);
    free(compute_diag_D_context);
    free(compute_sum_traces_for_F_curve_context);

    free(A);
    free(ldA);
    for( p = 0; p < m; p++ )
    {
        free(AX[p]);
        free(B[p]);
        free(B1[p]);
        free(B2[p]);
        free(WW1[p]);
        free(WW2[p]);
		free(DD[p]);
    }
    free(AX);
    free(B);
    free(ldB);
    free(B1);
    free(B2);
    free(WW1);
    free(WW2);
	free(DD);

    for ( i = 0; i < 10; i++ )
    {
        for ( p = 0; p < m; p++ )
        {
            free(D[i][p]);
        }
        free(D[i]);
    }
    for ( i = 0; i < 3; i++ )
    {
        free(CSL[i]);
    }

    free(FX);
    free(G);
	free(G0);
    free(H);
    free(W1);
    free(W2);
	free(X0);
    free(Xi);

    free(C1);
    free(C2);
    free(Ct);
    free(I);
    free(Lam);
    free(St);

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

// .. Function for for linear combination of two matrices.
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

// .. Function for computing diagonal of a matrix product diag(A^T*B).
static void diagonal_of_transpose_product_fun( double *A, int ldA, double *B, int ldB, double *C )
{
    int nt;

    //------------------------------------------------------------------

    nt = imin(nt_ddots,n-1);
    mkl_set_num_threads(num_mkl_threads_ddots);

    // .. Create the context for computing diagonal of a matrix product diag(A^T*B).
    diagonal_of_transpose_product_context->nt = nt;
    diagonal_of_transpose_product_context->n = n;
    diagonal_of_transpose_product_context->A = A;
    diagonal_of_transpose_product_context->ldA = ldA;
    diagonal_of_transpose_product_context->B = B;
    diagonal_of_transpose_product_context->ldB = ldB;
    diagonal_of_transpose_product_context->C = C;

    // .. Call the working function for computing diagonal of a matrix product diag(A^T*B).
    function = diagonal_of_transpose_product;
    argument = diagonal_of_transpose_product_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	diagonal_of_transpose_product( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    mkl_set_num_threads(max_num_threads+1);
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
    remove_diagonals_context->D = DD;

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
    multiple_gemms_context->ldB = ldB;

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
    traces_by_ddots_context->trans = 'T'; // We can transpose since B[p] are symmetric: tr(WW1[p]*B[p]) = tr(B[p]*WW1[p]) = tr(B[p]'*WW1[p]) = tr(WW1[p]'*B[p])
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

// .. Function for computing gradijent of F(X) and storing it in G.
static void gradF()
{
    int nt;

    //------------------------------------------------------------------

    nt = imin(nt_ddots,n-1);
    mkl_set_num_threads(num_mkl_threads_ddots);

    // .. Create the context for computing diagonal of a matrix product diag(A^T*B).
    diagonal_of_transpose_product_context->nt = nt;
    diagonal_of_transpose_product_context->n = n;
    diagonal_of_transpose_product_context->A = X;
    diagonal_of_transpose_product_context->ldA = ldX;
    diagonal_of_transpose_product_context->B = FX;
    diagonal_of_transpose_product_context->ldB = n;
    diagonal_of_transpose_product_context->C = I;

    // .. Call the working function for computing diagonal of a matrix product diag(A^T*B).
    function = diagonal_of_transpose_product;
    argument = diagonal_of_transpose_product_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	diagonal_of_transpose_product( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    nt = imin(max_num_threads,n*n-1);

    // .. Create the context for computing a product of a matrix A with diagonal matrix C, R = E + alpha * A * C.
    product_of_matrix_and_diagonal_context->nt = nt;
    product_of_matrix_and_diagonal_context->n = n;
    product_of_matrix_and_diagonal_context->E = FX;
    product_of_matrix_and_diagonal_context->ldE = n;
    product_of_matrix_and_diagonal_context->alpha = -1;
    product_of_matrix_and_diagonal_context->A = X;
    product_of_matrix_and_diagonal_context->ldA = ldX;
    product_of_matrix_and_diagonal_context->C = I;
    product_of_matrix_and_diagonal_context->R = G;
    product_of_matrix_and_diagonal_context->ldR = n;

    // .. Call the working function for computing a product of a matrix A with diagonal matrix C, R = E + alpha * A * C.
    function = product_of_matrix_and_diagonal;
    argument = product_of_matrix_and_diagonal_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	product_of_matrix_and_diagonal( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    mkl_set_num_threads(max_num_threads+1);

}

// .. Function for computing diagonal matrix Lambda stored as a vector Lam.
static void compute_Lam_fun()
{
    int nt;

    //------------------------------------------------------------------

    nt = imin(nt_ddots,n-1);
    mkl_set_num_threads(num_mkl_threads_ddots);

    // .. Create the context for computing diagonal matrix Lambda.
    compute_Lam_context->nt = nt;
    compute_Lam_context->n = n;
    compute_Lam_context->H = H;
    compute_Lam_context->ldH = n;
    compute_Lam_context->Lam = Lam;

    // .. Call the working function for computing diagonal matrix Lambda.
    function = compute_Lam;
    argument = compute_Lam_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	compute_Lam( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    mkl_set_num_threads(max_num_threads+1);

}

// .. Function for computing B1[p] and B2[p].
static void computeB1_B2()
{
    int nt;

    nt = imin(nt_gemms,m-1);
    mkl_set_num_threads(num_mkl_threads_gemms);

    //------------------------------------------------------------------

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
    multiple_gemms_context->X = H;
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

    // .. Create the context for multiple gemms.
    multiple_gemms_context->nt = nt;
    multiple_gemms_context->trans1 = 'T';
    multiple_gemms_context->trans2 = 'N';
    multiple_gemms_context->multiple_arg = 2;
	multiple_gemms_context->m = m;
	multiple_gemms_context->n = n;
    multiple_gemms_context->scal = dONE;
    multiple_gemms_context->A = WW1;
    multiple_gemms_context->ldA = ldB;
    multiple_gemms_context->B = B;
    multiple_gemms_context->ldB = ldB;
    multiple_gemms_context->X = X;
    multiple_gemms_context->ldX = ldX;
    multiple_gemms_context->C = B1;
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

    // .. Create the context for multiple gemms.
    multiple_gemms_context->nt = nt;
    multiple_gemms_context->trans1 = 'T';
    multiple_gemms_context->trans2 = 'N';
    multiple_gemms_context->multiple_arg = 2;
	multiple_gemms_context->m = m;
	multiple_gemms_context->n = n;
    multiple_gemms_context->scal = dONE;
    multiple_gemms_context->A = WW1;
    multiple_gemms_context->ldA = ldB;
    multiple_gemms_context->B = B;
    multiple_gemms_context->ldB = ldB;
    multiple_gemms_context->X = H;
    multiple_gemms_context->ldX = n;
    multiple_gemms_context->C = B2;
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

    mkl_set_num_threads(max_num_threads+1);

}

// .. Function for computing F on the geodesic or on the retraction.
static double F_curve( double t, int curve )
{
    int i, nt;
    double sum;

    //------------------------------------------------------------------

    nt = imin(max_num_threads,n-1);

    // .. Create the context for computing diagonal matrices stored in CSL.
    compute_diag_Ct_St_context->nt = nt;
    compute_diag_Ct_St_context->t = t;
    compute_diag_Ct_St_context->n = n;
	compute_diag_Ct_St_context->curve = curve;
    compute_diag_Ct_St_context->Lam = Lam;
    compute_diag_Ct_St_context->Ct = Ct;
    compute_diag_Ct_St_context->St = St;
    compute_diag_Ct_St_context->CSL = CSL;

    // .. Call the working function for computing diagonal matrices Ct and St.
    function = compute_diag_CSL;
    argument = compute_diag_Ct_St_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	compute_diag_CSL( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    nt = imin(max_num_threads,10*m*n-1);

    // .. Create the context for computing diagonal matrices D1[p],...,D10[p].
    compute_diag_D_context->nt = nt;
    compute_diag_D_context->m = m;
    compute_diag_D_context->n = n;
	compute_diag_D_context->curve = curve;
    compute_diag_D_context->B = B;
    compute_diag_D_context->B1 = B1;
    compute_diag_D_context->B2 = B2;
    compute_diag_D_context->CSL = CSL;
    compute_diag_D_context->D = D;

    // .. Call the working function for computing diagonal matrices D1[p],...,D10[p].
    function = compute_diag_D;
    argument = compute_diag_D_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	compute_diag_D( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    nt = imin(nt_ddots,10*m-1);
    mkl_set_num_threads(num_mkl_threads_ddots);

    // .. Create the context for computing sum of 10*m traces of diagonal matrices for F_curve.
    compute_sum_traces_for_F_curve_context->nt = nt;
    compute_sum_traces_for_F_curve_context->m = m;
    compute_sum_traces_for_F_curve_context->n = n;	
	compute_sum_traces_for_F_curve_context->curve = curve;
	compute_sum_traces_for_F_curve_context->t = t;
    compute_sum_traces_for_F_curve_context->CSL = CSL;
    compute_sum_traces_for_F_curve_context->D = D;
    compute_sum_traces_for_F_curve_context->loc_sum = loc_sum;

    // .. Call the working function for computing sum of 14*m traces of diagonal matrices for dF_geod.
    function = compute_sum_traces_for_F_curve;
    argument = compute_sum_traces_for_F_curve_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	compute_sum_traces_for_F_curve( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    mkl_set_num_threads(max_num_threads+1);

    sum = 0;
    for ( i = 0; i <= nt; i++ )
        sum += loc_sum[i];

    return sum;

}

double nelder_mead_curve( int curve, double tol, double fX )
{
    int fun_eval, nt, shrink;
    double alpha, beta, c, gamma, delta, F0, F1, Fc, Fe, Fr, pomd, t0, t0_old, t1, t1_old, tc, te, tr;

    alpha = 1.0;
    beta = 0.5;
    gamma = 2;
    delta = 0.5;

    t0 = 0.0;
    t1 = 1/sqrt(fX);
	t0_old = t0;
	t1_old = t1;

	F0 = fX;
	F1 = F_curve( t1, curve );
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
    		F1 = F_curve( t1, curve );
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
		Fr = F_curve(tr, curve);
		fun_eval++;

        // .. Reflect will never happen.

        // .. Expand
        if ( Fr < F0 ) 
        {
            te = c + gamma * (tr - c);
			Fe = F_curve(te, curve);
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
				Fc = F_curve(tc, curve);
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
				Fc = F_curve(tc, curve);
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
			F1 = F_curve(t1, curve);
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

double armijo_backtracking_curve( int curve, double F0 )
{
    int func_eval, max_steps, nt;
    double alpha, beta, dF_curveX, F1, nH, sigma;

    // .. dF_curveX=trace(FX'*H);
    dF_curveX = trace_of_product( 'T', FX, n, H, n );

    beta = 0.5;
    sigma = pow(2,-13);
    max_steps = 25;

    nH = dlange( "F", &n, &n, H, &n, W1);
    alpha = 1.0 / nH;

	F1 = F_curve(alpha, curve);
    func_eval = 1;

    while ( F1 > F0 + sigma * alpha * dF_curveX )
    {
        alpha = beta * alpha;
		F1 = F_curve(alpha, curve);
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

    return alpha;
}

// .. Function for computing a new point on geodesic or retraction, and the tangent on that point.
static void compute_new_point_tangent_curve( int curve, double t )
{
    int nt;

	//------------------------------------------------------------------

	nt = imin(max_num_threads,n*n-1);

	// .. Creat context for computing diagonal matrices Ct, St or (I+t^2*lam^2)^(-1/2).
    compute_diag_Ct_St_context->nt = nt;
    compute_diag_Ct_St_context->t = t;
	compute_diag_Ct_St_context->n = n;
    compute_diag_Ct_St_context->curve = curve;
    compute_diag_Ct_St_context->Lam = Lam;
    compute_diag_Ct_St_context->Ct = Ct;
    compute_diag_Ct_St_context->St = St;
    compute_diag_Ct_St_context->CSL = CSL;

	// .. Call the working function for computing diagonal matrices Ct, St or (I+t^2*lam^2)^(-1/2).
	function = compute_diag_Ct_St;
    argument = compute_diag_Ct_St_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	compute_diag_Ct_St( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------

    nt = imin(max_num_threads,n*n-1);

    // .. Create the context for computing point on and tangent to the geodesic or retraction.
    curve_pt_context->nt = nt;
    curve_pt_context->n = n;
	curve_pt_context->curve = curve;
	curve_pt_context->t = t;
    curve_pt_context->X = X;
    curve_pt_context->ldX = ldX;
    curve_pt_context->H = H;
    curve_pt_context->ldH = n;
    curve_pt_context->Ct = Ct;
    curve_pt_context->St = St;
    curve_pt_context->Lam = Lam;
    curve_pt_context->Xi = Xi;
    curve_pt_context->ldXi = n;

    // .. Call the working function for computing point on and tangent to the geodesic or retraction.
    function = curve_pt;
    argument = curve_pt_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	curve_pt( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------
}

// .. Function for computing vector transport of V along geodesic or retraction.
static void compute_vector_trans_fun( int curve, double t, double *X, int ldX, double *V, int ldV, double *pV, int ldpV )
{
    int nt;

    //------------------------------------------------------------------

	diagonal_of_transpose_product_fun( H, n, V, ldV, I );

    nt = imin(max_num_threads,n*n-1);

    // .. Create the context for computing point on and tangent to the geodesic or retraction.
    vector_trans_context->nt = nt;
    vector_trans_context->n = n;
	vector_trans_context->curve = curve;
	vector_trans_context->t = t;
    vector_trans_context->X = X;
    vector_trans_context->ldX = ldX;
    vector_trans_context->H = H;
    vector_trans_context->ldH = n;
	vector_trans_context->V = V;
    vector_trans_context->ldV = ldV;
	vector_trans_context->M = I;
    vector_trans_context->Ct = Ct;
    vector_trans_context->St = St;
    vector_trans_context->Lam = Lam;
    vector_trans_context->pV = pV;
    vector_trans_context->ldpV = ldpV;

    // .. Call the working function for computing point on and tangent to the geodesic or retraction.
    function = vector_trans;
    argument = vector_trans_context;

    // .. Wake the threads.
	barrier_wait(barrier);

	// .. Master thread is also working on its part.
	vector_trans( max_num_threads, -1, argument );

	// .. Synchronize the threads.
	barrier_wait(barrier);

    //------------------------------------------------------------------
}

int cg_jad_oblique_var_par( int curve, int linsear, int conj, int mm, int nm, int mklt_ddots, int t_ddots, int mklt_gemms, int t_gemms, double tol, double **Am, int *ldAm, double *Xm, int ldXm )
{
	int incx = 1;
    int iter;
	double tol_linsear = nm*dlamch( "E" );
    double beta, cond, fX, f0, FXX1, FXX2, Hess1, Hess2, res, ro, ro0, t;

    m = mm;
    n = nm;
	num_mkl_threads_ddots = mklt_ddots;
	nt_ddots = t_ddots;
	num_mkl_threads_gemms = mklt_gemms;
	nt_gemms = t_gemms;

    prepare_CPU( Am, ldAm, Xm, ldXm );

    F_X();
    gradF();

    scaling_of_matrix_fun( n, -1, G, n, H, n );

    compute_Lam_fun();

    computeB1_B2();

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
            if ( conj == 1 )
            {
                // .. Old approximation on manifold is in X0.
                dlacpy( "A", &n, &n, X, &ldX, X0, &n );
            }
		}
		compute_new_point_tangent_curve( curve, t );

        F_X();

        fX=F();

        cond = ( fX > tol * f0 ) && (iter<1000); 

        if ( !cond )
		{
            break;
		}

		if ( conj > 0 )
		{
            // .. Old gradient on manifold is in G0.
            dlacpy( "A", &n, &n, G, &n, G0, &n );
        }

		// .. diag(X'*FX) is in I.
        gradF();

		if ( conj == 0 )
		{
        	diagonal_of_transpose_product_fun( G, n, Xi, n, C1 );
        	diagonal_of_transpose_product_fun( Xi, n, Xi, n, C2 );

        	FXX1 = F_XX( G, n, Xi, n );
        	FXX2 = F_XX( Xi, n, Xi, n );

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
				compute_vector_trans_fun( curve, t, X0, n, G0, n, W1, n );
            }
            else
            {
				compute_vector_trans_fun( curve, t, X, ldX, G0, n, W1, n );
            }
            // .. Translated old gradient G0 is now in W1
            lin_comb_of_2matrices_fun( n, 1, G, n, -1, W1, n, W2, n );

			Hess1 = trace_of_product( 'T', W2, n, G, n );
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

        compute_Lam_fun();

        computeB1_B2();
    }

    finalize_CPU();

	return iter;

}
