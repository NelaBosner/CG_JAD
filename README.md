The codes implement original and parallel forms of Riemannian conjugate gradient method (CG) on the Stiefel and the oblique manifolds, each of them in 12 different variants. The program includes calls to MKL and to POSIX thread library functions. Headers describe usage of the functions.

CG on the Stiefel manifold is implemented in the following files:

    cg_jad_stiefel_var_org.cpp -> original form of the algorithm
    cg_jad_stiefel_var_org.h -> the header with description
    cg_jad_stiefel_var_par.cpp -> parallel form of the algorithm
    cg_jad_stiefel_var_par.h -> the header with description
    test_cg_jad_stiefel_var.cpp -> the testing program

CG on the oblique manifold is implemented in the following files:

    cg_jad_oblique_var_org.cpp -> original form of the algorithm
    cg_jad_oblique_var_org.h -> the header with description
    cg_jad_oblique_var_par.cpp -> parallel form of the algorithm
    cg_jad_oblique_var_par.h -> the header with description
    test_cg_jad_oblique_var.cpp -> the testing program
