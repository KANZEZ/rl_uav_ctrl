#include <stdio.h>
#include "myMPC_FORCESPRO/include/myMPC_FORCESPRO.h"
#include "myMPC_FORCESPRO/include/myMPC_FORCESPRO_memory.h"

int main()
{
    int return_val = 0;
    int i;
    int exitflag = 0;
    double A[2][2] = {{1.1, 1.0}, {0.0, 1.0}};
    double B[2] = {1.0, 0.5};
    double X[2] = {0.0};
    double U[1] = {0.0};
    double X_next[2] = {-4.0, 2.0};
    myMPC_FORCESPRO_params params;
    myMPC_FORCESPRO_info info;
    myMPC_FORCESPRO_output output;
    myMPC_FORCESPRO_mem * mem;

    mem = myMPC_FORCESPRO_internal_mem(0);

    for (i = 0; i < 30; i++)
    {
        X[0] = X_next[0];
        X[1] = X_next[1];
        params.minusA_times_x0[0] = -(A[0][0] * X[0] + A[0][1] * X[1]);
        params.minusA_times_x0[1] = -(A[1][0] * X[0] + A[1][1] * X[1]);
        exitflag = myMPC_FORCESPRO_solve(&params, &output, &info, mem, NULL);
        if (exitflag != 1)
        {
            printf("\n\nmyMPC_FORCESPRO did not return optimal solution at step %d. Exiting.\n", i + 1);
            return_val = 1;
            break;
        }
        U[0] = output.u0;
        X_next[0] = A[0][0] * X[0] + A[0][1] * X[1] + B[0] * U[0];
        X_next[1] = A[1][0] * X[0] + A[1][1] * X[1] + B[1] * U[0];
        printf("\n\nStep %d: OPTIMAL, Iterations %d, Output U: %4.6e, State X: %4.6e %4.6e\n", i + 1, info.it, U[0], X[0], X[1]);
    }
    return return_val;
}
