#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

/*
	A simple MPI example.
	TODO:
	1. Fill in the needed MPI code to make this run on any number of nodes.
	2. The answer must match the original serial version.
	3. Think of corner cases (valid but tricky values).

	Example input:
	./simple 2 10000

*/

int main(int argc, char **argv) {
	int rank, size, last_rank;

	if (argc < 3) {
		printf("This program requires two parameters:\n \
the start and end specifying a range of positive integers in which \
start is 2 or greater, and end is greater than start.\n");
		exit(1);
	}

    double time = MPI_Wtime();

	int start = atoi(argv[1]);
	int stop = atoi(argv[2]);


	if(start < 2 || stop <= start){
		printf("Start must be greater than 2 and the end must be larger than start.\n");
		exit(1);
	}

	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // If numProcesses is more than range it will not be used
    if (rank > (stop - start)) {
        MPI_Finalize();
        return 0;
    }

	// Perform the computation
    double sum = 0.0;

     if (rank == 0 && size != 1) {
         for (int i = 0; i < fmin(size-1, stop - start); ++i) {
             double number;
             MPI_Recv(&number, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
             sum = sum + number;
         }
         printf("%f\n", sum);
         printf("%f\n", MPI_Wtime() - time);
     } else if (size != 1) {
         int interval = (int) ceil((double)(stop - start) / (double)(size - 1));
         int end = start + interval * rank;

         if (rank == size - 1) {
             end = stop;
         }

         for (int i = start + interval * (rank - 1); i < end; i++) {
             //printf("i: %d\n", i);
             sum += 1.0/log(i);
             //printf("new sum: %f\n", sum);
         }


         MPI_Send(&sum, 1, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD);
     } else {
         for (int i = start; i < stop; ++i) {
             sum += 1.0/log(i);
         }
         printf("%f\n", sum);
         printf("%f\n", MPI_Wtime() - time);
     }

    MPI_Finalize();

	return 0;
}

