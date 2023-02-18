#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "chrono.c"
#include <assert.h>
#include <time.h>
#include "verificaKNN.c"

int nproc;      	// o número de processos MPI
int n;              // numero de pontos de uma matriz
int nq;             // numero de pontos de outro matriz
int k;              // numero de vizinhos
int D;              // dimensões
float **P;          // matriz P
float **Q;          // matriz Q
int ** vizinhos;	// vizinhos mais proximos
int processId; 	    // rank dos processos
float r;            // valor aleatorio para preecher as matrizes

chronometer_t knnTime;

#define DEBUG 0

int ** knn( float **Q, int nq, float **P, int n, int D, int k){
    int **vizinhos_aux = (int**)calloc(nq, sizeof(int*));
	
	int rangeQ = nq/nproc;
	int rangeIniQ = processId*rangeQ;
	int rangeFimQ = processId*rangeQ+rangeQ-1;

	int rangeP = n/nproc;
	int rangeIniP = processId*rangeP;
	int rangeFimP = processId*rangeP+rangeP-1;

    for (int i = 0; i < nq; i++)
        vizinhos_aux[i] = (int*)calloc(k, sizeof(int));

    if(processId == 0){
		MPI_Bcast(Q, nq*D, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(P, n*D, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}

	

    return vizinhos_aux;
}

void geraConjuntoDeDados( float **Q, int nq, int D ){
    for (int i = 0; i < nq; i++)
        for(int l = 0; l < D; l++){
            r = float(rand());
            Q[i][l] = r;
        }
}

int main(int argc, char *argv[]){

	if (argc < 5) {
		printf("usage: mpirun -np <np> %s <nq> <n> <D> <k>\n",
			argv[0]);
		return 0;
	}
	else {
        nq = atoi(argv[1]);
        n = atoi(argv[2]);
		D = atoi(argv[3]);
        k = atoi(argv[4]);
	}

    MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &processId);

	MPI_Status Stat;

    srand(777); 

	P = (float**)calloc(n, sizeof(float*));

    for(int i = 0; i < n; i++)
        P[i] = (float*)calloc(D, sizeof(float));

    geraConjuntoDeDados( P, n, D );
    
	Q = (float**)calloc(nq, sizeof(float*));

    for(int i = 0; i < nq; i++)
        Q[i] = (float*)calloc(D, sizeof(float));

    geraConjuntoDeDados( Q, nq, D );

	MPI_Barrier(MPI_COMM_WORLD);

	if(processId == 0){
		chrono_reset(&knnTime);
		chrono_start(&knnTime);
	}

    vizinhos = knn( Q, nq, P, n, D, k);

	MPI_Barrier(MPI_COMM_WORLD);

	if(processId == 0){
		verificaKNN(Q, nq, P, n, D, k, vizinhos);
		chrono_stop(&knnTime);
		chrono_reportTime(&knnTime, "knnTime");

		// calcular e imprimir a VAZAO (numero de operacoes/s)
		double total_time_in_seconds = (double)chrono_gettotal(&knnTime) /
									((double)1000 * 1000 * 1000);
	    double total_time_in_micro = (double)chrono_gettotal(&knnTime) /
									((double)1000);
		printf("total_time_in_seconds: %lf s\n", total_time_in_seconds);
		// printf("Latencia: %lf us/nmsg\n", (total_time_in_micro / nmsg)/2);
		// double MBPS = ((double)(nmsg*tmsg) / ((double)total_time_in_seconds*1000*1000));
		// printf("Throughput: %lf MB/s\n", MBPS*(nproc-1));
	}

    for(int i = 0; i < nq; i++)
        free(Q[i]);
    free(Q);

    for(int i = 0; i < n; i++)
        free(P[i]);
    free(P);

    for (int i = 0; i < nq; i++)
        free(vizinhos[i]);
    free(vizinhos);

	MPI_Finalize( );
	return 0;
}