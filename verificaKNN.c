#include <stdio.h>

void verificaKNN( float *Q[], int nq, float *P[], int n, int D, int k, int *R[] ) {
    printf( " ------------ VERIFICA KNN --------------- \n" );
    for( int conj=0; conj<nq; conj++ ) {
        printf( "knn[%d]: ", conj );
        for( int elem=0; elem<k; elem++ )
            printf( "%d ", R[conj][elem] );
        printf( "\n" );
    }
}