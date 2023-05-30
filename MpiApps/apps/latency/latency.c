#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>


#define MYBUFSIZE 4*1024*1025
#define WARMING_LOOPS 10
#define MAX_LOOPS (INT_MAX - WARMING_LOOPS)
char s_buf[MYBUFSIZE];
char r_buf[MYBUFSIZE];

int main(int argc,char *argv[])
{
    int myid, numprocs, i;
    int size;
    int loop;
    MPI_Status stat;

    double t_start=0.0, t_end=0.0;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);

    if ( argc < 3 ) {
       fprintf(stderr, "Usage: latency loop msg_size\n");
       MPI_Finalize();
       return 0;
    }    

    loop = atoi(argv[1]);
    if ((loop < 0) || (loop > MAX_LOOPS)) {
        fprintf(stderr, "Maximum loops count is %d, minimum is 0\n", MAX_LOOPS);
        MPI_Finalize();
        return 0;
    }

    size=atoi(argv[2]);
    if ((size < 0) || (size > MYBUFSIZE)) {
        fprintf(stderr, "Maximum message size is %d, minimum is 0\n", MYBUFSIZE);
        MPI_Finalize();
        return 0;
    }

    /* touch the data */
    for ( i=0; i<size; i++ ){
        s_buf[i]='a';
        r_buf[i]='b';
    }

    MPI_Barrier( MPI_COMM_WORLD);

    if (myid == 0)
    {
        for ( i=0; i < loop + WARMING_LOOPS; i++ ) {
            if ( i == WARMING_LOOPS ) t_start=MPI_Wtime();
            MPI_Send(s_buf, size, MPI_CHAR, 1, i, MPI_COMM_WORLD);
            MPI_Recv(r_buf, size, MPI_CHAR, 1, i + 1000, MPI_COMM_WORLD, &stat);
        }
        t_end=MPI_Wtime();

    }else{
        for ( i=0; i < loop + WARMING_LOOPS; i++ ) {
            MPI_Recv(r_buf, size, MPI_CHAR, 0, i, MPI_COMM_WORLD, &stat);
            MPI_Send(s_buf, size, MPI_CHAR, 0, i + 1000, MPI_COMM_WORLD);
        }
    }


    if ( myid == 0 ) {
       double latency;
       latency = (t_end-t_start)*1.0e6/(2.0*loop);
       fprintf(stdout,"%d\t%f\t\n", size, latency);
    }
    MPI_Finalize();
    return 0;
}
