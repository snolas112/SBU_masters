#include <mpi.h>
#include <stdio.h>
#include <cmath>
#include <ctime>
#include <stdlib.h>


void NaiveMult1(int n, int matrixA[][4096], int matrixB[][4096], int matrixC[][4096]);
void NaiveMult2(int n, int matrixA[][2048], int matrixB[][2048], int matrixC[][2048]);
void NaiveMult3(int n, int matrixA[][1024], int matrixB[][1024], int matrixC[][1024]);
void NaiveMult4(int n, int matrixA[][512], int matrixB[][512], int matrixC[][512]);
void Strassen2(int n, int A[][2048], int B[][2048], int P[][2048]);
void Strassen3(int n, int A[][1024], int B[][1024], int P[][1024]);


int main(int argc, char *argv[]) {
    int rank, size;
    

    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    double start_time;
    double end_time;

    start_time=MPI_Wtime();

    int root = 0;

    int n = 4096;
    
    int A[n][4096];
    int B[n][4096];
    //int C1[n][4096];
    int C2[n][4096];
    
    //make sure C is initialized to zero
    for (int i =0; i<n; i++){
        for (int j =0; j < n; j++){
            //C1[i][j] = 0;
            C2[i][j] = 0;
	    A[i][j] = 0;
            B[i][j] = 0;
        }
    }
    
    for (int i =0; i<n; i++){
        for (int j =0; j < n; j++){
            A[i][j] = 1;
            B[i][j] = 2;
        }
    }
    

    //naive matrix mult
    //NaiveMult1(n,A,B,C1);
    
    int S[10][2048][2048]; //level1 sums
    //Strassensums(n,A,B,S);
    for (int i =0; i<n/2; i++){
        for (int j=0; j<n/2; j++){
            S[0][i][j]=A[i][j]+A[i+(n/2)][j+(n/2)];
            S[1][i][j]=B[i][j]+B[i+(n/2)][j+(n/2)];
            S[2][i][j]=A[i+(n/2)][j]+A[i+(n/2)][j+(n/2)];
            S[3][i][j]=B[i][j+(n/2)]-B[i+(n/2)][j+(n/2)];
            S[4][i][j]=B[i+(n/2)][j]-B[i][j];
            S[5][i][j]=A[i][j]+A[i][j+(n/2)];
            S[6][i][j]=A[i+(n/2)][j]-A[i][j];
            S[7][i][j]=B[i][j]+B[i][j+(n/2)];
            S[8][i][j]=A[i][j+(n/2)]-A[i+(n/2)][j+(n/2)];
            S[9][i][j]=B[i+(n/2)][j]+B[i+(n/2)][j+(n/2)];
        }
    }
    //level1 products

    if (rank==1){
        //P2=S3B11
        int P2[2048][2048] = { 0 };
        int B11[2048][2048] = { 0 };
        for (int i=0;i<n/2;i++){
            for(int j=0; j<n/2; j++){
                 B11[i][j]=B[i][j];
             }
        }
        Strassen2(n/2,S[2],B11,P2);
        MPI_Send(P2,2048*2048,MPI_INT,0,1,MPI_COMM_WORLD);
    }

    if (rank==2){
        //P3=A11S4
        int P3[2048][2048] = { 0 };
        int A11[2048][2048] = { 0 };
        for (int i=0;i<n/2;i++){
            for(int j=0; j<n/2; j++){
                A11[i][j]=A[i][j];
            }
        }
        Strassen2(n/2,A11,S[3],P3);
        MPI_Send(P3,2048*2048,MPI_INT,0,1,MPI_COMM_WORLD);
    }

    if (rank==3){
         //P4=A22S5
         int P4[2048][2048] = { 0 };
         int A22[2048][2048] = { 0 };
         for (int i=0;i<n/2;i++){
            for(int j=0; j<n/2; j++){
                 A22[i][j]=A[i+(n/2)][j+(n/2)];
             }
        }
        Strassen2(n/2,A22,S[4],P4);
        MPI_Send(P4,2048*2048,MPI_INT,0,1,MPI_COMM_WORLD);
    }

    if (rank==4){
        //P5=S6B22
        int P5[2048][2048] = { 0 };
        int B22[2048][2048] = { 0 };
        for (int i=0;i<n/2;i++){
            for(int j=0; j<n/2; j++){
                 B22[i][j]=B[i+(n/2)][j+(n/2)];
             }
         }
         Strassen2(n/2,S[5],B22,P5);
         MPI_Send(P5,2048*2048,MPI_INT,0,1,MPI_COMM_WORLD);
    }

    if (rank == 5){
        //P6=S7S8
        int P6[2048][2048] = { 0 };
        Strassen2(n/2,S[6],S[7],P6);
    MPI_Send(P6,2048*2048,MPI_INT,0,1,MPI_COMM_WORLD);
    }

    if (rank == 6){
        //P7=S9S10
        int P7[2048][2048] = { 0 };
        Strassen2(n/2,S[8],S[9],P7);
        MPI_Send(P7,2048*2048,MPI_INT,0,1,MPI_COMM_WORLD);
    }    
 
     
    if (rank == root){
	int P1[2048][2048] = { 0 };
	Strassen2(n/2,S[0],S[1],P1);        

        int P2[2048][2048] = { 0 };
        int P3[2048][2048] = { 0 };
        int P4[2048][2048] = { 0 };
        int P5[2048][2048] = { 0 };
        int P6[2048][2048] = { 0 };
        int P7[2048][2048] = { 0 };
        
        MPI_Status status;
        MPI_Recv(P2,2048*2048,MPI_INT,1,1,MPI_COMM_WORLD,&status);
        MPI_Recv(P3,2048*2048,MPI_INT,2,1,MPI_COMM_WORLD,&status);
        MPI_Recv(P4,2048*2048,MPI_INT,3,1,MPI_COMM_WORLD,&status);
        MPI_Recv(P5,2048*2048,MPI_INT,4,1,MPI_COMM_WORLD,&status);
        MPI_Recv(P6,2048*2048,MPI_INT,5,1,MPI_COMM_WORLD,&status);
        MPI_Recv(P7,2048*2048,MPI_INT,6,1,MPI_COMM_WORLD,&status);

        //level 1 combine back to C
        for (int i=0; i<(n/2);i++){
             for(int j=0;j<(n/2);j++){
                //C11=P1+P4+P7-P5
                C2[i][j]=P1[i][j]+P4[i][j]+P7[i][j]-P5[i][j];
                //C12=P3+P5
                C2[i][j+(n/2)]=P3[i][j]+P5[i][j];
                //C21=P2+P4
                C2[i+(n/2)][j]=P2[i][j]+P4[i][j];
                //C22=P1+P3+P6-P2
                C2[i+(n/2)][j+(n/2)]=P1[i][j]+P3[i][j]+P6[i][j]-P2[i][j];
             }
        }
    }
    //for (int i=0; i<n; i++){
    //    for (int j=0; j<n; j++){
    //        if(C1[i][j] != C2[i][j]){
    //            printf("not equal");
    //        }
    //    }
    //}

    end_time=MPI_Wtime();

    double timing = (end_time-start_time)/CLOCKS_PER_SEC;
    
    printf("the start time is %d seconds, end time is %d, and total time is %d: ", end_time, start_time, timing);

    MPI_Finalize();
}

void NaiveMult1(int n, int matrixA[][4096], int matrixB[][4096], int matrixC[][4096]){
    for (int i = 0; i<n; i++){
        for (int j = 0; j<n; j++){
            for (int k = 0; k<n ; k++){
                matrixC[i][j] = matrixC[i][j] + matrixA[i][k]*matrixB[k][j];
            }
        }
    }
}

void NaiveMult2(int n, int matrixA[][2048], int matrixB[][2048], int matrixC[][2048]){
    
    for (int i = 0; i<n; i++){
        for (int j = 0; j<n; j++){
            for (int k = 0; k<n ; k++){
                matrixC[i][j] = matrixC[i][j] + matrixA[i][k]*matrixB[k][j];
            }
        }
    }
}

void NaiveMult3(int n, int matrixA[][1024], int matrixB[][1024], int matrixC[][1024]){
    for (int i = 0; i<n; i++){
        for (int j = 0; j<n; j++){
            for (int k = 0; k<n ; k++){
                matrixC[i][j] = matrixC[i][j] + matrixA[i][k]*matrixB[k][j];
            }
        }
    }
}

void NaiveMult4(int n, int matrixA[][512], int matrixB[][512], int matrixC[][512]){
    for (int i = 0; i<n; i++){
        for (int j = 0; j<n; j++){
            for (int k = 0; k<n ; k++){
                matrixC[i][j] = matrixC[i][j] + matrixA[i][k]*matrixB[k][j];
            }
        }
    }
}

void Strassen2(int n, int A[][2048], int B[][2048], int P[][2048]){
    int S[10][1024][1024] = { 0 }; //level2 sums
    //Strassensums(n,A,B,P);
    for (int i =0; i<n/2; i++){
        for (int j=0; j<n/2; j++){
            S[0][i][j]=A[i][j]+A[i+(n/2)][j+(n/2)];
            S[1][i][j]=B[i][j]+B[i+(n/2)][j+(n/2)];
            S[2][i][j]=A[i+(n/2)][j]+A[i+(n/2)][j+(n/2)];
            S[3][i][j]=B[i][j+(n/2)]-B[i+(n/2)][j+(n/2)];
            S[4][i][j]=B[i+(n/2)][j]-B[i][j];
            S[5][i][j]=A[i][j]+A[i][j+(n/2)];
            S[6][i][j]=A[i+(n/2)][j]-A[i][j];
            S[7][i][j]=B[i][j]+B[i][j+(n/2)];
            S[8][i][j]=A[i][j+(n/2)]-A[i+(n/2)][j+(n/2)];
            S[9][i][j]=B[i+(n/2)][j]+B[i+(n/2)][j+(n/2)];
        }
    }
    //Strassn products level 2
    //P1=S1S2
    int PP1[1024][1024] = { 0 };
    Strassen3(n/2,S[0],S[1],PP1);
    //P2=S3B11
    int PP2[1024][1024] = { 0 };
    int B11[1024][1024] = { 0 };
    for (int i=0;i<n/2;i++){
        for(int j=0; j<n/2; j++){
             B11[i][j]=B[i][j];
         }
    }
    Strassen3(n/2,S[2],B11,PP2);
    //P3=A11Sn/2
    int PP3[1024][1024] = { 0 };
    int A11[1024][1024] = { 0 };
    for (int i=0;i<n/2;i++){
        for(int j=0; j<n/2; j++){
            A11[i][j]=A[i][j];
        }
    }
    Strassen3(n/2,A11,S[3],PP3);
    //P4=A22S5
    int PP4[1024][1024] = { 0 };
    int A22[1024][1024] = { 0 };
    for (int i=0;i<n/2;i++){
        for(int j=0; j<n/2; j++){
            A22[i][j]=A[i+(n/2)][j+(n/2)];
        }
    }
    Strassen3(n/2,A22,S[4],PP4);
    //P5=S6B22
    int PP5[1024][1024] = { 0 };
    int B22[1024][1024] = { 0 };
    for (int i=0;i<n/2;i++){
        for(int j=0; j<n/2; j++){
             B22[i][j]=B[i+(n/2)][j+(n/2)];
        }
    }
    Strassen3(n/2,S[5],B22,PP5);
    //P6=S7S8
    int PP6[1024][1024] = { 0 };
    Strassen3(n/2,S[6],S[7],PP6);
    //P7=S9S10
    int PP7[1024][1024] = { 0 };
    Strassen3(n/2,S[8],S[9],PP7);
    
    //return P which is the sum of those products
    for (int i=0; i<(n/2);i++){
             for(int j=0;j<(n/2);j++){
                //C11=P1+P4+P7-P5
                P[i][j]=PP1[i][j]+PP4[i][j]+PP7[i][j]-PP5[i][j];
                //C12=P3+P5
                P[i][j+(n/2)]=PP3[i][j]+PP5[i][j];
                //C21=P2+P4
                P[i+(n/2)][j]=PP2[i][j]+PP4[i][j];
                //C22=P1+P3+P6-P2
                P[i+(n/2)][j+(n/2)]=PP1[i][j]+PP3[i][j]+PP6[i][j]-PP2[i][j];
             }
        }

}

void Strassen3(int n, int A[][1024], int B[][1024], int P[][1024]){
int S[10][512][512] = { 0 }; //level2 sums
    //Strassensums(n,A,B,P);
    for (int i =0; i<n/2; i++){
        for (int j=0; j<n/2; j++){
            S[0][i][j]=A[i][j]+A[i+(n/2)][j+(n/2)];
            S[1][i][j]=B[i][j]+B[i+(n/2)][j+(n/2)];
            S[2][i][j]=A[i+(n/2)][j]+A[i+(n/2)][j+(n/2)];
            S[3][i][j]=B[i][j+(n/2)]-B[i+(n/2)][j+(n/2)];
            S[4][i][j]=B[i+(n/2)][j]-B[i][j];
            S[5][i][j]=A[i][j]+A[i][j+(n/2)];
            S[6][i][j]=A[i+(n/2)][j]-A[i][j];
            S[7][i][j]=B[i][j]+B[i][j+(n/2)];
            S[8][i][j]=A[i][j+(n/2)]-A[i+(n/2)][j+(n/2)];
            S[9][i][j]=B[i+(n/2)][j]+B[i+(n/2)][j+(n/2)];
        }
    }
    //Strassn products level 2
    //P1=S1S2
    int PP1[512][512] = { 0 };
    NaiveMult4(n/2,S[0],S[1],PP1);
    //P2=S3B11
    int PP2[512][512] = { 0 };
    int B11[512][512] = { 0 };
    for (int i=0;i<n/2;i++){
        for(int j=0; j<n/2; j++){
             B11[i][j]=B[i][j];
         }
    }
    NaiveMult4(n/2,S[2],B11,PP2);
    //P3=A11Sn/2
    int PP3[512][512] = { 0 };
    int A11[512][512] = { 0 };
    for (int i=0;i<n/2;i++){
        for(int j=0; j<n/2; j++){
            A11[i][j]=A[i][j];
        }
    }
    NaiveMult4(n/2,A11,S[3],PP3);
    //P4=A22S5
    int PP4[512][512] = { 0 };
    int A22[512][512] = { 0 };
    for (int i=0;i<n/2;i++){
        for(int j=0; j<n/2; j++){
            A22[i][j]=A[i+(n/2)][j+(n/2)];
        }
    }
    NaiveMult4(n/2,A22,S[n/2],PP4);
    //P5=S6B22
    int PP5[512][512] = { 0 };
    int B22[512][512] = { 0 };
    for (int i=0;i<n/2;i++){
        for(int j=0; j<n/2; j++){
             B22[i][j]=B[i+(n/2)][j+(n/2)];
        }
    }
    NaiveMult4(n/2,S[5],B22,PP5);
    //P6=S7S8
    int PP6[512][512] = { 0 };
    NaiveMult4(n/2,S[6],S[7],PP6);
    //P7=S9S10
    int PP7[512][512] = { 0 };
    NaiveMult4(n/2,S[8],S[9],PP7);
    
    //return P which is the sum of those products
    for (int i=0; i<(n/2);i++){
             for(int j=0;j<(n/2);j++){
                //C11=P1+P4+P7-P5
                P[i][j]=PP1[i][j]+PP4[i][j]+PP7[i][j]-PP5[i][j];
                //C12=P3+P5
                P[i][j+(n/2)]=PP3[i][j]+PP5[i][j];
                //C21=P2+P4
                P[i+(n/2)][j]=PP2[i][j]+PP4[i][j];
                //C22=P1+P3+P6-P2
                P[i+(n/2)][j+(n/2)]=PP1[i][j]+PP3[i][j]+PP6[i][j]-PP2[i][j];
             }
        }


}
