#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <math.h>

using namespace std;

void 
usage(const char *name)
{
	std::cout << "usage: " << name
                  << " matrix-size nworkers"
                  << std::endl;
 	exit(-1);
}


int
main(int argc, char **argv)
{
  double begin = omp_get_wtime();
  const char *name = argv[0];

  if (argc < 3) usage(name);

  int matrix_size = atoi(argv[1]);

  int nworkers = atoi(argv[2]);

  std::cout << "Number of Threads : " 
            << nworkers
            << std::endl;

  omp_set_num_threads(nworkers);

  /******my code implementation ******/
  
  int m_size = matrix_size+1;
  //double min_ran = 0;     // minimum random number
  double max_ran = 20000; // maximum random number
  srand((unsigned int)time(NULL));

  double** aArray;   // input array
  vector<double> pi(m_size);       // pi array
  double** uArray;   // upper-triangular matrix
  double** lArray;   // lower-triangular matrix
 
  int i,j,k;


  aArray = (double**) malloc( sizeof(double*) * m_size);
  uArray = (double**) malloc( sizeof(double*) * m_size);
  lArray = (double**) malloc( sizeof(double*) * m_size);


  #pragma omp prallel for private(i) shared(aArray, uArray, lArray)
  for(i=0; i<m_size; i++){
    aArray[i] = (double*) malloc (sizeof(double) * m_size);
    uArray[i] = (double*) malloc (sizeof(double) * m_size);
    lArray[i] = (double*) malloc (sizeof(double) * m_size);
  }


  /******** initialize matrix********/
  for(int i=1; i<m_size; i++)
    for(int j=1; j<m_size; j++)
      aArray[i][j] = rand()*max_ran/(double)(RAND_MAX); // 0~20000 random double numbers

#pragma omp parallel num_threads(nworkers)
{
  #pragma omp for nowait
  for(i=1; i<m_size; i++)
    pi[i]=i;

  #pragma omp for nowait
  for(i=2; i<m_size; i++)
    for(j=1; j<i; j++)
      uArray[i][j] = 0;
     
  #pragma omp for
  for(i=1; i<m_size; i++){
    lArray[i][i] = 1;
    for(j=i+1; j<m_size; j++)
      lArray[i][j] = 0;
  }
}

  /******** calculation part ********/
  double max = 0;
  double temp = 0;
  int k_other;
  //#pragma omp parallel
//{
  double decomp_begin = omp_get_wtime();
  for(k=1; k<m_size; k++){
    k_other = k;
    max = 0;
    temp = 0;

    #pragma omp parallel for shared(aArray) lastprivate(max) //private(i, k, k_other)
    for(i=k; i<m_size; i++){
      if( max < fabs(aArray[i][k]) ){
        max = fabs(aArray[i][k]);
        k_other = i;
      }
    }
      	
    if( max == 0 ){
      cout<<"error(singular matrix)"<<endl; exit(0);
    }
    
      
    /*** swap pi ***/
    temp = pi[k];
    pi[k] = pi[k_other];
    pi[k_other] = temp;


    #pragma omp parallel num_threads(nworkers)
    {

    /*** swap input array***/
    #pragma omp for nowait
    for(i=1; i<m_size; i++){
      double t = aArray[k][i];
      aArray[k][i] = aArray[k_other][i];
      aArray[k_other][i] = t;
    }    
 
    /*** swap lower array***/
    #pragma omp for
    for(i=1; i<k; i++){
    double t = lArray[k][i];
    lArray[k][i] = lArray[k_other][i];
    lArray[k_other][i] = t;
    }

    uArray[k][k] = aArray[k][k];
    #pragma omp for //shared(aArray)
    for(i=k+1; i<m_size; i++){
      lArray[i][k] = aArray[i][k]  / uArray[k][k];
      uArray[k][i] = aArray[k][i];
    }
    #pragma omp for //shared(lArray, uArray)
    for(i=k+1; i<m_size; i++)    
      for(j=k+1; j<m_size; j++){
        aArray[i][j] = aArray[i][j] - lArray[i][k]*uArray[k][j];
      }
    }
  }
 double decomp_end = omp_get_wtime();
//}
double result = 0;
double t = 0;
/**** Compute L2,1 ****/
#pragma omp parallel for lastprivate(result)// reduction(+: result) private(i, j, t)
for(i=k+1; i<m_size; i++){
  result += sqrt(t);
  for(j=k+1; j<m_size; j++){
    t += (aArray[j][i] * aArray[j][i]);
  }
}

double end = omp_get_wtime();
cout<<"LU-decomposition time : "<< (decomp_end-decomp_begin)*1000<<"ms"<<endl;
cout<<"Entire process time : "<< (end-begin)*1000<<"ms"<<endl;// /( (CLOCKS_PER_SEC) )<<"s"<<endl;

return 0;
}
