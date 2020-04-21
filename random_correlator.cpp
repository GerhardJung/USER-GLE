/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// Correlates random numbers
// see Appendix B: J. Chem. Phys. 143, 243128 (2015)

#include <math.h>
#include "random_correlator.h"
#include "comm.h"
#include "random_mars.h"
#include "error.h"

using namespace LAMMPS_NS;

#define FT_METHOD

/* ---------------------------------------------------------------------- */

RanCor::RanCor(LAMMPS *lmp, int mem_count, double *mem_kernel, double precision) : Pointers(lmp)
{
  this->mem_count = mem_count;
  this->mem_kernel = mem_kernel;
  this->precision = precision;
  
  // init the coefficients for the correlation and the memory for the uncorrelated random numbers
  srand(time(NULL));
  init();
}

/* ---------------------------------------------------------------------- */

RanCor::~RanCor()
{
  delete [] a_coeff;
}

/* ---------------------------------------------------------------------- */

void RanCor::init()
{
  
#ifdef FT_METHOD
  init_acoeff();

#else  
  int i,n,s;
  
  // rescale the memory to simplify the optimization
  double norm = mem_kernel[0];
  for (int i=0; i<mem_count; i++) {
    mem_kernel[i]/=norm;
  }
  
  N = mem_count+5;
  a_coeff = new float[N];
  for (i=0; i< N; i++) a_coeff[i]=0.0;
  
  //optimize alpha-parameter for correlated noise

  NelderMeadOptimizer opt(N, precision, mem_kernel, mem_count);

  // request a simplex to start with
  Vector v(a_coeff,N);

  init_opt(opt,v,N);
  //opt.print_v();

  v = opt.step(v, min_function(v,N));
  printf("optimizer quality: %f\n",min_function(v,N));

  for ( i=0; i < N; i++ ) a_coeff[i] = v[i];

  for (i=0; i< N; i++) printf("%f ",v[i]);
  printf("\n");

  
  
  // scale back the memory and the parameter
  for (int i=0; i<mem_count; i++) {
    mem_kernel[i]*=norm;
  }
  for (int i=0; i<N; i++) {
    a_coeff[i] *= sqrt(norm);
  }
  
  FILE * out=fopen("ansatz.dat","w");
  for(n=0;n<mem_count;n++){
    double loc_sum = 0.0;
    for(s=0;s<N;s++){
      int ind = n+s;
      //if (n+s>2*mem_count-2) ind -= N;
      if (n+s>N-1) continue;
      loc_sum += a_coeff[s]*a_coeff[ind];
    }
    fprintf(out,"%d %f %f %f\n",n,mem_kernel[n],loc_sum,mem_kernel[n]-loc_sum);
        // correct the memory kernel to fullfill the fluctuation dissipation theorem
    mem_kernel[n] = loc_sum;
  }
  fclose(out);
#endif
  
}

/* ----------------------------------------------------------------------
   gaussian RN
------------------------------------------------------------------------- */

double RanCor::gaussian(double* normal, int firstindex)
{
  // calculate corr. random number
  int i,j;
  double ran = 0.0;
  j = firstindex;
  for (i=0; i<N; i++) {
    ran += normal[j]*a_coeff[i];
    j--;
    if (j <0) j += 2*mem_count-1;
  }
  return ran;
}

/* ---------------------------------------------------------------------- 
  initializes the alpha coefficients by fourier transformation
  ----------------------------------------------------------------------  */
void RanCor::init_acoeff() {
  
  N = 2*mem_count-1;
  
  a_coeff = new float[N];
  double * a_coeff_imag = new double[N];
  for (int i=0; i< N; i++){
    a_coeff[i]=0.0;
    a_coeff_imag[i]=0.0;
  }
  
  complex<double> FT_mem_kernel[N];
  
  forwardDFT(mem_kernel,N ,FT_mem_kernel);
  FILE * out;
  out=fopen("ft_mem.dat","w");
  int warn_flag = 0;
  for (int i=0; i< N; i++) {
    fprintf(out,"%d %f %f\n",i,FT_mem_kernel[i].real(),FT_mem_kernel[i].imag());
    if ( FT_mem_kernel[i].real() < 0 && warn_flag == 0 ) {
      error->warning(FLERR,"Some modes of the memory are < 0 in fix gle. Memory will be corrected. ");
      warn_flag = 1;
    }
  }
  fclose(out);
  complex<double> FT_a_coeff[N];
  
  for (int i=0; i<N;i++)
   FT_a_coeff[i] = sqrt(FT_mem_kernel[i]);
  
  out=fopen("ft_mem_sqrt.dat","w");
  for (int i=0; i< N; i++) {
    FT_a_coeff[i].imag(abs(FT_a_coeff[i].imag()));
    fprintf(out,"%d %f %f\n",i,FT_a_coeff[i].real(),FT_a_coeff[i].imag());
  }
  fclose(out);
  
  inverseDFT(FT_a_coeff,N, a_coeff, a_coeff_imag);
    out=fopen("as.dat","w");
  for (int i=0; i< N; i++) {
    fprintf(out,"%d %f %f\n",i,a_coeff[i],a_coeff_imag[i]);
  }
fclose(out);
  
  /*for(int s=0;s<N;s++){
    printf("%d %f\n",s,a_coeff[s]);
  }*/
  
  // print the memory
  out=fopen("ansatz.dat","w");
  int n,s;
  for(n=0;n<mem_count;n++){
    double loc_sum = 0.0;
    for(s=0;s<2*mem_count-1;s++){
      int ind = n+s;
      if (n+s>2*mem_count-2) ind -= N;
      loc_sum += a_coeff[s]*a_coeff[ind];
    }
    fprintf(out,"%d %.10f %.10f %.10f\n",n,mem_kernel[n],loc_sum,mem_kernel[n]-loc_sum);
        // correct the memory kernel to fullfill the fluctuation dissipation theorem
    mem_kernel[n] = loc_sum;
  }
   fclose(out);
   
   delete [] a_coeff_imag;
   
  //error->all(FLERR,"break");
}

/* ---------------------------------------------------------------------- 
  initializes the nm optimizer
  ----------------------------------------------------------------------  */

void RanCor::init_opt(NelderMeadOptimizer &opt, Vector v, int dimension) {
  
  opt.insert(v,min_function(v,dimension));
  
  double p = 10.0/(dimension*sqrt(2))*(sqrt(dimension+1)+dimension-1);
  double q = 10.0/(dimension*sqrt(2))*(sqrt(dimension+1)-1);

  int d,i;
  for (i=0; i<dimension; i++) {
    Vector new_v(a_coeff,dimension);
    for (d=0; d<dimension; d++) {
      if (d!=i) {
	new_v[d] += q;
      } else {
	new_v[d] += p;
      }
    }

    opt.insert(new_v,min_function(new_v,dimension));
  }
 
}
 
/* ---------------------------------------------------------------------- 
  function for the minimization process to determine correlation coefficients
  ----------------------------------------------------------------------  */

float RanCor::min_function(Vector v, int dimension)
{
  float sum=0.0;
  int n,s;
    
  for(n=0;n<mem_count;n++){
    double loc_sum = 0.0;
    for(s=0;s<dimension;s++){
      int ind = n+s;
      //if (n+s>dimension-1) ind -= dimension;
      if (n+s>dimension-1) continue;
      loc_sum += v[s]*v[ind];
    }
    loc_sum -= mem_kernel[n];
    sum += loc_sum*loc_sum;
  }
  
  return sum;
}

/* ---------------------------------------------------------------------- 
  performs a forward DFT of reell (and symmetric) input
  ----------------------------------------------------------------------  */
void RanCor::forwardDFT(double *data, int N, complex<double> *result) { 
  for (int k = -mem_count+1; k < mem_count; k++) { 
    result[k+mem_count-1].real(0.0);
    result[k+mem_count-1].imag(0.0);
    for (int n = -mem_count+1; n < mem_count; n++) { 
      double data_loc = 0.0;
      if (n<0) data_loc = data[abs(n)];
      else data_loc = data[n];
      result[k+mem_count-1].real( result[k+mem_count-1].real() + data_loc * cos(2*M_PI / N * n * k));
      result[k+mem_count-1].imag( result[k+mem_count-1].imag() - data_loc * sin(2*M_PI / N * n * k));
    } 
  } 
}

/* ---------------------------------------------------------------------- 
  performs a backward DFT with complex input (and reell output)
  ----------------------------------------------------------------------  */
void RanCor::inverseDFT(complex<double> *data, int N, float *result, double *result_imag) { 
  for (int n = -mem_count+1; n < mem_count; n++) { 
    result[n+mem_count-1] = 0.0; 
    for (int k = -mem_count+1; k < mem_count; k++) { 
      result[n+mem_count-1] += data[k+mem_count-1].real() * cos(2*M_PI / N * n * k) - data[k+mem_count-1].imag() * sin(2*M_PI / N * n * k);
      result_imag[n+mem_count-1] += data[k+mem_count-1].imag() * cos(2*M_PI / N * n * k) + data[k+mem_count-1].real() * sin(2*M_PI / N * n * k);
    } 
    result[n+mem_count-1] /= N;
    result_imag[n+mem_count-1] /= N;
  } 
}
