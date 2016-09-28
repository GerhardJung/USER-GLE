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
#include "random_correlater.h"
#include "comm.h"
#include "random_mars.h"
#include "error.h"


using namespace LAMMPS_NS;

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
  int i,n,s;
  
  // rescale the memory to simplify the optimization
  /*for (i=0; i<mem_count; i++) {
    mem_kernel[i] = 100*exp(-10.68*i*0.005)*cos(12.83*i*0.005);
  }*/
  /*double norm = mem_kernel[0];
 for (i=0; i<mem_count; i++) {
    mem_kernel[i] /= norm / 100.0;
  }*/
#ifdef SOLO_OPT
  double norm = mem_kernel[0];
  for (int i=0; i<mem_count; i++) {
    mem_kernel[i] /= norm;
  }
  a_coeff = new float[mem_count];
  for (int i=0; i< mem_count; i++) a_coeff[i]=0.0;
  
#else
   init_acoeff();

#endif
  

#ifdef SOLO_OPT
  for (int step = 10; step < mem_count; step+=10) {
    if (step > mem_count) continue;
    if (step + 10 > mem_count) step = mem_count;
#else
  int step = mem_count;
#endif
  
    //NelderMeadOptimizer opt(step, precision, mem_kernel);
  
    // request a simplex to start with
    //Vector v(a_coeff,step);
    
   // printf("optimizer quality: %f\n",min_function(v,step));
  //error->all(FLERR,"break");
    //init_opt(opt,v,step);
    
    //opt.print_v();
#ifdef SOLO_OPT
      while (!opt.done()) {
	v = opt.step(v, min_function(v,step));
	printf("optimizer quality: %f\n",min_function(v,step));
      }
  
      //  for ( i=0; i < step; i++ ) {
      //a_coeff[i] = v[i];
      //a_coeff[2*step-2-i] = v[i];
    }
  }
#endif
  // scale back the memory and the parameter
#ifdef SOLO_OPT
for (i=0; i<mem_count; i++) {
    mem_kernel[i]*=norm;
    a_coeff[i] *= sqrt(norm);
  }
#endif

  
}

/* ----------------------------------------------------------------------
   gaussian RN
------------------------------------------------------------------------- */
double RanCor::gaussian(double* normal, int lastindex)
{
  // calculate corr. random number
  int i,j;
  double ran = 0.0;
  j = lastindex;
  for (i=0; i<2*mem_count-1; i++) {
    ran += normal[j]*a_coeff[i];
    j--;
    if (j == -1) j = 2*mem_count-1;
  }
  return ran;
}

/* ---------------------------------------------------------------------- 
  initializes the alpha coefficients by fourier transformation
  ----------------------------------------------------------------------  */
void RanCor::init_acoeff() {
  
  int N = 2*mem_count-1;
  
  a_coeff = new double[N];
  for (int i=0; i< N; i++) a_coeff[i]=0.0;
  
  complex<double> FT_mem_kernel[N];
  
  forwardDFT(mem_kernel,N ,FT_mem_kernel);
  
  complex<double> FT_a_coeff[N];
  
  for (int i=0; i<N;i++)
    FT_a_coeff[i] = sqrt(FT_mem_kernel[i]);
  
  inverseDFT(FT_a_coeff,N, a_coeff);
  
  /*for(int s=0;s<N;s++){
    printf("%d %f\n",s,a_coeff[s]);
  }*/
  
  int n,s;
  for(n=0;n<mem_count;n++){
    float loc_sum = 0.0;
    for(s=0;s<N;s++){
      if (n-s>0) continue;
      loc_sum += a_coeff[s]*a_coeff[n-s+N-1];
    }
    printf("%d %f %f\n",n,mem_kernel[n],loc_sum);
  }
  
  //error->all(FLERR,"break");
}

/* ---------------------------------------------------------------------- 
  initializes the nm optimizer
  ----------------------------------------------------------------------  */

void RanCor::init_opt(NelderMeadOptimizer &opt, Vector v, int dimension) {
  
  opt.insert(v,min_function(v,dimension));
  
#ifdef SOLO_OPT
  double factor = 0.01;
#else
  double factor = 0.0001;
#endif
  double p = factor/(dimension*sqrt(2))*(sqrt(dimension+1)+dimension-1);
  double q = factor/(dimension*sqrt(2))*(sqrt(dimension+1)-1);

  int d,i;
  for (i=0; i<dimension; i++) {
    //Vector new_v(a_coeff,dimension);
    for (d=0; d<dimension; d++) {
      if (d!=i) {
	//new_v[d] += q;
      } else {
	//new_v[d] += p;
      }
    }

    //opt.insert(new_v,min_function(new_v,dimension));
  }
 
}
 
/* ---------------------------------------------------------------------- 
  function for the minimization process to determine correlation coefficients
  ----------------------------------------------------------------------  */
float RanCor::min_function(Vector v, int dimension)
{
  float sum=0.0;
  int n,s;
  for(n=0;n<dimension;n++){
    float loc_sum = 0.0;
    for(s=0;s<2*dimension-1;s++){
      if(n+s>=2*dimension-1) continue;
      int sp = s-dimension+1;
      int snp = s+n-dimension+1;
      float left,right;
      if (sp > 0) left = v[2*dimension-2-s];
      else left = v[s];
      if (snp > 0) right = v[2*dimension-2-s-n];
      else right = v[s+n];
      loc_sum += left*right;
    }
    //printf("%f - %f\n", loc_sum, mem_kernel[n]);
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
void RanCor::inverseDFT(complex<double> *data, int N, double *result) { 
  for (int n = -mem_count+1; n < mem_count; n++) { 
    result[n+mem_count-1] = 0.0; 
    for (int k = -mem_count+1; k < mem_count; k++) { 
      result[n+mem_count-1] += data[k+mem_count-1].real() * cos(2*M_PI / N * n * k) - data[k+mem_count-1].imag() * sin(2*M_PI / N * n * k);
    } 
    result[n+mem_count-1] /= N;
  } 
}

