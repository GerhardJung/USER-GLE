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
  
  
#ifndef PRONY
  // rescale memory (for optimizer)
  double norm = mem_kernel[0];
  for (int i=0; i<mem_count; i++) {
    mem_kernel[i] /= norm;
  }
  
  // init coeff by fourier transformation
  init_acoeff();

  // init optimizer
  NelderMeadOptimizer opt(2*mem_count-1, precision, mem_kernel, mem_count);
  
  // request a simplex to start with
  Vector v(a_coeff,2*mem_count-1);
  init_opt(opt,v,2*mem_count-1);

  // optimize
  v = opt.step(v, min_function(v,2*mem_count-1));
  printf("optimizer quality: %f\n",min_function(v,2*mem_count-1)); 
  
  for (i=0; i<2*mem_count-1; i++) {
    a_coeff[i] = v[i];
  }
  
  // print the memory
  for(n=0;n<mem_count;n++){
    double loc_sum = 0.0;
    for(s=0;s<2*mem_count-1;s++){
      if (n-s>0) continue;
      loc_sum += a_coeff[s]*a_coeff[n-s+2*mem_count-2];
    }
    printf("%d %f %f\n",n,mem_kernel[n],loc_sum);
  }
  
  for (i=0; i<mem_count; i++) {
    mem_kernel[i]*=norm;
  }
  for (i=0; i<2*mem_count-1; i++) {
    a_coeff[i]*=sqrt(norm);
  }
#else //Prony-Series
  double norm = mem_kernel[0];
  for (int i=0; i<mem_count; i++) {
    mem_kernel[i] /= norm;
  }

  int n_cos = 8;
  int n_sin = 0;
  
  
  
  int dim_cos = n_cos*3;
  int dim_sin = n_sin*4;
  int dim_tot = dim_cos + dim_sin;
  
  a_coeff = new double[dim_tot];
  for (int i=0; i<dim_cos; i+=3) {
    a_coeff[i]=1.0/n_cos;
    a_coeff[i+1]=-0.1;
    a_coeff[i+2]=0.1;
  }
  for (int i=dim_cos; i<dim_tot; i+=4) {
    a_coeff[i]=1.0;
    a_coeff[i+1]=-0.1;
    a_coeff[i+2]=0.1;
    a_coeff[i+3]=0;
  }
  
  for (n=0; n<mem_count; n++) {
    double sum = 0.0;
    for(s=0;s<dim_cos;s+=3){
      sum += a_coeff[s]*exp(a_coeff[s+1]*n)*cos(a_coeff[s+2]*n);
    }
    for(s=dim_cos;s<dim_tot;s+=4){
      if (n>=a_coeff[i+3])
	sum += a_coeff[s]*exp(a_coeff[s+1]*n)*sin(a_coeff[s+2]*(n-a_coeff[i+3]));
    }
    printf("sum %f mem %f\n",sum, mem_kernel[n]);
  }
  
  // init optimizer
  NelderMeadOptimizer opt(dim_cos,dim_tot, precision, mem_kernel, mem_count);
  
  // request a simplex to start with
  Vector v(a_coeff,dim_tot);
  init_opt(opt,v,dim_cos,dim_tot);

  v = opt.step(v, min_function(v,dim_cos,dim_tot));
  printf("optimizer quality: %f\n",min_function(v,dim_cos,dim_tot)); 
  
  for (int i=0; i< dim_tot; i++) a_coeff[i]=v[i];
  
  for (int i=0; i<dim_cos; i+=3) {
    printf("cos coeff: %f %f %f\n",a_coeff[i], a_coeff[i+1], a_coeff[i+2]);
  }
  for (int i=dim_cos; i<dim_tot; i+=4) {
    printf("sin coeff: %f %f %f %f\n",a_coeff[i], a_coeff[i+1], a_coeff[i+2],a_coeff[i+3]);
  }
  FILE * out;
  out=fopen("ansatz.dat","w");
  for (n=0; n<mem_count; n++) {
    double sum = 0;
    for(s=0;s<dim_cos;s+=3){
      sum += a_coeff[s]*exp(a_coeff[s+1]*n)*cos(a_coeff[s+2]*n);
    }
    for(s=dim_cos;s<dim_tot;s+=4){
      if (n>=a_coeff[i+3])
	sum += a_coeff[s]*exp(a_coeff[s+1]*n)*sin(a_coeff[s+2]*n);
    }
    fprintf(out,"%d %f %f\n",n,sum, mem_kernel[n]);
  }
  fclose(out);

  error->all(FLERR,"break");
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
    if (j == -1) j = 2*mem_count-2;
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
    double loc_sum = 0.0;
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
void RanCor::init_opt(NelderMeadOptimizer &opt, Vector v, int dim_cos, int dim_tot) {
  
  opt.insert(v,min_function(v,dim_cos,dim_tot));
  
  double factor = 1;

  double p = factor/(dim_tot*sqrt(2))*(sqrt(dim_tot+1)+dim_tot-1);
  double q = factor/(dim_tot*sqrt(2))*(sqrt(dim_tot+1)-1);

  int d,i;
  for (i=0; i<dim_tot; i++) {
    Vector new_v(a_coeff,dim_tot);
    for (d=0; d<dim_cos; d++) {
      if (d!=i) {
	new_v[d] += q;
      } else {
	new_v[d] += p;
      }
    }
    for (d=dim_cos; d<dim_tot; d++) {
      int factor = 1;
      if ((d-dim_cos)%3==0) factor = 100;
      if (d!=i) {
	new_v[d] += q*factor;
      } else {
	new_v[d] += p*factor;
      }
      
    }

    opt.insert(new_v,min_function(new_v,dim_cos,dim_tot));
  }
}
 
/* ---------------------------------------------------------------------- 
  function for the minimization process to determine correlation coefficients
  ----------------------------------------------------------------------  */
/*double RanCor::min_function(Vector v, int dimension)
{
  int n,s;
  double sum = 0.0;
  for(n=0;n<(dimension+1)/2;n++){
    double loc_sum = 0.0;
    for(s=0;s<dimension;s++){
      if (n-s>0) continue;
      loc_sum += v[s]*v[n-s+dimension-1];
    }
    loc_sum -= mem_kernel[n];   
    sum += loc_sum*loc_sum;
  }

  return sum;
}*/
double RanCor::min_function(Vector v, int dim_cos, int dim_tot) {
	  int n,s;
	  double sum = 0.0;
	  for(n=0;n<mem_count;n++){
	    double loc_sum = 0.0;
	    for(s=0;s<dim_cos;s+=3){
	      loc_sum += a_coeff[s]*exp(a_coeff[s+1]*n)*cos(a_coeff[s+2]*n);
	    }
	    for(s=dim_cos;s<dim_tot;s+=4){
	      if (n>=a_coeff[s+3])
	      loc_sum += a_coeff[s]*exp(a_coeff[s+1]*n)*sin(a_coeff[s+2]*n);
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
void RanCor::inverseDFT(complex<double> *data, int N, double *result) { 
  for (int n = -mem_count+1; n < mem_count; n++) { 
    result[n+mem_count-1] = 0.0; 
    for (int k = -mem_count+1; k < mem_count; k++) { 
      result[n+mem_count-1] += data[k+mem_count-1].real() * cos(2*M_PI / N * n * k) - data[k+mem_count-1].imag() * sin(2*M_PI / N * n * k);
    } 
    result[n+mem_count-1] /= N;
  } 
}

