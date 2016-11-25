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

RanCor::RanCor(LAMMPS *lmp, int mem_count, double *mem_kernel, double precision, double t_target) : Pointers(lmp)
{
  this->mem_count = mem_count;
  this->mem_kernel = mem_kernel;
  this->precision = precision;
  this->t_target = t_target;
  
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
    //mem_kernel[i] =exp(-19.30*i*0.005)*cos(28.25*i*0.005);
  }
  
  // init coeff by fourier transformation
  init_acoeff();

  // init optimizer
  /*NelderMeadOptimizer opt(2*mem_count-1, precision, mem_kernel, mem_count);
  
  // request a simplex to start with
  Vector v(a_coeff,2*mem_count-1);
  init_opt(opt,v,2*mem_count-1);

  // optimize
  v = opt.step(v, min_function(v,2*mem_count-1));
  printf("optimizer quality: %f\n",min_function(v,2*mem_count-1)); 
  
  for (i=0; i<2*mem_count-1; i++) {
    a_coeff[i] = v[i];
  }*/
  
  // print the memory
  FILE * out;
  out=fopen("ansatz.dat","w");
  int N = mem_count-1;
  for(n=0;n<mem_count;n++){
    double loc_sum = 0.0;
    for(s=-N;s<N;s++){
      if (n-s> N) continue;
      loc_sum += a_coeff[s+N]*a_coeff[n-s+N];
    }
    fprintf(out,"%d %f %f %f\n",n,mem_kernel[n],loc_sum,mem_kernel[n]-loc_sum);
  }
   fclose(out);
  
  /*for (i=0; i<mem_count; i++) {
    mem_kernel[i]*=norm;
  }
  for (i=0; i<2*mem_count-1; i++) {
    a_coeff[i]*=sqrt(norm);
  }*/
  
  // correct parameters to fullfil FDT
  printf("%f\n",t_target);
  for (i=0; i<2*mem_count-1; i++) {
    a_coeff[i]*=sqrt(t_target);
  }
  
#else //Prony-Series
  double *mem_save = new double[mem_count]; 
  double norm = mem_kernel[0];
  for (int i=0; i<mem_count; i++) {
    mem_kernel[i] /= norm;
    mem_save[i]=mem_kernel[i];
  }

  a_coeff = new double[6];
  
  int N_prob = 5;
  
  for (int j=0; j<N_prob;j++) {
  
    int dim_cos = 3;
    int dim_sin = 3;
    int dim_tot = dim_cos + dim_sin;
  
    a_coeff[0] = 0.1;
    a_coeff[1] = 0.1;
    a_coeff[2] = 0.1;
    a_coeff[3] = 0.1;
    a_coeff[4] = 0.1;
    a_coeff[5] = 0.1;
  
    // init optimizer
    NelderMeadOptimizer opt(dim_cos,dim_tot, precision, mem_kernel, mem_count);
  
    // request a simplex to start with
    Vector v(a_coeff,dim_tot);
    init_opt(opt,v,dim_cos,dim_tot);

    v = opt.step(v, min_function(v,dim_cos,dim_tot));
    printf("optimizer quality: %f\n",min_function(v,dim_cos,dim_tot)); 
    
    
    for (i=0; i<dim_tot; i++) a_coeff[i] = v[i];
    
    for (n=0; n<mem_count; n++) {
      double sum = 0;
      for(s=0;s<dim_cos;s+=3){
	sum += a_coeff[s]*exp(-a_coeff[s+1]*n)*cos(a_coeff[s+2]*n);
      }
      for(s=dim_cos;s<dim_tot;s+=3){
	sum += a_coeff[s]*exp(-a_coeff[s+1]*n)*sin(a_coeff[s+2]*n);
      }
      mem_kernel[n]-=sum;
    }
  
  }
  
  FILE * out;
  out=fopen("ansatz.dat","w");
  for (n=0; n<mem_count; n++) {
    fprintf(out,"%d %f %f %f\n",n,mem_save[n],mem_save[n]-mem_kernel[n],mem_kernel[n]);
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
  
  int N = mem_count-1;
  
  a_coeff = new double[2*N+1];
  for (int i=0; i<= 2*N; i++) a_coeff[i]=0.0;
  
  complex<double> FT_mem_kernel[2*N];
  
  forwardDFT(mem_kernel,N ,FT_mem_kernel);
  FILE * out;
  out=fopen("ft_mem.dat","w");
  for (int i=0; i< 2*N; i++) {
    fprintf(out,"%d %f\n",i,FT_mem_kernel[i].real());
  }
  fclose(out);
  
  complex<double> FT_a_coeff[2*N];
  
  for (int i=0; i<2*N;i++)
   FT_a_coeff[i] = sqrt(FT_mem_kernel[i]);
  
  out=fopen("sqrt_ft_mem.dat","w");
  for (int i=0; i< 2*N; i++) {
    fprintf(out,"%d %f %f\n",i,FT_a_coeff[i].real(),FT_a_coeff[i].imag());
  }
  fclose(out);
  
  inverseDFT(FT_a_coeff,N, a_coeff);
  out=fopen("as.dat","w");
  for (int i=0; i<= 2*N; i++) {
    fprintf(out,"%d %f\n",i,a_coeff[i]);
  }
  fclose(out);
  
  /*for(int s=0;s<N;s++){
    printf("%d %f\n",s,a_coeff[s]);
  }*/
  
  /*int n,s;
  for(n=0;n<mem_count;n++){
    double loc_sum = 0.0;
    for(s=0;s<N;s++){
      if (n-s>0) continue;
      loc_sum += a_coeff[s]*a_coeff[n-s+N-1];
    }
    printf("%d %f %f\n",n,mem_kernel[n],loc_sum);
  }*/
  
  //error->all(FLERR,"break");
}

/* ---------------------------------------------------------------------- 
  initializes the nm optimizer
  ----------------------------------------------------------------------  */
void RanCor::init_opt(NelderMeadOptimizer &opt, Vector v, int dim_cos, int dim_tot) {
  
  opt.insert(v,min_function(v,dim_cos,dim_tot));
  
  double factor = 0.5;

  double p = factor/(dim_tot*sqrt(2))*(sqrt(dim_tot+1)+dim_tot-1);
  double q = factor/(dim_tot*sqrt(2))*(sqrt(dim_tot+1)-1);

  int d,i;
  for (i=0; i<dim_tot; i++) {
    Vector new_v(a_coeff,dim_tot);
    for (d=0; d<dim_tot; d++) {
      if (d!=i) {
	new_v[d] += q;
      } else {
	new_v[d] += p;
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
	      loc_sum += v[s]*exp(-v[s+1]*n)*cos(v[s+2]*n);
	    }
	    for(s=dim_cos;s<dim_tot;s+=3){
	      loc_sum += v[s]*exp(-v[s+1]*n)*sin(v[s+2]*n);
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
  for (int k = -N; k < N; k++) { 
    result[k+N].real(0.0);
    result[k+N].imag(0.0);
    for (int n = -N; n < N; n++) { 
      double data_loc = 0.0;
      if(n<0) data_loc = data[abs(n)];
      else data_loc = data[n];
      result[k+N].real( result[k+N].real() + data_loc * cos(M_PI / N * n * k));
      result[k+N].imag( result[k+N].imag() - data_loc * sin(M_PI / N * n * k));
    } 
  } 
}

/* ---------------------------------------------------------------------- 
  performs a backward DFT with complex input (and reell output)
  ----------------------------------------------------------------------  */
void RanCor::inverseDFT(complex<double> *data, int N, double *result) { 
  for (int n = -N; n <= N; n++) { 
    result[n+N] = 0.0; 
    for (int k = -N; k < N; k++) { 
      result[n+N] += data[k+N].real() * cos(M_PI / N * n * k)-data[k+N].imag() * sin(M_PI / N * n * k);
    } 
    result[n+N] /= (2*N);
  } 
}

