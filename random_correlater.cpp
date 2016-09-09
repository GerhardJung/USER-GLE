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
  
  a_coeff = new float[mem_count];
  for (i=0; i< mem_count; i++) a_coeff[i]=0.0;
  
  for (int step = 10; step <= mem_count; step+=10) {
    NelderMeadOptimizer opt(step, precision, mem_kernel);
  
    // request a simplex to start with
    Vector v(a_coeff,step);
    for (i=0; i< step; i++) printf("%f ",v[i]);
      printf("\n");

    init_opt(opt,v,step);
    //opt.print_v();

      while (!opt.done()) {
	v = opt.step(v, min_function(v,step));
	printf("optimizer quality: %f\n",min_function(v,step));
      }
  
    for ( i=0; i < step; i++ ) a_coeff[i] = v[i];
  
    for(n=0;n<mem_count;n++){
      float loc_sum = 0.0;
      for(s=0;s<mem_count;s++){
	if(n+s>=mem_count) continue;
	loc_sum += a_coeff[s]*a_coeff[s+n];
      }
    
      printf("%d %f %f\n",n,mem_kernel[n],loc_sum);
    }
  
    for (i=0; i< step; i++) printf("%f ",v[i]);
    printf("\n");

  }
  
  
  
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
  for (i=0; i<mem_count; i++) {
    ran += normal[j]*a_coeff[i];
    j--;
    if (j<0) j=mem_count-1;
  }

  return ran;
}

/* ---------------------------------------------------------------------- 
  initializes the nm optimizer
  ----------------------------------------------------------------------  */

void RanCor::init_opt(NelderMeadOptimizer &opt, Vector v, int dimension) {
  
  opt.insert(v,min_function(v,dimension));
  
  double p = 1.0/(dimension*sqrt(2))*(sqrt(dimension+1)+dimension-1);
  double q = 1.0/(dimension*sqrt(2))*(sqrt(dimension+1)-1);

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
  for(n=0;n<dimension;n++){
    float loc_sum = 0.0;
    for(s=0;s<dimension;s++){
      if(n+s>=dimension) continue;
      loc_sum += v[s]*v[s+n];
    }
    loc_sum -= mem_kernel[n];
    sum += loc_sum*loc_sum;
  }
  return sum;
}
