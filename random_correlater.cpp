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
  for (i=0; i< mem_count; i++) a_coeff[i]=0.5;
  
  NelderMeadOptimizer opt(mem_count, precision);
  
  // request a simplex to start with
  Vector v(a_coeff,mem_count);

  while (!opt.done()) {
    //printf("%f %f %f %f %f %f %f\n",v[0],v[1],v[2],v[3],v[4],v[5],f(v));
    v = opt.step(v, min_function(v));
  }
  
  for ( i=0; i < mem_count; i++ ) a_coeff[i] = v[i];
  
  /* for(n=0;n<mem_count;n++){
    float loc_sum = 0.0;
    for(s=0;s<mem_count;s++){
      if(n+s>=mem_count) continue;
      loc_sum += v[s]*v[s+n];
    }
    printf("%d %f %f\n",n,mem_kernel[n],loc_sum);
  } */
  
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
  function for the minimization process to determine correlation coefficients
  ----------------------------------------------------------------------  */

float RanCor::min_function(Vector v)
{
  float sum=0.0;
  int n,s;
  for(n=0;n<mem_count;n++){
    float loc_sum = 0.0;
    for(s=0;s<mem_count;s++){
      if(n+s>=mem_count) continue;
      loc_sum += v[s]*v[s+n];
    }
    loc_sum -= mem_kernel[n];
    sum += loc_sum*loc_sum;
  }
  return sum;
}

 
