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

// Correlated random number generator
// see Appendix B: J. Chem. Phys. 143, 243128 (2015)

#include <math.h>
#include "random_correlated.h"
#include "comm.h"
#include "random_mars.h"
#include "error.h"
#include "nm_optimizer.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

RanCor::RanCor(LAMMPS *lmp, int seed, int mem_count, double *mem_kernel, double precision) : Pointers(lmp)
{
  this->seed = seed;
  this->mem_count = mem_count;
  this->mem_kernel = mem_kernel;
  this->precision = precision;
   
  // create basic random number generator
  random = new RanMars(lmp,seed);
  
  // init the coefficients for the correlation
  int i;
  a_coeff = new double[mem_count];
  for (i=0; i< mem_count; i++) a_coeff[i]=0;
  init();
}

/* ---------------------------------------------------------------------- */

RanCor::~RanCor()
{
  
}

/* ---------------------------------------------------------------------- */

void RanCor::init()
{
  int i;
  printf("start optimize\n");
  
  NelderMeadOptimizer opt(mem_count, precision);
  
}

 
