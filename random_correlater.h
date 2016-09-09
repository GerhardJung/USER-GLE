/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_RANCOR_H
#define LMP_RANCOR_H

#include "pointers.h"
#include "nm_optimizer.h"

namespace LAMMPS_NS {

class RanCor : protected Pointers {
 public:
  RanCor(class LAMMPS *, int, double*, double);
  ~RanCor();
  double gaussian(double*, int);

 private:
  int mem_count;
  double *mem_kernel;
  
  double precision;
  float *a_coeff;
  
  void init();
  float min_function(Vector, int);
  void init_opt(NelderMeadOptimizer &opt, Vector v, int);
  
  FILE * test_out;
  
};

}

#endif

/* ERROR/WARNING messages:

E: Invalid seed for Marsaglia random # generator

The initial seed for this random number generator must be a positive
integer less than or equal to 900 million.

*/
