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

#ifdef FIX_CLASS

FixStyle(brownian,FixBrownian)

#else

#ifndef LMP_FIX_BROWNIAN_H
#define LMP_FIX_BROWNIAN_H

#include "fix.h"

namespace LAMMPS_NS {

class FixBrownian : public Fix {
 public:
  FixBrownian(class LAMMPS *, int, char **);
  virtual ~FixBrownian();
  int setmask();
  void init();
  void setup(int);
  virtual void initial_integrate(int);
  virtual void final_integrate();
  

 protected:
  double D;
  double temp;

  class RanMars *random;
  class RanCor *random_correlator;
  int seed;

};

}

#endif
#endif
