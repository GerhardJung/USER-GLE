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

FixStyle(gle/noise,FixGLENoise)

#else

#ifndef LMP_FIX_GLE_NOISE_H
#define LMP_FIX_GLE_NOISE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixGLENoise : public Fix {
 public:
  FixGLENoise(class LAMMPS *, int, char **);
  virtual ~FixGLENoise();
  int setmask();
  virtual void init();
  virtual void initial_integrate(int);
  virtual void final_integrate();
  virtual double compute_vector(int);

  double memory_usage();
  void grow_arrays(int);

 protected:
  double dtv,dtf;
  double t_target;

  int aux_terms;
  double aux_a_self;
  double aux_a_cross;
  double *aux_c11;
  double *aux_c21;
  double *aux_c22;
  double *aux_c24;
  double *aux_lam1;
  double *aux_lam2;

  double **q_aux;

  class RanMars *random;
  double **ran_self;
  double **ran_cross;
  
  double **chol_decomp;
  
  double *v_step;
  double *f_step;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Fix gld series type must be pprony for now

Self-explanatory.

E: Fix gld prony terms must be > 0

Self-explanatory.

E: Fix gld start temperature must be >= 0

Self-explanatory.

E: Fix gld stop temperature must be >= 0

Self-explanatory.

E: Fix gld needs more prony series coefficients

Self-explanatory.

E: Fix gld c coefficients must be >= 0

Self-explanatory.

E: Fix gld tau coefficients must be > 0

Self-explanatory.

E: Cannot zero gld force for zero atoms

There are no atoms currently in the group.

*/
