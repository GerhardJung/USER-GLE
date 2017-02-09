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

FixStyle(gle/pair/aux,FixGLEPairAux)

#else

#ifndef LMP_FIX_GLE_PAIR_AUX_H
#define LMP_FIX_GLE_PAIR_AUX_H

#include "fix.h"

namespace LAMMPS_NS {

class FixGLEPairAux : public Fix {
 public:
  FixGLEPairAux(class LAMMPS *, int, char **);
  virtual ~FixGLEPairAux();
  int setmask();
  virtual void init();
  virtual void initial_integrate(int);
  virtual void final_integrate();
  virtual double compute_vector(int);

  double memory_usage();
  void grow_arrays(int);
  void copy_arrays(int, int, int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);
  int pack_restart(int, double *);
  void unpack_restart(int, int);
  int size_restart(int);
  int maxsize_restart();
  void init_q_aux();

 protected:
  double dtv,dtf;
  double t_target;

  int aux_terms;
  double aux_a_self;
  double aux_a_cross;
  double *aux_c_self;
  double *aux_c_cross;
  double *aux_lam;

  double **q_aux;

  class RanMars *random;
  double *ran_save;
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
