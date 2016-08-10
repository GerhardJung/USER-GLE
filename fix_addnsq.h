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

FixStyle(addnsq,FixAddNsq)

#else

#ifndef LMP_FIX_ADDNSQ_H
#define LMP_FIX_ADDNSQ_H

#include "fix.h"

namespace LAMMPS_NS {

class FixAddNsq : public Fix {
 public:
  FixAddNsq(class LAMMPS *, int, char **);
  ~FixAddNsq();
  int setmask();
  void init();
  void setup(int);
  void min_setup(int);
  void post_force(int);
  void min_post_force(int);
  double memory_usage();

 private:
  int me,nmax;
  // information of atoms in the group
  int ngroup_glo;
  tagint *group_ids;
  int *indices_group;
  
  int checkpoint; //indice after which the force will be computed (for printout)
  
  // global atom data (position)
  double **group_x;
  double **group_x_loc;
  // calculated force/energy/virial
  double **group_f;
  double **group_e;
  double **group_v;
  
  // potential list to calculate
  int n_pot;
  int *type_pot;
  double **coeff_pot;
  
  void calc_ljoff(int i, int j, int p, double rsq, double &pair_force, double &energy);
  void calc_harmonic(int i, int j, int p, double rsq, double &pair_force, double &energy);
  
  double **array;

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Region ID for fix addforce does not exist

Self-explanatory.

E: Variable name for fix addforce does not exist

Self-explanatory.

E: Variable for fix addforce is invalid style

Self-explanatory.

E: Cannot use variable energy with constant force in fix addforce

This is because for constant force, LAMMPS can compute the change
in energy directly.

E: Must use variable energy with fix addforce

Must define an energy vartiable when applyting a dynamic
force during minimization.

*/
