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

FixStyle(gle/pair/jung,FixGLEPairJung)

#else

#ifndef LMP_FIX_GLE_PAIR_JUNG_H
#define LMP_FIX_GLE_PAIR_JUNG_H

#include "fix.h"
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigen>

namespace LAMMPS_NS {

class FixGLEPairJung : public Fix {
 public:
  FixGLEPairJung(class LAMMPS *, int, char **);
  virtual ~FixGLEPairJung();
  int setmask();
  virtual void init();
  virtual void initial_integrate(int);
  virtual void final_integrate();
  virtual double compute_vector(int);

  double memory_usage();
  void grow_arrays(int);

 protected:
  double dtv,dtf, int_a,int_b;
  double t_target;

  FILE * input;
  char* keyword;
  int memory_flag;
  
  double dStart,dStep,dStop;
  int Nd;
  double tStart,tStep,tStop;
  int Nt;
  int Niter;
  
  double *self_data;
  double *cross_data;
  
  int Nupdate;
  double **ran;
  double *fd;
  double *fr;
  double **x_save;
  int lastindex;
  double **r_step;
  double **f_step;
  double **r_save;

  class RanMars *random;
  
  // cholsky decomp
  Eigen::MatrixXd A;
  Eigen::MatrixXd Aps;
 
  
  // Timing 
  int me;
  double t1,t2;
  double time_read;
  double time_init;
  double time_int_rel1;
  double time_dist_update;
  double time_int_rel2;
  
  void read_input();
  void update_cholesky();
  void distance_update();
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
