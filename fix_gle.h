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

FixStyle(gle,FixGLE)

#else

#ifndef LMP_FIX_GLE_H
#define LMP_FIX_GLE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixGLE : public Fix {
 public:
  FixGLE(class LAMMPS *, int, char **);
  virtual ~FixGLE();
  int setmask();
  void init();
  void setup(int);
  virtual void post_force(int);
  void reset_target(double);
  void reset_dt();
  int modify_param(int, char **);
  virtual double compute_scalar();
  double memory_usage();
  virtual void *extract(const char *, int &);
  void grow_arrays(int);
  void copy_arrays(int, int, int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);

 protected:
  double **save_velocity; //used to save peratom velocities
  double **save_random; //used to save peratom random numbers
  int lastindex;
   
  int flangevin_allocated;
  double t_start,t_period,t_stop,t_target;
  int mem_count;
  FILE * mem_file;
  double *mem_kernel;
  double mem_dt;
  double *gfactor1,*gfactor2;
  double energy,energy_onestep;
  double tsqrt;
  int tstyle,tvar;

  int maxatom1,maxatom2;
  double **flangevin;
  double *tforce;
  int nvalues;

  char *id_temp;
  class Compute *temperature;

  class RanMars *random;
  class RanCor *random_correlator;
  int seed;
  double precision;

  void compute_target();
  void read_mem_file();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Fix langevin period must be > 0.0

The time window for temperature relaxation must be > 0

W: Energy tally does not account for 'zero yes'

The energy removed by using the 'zero yes' flag is not accounted
for in the energy tally and thus energy conservation cannot be
monitored in this case.

E: Fix langevin omega requires atom style sphere

Self-explanatory.

E: Fix langevin angmom requires atom style ellipsoid

Self-explanatory.

E: Variable name for fix langevin does not exist

Self-explanatory.

E: Variable for fix langevin is invalid style

It must be an equal-style variable.

E: Fix langevin omega requires extended particles

One of the particles has radius 0.0.

E: Fix langevin angmom requires extended particles

This fix option cannot be used with point paritlces.

E: Cannot zero GLE force of 0 atoms

The group has zero atoms, so you cannot request its force
be zeroed.

E: Fix langevin variable returned negative temperature

Self-explanatory.

E: Could not find fix_modify temperature ID

The compute ID for computing temperature does not exist.

E: Fix_modify temperature ID does not compute temperature

The compute ID assigned to the fix must compute temperature.

W: Group for fix_modify temp != fix group

The fix_modify command is specifying a temperature computation that
computes a temperature on a different group of atoms than the fix
itself operates on.  This is probably not what you want to do.

*/
