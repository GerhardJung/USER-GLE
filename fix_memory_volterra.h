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

#ifdef FIX_CLASS

FixStyle(memory/volterra,FixMemoryVolterra)

#else

#ifndef LMP_FIX_MEMORY_VOLTERRA_H
#define LMP_FIX_MEMORY_VOLTERRA_H

#include "stdio.h"
#include "fix.h"

namespace LAMMPS_NS {

class FixMemoryVolterra : public Fix {
 public:
  FixMemoryVolterra(class LAMMPS *, int, char **);
  ~FixMemoryVolterra();
  int setmask();
  void init();
  void setup(int);
  void end_of_step();
  double compute_array(int,int);
  void reset_timestep(bigint);
  double memory_usage();
  void grow_arrays(int);
  void copy_arrays(int, int, int);

 private:
  int nrepeat,nfreq,nevery_corr;
  
  double **corr; //used to store correlation
  double **mem;  //used to memory

  bigint nextvalid();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Cannot open fix ave/correlate file %s

The specified file cannot be opened.  Check that the path and name are
correct.

E: Compute ID for fix ave/correlate does not exist

Self-explanatory.

E: Fix ave/correlate compute does not calculate a scalar

Self-explanatory.

E: Fix ave/correlate compute does not calculate a vector

Self-explanatory.

E: Fix ave/correlate compute vector is accessed out-of-range

The index for the vector is out of bounds.

E: Fix ID for fix ave/correlate does not exist

Self-explanatory.

E: Fix ave/correlate fix does not calculate a scalar

Self-explanatory.

E: Fix ave/correlate fix does not calculate a vector

Self-explanatory.

E: Fix ave/correlate fix vector is accessed out-of-range

The index for the vector is out of bounds.

E: Fix for fix ave/correlate not computed at compatible time

Fixes generate their values on specific timesteps.  Fix ave/correlate
is requesting a value on a non-allowed timestep.

E: Variable name for fix ave/correlate does not exist

Self-explanatory.

E: Fix ave/correlate variable is not equal-style variable

Self-explanatory.

E: Fix ave/correlate missed timestep

You cannot reset the timestep to a value beyond where the fix
expects to next perform averaging.

*/