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

FixStyle(ave/correlate/peratom,FixAveCorrelatePeratom)

#else

#ifndef LMP_FIX_AVE_CORRELATE_PERATOM_H
#define LMP_FIX_AVE_CORRELATE_PERATOM_H

#include "stdio.h"
#include "fix.h"

namespace LAMMPS_NS {

class FixAveCorrelatePeratom : public Fix {
 public:
  FixAveCorrelatePeratom(class LAMMPS *, int, char **);
  ~FixAveCorrelatePeratom();
  int setmask();
  void init();
  void setup(int);
  void end_of_step();
  double compute_array(int,int);
  void reset_timestep(bigint);
  double memory_usage();
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);
  void grow_arrays(int);
  void copy_arrays(int, int, int);
  
 protected:
  char *id_fix;
  class FixStore *fix;

 private:
  int me,nvalues;
  int nrepeat,nfreq;
  int nav,nsave;
  bigint nvalid;
  int *which,*argindex,*value2index;
  char **ids;
  FILE *fp;
  
  double **array; //used for peratom quantities

  int type,ave,startstep,overwrite, dynamics, memory_flag, memory_switch;
  int include_orthogonal, include_memory;
  double prefactor;
  int bin;
  double range_lower,range_upper;
  char *title1,*title2,*title3;
  long filepos;

  int lastindex;       // index in values ring of latest time sample
  int nsample;         // number of time samples in values ring

  int npair;           // number of correlation pairs to calculate
  int *count;
  double **corr;
   
  int *save_count;     // saved values at Nfreq for output via compute_array()
  double **save_corr;
  
  int ngroup_glo;
  tagint *group_ids;
  double *group_mass;
  double **group_data_loc,**group_data;

  void accumulate(int *indices_group, int ngroup_loc);
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