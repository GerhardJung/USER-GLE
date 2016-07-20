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
  
  void write_restart(FILE *);
  void restart(char *);

 private:
  int me,nvalues,nprocs;
  int nrepeat,nfreq;
  int nav,nsave;
  bigint nvalid;
  int *which,*argindex,*value2index;
  char **ids;
  FILE *fp;
  
  double **array; //used for peratom quantities

  int type,ave,startstep,overwrite, dynamics, memory_switch, variable_flag;
  int include_orthogonal;
  double prefactor;
  
  double **alpha;		//for orthogonal dynamics
  double **epsilon;
  double **kappa;
  double **zeta;
  double **norm;
  
  int variable_nvalues;		//for variable dependence of the correlation
  int variable_value2index;
  char *variable_id;
  int bins;
  int factor;
  double range;
  double **variable_store;
  
  char *title1,*title2,*title3;
  long filepos;
  
  int mean_flag;
  FILE *mean_file;
  double *mean_count;
  double *mean;
  long mean_filepos;

  int lastindex;       // index in values ring of latest time sample
  int firstindex;	// index in values ring of oldest time sample
  int nsample;         // number of time samples in values ring

  int npair;           // number of correlation pairs to calculate
  double *local_count,*global_count,*save_count;
  double **local_corr,**global_corr,**save_corr;
  int corr_length;
  
  int ngroup_glo;
  tagint *group_ids;
  double *group_mass;
  double **group_data_loc,**group_data;

  void accumulate(int *indices_group, int ngroup_loc);
  bigint nextvalid();
  void calc_mean(int *indices_group, int ngroup_loc);
  void decompose(double *res_data, double *dr, double *inp_data);
  int first;
  
  //timing
  double t1;
  double t2;
  double time_init_compute;
  double calc_write_nvalues;
  double write_var;
  double write_orthogonal;
  double reduce_write_global;
  double time_calc;
  double time_red_calc;
  double time_calc_mean;
  double time_total; 
  
  template <typename T> int sgn(T val);
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