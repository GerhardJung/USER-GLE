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

#ifdef COMPUTE_CLASS

ComputeStyle(com_local,ComputeCOMLocal)

#else

#ifndef LMP_COMPUTE_COM_LOC_H
#define LMP_COMPUTE_COM_LOC_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeCOMLocal : public Compute {
 public:
  ComputeCOMLocal(class LAMMPS *, int, char **);
  ~ComputeCOMLocal();
  void init();
  void compute_vector();

 private:
   int mode;
  double radius, r2;
  int jgroup,jgroupbit;
  int nmax;
  double *pos_group_loc;
  double *pos_group_glo;
  double *count_loc;
  double *com_loc;
  double *count_glo;
  double *com_glo;
  
  // neighbor list
  int irequest;
  NeighList *list;
  
  int ngroup_loc, ngroup_glo,ngroup_scan;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

*/
