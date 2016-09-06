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

ComputeStyle(radf,ComputeRADF)

#else

#ifndef LMP_COMPUTE_RADF_H
#define LMP_COMPUTE_RADF_H

#include <stdio.h>
#include "compute.h"

namespace LAMMPS_NS {

class ComputeRADF : public Compute {
 public:
  ComputeRADF(class LAMMPS *, int, char **);
  ~ComputeRADF();
  void init();
  void init_list(int, class NeighList *);
  void compute_array();

 private:
  int nbin_r;            // # of radial radf bins
  int nbin_a;            // # of angular radf bins
  int npairs;            // # of radf pairs
  double delr,delrinv;   // bin width and its inverse
  double dela,delainv;   // bin width and its inverse
  int ***radfpair;        // map 2 type pair to radf pair for each histo
  int **nradfpair;        // # of histograms for each type pair
  int *ilo,*ihi,*jlo,*jhi;
  double **hist;         // histogram bins
  double **histall;      // summed histogram bins across all procs

  int *typecount;
  int *icount,*jcount;
  int *duplicates;
  
  int max_icount;
  double **ilist_x;

  class NeighList *list; // half neighbor list
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Compute radf requires a pair style be defined

Self-explanatory.

*/
