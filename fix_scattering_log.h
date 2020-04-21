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

FixStyle(scattering/log,FixScatteringLog)

#else

#ifndef LMP_FIX_SCATTERING_LOG_H
#define LMP_FIX_SCATTERING_LOG_H

#include "fix.h"
#include <vector>


#define AllocMem(a, n, t) a = (t *) malloc ((n) * sizeof (t))

#define AllocMem2(a, n1, n2, t)\
	  AllocMem (a, n1, t *);\
	  AllocMem (a[0], n1 * n2, t);\
	  for (k = 1; k < n1; k ++) a[k] = a[k - 1] + n2;
	  
#define Sqr(x) ((x) * (x))



typedef double real;

namespace LAMMPS_NS {

class FixScatteringLog : public Fix {
  public:
    FixScatteringLog(class LAMMPS *, int, char **);
    ~FixScatteringLog();
    int setmask();
    void init();
    void setup(int);
    void end_of_step();
    

  protected:

    real *valST;
    int nFunCorr;
    int nModes;
    real channel_w;
    
    real *densityProfile;
    int profileBins;
    int profileCount;
    
    real **strucFac;

    void AllocArrays();
    void EvalSpacetimeCorr ();
    void add (real * val, int k);
    void AccumSpacetimeCorr ();
    void ZeroSpacetimeCorr ();
    void ZeroSpacetimeCorrIn ();
    void PrintSpacetimeCorr (FILE *fp);
    void PrintSpacetimeCorrIn (FILE *fp);
    void PrintDensityProfile (FILE *fp);
    void PrintStrucFac (FILE *fp);
    void output();
    
    int N_blocks;
    int N_levels;
    real ** blocking_sum;
    real * pos_save;
    real ** blocking_sumz;
    real ** correlationIn;
    int ** countIn;
    int * insertindex;
    
    int kmax;
    int t_loc;
    
    real ** shift;
    real ** accumulator;
    int * naccumulator;
    real ** correlation;
    int * countcor;

  };
}

#endif 
#endif
