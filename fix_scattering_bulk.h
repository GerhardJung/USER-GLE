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

FixStyle(scattering/bulk,FixScatteringBulk)

#else

#ifndef LMP_FIX_SCATTERING_BULK_H
#define LMP_FIX_SCATTERING_BULK_H

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

class FixScatteringBulk : public Fix {
  public:
    FixScatteringBulk(class LAMMPS *, int, char **);
    ~FixScatteringBulk();
    int setmask();
    void init();
    void setup(int);
    void end_of_step();

  protected:

    real *valST;
    real *valVEL;
    int nFunCorr;
    
    real *strucFac;

    void AllocArrays();
    void EvalSpacetimeCorr ();
    void add (real * val, int k);
    void addVACF (real * val, int k, int N);
    void AccumSpacetimeCorr ();
    void ZeroSpacetimeCorr ();
    void ZeroSpacetimeCorrIn ();
    void ZeroSpacetimeCorrIn2 ();
    void PrintSpacetimeCorr (FILE *fp);
    void PrintSpacetimeCorrIn (FILE *fp);
    void PrintSpacetimeCorrIn2 (FILE *fp);
    void PrintStrucFac (FILE *fp);
    
    int N_blocks;
    int N_count;
    int N_levels;
    int N_levels_msd;
    int N_cor;
    int dmin;
    real ** blocking_sum;
    real * pos_save;
    
    int nrelax;

    real ** correlationIn;
    real ** MSD;
    real ** NGP;
    
    int count;
    int ** countIn;
    int * countMSD;
    int * countNGP;

    
    int kmax;
    int t_loc;
    
    real ** shift;
    real ** accumulator;
    int * naccumulator;
    real ** correlation;
    int * insertindex;
    int * countcor;
    
    real ** shiftVACF;
    real ** accumulatorVACF;
    int * naccumulatorVACF;
    real ** correlationVACF;
    int * insertindexVACF;
    int * countcorVACF;

    real ** pos_save2;
    real ** correlationIn2;
    int * countIn2;
    int lastindex;
  };
}

#endif 
#endif
