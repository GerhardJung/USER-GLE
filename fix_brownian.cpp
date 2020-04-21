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

/* ----------------------------------------------------------------------
   Contributing authors: Carolyn Phillips (U Mich), reservoir energy tally
                         Aidan Thompson (SNL) GJF formulation
------------------------------------------------------------------------- */

#include <mpi.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "fix_brownian.h"
#include "math_extra.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "domain.h"
#include "region.h"
#include "respa.h"
#include "comm.h"
#include "input.h"
#include "variable.h"
#include "random_correlater.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"
#include "group.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NOBIAS,BIAS};
enum{CONSTANT,EQUAL,ATOM};

/* ---------------------------------------------------------------------- */

FixBrownian::FixBrownian(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 6) error->all(FLERR,"Illegal fix brownian command");

  // set fix properties
  
  global_freq = 1;
  nevery = 1;
  peratom_freq = 1;
  peratom_flag = 1;
  
  // read input parameter
  D = force->numeric(FLERR,arg[3]);

  temp = force->numeric(FLERR,arg[4]);
  
  seed = force->inumeric(FLERR,arg[5]);
  
  // optional parameter
  
  if (seed <= 0) error->all(FLERR,"Illegal fix brownian command");
  
  // initialize correlated RNG with processor-unique seed
  random = new RanMars(lmp,seed + comm->me);

}

/* ---------------------------------------------------------------------- */

FixBrownian::~FixBrownian()
{


}

/* ---------------------------------------------------------------------- */

int FixBrownian::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */


/* ---------------------------------------------------------------------- */

void FixBrownian::init()
{

  
}

/* ---------------------------------------------------------------------- */

void FixBrownian::setup(int vflag)
{

}

/* ---------------------------------------------------------------------- */

void FixBrownian::initial_integrate(int vflag)
{
  int n,d,m;
  double **v = atom->v;
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  double *mass = atom->mass;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  tagint *tag = atom->tag;
  


  for ( n = 0; n < nlocal; n++) {
    int itag = tag[n]-1;
    if (mask[n] & groupbit) {
      
    }
  }

  
}

/* ---------------------------------------------------------------------- */

void FixBrownian::final_integrate()
{
  int n,d,m;
  double **v = atom->v;
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  double *mass = atom->mass;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  tagint *tag = atom->tag;
  int itag;

  for ( n = 0; n < nlocal; n++) {
   itag = tag[n]-1;
    if (mask[n] & groupbit) {

    }
  }
  
  
}