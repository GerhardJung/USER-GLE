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

#include <string.h>
#include <stdlib.h>
#include "fix_addactivity.h"
#include "atom.h"
#include "atom_masks.h"
#include "accelerator_kokkos.h"
#include "random_mars.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "region.h"
#include "respa.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "force.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{ABP};

/* ---------------------------------------------------------------------- */

FixAddActivity::FixAddActivity(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 8) error->all(FLERR,"Illegal fix addactivity command");
  
  // orientation needed to add activity 
  
  if (!atom->mu_flag)
    error->all(FLERR,"Fix addactivity requires atom attribute mu");

  style = 0;
  Fact = 0.0;
  D = 0.0;
  
  int iarg = 3;
  if (strcmp(arg[iarg],"style") == 0) {
    if (strcmp(arg[iarg+1],"ABP") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix addactivity command");
      style = ABP;
      Fact = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      D = utils::numeric(FLERR,arg[iarg+2],false,lmp);
    } else error->all(FLERR,"Illegal fix addactivity command");
    iarg += 4;
  } else error->all(FLERR,"Illegal fix addactivity command");

  // optional args
  
  // seed for random numbers
  int seed = utils::inumeric(FLERR,arg[iarg],false,lmp);
  iarg++;

  nevery = 1;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"every") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix addactivity command");
      nevery = atoi(arg[iarg+1]);
      if (nevery <= 0) error->all(FLERR,"Illegal fix addactivity command");
      iarg += 2;
    } else error->all(FLERR,"Illegal fix addactivity command");
  }
  
  // initialize Marsaglia RNG with processor-unique seed
  random = new RanMars(lmp,seed + comm->me);
  
  memory->create(sforce,maxatom,4,"addactivity:sforce");

}

/* ---------------------------------------------------------------------- */

FixAddActivity::~FixAddActivity()
{

}

/* ---------------------------------------------------------------------- */

int FixAddActivity::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= THERMO_ENERGY;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAddActivity::init()
{
  
}

/* ---------------------------------------------------------------------- */

void FixAddActivity::setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAddActivity::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAddActivity::post_force(int vflag)
{
  double **x = atom->x;
  double **f = atom->f;
  double **mu = atom-> mu;
  double *rad = atom-> radius;
  double **omega = atom->omega;
  int *mask = atom->mask;
  imageint *image = atom->image;
  int nlocal = atom->nlocal;

  if (update->ntimestep % nevery) return;
  
  // reallocate sforce array if necessary

  if (atom->nmax > maxatom) {
    maxatom = atom->nmax;
    memory->destroy(sforce);
    memory->create(sforce,maxatom,4,"addactivity:sforce");
  }

  // set force vector
  
  if (style == ABP) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
	sforce[i][0] = Fact*mu[i][0];
	sforce[i][1] = Fact*mu[i][1];
	sforce[i][2] = Fact*mu[i][2];
      }
    }
  }
  
  // add force

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      f[i][0] += sforce[i][0];
      f[i][1] += sforce[i][1];
      f[i][2] += sforce[i][2];
    }
  }
  
  // set angular velocity w = sqrt(3D_r)*\zeta(t) (projected on velocity of jth particle)
  double eta0, eta1, eta2;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      //printf("%f\n",rad[i]);
      double Dr = 3.0/4.0*D*1/(rad[i]*rad[i]);
      Fnoise = sqrt(2.0*Dr/update->dt);
      //eta0 = Fnoise * random->gaussian();
      //eta1 = Fnoise * random->gaussian();
      //eta2 = Fnoise * random->gaussian();
      omega[i][0] = Fnoise * random->gaussian();
      omega[i][1] = Fnoise * random->gaussian();
      omega[i][2] = Fnoise * random->gaussian();
    }
  }
  
  
}

/* ---------------------------------------------------------------------- */

void FixAddActivity::min_post_force(int vflag)
{
  post_force(vflag);
}
