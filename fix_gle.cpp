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
#include "fix_gle.h"
#include "math_extra.h"
#include "atom.h"
#include "atom_vec_ellipsoid.h"
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
#include "random_mars.h"
#include "memory.h"
#include "error.h"
#include "group.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NOBIAS,BIAS};
enum{CONSTANT,EQUAL,ATOM};

#define SINERTIA 0.4          // moment of inertia prefactor for sphere
#define EINERTIA 0.2          // moment of inertia prefactor for ellipsoid

/* ---------------------------------------------------------------------- */

FixGLE::FixGLE(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 7) error->all(FLERR,"Illegal fix langevin command");

  dynamic_group_allow = 1;
  scalar_flag = 1;
  global_freq = 1;
  extscalar = 1;
  nevery = 1;

  t_start = force->numeric(FLERR,arg[3]);
  t_target = t_start;
  tstyle = CONSTANT;


  t_stop = force->numeric(FLERR,arg[4]);
  t_period = force->numeric(FLERR,arg[5]);
  seed = force->inumeric(FLERR,arg[6]);

  if (t_period <= 0.0) error->all(FLERR,"Fix langevin period must be > 0.0");
  if (seed <= 0) error->all(FLERR,"Illegal fix langevin command");
  
  

  // initialize Marsaglia RNG with processor-unique seed

  random = new RanMars(lmp,seed + comm->me);

  // allocate per-type arrays for force prefactors

  gfactor1 = new double[atom->ntypes+1];
  gfactor2 = new double[atom->ntypes+1];
  ratio = new double[atom->ntypes+1];
  for (int i = 1; i <= atom->ntypes; i++) ratio[i] = 1.0;

  // set temperature = NULL, user can override via fix_modify if wants bias

  id_temp = NULL;
  temperature = NULL;

  energy = 0.0;

  // flangevin is unallocated until first call to setup()
  // compute_scalar checks for this and returns 0.0 
  // if flangevin_allocated is not set

  flangevin = NULL;
  flangevin_allocated = 0;
  tforce = NULL;
  maxatom1 = maxatom2 = 0;

}

/* ---------------------------------------------------------------------- */

FixGLE::~FixGLE()
{
  delete random;
  delete [] gfactor1;
  delete [] gfactor2;
  delete [] ratio;
  delete [] id_temp;
  memory->destroy(flangevin);
  memory->destroy(tforce);

}

/* ---------------------------------------------------------------------- */

int FixGLE::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= END_OF_STEP;
  mask |= THERMO_ENERGY;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixGLE::init()
{

  // set force prefactors

  if (!atom->rmass) {
    for (int i = 1; i <= atom->ntypes; i++) {
      gfactor1[i] = -atom->mass[i] / t_period / force->ftm2v;
      gfactor2[i] = sqrt(atom->mass[i]) *
        sqrt(24.0*force->boltz/t_period/update->dt/force->mvv2e) /
        force->ftm2v;
      gfactor1[i] *= 1.0/ratio[i];
      gfactor2[i] *= 1.0/sqrt(ratio[i]);
    }
  }

}

/* ---------------------------------------------------------------------- */

void FixGLE::setup(int vflag)
{
  
  post_force(vflag);

}

/* ---------------------------------------------------------------------- */

void FixGLE::post_force(int vflag)
{
  double gamma1,gamma2;

  double **v = atom->v;
  double **f = atom->f;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;


  double fdrag[3],fran[3];


  compute_target();

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      gamma1 = gfactor1[type[i]];
      gamma2 = gfactor2[type[i]] * tsqrt;

      fran[0] = gamma2*(random->uniform()-0.5);
      fran[1] = gamma2*(random->uniform()-0.5);
      fran[2] = gamma2*(random->uniform()-0.5);

      fdrag[0] = gamma1*v[i][0];
      fdrag[1] = gamma1*v[i][1];
      fdrag[2] = gamma1*v[i][2];

      f[i][0] += fdrag[0] + fran[0];
      f[i][1] += fdrag[1] + fran[1];
      f[i][2] += fdrag[2] + fran[2];

    }
  }

}

/* ----------------------------------------------------------------------
   set current t_target and t_sqrt
------------------------------------------------------------------------- */

void FixGLE::compute_target()
{
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double delta = update->ntimestep - update->beginstep;
  if (delta != 0.0) delta /= update->endstep - update->beginstep;

  // if variable temp, evaluate variable, wrap with clear/add
  // reallocate tforce array if necessary

  t_target = t_start + delta * (t_stop-t_start);
  tsqrt = sqrt(t_target);
 
}

/* ----------------------------------------------------------------------
   tally energy transfer to thermal reservoir
------------------------------------------------------------------------- */

void FixGLE::end_of_step()
{

}

/* ---------------------------------------------------------------------- */

void FixGLE::reset_target(double t_new)
{
  t_target = t_start = t_stop = t_new;
}

/* ---------------------------------------------------------------------- */

void FixGLE::reset_dt()
{
  if (atom->mass) {
    for (int i = 1; i <= atom->ntypes; i++) {
      gfactor2[i] = sqrt(atom->mass[i]) *
        sqrt(24.0*force->boltz/t_period/update->dt/force->mvv2e) /
        force->ftm2v;
      gfactor2[i] *= 1.0/sqrt(ratio[i]);
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixGLE::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"temp") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal fix_modify command");
    delete [] id_temp;
    int n = strlen(arg[1]) + 1;
    id_temp = new char[n];
    strcpy(id_temp,arg[1]);

    int icompute = modify->find_compute(id_temp);
    if (icompute < 0)
      error->all(FLERR,"Could not find fix_modify temperature ID");
    temperature = modify->compute[icompute];

    if (temperature->tempflag == 0)
      error->all(FLERR,
                 "Fix_modify temperature ID does not compute temperature");
    if (temperature->igroup != igroup && comm->me == 0)
      error->warning(FLERR,"Group for fix_modify temp != fix group");
    return 2;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

double FixGLE::compute_scalar()
{
  return 0.0;
}

/* ----------------------------------------------------------------------
   extract thermostat properties
------------------------------------------------------------------------- */

void *FixGLE::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str,"t_target") == 0) {
    return &t_target;
  }
  return NULL;
}

/* ----------------------------------------------------------------------
   memory usage of tally array
------------------------------------------------------------------------- */

double FixGLE::memory_usage()
{
  double bytes = 0.0;
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate atom-based array for franprev
------------------------------------------------------------------------- */

void FixGLE::grow_arrays(int nmax)
{
  
}

/* ----------------------------------------------------------------------
   copy values within local atom-based array
------------------------------------------------------------------------- */

void FixGLE::copy_arrays(int i, int j, int delflag)
{
  
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixGLE::pack_exchange(int i, double *buf)
{
  return nvalues;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixGLE::unpack_exchange(int nlocal, double *buf)
{
  return nvalues;
}
