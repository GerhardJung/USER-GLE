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

FixGLE::FixGLE(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 8) error->all(FLERR,"Illegal fix langevin command");

  // set fix properties
  
  dynamic_group_allow = 1;
  scalar_flag = 1;
  global_freq = 1;
  extscalar = 1;
  nevery = 1;
  
  // read input parameter

  t_start = force->numeric(FLERR,arg[3]);
  t_target = t_start;
  tstyle = CONSTANT;
  t_stop = force->numeric(FLERR,arg[4]);
  t_period=1;
  
  mem_count = force->numeric(FLERR,arg[6]);
  mem_file = fopen(arg[5],"r");
  mem_kernel = new double[mem_count];
  read_mem_file();
  
  seed = force->inumeric(FLERR,arg[7]);
  
  if (t_period <= 0.0) error->all(FLERR,"Fix langevin period must be > 0.0");
  if (seed <= 0) error->all(FLERR,"Illegal fix langevin command");
  
  // initialize correlated RNG with processor-unique seed
  
  random = new RanMars(lmp,seed + comm->me);
  precision = 0.005;
  random_correlator = new RanCor(lmp,mem_count, mem_kernel, precision);
  
  // allocate and init per-atom arrays (velocity and normal random number)
  
  save_velocity = NULL;
  save_random = NULL;
  comm->maxexchange_fix += 6*mem_count;
  grow_arrays(atom->nmax);
  atom->add_callback(0);
  
  lastindex = 0;
  int nlocal= atom->nlocal, n, d,m;
  for ( n=0; n<nlocal; n++ )
    for ( d=0; d<3; d++ ) 
      for ( m=0; m<mem_count; m++ ) {
	save_velocity[n][d*mem_count+m] = 0.0;
	save_random[n][d*mem_count+m] = 0.0;
      }

  // allocate and init per-type arrays for force prefactors
  
  gfactor1 = new double[atom->ntypes+1];
  gfactor2 = new double[atom->ntypes+1];

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
  comm->maxexchange_fix -= 6*mem_count;
  atom->delete_callback(id,0);
  delete random;
  delete random_correlator;
  delete [] gfactor1;
  delete [] gfactor2;
  delete [] id_temp;
  memory->destroy(flangevin);
  memory->destroy(tforce);

}

/* ---------------------------------------------------------------------- */

int FixGLE::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= THERMO_ENERGY;
  return mask;
}

/* ---------------------------------------------------------------------- */


void FixGLE::read_mem_file()
{
  // skip first lines starting with #
  char buf[0x1000];
  long filepos;
  filepos = ftell(mem_file);
  while (fgets(buf, sizeof(buf), mem_file) != NULL) {
    if (buf[0] != '#') {
      fseek(mem_file,filepos,SEEK_SET);
      break;
    }
    filepos = ftell(mem_file);
  } 
  
  //read memory
  int i;
  double t,t_old, mem;
  t = t_old = mem = 0;
  for(i=0; i<mem_count; i++){
    t_old = t;
    fscanf(mem_file,"%lf %lf\n",&t,&mem);
    if (t - t_old - update->dt > 10E-10) error->all(FLERR,"memory needs resolution similar to timestep");
    mem_kernel[i] = mem;
  }
  
}

/* ---------------------------------------------------------------------- */

void FixGLE::init()
{

  // set force prefactors

  if (!atom->rmass) {
    for (int i = 1; i <= atom->ntypes; i++) {
      gfactor1[i] = -atom->mass[i]*update->dt;
      gfactor2[i] = sqrt(atom->mass[i]);
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
  int n,d,m;
  double gamma1,gamma2;
  double **v = atom->v;
  double **f = atom->f;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double fdrag[3],fran[3];
  
  // update velocity and random number
  for ( n=0; n<nlocal; n++ ) {
    for ( d=0; d<3; d++ ) {
      save_velocity[n][d*mem_count+lastindex] = v[n][d];
      save_random[n][d*mem_count+lastindex] = random->gaussian();
    }
  }

  compute_target();

  for ( n = 0; n < nlocal; n++) {
    if (mask[n] & groupbit) {
      gamma1 = gfactor1[type[n]];
      gamma2 = gfactor2[type[n]] * tsqrt;

      fran[0] = gamma2*random_correlator->gaussian(&save_random[n][0],lastindex);
      fran[1] = gamma2*random_correlator->gaussian(&save_random[n][mem_count],lastindex);
      fran[2] = gamma2*random_correlator->gaussian(&save_random[n][2*mem_count],lastindex);

      double mem_sum;
      int j;
      for ( d=0; d<3; d++ ) {
	mem_sum = 0.0;
	j = lastindex;
	for ( m=0; m<mem_count; m++ ) {
	  mem_sum += save_velocity[n][d*mem_count+j]*mem_kernel[m];
	  //if (n == 0 && d == 0) printf("%d %f %f\n",m,save_velocity[n][d*mem_count+j],mem_kernel[m]);
	  j--;
	  if (j < 0) j = mem_count -1;
	}
	fdrag[d] = gamma1*mem_sum;
      }

      f[n][0] += fdrag[0] + fran[0];
      f[n][1] += fdrag[1] + fran[1];
      f[n][2] += fdrag[2] + fran[2];

    }
  }
  
  lastindex++;
  if (lastindex==mem_count) lastindex=0;

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
      gfactor2[i] = sqrt(atom->mass[i]);
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
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixGLE::memory_usage() {
  double bytes = atom->nmax * 6 * mem_count * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate atom-based array
------------------------------------------------------------------------- */

void FixGLE::grow_arrays(int nmax) {
  memory->grow(save_velocity,nmax,3*mem_count,"fix/gle:save_velocity");
  memory->grow(save_random,nmax,3*mem_count,"fix/gle:save_random");
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
  int offset = 0;
  int d,m;
  // pack velocity
  for ( m=0; m<mem_count; m++ ) {
    for ( d=0; d<3; d++ ) { 
      buf[offset++] = save_velocity[i][d+3*m];
    }
  }
  
  // pack random number
  for ( m=0; m<mem_count; m++ ) {
    buf[offset++] = save_random[i][m];
  }
  
  return offset;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixGLE::unpack_exchange(int nlocal, double *buf)
{
  int offset = 0;
  int d,m;
  // pack velocity
  for ( m=0; m<mem_count; m++ ) {
    for ( d=0; d<3; d++ ) { 
      save_velocity[nlocal][d+3*m] = buf[offset++];
    }
  }
  
  // pack normal random number
  for ( m=0; m<mem_count; m++ ) {
    save_random[nlocal][m] = buf[offset++];
  }
  
  return offset;
}
