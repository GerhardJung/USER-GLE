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
  
  global_freq = 1;
  nevery = 1;
  peratom_freq = 1;
  peratom_flag = 1;
  size_peratom_cols = 6;
  
  // read input parameter

  t_start = force->numeric(FLERR,arg[3]);
  t_target = t_start;
  tstyle = CONSTANT;
  t_stop = force->numeric(FLERR,arg[4]);
  
  mem_count = force->numeric(FLERR,arg[6]);
  mem_file = fopen(arg[5],"r");
  mem_kernel = new double[mem_count];
  read_mem_file();
  
  seed = force->inumeric(FLERR,arg[7]);
  
  // optional parameter
  int restart = 0;
  int iarg = 8;
  force_flag = 1;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"restart") == 0) {
      restart = 1;
      iarg += 1;
    }
  }
  
  if (seed <= 0) error->all(FLERR,"Illegal fix langevin command");
  
  // initialize correlated RNG with processor-unique seed
  random = new RanMars(lmp,seed + comm->me);
  precision = 0.000002;
  random_correlator = new RanCor(lmp,mem_count, mem_kernel, precision);
  
  // allocate and init per-atom arrays (velocity and normal random number)
  
  save_position = NULL;
  save_random = NULL;
  comm->maxexchange_fix += 6*mem_count;
  grow_arrays(atom->nmax);
  //atom->add_callback(0);
  
  
  lastindex_p = firstindex_r  = 0;
  int nlocal= atom->nlocal, n,d,m;
  nmax = atom->nmax;

  //memory for force save
  memory->create(array,nmax,6,"fix_gle:array");
  array_atom = array;
  
  // initialize array to zero
  for (int i = 0; i < nlocal; i++)
    for (int k = 0; k < 6; k++) array[i][k] = 0.0;
    
  int *type = atom->type;
  double *mass = atom->mass;
  gjffac = 1.0/(1.0+mem_kernel[0]*update->dt/2.0/mass[type[0]]);

}

/* ---------------------------------------------------------------------- */

FixGLE::~FixGLE()
{

  //atom->delete_callback(id,0);
  delete random;
  delete random_correlator;
  delete [] mem_kernel;
  memory->destroy(save_random);
  memory->destroy(save_position);
  memory->destroy(array);

}

/* ---------------------------------------------------------------------- */

int FixGLE::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
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
  t = t_old = mem = 0.0;
  for(i=0; i<mem_count; i++){
    t_old = t;
    fscanf(mem_file,"%lf %lf\n",&t,&mem);
    //if (abs(t - t_old - update->dt) > 10E-10 && t_old != 0.0) error->all(FLERR,"memory needs resolution similar to timestep");
    mem_kernel[i] = mem;
  }
}

/* ---------------------------------------------------------------------- */

void FixGLE::init()
{

  
}

/* ---------------------------------------------------------------------- */

void FixGLE::setup(int vflag)
{

  int nlocal= atom->nlocal, n,d,m;
  double **x = atom->x;
  for ( n=0; n<nlocal; n++ )
    for ( d=0; d<3; d++ ) {
      for ( m=0; m<=mem_count; m++ ) {
	save_position[n][d*(mem_count+1)+m] = x[n][d];
      }
      for ( m=0; m<2*mem_count-1; m++ ) {
	save_random[n][d*(2*mem_count-1)+m] = random->gaussian();
      }
    }
  
}

/* ---------------------------------------------------------------------- */

void FixGLE::initial_integrate(int vflag)
{
  int n,d,m;
  double **v = atom->v;
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  double *mass = atom->mass;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double fdrag[3],fran[3];
  
  // update random numbers
  for ( n=0; n<nlocal; n++ ) {
    for ( d=0; d<3; d++ ) {
      save_random[n][d*(2*mem_count-1)+firstindex_r] = random->gaussian();
    }
  }

  for ( n = 0; n < nlocal; n++) {
    if (mask[n] & groupbit) {

      // calculate correlated noise 
      fran[0] = array[n][0] = random_correlator->gaussian(&save_random[n][0],firstindex_r);
      fran[1] = array[n][1] = random_correlator->gaussian(&save_random[n][2*mem_count-1],firstindex_r);
      fran[2] = array[n][2] = random_correlator->gaussian(&save_random[n][4*mem_count-2],firstindex_r);
      
      for (d = 0; d<3;d++) {
	
	fdrag[d]=0;
	int tn1 = lastindex_p;
	int tn = lastindex_p-1;
	if (tn < 0) tn = mem_count;
	
	for (m = 1; m<mem_count;m++) {
	  fdrag[d]+=(save_position[n][d*(mem_count+1)+tn1]-save_position[n][d*(mem_count+1)+tn])*mem_kernel[m];
	  tn1--;
	  tn--;
	  if (tn1 < 0) tn1 = mem_count;
	  if (tn < 0) tn = mem_count;
	}
	
	array[n][3+d]=fdrag[d];
	
	x[n][d] += gjffac*update->dt*v[n][d] 
	+ gjffac*sqrt(update->dt)*update->dt/2.0/mass[type[0]]*fran[d] 
	- gjffac*update->dt/2.0/mass[type[0]]*fdrag[d];
	
      }
      
    }
    
  }

  
  firstindex_r++;
  if (firstindex_r==2*mem_count-1) firstindex_r=0;
  lastindex_p++;
  if (lastindex_p==mem_count+1) lastindex_p=0;

}

/* ---------------------------------------------------------------------- */

void FixGLE::final_integrate()
{
  int n,d,m;
  double **v = atom->v;
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  double *mass = atom->mass;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  
  // update positions numbers
  for ( n=0; n<nlocal; n++ ) {
    for ( d=0; d<3; d++ ) {
      save_position[n][d*(mem_count+1)+lastindex_p] = x[n][d]; 
    }
  }

  for ( n = 0; n < nlocal; n++) {
    if (mask[n] & groupbit) {
      for (d = 0; d<3;d++) {
	int tn = lastindex_p-1;
	if (tn < 0) tn = mem_count;
	//array[n][3+d] += (save_position[n][d*(mem_count+1)+lastindex_p]-save_position[n][d*(mem_count+1)+tn])*mem_kernel[0];
	v[n][d] = 2.0/update->dt*(save_position[n][d*(mem_count+1)+lastindex_p]-save_position[n][d*(mem_count+1)+tn])-v[n][d];
	  
	if (force_flag) {
	  int tn1 = lastindex_p;
	  double vn1 = v[n][d];
	  double fdrag = vn1*mem_kernel[m]*update->dt;
	  double vn = 2.0/update->dt*(save_position[n][d*(mem_count+1)+tn1]-save_position[n][d*(mem_count+1)+tn])-vn1;
	  for (m = 0; m<mem_count;m++) {
	    fdrag -= vn1*mem_kernel[m]*update->dt;
	    vn1 = vn;
	    tn1 = tn;
	    tn--;
	    if (tn < 0) tn = mem_count;
	    vn = 2.0/update->dt*(save_position[n][d*(mem_count+1)+tn1]-save_position[n][d*(mem_count+1)+tn])-vn1;
	  }
	  //array[n][3+d] = fdrag;
	  array[n][3+d] /= - sqrt(update->dt);
	  //array[n][d] *= sqrt(update->dt);
	  f[n][d] += array[n][d] + array[n][d+3];
	}
      }
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
      gfactor2[i] = 1.0;
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
  
  memory->grow(save_position,nmax,3*(mem_count+1),"fix/gle:save_position");
  memory->grow(save_random,nmax,6*mem_count-3,"fix/gle:save_random");

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
  for ( d=0; d<3; d++ ) { 
    for ( m=0; m<mem_count+1; m++ ) {
      buf[offset++] = save_position[i][d*(mem_count+1)+m];
    }
  }
  
  // pack random number
  for ( d=0; d<3; d++ ) { 
    for ( m=0; m<2*mem_count-1; m++ ) {
      buf[offset++] = save_random[i][d*(2*mem_count-1)+m];
    }
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
  for ( d=0; d<3; d++ ) { 
    for ( m=0; m<mem_count; m++ ) {
      save_position[nlocal][d*(mem_count+1)+m] = buf[offset++];
    }
  }
  
  // pack normal random number
  for ( d=0; d<3; d++ ) { 
    for ( m=0; m<2*mem_count-1; m++ ) {
      save_random[nlocal][d*(2*mem_count-1)+m] = buf[offset++];
    }
  }
  
  return offset;
}
