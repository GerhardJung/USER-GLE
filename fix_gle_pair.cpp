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
#include "fix_gle_pair.h"
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

/* ---------------------------------------------------------------------- */

FixGLEPair::FixGLEPair(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 8) error->all(FLERR,"Illegal fix langevin command");

  // set fix properties
  
  global_freq = 1;
  nevery = 1;
  peratom_freq = 1;
  peratom_flag = 1;
  size_peratom_cols = 3*atom->nlocal;
  
  // read input parameter
  t_target = force->numeric(FLERR,arg[3]);
  tsqrt = sqrt(t_target);
  
  printf("checkpoint01\n");
  
  pot_count = force->numeric(FLERR,arg[5]);
  pot_file = fopen(arg[4],"r");
  dist_tabulated = new double[pot_count];
  pot_tabulated = new double[pot_count];
  phi_tabulated = new double[pot_count];
  read_pot_file();
  rcut = force->numeric(FLERR,arg[6]);
  r2cut = rcut*rcut;
  
  printf("checkpoint02\n");
  
  mem_count = force->numeric(FLERR,arg[8]);
  mem_file = fopen(arg[7],"r");
  mem_kernel = new double[mem_count];
  printf("checkpoint021\n");
  read_mem_file();
  
  printf("checkpoint022\n");
  
  seed = force->inumeric(FLERR,arg[9]);
  
  // optional parameter
  int restart = 0;
  int iarg = 10;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"restart") == 0) {
      restart = 1;
      iarg += 1;
    }
  }
  
  printf("checkpoint03\n");
 
  if (seed <= 0) error->all(FLERR,"Illegal fix gle/pair command");
  
  // initialize correlated RNG with processor-unique seed
  random = new RanMars(lmp,seed + comm->me);
  precision = 0.000002;
  random_correlator = new RanCor(lmp,mem_count, mem_kernel, precision);
  
  // allocate and init per-atom arrays (velocity and normal random number)
  
  save_velocity = NULL;
  save_random = NULL;
  //comm->maxexchange_fix += 9*mem_count;
  printf("%d, %d\n",atom->nlocal,atom->nmax);
  grow_arrays(atom->nlocal);
  //atom->add_callback(0);
  
  printf("checkpoint1\n");
  
  lastindex_v = firstindex_r  = 0;
  int nlocal= atom->nlocal, i,j,d,t;
  for ( i=0; i<nlocal; i++ )
    for ( j=i+1; j<nlocal; j++ ) {
      for ( d=0; d<3; d++ ) {
	for ( t=0; t<mem_count; t++ ) {
	  save_velocity[i][j*3*mem_count+d*mem_count+t] = 0.0;
	}
      }
      for ( t=0; t<2*mem_count-1; t++ ) {
	save_random[i][j*(2*mem_count-1)+t] = random->gaussian();
      }
    }

  printf("checkpoint2\n");
  
  //memory for force save
  memory->create(array,atom->nlocal,3*atom->nlocal,"fix_gle_pair:array");
  array_atom = array;

}

/* ---------------------------------------------------------------------- */

FixGLEPair::~FixGLEPair()
{

  //atom->delete_callback(id,0);
  delete random;
  delete random_correlator;
  delete [] dist_tabulated;
  delete [] pot_tabulated;
  delete [] phi_tabulated;
  delete [] mem_kernel;
  memory->destroy(save_random);
  memory->destroy(save_velocity);
  memory->destroy(array);

}

/* ---------------------------------------------------------------------- */

int FixGLEPair::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= THERMO_ENERGY;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixGLEPair::read_pot_file()
{
  // skip first lines starting with #
  char buf[0x1000];
  long filepos;
  filepos = ftell(pot_file);
  while (fgets(buf, sizeof(buf), pot_file) != NULL) {
    if (buf[0] != '#') {
      fseek(pot_file,filepos,SEEK_SET);
      break;
    }
    filepos = ftell(pot_file);
  } 
  
  //read potentials
  int r;
  double dist,pot,phi;
  for(r=0; r<pot_count; r++){
    fscanf(pot_file,"%lf %lf %lf\n",&dist,&pot,&phi);
    dist_tabulated[r] = dist;
    pot_tabulated[r] = pot;
    phi_tabulated[r] = phi;
  }
}

/* ---------------------------------------------------------------------- */

void FixGLEPair::read_mem_file()
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
  int t;
  double dt,dt_old, mem;
  dt = dt_old = mem = 0.0;
  for(t=0; t<mem_count; t++){
    dt_old = dt;
    fscanf(mem_file,"%lf %lf\n",&dt,&mem);
    if (abs(dt - dt_old - update->dt) > 10E-10 && dt_old != 0.0) error->all(FLERR,"memory needs resolution similar to timestep");
    mem_kernel[t] = mem;
  }
}

/* ---------------------------------------------------------------------- */

void FixGLEPair::init()
{

}

/* ---------------------------------------------------------------------- */

void FixGLEPair::setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixGLEPair::post_force(int vflag)
{
  int i,j,d,t,tn;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double xtmp,ytmp,ztmp,delx,dely,delz;
  double vxtmp,vytmp,vztmp,delvx,delvy,delvz;
  double delvx_p, delvy_p, delvz_p;
  double rsq, dist, vrp;
  double fpair, phipair, rran;
  double fcon[3],fdis[3],fran[3];
  
  // calculate forces by iterating over every pair of particles
  for ( i=0; i<nlocal; i++ ) {
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    
    vxtmp = v[i][0];
    vytmp = v[i][1];
    vztmp = v[i][2];
    
    for ( j=i+1; j<nlocal; j++ ) {
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      domain->minimum_image(delx,dely,delz);
      rsq = delx*delx + dely*dely + delz*delz;
      dist = sqrt (rsq);
      
      delvx = vxtmp - v[j][0];
      delvy = vytmp - v[j][1];
      delvz = vztmp - v[j][2];
      
      // calculate parallel velocity component
      vrp = delvx*delx + delvy*dely +delvz*delz;
      delvx_p = vrp * delx / rsq;
      delvy_p = vrp * dely / rsq;
      delvz_p = vrp * delz / rsq;
      
      // update parallel velocity component
      save_velocity[i][j*3*mem_count+lastindex_v] = delvx_p;
      save_velocity[i][j*3*mem_count+mem_count+lastindex_v] = delvy_p;
      save_velocity[i][j*3*mem_count+2*mem_count+lastindex_v] = delvz_p;
      // update random number
      save_random[i][j*(2*mem_count-1)+firstindex_r] = random->gaussian();
      
      // reset array
      array[i][j] = 0;
      array[i][j+nlocal] = 0;
      array[i][j+2*nlocal] = 0;
      array[j][i] = 0;
      array[j][i+nlocal] = 0;
      array[j][i+2*nlocal] = 0;
      
      // calculate forces
      if (rsq < r2cut) {
	
	// conservative
	double tmp = (1-dist/rcut)*(1-dist/rcut);
	tmp = tmp*tmp;
	fpair = 795.69*(1+4*dist/rcut)*tmp/dist;
	fcon[0] = fpair * delx;
	fcon[1] = fpair * dely;
	fcon[2] = fpair * delz;
	
	// dissipative
	phipair = 36600*pow(1-dist/rcut,3.84);
	tn = lastindex_v;
	fdis[0]= 0.5*mem_kernel[0]*save_velocity[i][j*3*mem_count+tn]*update->dt;
	fdis[1]= 0.5*mem_kernel[0]*save_velocity[i][j*3*mem_count+mem_count+tn]*update->dt;
	fdis[2]= 0.5*mem_kernel[0]*save_velocity[i][j*3*mem_count+2*mem_count+tn]*update->dt;
	tn--;
	if (tn < 0) tn=mem_count-1;
	for (t=1; t<mem_count; t++) {
	  fdis[0] += mem_kernel[t]*save_velocity[i][j*3*mem_count+tn]*update->dt;
	  fdis[1] += mem_kernel[t]*save_velocity[i][j*3*mem_count+mem_count+tn]*update->dt;
	  fdis[2] += mem_kernel[t]*save_velocity[i][j*3*mem_count+2*mem_count+tn]*update->dt;
	  tn--;
	  if (tn < 0) tn=mem_count-1;
	}
	fdis[0] *= -phipair;
	fdis[1] *= -phipair;
	fdis[2] *= -phipair;
	
	// random
	rran = random_correlator->gaussian(&save_random[i][j*(2*mem_count-1)],firstindex_r)/dist;
	fran[0] = sqrt(phipair)*rran*delx;
	fran[1] = sqrt(phipair)*rran*dely;
	fran[2] = sqrt(phipair)*rran*delz;
	
	//printf("%d %d: dist: %f con: %f, diss: %f, ran: %f\n",i,j,dist,fcon[0],fdis[0],fran[0]); 
	
	f[i][0] += fcon[0] + fdis[0] + fran[0];
	f[i][1] += fcon[1] + fdis[1] + fran[1];
	f[i][2] += fcon[2] + fdis[2] + fran[2];
	
	f[j][0] -= fcon[0] + fdis[0] + fran[0];
	f[j][1] -= fcon[1] + fdis[1] + fran[1];
	f[j][2] -= fcon[2] + fdis[2] + fran[2];
	
	array[i][j] = fcon[0]+fdis[0] + fran[0];
	array[i][j+nlocal] = fcon[1]+ fdis[1] + fran[1];
	array[i][j+2*nlocal] = fcon[2]+ fdis[2] + fran[2];
	
	//if (i==0 && j==2) printf("pair output: %f %f %f %f\n",dist,array[i][j],array[i][j+nlocal],array[i][j+2*nlocal]);
	
	array[j][i] = -fcon[0]- fdis[0] - fran[0];
	array[j][i+nlocal] = -fcon[1]- fdis[1] - fran[1];
	array[j][i+2*nlocal] = -fcon[2]- fdis[2] - fran[2];
	
      }
    }
    
  }
  
  lastindex_v++;
  if (lastindex_v==mem_count) lastindex_v=0;
  firstindex_r++;
  if (firstindex_r==2*mem_count-1) firstindex_r=0;

}

/* ---------------------------------------------------------------------- */

void FixGLEPair::reset_dt()
{

}

/* ----------------------------------------------------------------------
   extract thermostat properties
------------------------------------------------------------------------- */

void *FixGLEPair::extract(const char *str, int &dim)
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

double FixGLEPair::memory_usage() {
  double bytes = atom->nmax * 9 * mem_count * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate atom-based array
------------------------------------------------------------------------- */

void FixGLEPair::grow_arrays(int nmax) {
  
  memory->grow(save_velocity,nmax,3*mem_count*(nmax-1),"fix/gle:save_velocity");
  memory->grow(save_random,nmax,(2*mem_count-1)*(nmax-1),"fix/gle:save_random");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based array
------------------------------------------------------------------------- */

void FixGLEPair::copy_arrays(int i, int j, int delflag)
{
  
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixGLEPair::pack_exchange(int i, double *buf)
{
  int offset = 0;
  int d,m;
  // pack velocity
  for ( d=0; d<3; d++ ) { 
    for ( m=0; m<mem_count; m++ ) {
      buf[offset++] = save_velocity[i][d*mem_count+m];
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

int FixGLEPair::unpack_exchange(int nlocal, double *buf)
{
  int offset = 0;
  int d,m;
  // pack velocity
  for ( d=0; d<3; d++ ) { 
    for ( m=0; m<mem_count; m++ ) {
      save_velocity[nlocal][d*mem_count+m] = buf[offset++];
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
