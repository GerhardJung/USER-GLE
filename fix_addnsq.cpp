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

/* missing features:

- parallelizing the main nsq for-loop (not that important because this fix is meant for very small groups) 

*/

#include <string.h>
#include <stdlib.h>
#include "fix_addnsq.h"
#include "atom.h"
#include "atom_masks.h"
#include "update.h"
#include "modify.h"
#include "input.h"
#include "domain.h"
#include "pair.h"
#include "bond.h"
#include "variable.h"
#include "memory.h"
#include <algorithm>    // std::find
#include "error.h"
#include "force.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{LJOFF,HARMONIC};

/* ---------------------------------------------------------------------- */

FixAddNsq::FixAddNsq(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  
  MPI_Comm_rank(world,&me);
    
  if (narg < 4) error->all(FLERR,"Illegal fix addforce command");

  global_freq = 1;
  peratom_freq = 1;

  // args

  // optional args
  
  n_pot = 0;
  checkpoint = -1;
  memory->create(type_pot,narg,"addnsq:type_pot");
  memory->create(coeff_pot,narg,6,"addnsq:coeff_pot");

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"lj/off") == 0) {
      if (iarg+5 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      n_pot++;
      type_pot[n_pot-1]=LJOFF;
      double epsilon = force->numeric(FLERR,arg[iarg+1]);
      double sigma = force->numeric(FLERR,arg[iarg+2]);
      double offset = force->numeric(FLERR,arg[iarg+3]);
      double rcut = force->numeric(FLERR,arg[iarg+4]);
      // calculate relevant 
      coeff_pot[n_pot-1][0] = rcut*rcut;
      coeff_pot[n_pot-1][1] = offset;
      coeff_pot[n_pot-1][2] = 48.0 * epsilon * pow(sigma,12.0);
      coeff_pot[n_pot-1][3] = 24.0 * epsilon * pow(sigma,6.0);
      coeff_pot[n_pot-1][4] = 4.0 * epsilon * pow(sigma,12.0);
      coeff_pot[n_pot-1][5] = 4.0 * epsilon * pow(sigma,6.0);
      iarg += 5;
      if ( iarg < narg && strcmp(arg[iarg],"checkpoint") == 0) { 
	checkpoint = n_pot-1;
	peratom_flag = 1;
	size_peratom_cols = 3;
	iarg ++;
      }
    } else if (strcmp(arg[iarg],"harmonic") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      n_pot++;
      type_pot[n_pot-1]=HARMONIC;
      coeff_pot[n_pot-1][0]=force->numeric(FLERR,arg[iarg+1]);
      coeff_pot[n_pot-1][1]=force->numeric(FLERR,arg[iarg+2]);
      coeff_pot[n_pot-1][2]=0.0;
      coeff_pot[n_pot-1][3]=0.0;
      coeff_pot[n_pot-1][4]=0.0;
      iarg += 3;
      if ( iarg < narg && strcmp(arg[iarg],"checkpoint") == 0) { 
	checkpoint = n_pot-1;
	peratom_flag = 1;
	size_peratom_cols = 3;
	iarg ++;
      }
    } else error->all(FLERR,"Illegal fix addnsq command");
  }

  
  // find the number of atoms in the group
  int a,d;
  int nlocal= atom->nlocal;
  int *mask= atom->mask;
  int ngroup_loc=0;
  
  // calculate the number of local/global atoms belonging to the group
  for (a= 0; a < nlocal; a++) {
    if(mask[a] & groupbit) {
      ngroup_loc++;
    }
  }
  MPI_Allreduce(&ngroup_loc, &ngroup_glo, 1, MPI_INT, MPI_SUM, world);
  
  memory->create(indices_group,ngroup_glo,"addnsq:indices_group");

  //create storage for global atom data (position, force)
  memory->create(group_x,ngroup_glo,3,"addnsq:group_x");
  memory->create(group_x_loc,ngroup_glo,3,"addnsq:group_x_loc");
  memory->create(group_f,ngroup_glo,3*n_pot,"addnsq:group_f");
  memory->create(group_e,ngroup_glo,n_pot,"addnsq:group_e");
  memory->create(group_v,ngroup_glo,6*n_pot,"addnsq:group_v");
  
  //create array if checkpoint is necessary
  if ( peratom_flag  ) {
    nmax = atom->nmax;
    memory->create(array,nmax,3,"addnsq:array");
    array_atom = array;
  }
 
}

/* ---------------------------------------------------------------------- */

FixAddNsq::~FixAddNsq()
{
  
  memory->destroy(group_x);
  memory->destroy(group_x_loc);
  
  memory->destroy(indices_group);
  
  memory->destroy(group_f);
  memory->destroy(group_e);
  memory->destroy(group_v);
  
  if (peratom_flag) memory->destroy(array);
  

}

/* ---------------------------------------------------------------------- */

int FixAddNsq::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= THERMO_ENERGY;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAddNsq::init()
{
  // check variables

 

  // set index and check validity of region

}

/* ---------------------------------------------------------------------- */

void FixAddNsq::setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAddNsq::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAddNsq::post_force(int vflag)
{
  // find relevant particles // find group-member on each processor
  int a,d;
  int nlocal= atom->nlocal;
  int *mask= atom->mask;
  int ngroup_loc=0,ngroup_scan=0;
  for (a= 0; a < nlocal; a++) {
    if(mask[a] & groupbit) {
      indices_group[ngroup_loc]=a;
      ngroup_loc++;
    }
  }
  MPI_Exscan(&ngroup_loc,&ngroup_scan,1,MPI_INT, MPI_SUM, world);
  
  // update global atom data (position)
  double **x = atom->x;
  for (a=0; a<ngroup_glo; a++)
    for (d=0; d<3; d++)
      group_x[a][d]=group_x_loc[a][d]=0.0;
  for (a= 0; a < ngroup_loc; a++) {
    group_x_loc[a+ngroup_scan][0] = x[indices_group[a]][0];
    group_x_loc[a+ngroup_scan][1] = x[indices_group[a]][1];
    group_x_loc[a+ngroup_scan][2] = x[indices_group[a]][2];
  }
  MPI_Allreduce(&group_x_loc[0][0], &group_x[0][0], ngroup_glo*3, MPI_DOUBLE, MPI_SUM, world);
  
  //reset arrays from last calculation
  for (a= 0; a < ngroup_glo; a++)
    for (d=0; d<3*n_pot; d++)
      group_f[a][d] = 0.0;
  for (a=0; a<ngroup_glo; a++)
    for (d=0; d<n_pot; d++)
      group_e[a][d]=0.0;
  for (a=0; a<ngroup_glo; a++)
    for (d=0; d<6*n_pot; d++)
      group_v[a][d]=0.0;
  
  // calculate potentials (with nsq)
  int i,j,p;
  for (i=0; i<ngroup_glo; i++)
    for (j=i+1; j<ngroup_glo; j++){
      //if (me==0) printf("%d: x=%f y=%f z=%f\n",i,group_x[i][0],group_x[i][1],group_x[i][2]);
      //if (me==0) printf("%d: x=%f y=%f z=%f\n",j,group_x[j][0],group_x[j][1],group_x[j][2]);
      double dx = group_x[i][0] - group_x[j][0];
      double dy = group_x[i][1] - group_x[j][1];
      double dz = group_x[i][2] - group_x[j][2];
      //if (me==0) printf("i:%d j:%d x=%f y=%f z=%f\n",i,j,dx,dy,dz);
      domain->minimum_image(dx,dy,dz);
      double rsq = dx*dx + dy*dy + dz*dz;
      //if (me==0) printf("i:%d j:%d x=%f y=%f z=%f -> rsq=%f\n",i,j,dx,dy,dz,rsq);
      for (p=0; p<n_pot; p++){
	double pair_force=0.0;
	double energy=0.0;
	if (type_pot[p]==LJOFF) {
	  calc_ljoff(i,j, p, rsq, pair_force, energy);
	} else if (type_pot[p]==HARMONIC) {
	  calc_harmonic(i,j, p ,rsq, pair_force, energy);
	}

	group_f[i][3*p] += dx*pair_force;
	group_f[i][1+3*p] += dy*pair_force;
	group_f[i][2+3*p] += dz*pair_force;
	
	
        group_f[j][3*p] -= dx*pair_force;
        group_f[j][1+3*p] -= dy*pair_force;
        group_f[j][2+3*p] -= dz*pair_force;
	
	group_e[i][p] += energy/2.0;
	group_e[j][p] += energy/2.0;
	
	group_v[i][6*p] += dx*dx*pair_force/2.0;
	group_v[i][6*p+1] += dy*dy*pair_force/2.0;
	group_v[i][6*p+2] += dz*dz*pair_force/2.0;
	group_v[i][6*p+3] += dx*dy*pair_force/2.0;
	group_v[i][6*p+4] += dx*dz*pair_force/2.0;
	group_v[i][6*p+5] += dy*dz*pair_force/2.0;
	
	group_v[j][6*p] += dx*dx*pair_force/2.0;
	group_v[j][6*p+1] += dy*dy*pair_force/2.0;
	group_v[j][6*p+2] += dz*dz*pair_force/2.0;
	group_v[j][6*p+3] += dx*dy*pair_force/2.0;
	group_v[j][6*p+4] += dx*dz*pair_force/2.0;
	group_v[j][6*p+5] += dy*dz*pair_force/2.0;

      }
    }
    
  // include calculated force/energy to global arrays
  double **f = atom->f;
  for ( a=0; a<ngroup_loc; a++ ) {
    for ( p=0; p<n_pot; p++ ) {
      if ( p == checkpoint && a==0 ) {
	//create new array if it is too small
	if (atom->nlocal > nmax) {
	  nmax = atom->nmax;
	  memory->destroy(array);
	  memory->create(array,nmax,3,"addnsq:array");
	  array_atom = array;
	}
	int i;
	for ( i=0; i<nlocal; i++ ) {
	  array[i][0] = f[i][0];
	  array[i][1] = f[i][1];
	  array[i][2] = f[i][2];
	}
      }
      f[indices_group[a]][0] += group_f[a+ngroup_scan][3*p];
      f[indices_group[a]][1] += group_f[a+ngroup_scan][1+3*p];
      f[indices_group[a]][2] += group_f[a+ngroup_scan][2+3*p];
      
      if (type_pot[p]==LJOFF) {
	if (force->pair->eflag_either) {
	  if (force->pair->eflag_global) {
	    force->pair->eng_vdwl += group_e[a+ngroup_scan][p];
	  }
	  if (force->pair->eflag_atom) {
	    force->pair->eatom[indices_group[a]] += group_e[a+ngroup_scan][p];
	  }
	} 
	
	if (force->pair->vflag_global) {
	  force->pair->virial[0] += group_v[a+ngroup_scan][6*p];
	  force->pair->virial[1] += group_v[a+ngroup_scan][6*p+1];
          force->pair->virial[2] += group_v[a+ngroup_scan][6*p+2];
          force->pair->virial[3] += group_v[a+ngroup_scan][6*p+3];
          force->pair->virial[4] += group_v[a+ngroup_scan][6*p+4];
          force->pair->virial[5] += group_v[a+ngroup_scan][6*p+5];
	  
	}
	
	if (force->pair->vflag_atom) {
	  force->pair->vatom[indices_group[a]][0] += group_v[a+ngroup_scan][6*p];
	  force->pair->vatom[indices_group[a]][1] += group_v[a+ngroup_scan][6*p];
	  force->pair->vatom[indices_group[a]][2] += group_v[a+ngroup_scan][6*p];
	  force->pair->vatom[indices_group[a]][3] += group_v[a+ngroup_scan][6*p];
	  force->pair->vatom[indices_group[a]][4] += group_v[a+ngroup_scan][6*p];
	  force->pair->vatom[indices_group[a]][5] += group_v[a+ngroup_scan][6*p];
	}
	
      }
      if (type_pot[p]==HARMONIC) { //TODO set flags (please not manually...)
	if (force->bond->eflag_either) {
	  if (force->bond->eflag_global) {
	    force->bond->energy += group_e[a+ngroup_scan][p];
	  }
	  if (force->bond->eflag_atom) {
	    force->bond->eatom[indices_group[a]] += group_e[a+ngroup_scan][p];
	  }
	} 
	
	if (force->bond->vflag_global) {
	  force->bond->virial[0] += group_v[a+ngroup_scan][6*p];
	  force->bond->virial[1] += group_v[a+ngroup_scan][6*p+1];
          force->bond->virial[2] += group_v[a+ngroup_scan][6*p+2];
          force->bond->virial[3] += group_v[a+ngroup_scan][6*p+3];
          force->bond->virial[4] += group_v[a+ngroup_scan][6*p+4];
          force->bond->virial[5] += group_v[a+ngroup_scan][6*p+5];
	  
	}
	
	if (force->bond->vflag_atom) {
	  force->bond->vatom[indices_group[a]][0] += group_v[a+ngroup_scan][6*p];
	  force->bond->vatom[indices_group[a]][1] += group_v[a+ngroup_scan][6*p];
	  force->bond->vatom[indices_group[a]][2] += group_v[a+ngroup_scan][6*p];
	  force->bond->vatom[indices_group[a]][3] += group_v[a+ngroup_scan][6*p];
	  force->bond->vatom[indices_group[a]][4] += group_v[a+ngroup_scan][6*p];
	  force->bond->vatom[indices_group[a]][5] += group_v[a+ngroup_scan][6*p];
	}
	
      }
      
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixAddNsq::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAddNsq::calc_ljoff(int i, int j, int p, double rsq, double &pair_force, double &energy)
{
  
  if (rsq < coeff_pot[p][0]) {
    
    double r = sqrt(rsq);
    
    if(!force->pair->eflag_either && r<coeff_pot[p][1]){
      printf("distance = %f < offset = %f !\n",r,coeff_pot[p][1]);
      error->all(FLERR,"Distance between particles too small");
    }
    
    double rinv_norm = 1.0/r;
    double rinv = 1.0/(r-coeff_pot[p][1]);
    double r2inv = rinv*rinv;
    r2inv = r2inv*r2inv*r2inv;
    
    pair_force = r2inv * (coeff_pot[p][2]*r2inv - coeff_pot[p][3]);
    pair_force *= rinv*rinv_norm;
    
    energy = r2inv * ( coeff_pot[p][4]*r2inv - coeff_pot[p][5] ) - coeff_pot[p][1];
    
    //printf("nsq: i:%d j:%d %f %f %f\n",indices_group[i],indices_group[j],r,pair_force,energy);
  
  }
}

/* ---------------------------------------------------------------------- */

void FixAddNsq::calc_harmonic(int i, int j, int p, double rsq, double &pair_force, double &energy)
{
  
  double r = sqrt(rsq);
  double dr = r - coeff_pot[p][1];
  double rk = coeff_pot[p][0] * dr;

  // force & energy

  if (r > 0.0) pair_force = -2.0*rk/r;
  else pair_force = 0.0;

  if (force->pair->eflag_either) energy = rk*dr;
    
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixAddNsq::memory_usage()
{
  double bytes = 0.0;
  return bytes;
}
