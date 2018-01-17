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

#include "compute_com_local.h"
#include "string.h"
#include "update.h"
#include "group.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "atom.h"
#include "domain.h"

using namespace LAMMPS_NS;

enum{SINGLE,GROUP,GROUPSHAPE};

/* ---------------------------------------------------------------------- */

ComputeCOMLocal::ComputeCOMLocal(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg < 6) error->all(FLERR,"Illegal compute com_local command");

  vector_flag = 1;
  extvector = 0;
  
  
  if (strcmp(arg[3],"single") == 0) {
    mode = SINGLE;
  } else if (strcmp(arg[3],"group") == 0) {
    mode = GROUP;
  } else if (strcmp(arg[3],"group/shape") == 0) {
    mode = GROUPSHAPE;
  } else error->all(FLERR,"Illegal compute com_local command");
  radius = force->numeric(FLERR,arg[4]);
  r2 = radius*radius;
  jgroup = group->find(arg[5]);
  jgroupbit = group->bitmask[jgroup];
  
  if (mode == GROUPSHAPE && radius > neighbor->cutneighmax) {
    error->all(FLERR,"Radius larger then maximum neighbor cutoff!\n");
  }
  
  // determine group members
  if (mode == SINGLE) {
    int i;
    int nlocal = atom->nlocal;
    int *mask = atom->mask;
    ngroup_loc = ngroup_glo = ngroup_scan = 0;
    for (i=0; i<nlocal; i++) {
      if(mask[i] & groupbit) {
	ngroup_loc++;
      }
    }
    MPI_Allreduce(&ngroup_loc, &ngroup_glo, 1, MPI_INT, MPI_SUM, world);
    if ( ngroup_glo == 0 ) error->all(FLERR,"Illegal compute com_local command: No group members");
  } else {
    ngroup_glo = 1;
  }
  
  // allocate memory
  size_vector = 4*ngroup_glo;
  memory->create(pos_group_loc,ngroup_glo*3,"com_local/atom:x_group_loc");
  memory->create(pos_group_glo,ngroup_glo*3,"com_local/atom:x_group_glo");
  memory->create(com_loc,ngroup_glo*4,"com_local/atom:com_loc");
  memory->create(com_glo,ngroup_glo*4,"com_local/atom:com_glo");
  memory->create(count_loc,ngroup_glo,"com_local/atom:count_loc");
  memory->create(count_glo,ngroup_glo,"com_local/atom:count_glo");
  vector = com_glo;
 
}

/* ---------------------------------------------------------------------- */

ComputeCOMLocal::~ComputeCOMLocal()
{
  memory->destroy(pos_group_loc);
  memory->destroy(pos_group_glo);
  memory->destroy(com_loc);
  memory->destroy(com_glo);
  memory->destroy(count_loc);
  memory->destroy(count_glo);
}

/* ---------------------------------------------------------------------- */

void ComputeCOMLocal::init()
{
  if (mode == GROUPSHAPE) {
    // need a full neighbor list, built whenever re-neighboring occurs
    irequest = neighbor->request(this);
    neighbor->requests[irequest]->pair = 0;
    neighbor->requests[irequest]->fix = 1;
    neighbor->requests[irequest]->half = 0;
    neighbor->requests[irequest]->full = 1;
  }
}

/* ---------------------------------------------------------------------- */

void ComputeCOMLocal::compute_vector() 
{

  int i,j;
  double massone;
  double xtmp, ytmp, ztmp, delx, dely, delz, rsq;
  int *indices_group;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  double **x = atom->x;
  int *type = atom->type;
  imageint *image = atom->image;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  
  // determine group members (dynamically)
  if (mode == SINGLE) {
    ngroup_loc = ngroup_scan = 0;
    memory->grow(indices_group,ngroup_loc,"com_local/atom:indices_group");
    for (i=0; i<nlocal; i++) {
      if(mask[i] & groupbit) {
	ngroup_loc++;
	memory->grow(indices_group,ngroup_loc,"com_local/atom:indices_group");
	indices_group[ngroup_loc-1]=i;
      }
    }
    MPI_Exscan(&ngroup_loc,&ngroup_scan,1,MPI_INT, MPI_SUM, world);
  } else {
    ngroup_glo = 1;
  }
  
  // scatter particle information to all processors
  for (i=0; i< ngroup_glo; i++) {
    pos_group_loc[3*i] = pos_group_glo[3*i] = com_loc[4*i] =  com_glo[4*i] = 0.0;
    pos_group_loc[3*i+1] = pos_group_glo[3*i+1] = com_loc[4*i+1] = com_glo[4*i+1] = 0.0;
    pos_group_loc[3*i+2] = pos_group_glo[3*i+2] = com_loc[4*i+2]= com_glo[4*i+2] = 0.0;
    com_loc[4*i+3]= com_glo[4*i+3] = 0.0;
    count_loc[i] = 0.0;
    count_glo[i] = 0.0;
  }
  // determine center of mass of the particles
  if (mode == SINGLE) {
    for (i=0; i< ngroup_loc; i++) {
      pos_group_loc[3*(i+ngroup_scan)]=x[indices_group[i]][0];
      pos_group_loc[3*(i+ngroup_scan)+1]=x[indices_group[i]][1];
      pos_group_loc[3*(i+ngroup_scan)+2]=x[indices_group[i]][2];
    }
    MPI_Allreduce(pos_group_loc, pos_group_glo, 3*ngroup_glo, MPI_DOUBLE, MPI_SUM, world);
  } else {
    double xcm[3];
    double masstotal = group->mass(igroup);
    group->xcm(igroup,masstotal,xcm);
    pos_group_glo[0] = xcm[0];
    pos_group_glo[1] = xcm[1];
    pos_group_glo[2] = xcm[2];
  }
  
  // calculate local density
  if (mode == GROUPSHAPE) {
    // iterate over neighbour list
    int ii,jj,inum,jnum,itype,jtype;
    double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
    double rsq,r,rinv,rinv_norm,r2inv,r6inv,forcelj,factor_lj;
    int *ilist,*jlist,*numneigh,**firstneigh;

    double **x = atom->x;
    double **f = atom->f;
    int *type = atom->type;
    int nlocal = atom->nlocal;
    double *special_lj = force->special_lj;
    int newton_pair = force->newton_pair;
    
    list = neighbor->lists[irequest];
    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;
    
    int iparticle;

    // loop over neighbors of my atoms

    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      // iterate over all particles in jgroup and check, whether they are close to an i particle
      //printf("mask(i)=%d\n",mask[i]);
      if (mask[i] & jgroupbit) {
	xtmp = x[i][0];
	ytmp = x[i][1];
	ztmp = x[i][2];
	itype = type[i];
	jlist = firstneigh[i];
	jnum = numneigh[i];
	massone = 0.0;
	if (rmass) massone = rmass[i];
	else massone = mass[itype];
	iparticle = 0;
	//printf("salt i %d\n",i);
	for (jj = 0; jj < jnum; jj++) {
	  j = jlist[jj];
	  factor_lj = special_lj[sbmask(j)];
	  j &= NEIGHMASK;
	  //printf("mask[j] %d\n",mask[j]);
	  if (mask[j] & groupbit) {
	    delx = xtmp - x[j][0];
	    dely = ytmp - x[j][1];
	    delz = ztmp - x[j][2];
	    rsq = delx*delx + dely*dely + delz*delz;
	    jtype = type[j];

	    //printf("rsq: %f\n",rsq);
	    
	    if (rsq < r2) {
	      iparticle = 1;
	      break;
	    }
	  }
	}
	
	// include particle if it is in the cone
	if (iparticle == 1) {
	  delx = xtmp - pos_group_glo[0];
	  dely = ytmp - pos_group_glo[1];
	  delz = ztmp - pos_group_glo[2];
	  domain->minimum_image(delx,dely,delz);
	  count_loc[0]+= massone;
	  //printf("%f\n",massone);
	  com_loc[0] += delx * massone;
	  com_loc[1] += dely * massone;
	  com_loc[2] += delz * massone;
	}
      }
    }
    
  } else {
    for (j = 0; j < nlocal; j++) {
      if (mask[j] & jgroupbit) {
	massone = 0.0;
	if (rmass) massone = rmass[j];
	else massone = mass[type[j]];
	xtmp = x[j][0];
	ytmp = x[j][1];
	ztmp = x[j][2];
	for (i = 0; i < ngroup_glo; i++) {
	  delx = xtmp - pos_group_glo[3*i];
	  dely = ytmp - pos_group_glo[3*i+1];
	  delz = ztmp - pos_group_glo[3*i+2];

	  domain->minimum_image(delx,dely,delz);
	  rsq = delx*delx + dely*dely + delz*delz;
	  if (rsq < r2) { // particle j is in the neighbourhood of a particle i -> contribution to local com
	    count_loc[i]+= massone;
	    //printf("%f\n",massone);
	    com_loc[4*i] += delx * massone;
	    com_loc[4*i+1] += dely * massone;
	    com_loc[4*i+2] += delz * massone;
	  }
	}
      }
    }
  }
  MPI_Allreduce(com_loc, com_glo, 4*ngroup_glo, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(count_loc, count_glo, ngroup_glo, MPI_DOUBLE, MPI_SUM, world);
  
  for (i=0; i< ngroup_glo; i++) {
    if (count_glo[i] != 0.0) {
      com_glo[4*i] /= count_glo[i];
      com_glo[4*i+1] /= count_glo[i];
      com_glo[4*i+2] /= count_glo[i];
      com_glo[4*i+3] = count_glo[i];
    }
  }
  
  if (mode == SINGLE) memory->destroy(indices_group);
  
  
}
