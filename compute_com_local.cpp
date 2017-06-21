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
#include "update.h"
#include "group.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "atom.h"
#include "domain.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeCOMLocal::ComputeCOMLocal(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg < 5) error->all(FLERR,"Illegal compute com_local command");

  vector_flag = 1;
  extvector = 0;
  
  radius = force->numeric(FLERR,arg[3]);
  r2 = radius*radius;
  jgroup = group->find(arg[4]);
  jgroupbit = group->bitmask[jgroup];
  
  // determine group members
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
  size_vector = 4*ngroup_glo;
  
  // allocate memory
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
  double *mass = atom->mass;
  
  // determine group members (dynamically)
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
  
  // scatter particle information to all processors
  for (i=0; i< ngroup_glo; i++) {
    pos_group_loc[3*i] = pos_group_glo[3*i] = com_loc[4*i] =  com_glo[4*i] = 0.0;
    pos_group_loc[3*i+1] = pos_group_glo[3*i+1] = com_loc[4*i+1] = com_glo[4*i+1] = 0.0;
    pos_group_loc[3*i+2] = pos_group_glo[3*i+2] = com_loc[4*i+2]= com_glo[4*i+2] = 0.0;
    com_loc[4*i+3]= com_glo[4*i+3] = 0.0;
    count_loc[i] = 0.0;
    count_glo[i] = 0.0;
  }
  for (i=0; i< ngroup_loc; i++) {
    pos_group_loc[3*(i+ngroup_scan)]=x[indices_group[i]][0];
    pos_group_loc[3*(i+ngroup_scan)+1]=x[indices_group[i]][1];
    pos_group_loc[3*(i+ngroup_scan)+2]=x[indices_group[i]][2];
  }
  MPI_Allreduce(pos_group_loc, pos_group_glo, 3*ngroup_glo, MPI_DOUBLE, MPI_SUM, world);
  
  // calculate local density
  for (j = 0; j < nlocal; j++) {
    if (mask[j] & jgroupbit) {
      massone = mass[type[j]];
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
	  com_loc[4*i] += delx * massone;
	  com_loc[4*i+1] += dely * massone;
	  com_loc[4*i+2] += delz * massone;
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
  
  memory->destroy(indices_group);
  
  
}
