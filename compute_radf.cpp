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
   Contributing authors: Paul Crozier (SNL), Jeff Greathouse (SNL)
------------------------------------------------------------------------- */
#include "string.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include "compute_radf.h"
#include "atom.h"
#include "update.h"
#include "force.h"
#include "pair.h"
#include "domain.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "group.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

ComputeRADF::ComputeRADF(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg < 6 || (narg-6) % 2) error->all(FLERR,"Illegal compute radf command");

  array_flag = 1;
  extarray = 0;
  
  // gather radial information
  nbin_r = force->inumeric(FLERR,arg[3]);
  if (nbin_r < 1) error->all(FLERR,"Illegal compute radf command");
  
  if (strcmp(arg[4],"cutoff") == 0) {
    if (force->pair) delr = force->pair->cutforce / nbin_r;
    else error->all(FLERR,"Compute radf requires a pair style be defined");
  } else {
    delr = force->numeric(FLERR,arg[4]) / nbin_r;
    if (force->pair && delr > force->pair->cutforce / nbin_r) {
      error->warning(FLERR,"RADF maximum radius > maximum cutoff. Large skin required!");
    } 
    if ( !force->pair ) {
      error->all(FLERR,"Compute radf requires a pair style be defined");
    }
  }
  delrinv = 1.0/delr;
  
  // gather angular information
  nbin_a = force->inumeric(FLERR,arg[5]);
  dela = MY_PI / nbin_a;
  delainv = 1.0/dela;

  if (narg == 6) npairs = 1;
  else npairs = (narg-6)/2;

  size_array_rows = nbin_r*nbin_a;
  size_array_cols = 2 + 2*npairs;

  int ntypes = atom->ntypes;
  memory->create(radfpair,npairs,ntypes+1,ntypes+1,"radf:radfpair");
  memory->create(nradfpair,ntypes+1,ntypes+1,"radf:nradfpair");
  ilo = new int[npairs];
  ihi = new int[npairs];
  jlo = new int[npairs];
  jhi = new int[npairs];

  if (narg == 6) {
    ilo[0] = 1; ihi[0] = ntypes;
    jlo[0] = 1; jhi[0] = ntypes;
    npairs = 1;

  } else {
    npairs = 0;
    int iarg = 6;
    while (iarg < narg) {
      force->bounds(arg[iarg],atom->ntypes,ilo[npairs],ihi[npairs]);
      force->bounds(arg[iarg+1],atom->ntypes,jlo[npairs],jhi[npairs]);
      if (ilo[npairs] > ihi[npairs] || jlo[npairs] > jhi[npairs])
        error->all(FLERR,"Illegal compute radf command");
      npairs++;
      iarg += 2;
    }
  }

  int i,j;
  for (i = 1; i <= ntypes; i++)
    for (j = 1; j <= ntypes; j++)
      nradfpair[i][j] = 0;

  for (int m = 0; m < npairs; m++)
    for (i = ilo[m]; i <= ihi[m]; i++)
      for (j = jlo[m]; j <= jhi[m]; j++)
        radfpair[nradfpair[i][j]++][i][j] = m;

  memory->create(hist,npairs,nbin_r*nbin_a,"radf:hist");
  memory->create(histall,npairs,nbin_r*nbin_a,"radf:histall");
  memory->create(array,nbin_r*nbin_a,2+2*npairs,"radf:array");
  typecount = new int[ntypes+1];
  icount = new int[npairs];
  jcount = new int[npairs];
  duplicates = new int[npairs];
}

/* ---------------------------------------------------------------------- */

ComputeRADF::~ComputeRADF()
{
  memory->destroy(radfpair);
  memory->destroy(nradfpair);
  delete [] ilo;
  delete [] ihi;
  delete [] jlo;
  delete [] jhi;
  memory->destroy(hist);
  memory->destroy(histall);
  memory->destroy(array);
  delete [] typecount;
  delete [] icount;
  delete [] jcount;
  delete [] duplicates;
}

/* ---------------------------------------------------------------------- */

void ComputeRADF::init()
{
  int i,j,m;
  
  // set 1st column of output array to bin coords
  for (int i = 0; i < nbin_a; i++)
    for (int j = 0; j < nbin_r; j++) {
      array[i*nbin_r+j][0] = (i+0.5) * dela;
      array[i*nbin_r+j][1] = (j+0.5) * delr;
    }

  // count atoms of each type that are also in group

  int *mask = atom->mask;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int ntypes = atom->ntypes;

  for (i = 1; i <= ntypes; i++) typecount[i] = 0;
  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) typecount[type[i]]++;

  // icount = # of I atoms participating in I,J pairs for each histogram
  // jcount = # of J atoms participating in I,J pairs for each histogram
  // duplicates = # of atoms in both groups I and J for each histogram

  for (m = 0; m < npairs; m++) {
    icount[m] = 0;
    for (i = ilo[m]; i <= ihi[m]; i++) icount[m] += typecount[i];
    jcount[m] = 0;
    for (i = jlo[m]; i <= jhi[m]; i++) jcount[m] += typecount[i];
    duplicates[m] = 0;
    for (i = ilo[m]; i <= ihi[m]; i++)
      for (j = jlo[m]; j <= jhi[m]; j++)
        if (i == j) duplicates[m] += typecount[i];
  }

  int *scratch = new int[npairs];
  MPI_Allreduce(icount,scratch,npairs,MPI_INT,MPI_SUM,world);
  for (i = 0; i < npairs; i++) icount[i] = scratch[i];
  MPI_Allreduce(jcount,scratch,npairs,MPI_INT,MPI_SUM,world);
  for (i = 0; i < npairs; i++) jcount[i] = scratch[i];
  MPI_Allreduce(duplicates,scratch,npairs,MPI_INT,MPI_SUM,world);
  for (i = 0; i < npairs; i++) duplicates[i] = scratch[i];
  delete [] scratch;
  
  // create lists for angle detection
  max_icount = 0;
  for (i = 0; i < npairs; i++) if (icount[i] > max_icount) max_icount = icount[i]; 
  memory->create(ilist_x,npairs,max_icount*4,"radf:ilist_x");

  // need an occasional half neighbor list

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->occasional = 1;
  
}

/* ---------------------------------------------------------------------- */

void ComputeRADF::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeRADF::compute_array()
{
  int i,j,m,ii,jj,inum,jnum,itype,jtype,ipair,jpair,ibin_r,ibin_a,ihisto;
  double xtmp,ytmp,ztmp,delx,dely,delz,rx,ry,rz,r,rsq;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double factor_lj,factor_coul;

  invoked_array = update->ntimestep;

  // invoke half neighbor list (will copy or build if necessary)

  neighbor->build_one(list);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // zero the histogram counts

  for (i = 0; i < npairs; i++)
    for (j = 0; j < nbin_r*nbin_a; j++)
      hist[i][j] = 0;

  // tally the RADF
  // both atom i and j must be in fix group
  // itype,jtype must have been specified by user
  // consider I,J as one interaction even if neighbor pair is stored on 2 procs
  // tally I,J pair each time I is central atom, and each time J is central

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int ntypes = atom->ntypes;
  
    
  // find closest pairs of i atoms (to define the angle)
  int *icount_loc = new int[npairs];
  int *icount_scan = new int[npairs];
  for (m = 0; m < npairs; m++) {
    icount_loc[m] = icount_scan[m] = 0;
  }
  
  for (i = 1; i <= ntypes; i++) typecount[i] = 0;
  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) typecount[type[i]]++;

  for (m = 0; m < npairs; m++) {
    icount_loc[m] = 0;
    for (i = ilo[m]; i <= ihi[m]; i++) icount_loc[m] += typecount[i];
  }
  MPI_Exscan(icount_loc,icount_scan,npairs,MPI_INT, MPI_SUM, world);
  
  // update global atom data (position)
  for (m = 0; m < npairs; m++) {
    int counter = 0;
    for (i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit &&  ilo[m] <= type[i] && ihi[m] >= type[i]) {
	ilist_x[m][ 4*(counter+icount_scan[m]) + 0] = i;
	ilist_x[m][ 4*(counter+icount_scan[m]) + 1] = x[i][0];
	ilist_x[m][ 4*(counter+icount_scan[m]) + 2] = x[i][1];
	ilist_x[m][ 4*(counter+icount_scan[m]) + 3] = x[i][2];
	counter++;
      }
    } 
  }
  int **scratch;
  memory->create(scratch,npairs,4*max_icount,"radf:scratch");
  MPI_Allreduce(&ilist_x[0][0],&scratch[0][0],npairs*4*max_icount,MPI_INT,MPI_SUM,world);
  
  for (m = 0; m < npairs; m++) {
    for (i = 0; i < icount[m]; i++) {
      ilist_x[m][4*i + 0] = scratch[m][4*i + 0];
      double min_dist = 1000000.0;
      for (j = 0; j < icount[m]; j++) {
	rx = scratch[m][4*i + 1] - scratch[m][4*j + 1];
	ry = scratch[m][4*i + 2] - scratch[m][4*j + 2];
	rz = scratch[m][4*i + 3] - scratch[m][4*j + 3];
	domain->minimum_image(rx,ry,rz);
	rsq = rx*rx + ry*ry + rz*rz;
	if (rsq < min_dist) {
	  ilist_x[m][4*i + 0] = i;
	  ilist_x[m][4*i + 1] = rx;
	  ilist_x[m][4*i + 2] = ry;
	  ilist_x[m][4*i + 3] = rz;
	  min_dist = rsq;
	}
      }
    }
  }
  memory->destroy(scratch);


  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  for (m = 0; m < npairs; m++) {
    for (ii = 0; ii < icount_loc[m]; ii++) {
      i = ilist_x[m][4*(ii + icount_scan[m]) + 0];
      rx = ilist_x[m][4*(ii + icount_scan[m]) + 1];
      ry = ilist_x[m][4*(ii + icount_scan[m]) + 2];
      rz = ilist_x[m][4*(ii + icount_scan[m]) + 3];
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      itype = type[i];
    
      jlist = firstneigh[i];
      jnum = numneigh[i];
      
      for (jj = 0; jj < jnum; jj++) {
	j = jlist[jj];
	factor_lj = special_lj[sbmask(j)];
	factor_coul = special_coul[sbmask(j)];
	j &= NEIGHMASK;

	// if both weighting factors are 0, skip this pair
	// could be 0 and still be in neigh list for long-range Coulombics
	// want consistency with non-charged pairs which wouldn't be in list

	if (factor_lj == 0.0 && factor_coul == 0.0) continue;

	if (!(mask[j] & groupbit)) continue;
	jtype = type[j];
	ipair = nradfpair[itype][jtype];
	jpair = nradfpair[jtype][itype];
	if (!ipair && !jpair) continue;

	delx = xtmp - x[j][0];
	dely = ytmp - x[j][1];
	delz = ztmp - x[j][2];
	r = sqrt(delx*delx + dely*dely + delz*delz);
	ibin_r = static_cast<int> (r*delrinv);
	if (ibin_r >= nbin_r) continue;
      
 
      
	ibin_a = 0;

	if (ipair)
	  for (ihisto = 0; ihisto < ipair; ihisto++)
	    hist[radfpair[ihisto][itype][jtype]][ibin_r+ibin_a*nbin_r] += 1.0;
	if (newton_pair || j < nlocal) {
	  if (jpair)
	    for (ihisto = 0; ihisto < jpair; ihisto++)
	      hist[radfpair[ihisto][jtype][itype]][ibin_r+ibin_a*nbin_r] += 1.0;
	}
      }
    }
  }

  // sum histograms across procs

  MPI_Allreduce(hist[0],histall[0],npairs*nbin_r*nbin_a,MPI_DOUBLE,MPI_SUM,world);

  // convert counts to g(r) and coord(r) and copy into output array
  // vfrac = fraction of volume in shell m
  // npairs = number of pairs, corrected for duplicates
  // duplicates = pairs in which both atoms are the same

  double constant,vfrac,gr,ncoord,rlower,rupper,normfac;

  if (domain->dimension == 3) {
    constant = 4.0*MY_PI / (3.0*domain->xprd*domain->yprd*domain->zprd);

    for (m = 0; m < npairs; m++) {
      normfac = (icount[m] > 0) ? static_cast<double>(jcount[m])
                - static_cast<double>(duplicates[m])/icount[m] : 0.0;
      ncoord = 0.0;
      for (ibin_a = 0; ibin_a < nbin_a; ibin_a++) {
	for (ibin_r = 0; ibin_r < nbin_r; ibin_r++) {
	  rlower = ibin_r*delr;
	  rupper = (ibin_r+1)*delr;
	  vfrac = constant * (rupper*rupper*rupper - rlower*rlower*rlower);
	  if (vfrac * normfac != 0.0)
	    gr = histall[m][ ibin_r + nbin_r*ibin_a ] / (vfrac * normfac * icount[m]);
	  else gr = 0.0;
	  if (icount[m] != 0)
	    ncoord += gr * vfrac * normfac;
	  array[ibin_r + nbin_r*ibin_a][2+2*m] = gr;
	  array[ibin_r + nbin_r*ibin_a][3+2*m] = ncoord;
	}
      }
    }
  } 
}
