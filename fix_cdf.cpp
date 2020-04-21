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

#include "fix_cdf.h"
#include "string.h"
#include "update.h"
#include "group.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "atom.h"
#include "domain.h"
#include "input.h"
#include "comm.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixCDF::FixCDF(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  
  if (narg < 9) error->all(FLERR,"Illegal compute cdf command");
  
  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);
  
  nevery = force->inumeric(FLERR,arg[3]);
  
  nbin_x = force->inumeric(FLERR,arg[4]);
  range_x = force->numeric(FLERR,arg[5]);
  nbin_r = force->inumeric(FLERR,arg[6]);
  range_r = force->numeric(FLERR,arg[7]);
  
  jgroup = group->find(arg[8]);
  jgroupbit = group->bitmask[jgroup];
  
  print = 0;
  
  //printf("out: %s\n",arg[9]);
  
  if (strcmp(arg[9],"file") == 0) {
    if (11 > narg) error->all(FLERR,"Illegal compute cdf/file command");
    print = 1;
    //printf("me: %d\n",me);
    if (me == 0) {
      //printf("Fix cdf, option file\n");
      out = fopen(arg[10],"w");
      if (out == NULL) {
        char str[128];
        sprintf(str,"Cannot open compute cdf/file file %s",arg[10]);
        error->one(FLERR,str);
      }
    } 
  }
  
  // allocate memory
  size_vector = nbin_x*nbin_r;
  memory->create(data_histo,size_vector,"cdf/data_histo");
  memory->create(data_histo_loc,size_vector,"cdf/data_histo_loc");
  memory->create(data_velx,size_vector,"cdf/data_velx");
  memory->create(data_vely,size_vector,"cdf/data_vely");
  memory->create(data_velz,size_vector,"cdf/data_velz");
  memory->create(data_vel_locx,size_vector,"cdf/data_vel_locx");
  memory->create(data_vel_locy,size_vector,"cdf/data_vel_locy");
  memory->create(data_vel_locz,size_vector,"cdf/data_vel_locz");
  
    memory->create(data_pressx,size_vector,"cdf/data_pressx");
    memory->create(data_pressyz,size_vector,"cdf/data_pressyz");
    
    memory->create(data_press_locx,size_vector,"cdf/data_press_locx");
    memory->create(data_press_locyz,size_vector,"cdf/data_press_locyz");
  
  count = nevery;
}

/* ---------------------------------------------------------------------- */

FixCDF::~FixCDF()
{
  memory->destroy(data_histo);
  memory->destroy(data_histo_loc);
  memory->destroy(data_velx);
  memory->destroy(data_vely);
  memory->destroy(data_velz);
  memory->destroy(data_vel_locx);
  memory->destroy(data_vel_locy);
  memory->destroy(data_vel_locz);
      memory->destroy(data_pressx);
    memory->destroy(data_pressyz);
    
    memory->destroy(data_press_locx);
    memory->destroy(data_press_locyz);
  if (me==0) fclose(out);
}

/* ---------------------------------------------------------------------- */

int FixCDF::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixCDF::init()
{

}

/* ---------------------------------------------------------------------- */

void FixCDF::end_of_step() 
{
  if (count == nevery) {
    // determine com of the colloid
    int n;
    double xcm[3];
    double masstotal = group->mass(igroup);
    group->xcm(igroup,masstotal,xcm);
    
    for (n=0; n<nbin_x*nbin_r; n++) {
      data_histo[n] = 0;
      data_histo_loc[n] = 0;
      data_velx[n] = 0.0;
      data_vel_locx[n] = 0.0;
      data_vely[n] = 0.0;
      data_vel_locy[n] = 0.0;
      data_velz[n] = 0.0;
      data_vel_locz[n] = 0.0;
      
      data_pressx[n] = 0.0;
      data_press_locx[n] = 0.0;
      
            data_pressyz[n] = 0.0;
      data_press_locyz[n] = 0.0;
    }
    
    double delx, dely, delz;
    int i;
    double **x = atom->x;
    double **v = atom->v;
    int nlocal = atom->nlocal;
    int * mask = atom->mask;
    double **vatom = force->pair->vatom;
    for (i=0; i<nlocal; i++) {
      if(mask[i] & jgroupbit) {
        delx = x[i][0] -xcm[0];
        dely = x[i][1] -  xcm[1];
        delz =  x[i][2] - xcm[2];  
        
        domain->minimum_image(delx,dely,delz);
    
        double data_x = delx + range_x;
        double data_r = sqrt(dely*dely + delz*delz);
        int bin_x=-1;
        if (data_x > 0.0)
          bin_x = data_x*((double) nbin_x)/(2.0*range_x);
        int bin_r = data_r*((double) nbin_r)/(range_r);
        if (bin_x >= 0 && bin_x <nbin_x && bin_r >= 0 && bin_r <nbin_r) {
          data_histo_loc[bin_x*nbin_r + bin_r]++;
          data_vel_locx[bin_x*nbin_r + bin_r]+=v[i][0];
          data_vel_locy[bin_x*nbin_r + bin_r]+=sqrt(v[i][1]*v[i][1]+v[i][2]*v[i][2]);
          double sign = v[i][1]*dely+v[i][2]*delz;
          double dx = dely*dely + delz*delz;
          data_vel_locz[bin_x*nbin_r + bin_r]=sign/sqrt(dx);
	  
	  data_vel_locx[bin_x*nbin_r + bin_r]+=vatom[i][0];
	  data_vel_locx[bin_x*nbin_r + bin_r]+=v[i][0]*v[i][0];
          data_vel_locyz[bin_x*nbin_r + bin_r]+=(vatom[i][1]+vatom[i][2])/2.0;
	  data_vel_locyz[bin_x*nbin_r + bin_r]+=(v[i][1]*v[i][1]+v[i][2]*v[i][2])/2.0;
        }
      }
    }
    
    //printf("test1\n");
    
    MPI_Allreduce(data_histo_loc, data_histo, nbin_x*nbin_r, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(data_vel_locx, data_velx, nbin_x*nbin_r, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(data_vel_locy, data_vely, nbin_x*nbin_r, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(data_vel_locz, data_velz, nbin_x*nbin_r, MPI_DOUBLE, MPI_SUM, world);
    
        MPI_Allreduce(data_press_locx, data_pressx, nbin_x*nbin_r, MPI_DOUBLE, MPI_SUM, world);
	 MPI_Allreduce(data_press_locyz, data_pressyz, nbin_x*nbin_r, MPI_DOUBLE, MPI_SUM, world);
    
    //printf("test2\n");
    
    int r,xloc;
    for (xloc=0; xloc<nbin_x; xloc++) {
      for (r=0; r<nbin_r; r++) {
        if ( data_histo[xloc*nbin_r + r] != 0 ) {
          data_velx[xloc*nbin_r + r] /= data_histo[xloc*nbin_r + r];
          data_vely[xloc*nbin_r + r] /= data_histo[xloc*nbin_r + r];
          data_velz[xloc*nbin_r + r] /= data_histo[xloc*nbin_r + r];
	  
	  data_pressx[xloc*nbin_r + r] /= data_histo[xloc*nbin_r + r];
	  data_pressyz[xloc*nbin_r + r] /= data_histo[xloc*nbin_r + r];
        }
      }
    }
    
    //printf("test3\n");

    if (print && (me == 0) ) {
      fprintf(out,"t=%d\n",update->ntimestep);
      for (xloc=0; xloc<nbin_x; xloc++) {
        for (r=0; r<nbin_r; r++) {
          fprintf(out,"%f %f %f %f %f %f %f %f\n",xloc/((double) nbin_x)*(2.0*range_x)-range_x,r/((double) nbin_r)*(range_r),data_histo[xloc*nbin_r + r],data_velx[xloc*nbin_r + r],data_vely[xloc*nbin_r + r],data_velz[xloc*nbin_r + r],data_pressx[xloc*nbin_r + r],data_pressyz[xloc*nbin_r + r]);
        }
        fprintf(out,"\n");
      }
      fprintf(out,"\n\n");
    }
    count = 0;
  } 
  count ++;
}
