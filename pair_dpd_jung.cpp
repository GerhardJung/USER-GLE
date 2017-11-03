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
   Contributing author: Kurt Smith (U Pittsburgh)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "pair_dpd_jung.h"
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "update.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"
#include <Eigen/Dense>
#include <Eigen/Eigen>

using namespace Eigen;
using namespace LAMMPS_NS;
using namespace std;

#define EPSILON 1.0e-10

/* ---------------------------------------------------------------------- */

PairDPDJung::PairDPDJung(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 1;
  random = NULL;
  
  MPI_Comm_rank(world,&me);
  time_mvm = 0.0;
  time_inv = 0.0;
  
  
}

/* ---------------------------------------------------------------------- */

PairDPDJung::~PairDPDJung()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(a0);
    memory->destroy(gamma);
    memory->destroy(sigma);
  }

  if (random) delete random;
}

/* ---------------------------------------------------------------------- */

void PairDPDJung::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype,itag,jtag;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair_c,fpair_r,fpair_d;
  double vxtmp,vytmp,vztmp,delvx,delvy,delvz;
  double rsq,r,rinv,dot,wd,randnum,factor_dpd;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int *type = atom->type;
  int *tag = atom->tag;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double dtinvsqrt = 1.0/sqrt(update->dt);
  
  my_list = neighbor->lists[irequest];

  inum = my_list->inum;
  ilist = my_list->ilist;
  numneigh = my_list->numneigh;
  firstneigh = my_list->firstneigh;
  
  // loop over neighbors of my atoms, to determine conservative force input vector (only in the first timestep)
  if (!initialized) {
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      itype = type[i];
      itag = tag[i]-1;
      jlist = firstneigh[i];
      jnum = numneigh[i];

      for (jj = 0; jj < jnum; jj++) {
	j = jlist[jj];
	factor_dpd = special_lj[sbmask(j)];
	j &= NEIGHMASK;

	delx = xtmp - x[j][0];
	dely = ytmp - x[j][1];
	delz = ztmp - x[j][2];
	rsq = delx*delx + dely*dely + delz*delz;
	jtype = type[j];
	jtag = tag[j] -1;

	if (rsq < cutsq[itype][jtype]) {
	  r = sqrt(rsq);
	  if (r < EPSILON) continue;     // r can be 0.0 in DPD systems
	  rinv = 1.0/r;
	  dot = delx*delvx + dely*delvy + delz*delvz;
	  wd = 1.0 - r/cut[itype][jtype];
	  randnum = random->gaussian();

	  // conservative force = a0 * wd
	  // drag force = -gamma * wd^2 * (delx dot delv) / r
	  //random force = sigma * wd * rnd * dtinvsqrt;

	  fpair_c = a0[itype][jtype]*wd;
	  fpair_r = sigma[itype][jtype]*wd*randnum*dtinvsqrt;
	  fpair_c *= factor_dpd*rinv;
	  fpair_r *= factor_dpd*rinv;

	  fcn1[3*itag] += delx*fpair_c;
	  fcn1[3*itag+1] += dely*fpair_c;
	  fcn1[3*itag+2] += delz*fpair_c;
	  
	  frn1[3*itag] += delx*fpair_r;
	  frn1[3*itag+1] += dely*fpair_r;
	  frn1[3*itag+2] += delz*fpair_r;
	  if (newton_pair || j < nlocal) {
	    fcn1[3*jtag] -= delx*fpair_c;
	    fcn1[3*jtag+1] -= dely*fpair_c;
	    fcn1[3*jtag+2] -= delz*fpair_c;
	    
	    frn1[3*jtag] -= delx*fpair_r;
	    frn1[3*jtag+1] -= dely*fpair_r;
	    frn1[3*jtag+2] -= delz*fpair_r;
	  }
	}
      }
    }
    initialized = 1;
  }
  
  // new timestep: migrate conservative force and position and reset random force
  for (i=0; i<nlocal; i++) {
    fn[3*i] = 0.0;
    fn[3*i+1] = 0.0;
    fn[3*i+2] = 0.0;
    drn[3*i] = 0.0;
    drn[3*i+1] = 0.0;
    drn[3*i+2] = 0.0;
    fcn[3*i] = fcn1[3*i];
    fcn[3*i+1] = fcn1[3*i+1];
    fcn[3*i+2] = fcn1[3*i+2];
    fcn1[3*i] = 0.0;
    fcn1[3*i+1] = 0.0;
    fcn1[3*i+2] = 0.0;
    frn[3*i] = frn1[3*i];
    frn[3*i+1] = frn1[3*i+1];
    frn[3*i+2] = frn1[3*i+2];
    frn1[3*i] = 0.0;
    frn1[3*i+1] = 0.0;
    frn1[3*i+2] = 0.0;
    fdn[3*i] = 0.0;
    fdn[3*i+1] = 0.0;
    fdn[3*i+2] = 0.0;
  }
  
  /*  double **res_matrix;
  int size = atom->nlocal*3;
  res_matrix = new double*[size];
  for(int i = 0; i < size; ++i){
    res_matrix[i] = new double[size];
    for (int j=0; j<size; j++) {
      res_matrix[i][j] = 0.0;
    }
  }
  create_matrix(res_matrix);
  res_matrix_eigen = MatrixXd::Zero(size,size);
  for (i=0; i<3*atom->nlocal; i++) {
    //printf("%d ",i);
    for (j=0; j<3*atom->nlocal; j++) {
      //printf("%f ",res_matrix[i][j]);
      res_matrix_eigen(i,j) = res_matrix[i][j];
    }
    //printf("\n");
  }*/
  
  for (i=0; i<atom->nlocal; i++) {
    itag = tag[i] -1;
    fn[3*itag] = update->dt*v[i][0] + update->dt*update->dt/2.0*fcn[3*itag] + update->dt*update->dt/2.0*frn[3*itag];
    fn[3*itag+1] = update->dt*v[i][1] + update->dt*update->dt/2.0*fcn[3*itag+1] + update->dt*update->dt/2.0*frn[3*itag+1];
    fn[3*itag+2] = update->dt*v[i][2] + update->dt*update->dt/2.0*fcn[3*itag+2] + update->dt*update->dt/2.0*frn[3*itag+2];
    
    //printf("%f %f %f\n",f_step[3*i],f_step[3*i+1],f_step[3*i+2]);
  }
  t2 = MPI_Wtime();
  time_mvm += t2 -t1;
  
  compute_inverse(fn,drn);
  
  // integrate position
  for (i=0; i<atom->nlocal; i++) {
    itag = tag[i] -1;
    x[i][0] += drn[3*itag];
    x[i][1] += drn[3*itag+1];
    x[i][2] += drn[3*itag+2];
    //printf("x: %d %f %f %f\n",i,x[i][0],x[i][1],x[i][2]);
  }
  // update positions in neighbor list
  // domain->pbc();
  // comm->exchange();
  // comm->borders();
  // neighbor->build();
  
  // update conservative and random force
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    itag = tag[i]-1;
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_dpd = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];
      jtag = tag[j] -1;

      if (rsq < cutsq[itype][jtype]) {
        r = sqrt(rsq);
        if (r < EPSILON) continue;     // r can be 0.0 in DPD systems
        rinv = 1.0/r;
        dot = delx*delvx + dely*delvy + delz*delvz;
        wd = 1.0 - r/cut[itype][jtype];
        randnum = random->gaussian();

        // conservative force = a0 * wd
        // drag force = -gamma * wd^2 * (delx dot delv) / r
        // random force = sigma * wd * rnd * dtinvsqrt;

        fpair_c = a0[itype][jtype]*wd;
	fpair_r = sigma[itype][jtype]*wd*randnum*dtinvsqrt;
	fpair_c *= factor_dpd*rinv;
	fpair_r *= factor_dpd*rinv;

	fcn1[3*itag] += delx*fpair_c;
	fcn1[3*itag+1] += dely*fpair_c;
	fcn1[3*itag+2] += delz*fpair_c;
	  
	frn1[3*itag] += delx*fpair_r;
	frn1[3*itag+1] += dely*fpair_r;
	frn1[3*itag+2] += delz*fpair_r;
	if (newton_pair || j < nlocal) {
	  fcn1[3*jtag] -= delx*fpair_c;
	  fcn1[3*jtag+1] -= dely*fpair_c;
	  fcn1[3*jtag+2] -= delz*fpair_c;
	    
	  frn1[3*jtag] -= delx*fpair_r;
	  frn1[3*jtag+1] -= dely*fpair_r;
	  frn1[3*jtag+2] -= delz*fpair_r;
	}
        
        if (eflag) {
          // unshifted eng of conservative term:
          // evdwl = -a0[itype][jtype]*r * (1.0-0.5*r/cut[itype][jtype]);
          // eng shifted to 0.0 at cutoff
          evdwl = 0.5*a0[itype][jtype]*cut[itype][jtype] * wd*wd;
          evdwl *= factor_dpd;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair_c,delx,dely,delz);
      }
    }
  }
  
  // update dissipative force
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    itag = tag[i] -1;
    vxtmp = drn[3*itag];
    vytmp = drn[3*itag+1];
    vztmp = drn[3*itag+2];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_dpd = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];
      jtag = tag[j] -1;

      if (rsq < cutsq[itype][jtype]) {
        r = sqrt(rsq);
        if (r < EPSILON) continue;     // r can be 0.0 in DPD systems
        rinv = 1.0/r;
        delvx = vxtmp - drn[3*jtag];
        delvy = vytmp - drn[3*jtag+1];
        delvz = vztmp - drn[3*jtag+2];
        dot = delx*delvx + dely*delvy + delz*delvz;
        wd = 1.0 - r/cut[itype][jtype];
        fpair_d = -gamma[itype][jtype]*wd*wd*dot*rinv;
	//printf("wd %f\n",wd*wd);
        fpair_d *= factor_dpd*rinv;

	//printf("%f\n",delx*fpair);
	
        fdn[3*itag] += delx*fpair_d;
        fdn[3*itag+1] += dely*fpair_d;
        fdn[3*itag+2] += delz*fpair_d;
        if (newton_pair || j < nlocal) {
          fdn[3*jtag] -= delx*fpair_d;
          fdn[3*jtag+1] -= dely*fpair_d;
          fdn[3*jtag+2] -= delz*fpair_d;
        }
      }
    }
  }
  
  // integrate velocity
  for (i=0; i<atom->nlocal; i++) {
    itag = tag[i] -1;
    v[i][0] += update->dt/2.0*(fcn[3*itag]+fcn1[3*itag]) + fdn[3*itag] + update->dt/2.0*(frn[3*itag]+frn1[3*itag]);
    v[i][1] += update->dt/2.0*(fcn[3*itag+1]+fcn1[3*itag+1]) + fdn[3*itag+1] + update->dt/2.0*(frn[3*itag+1]+frn1[3*itag+1]);
    v[i][2] += update->dt/2.0*(fcn[3*itag+2]+fcn1[3*itag+2]) + fdn[3*itag+2] + update->dt/2.0*(frn[3*itag+2]+frn1[3*itag+2]);
    //printf("v: %d %f %f %f\n",i,v[i][0],v[i][1],v[i][2]);
  }
  
  // test step
  /*double *test_input = new double[3*atom->nlocal];
  double *test_output = new double[3*atom->nlocal];
  for (i=0; i<3*atom->nlocal; i++) {
    test_input[i]=0.0;
    test_output[i]=0.0;
  }
  test_input[0] = 1.0;
  test_input[3] = 0.5;
  
  //printf("test direkt\n");
  compute_step(f_step,test_output);
  for (i=0; i<3*atom->nlocal; i++) {
    //printf("%f ",test_output[i]);
  }
  //printf("\n\n");
  

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> res_eigen(res_matrix_eigen);
	Eigen::MatrixXd res_eigen_eigenvector = res_eigen.eigenvectors().real();
	if (res_eigen.info()!=0) {
	  cout << res_matrix_eigen << endl;
	  error->all(FLERR,"Hk is not positive-definite in pair/gle\n");
	}  
	Eigen::MatrixXd res_eigen_diag = Eigen::MatrixXd::Zero(size,size);
	for (int i=0; i<size; i++) {
	  res_eigen_diag(i,i) = sqrt(res_eigen.eigenvalues().real()(i));
	}
	MatrixXd f_res_eigen = res_eigen_eigenvector * res_eigen_diag * res_eigen_eigenvector.transpose();
	//cout << f_res_eigen << endl;
	//cout << f_res_eigen*f_res_eigen.transpose() << endl;
	 //printf("\n");
	//cout << res_matrix_eigen << endl;
	
	  //printf("test matrix\n");
  for (i=0; i<3*atom->nlocal; i++) {
    test_output[i] = 0.0;
    for (j=0; j<3*atom->nlocal; j++) {
      test_output[i] += res_matrix[i][j]*f_step[j];
    }
    //printf("%f ",test_output[i]);
  }
  //printf("\n");
  
  
  // test inverse
  //printf("test inverse\n");
  //compute_inverse(test_input, test_output);
  //compute_inverse(test_output, test_input);
  for (i=0; i<3*atom->nlocal; i++) {
    //printf("%f ",test_input[i]);
  }
  //printf("\n");*/
  
  //printf("%f %f\n",update->nsteps,update->ntimestep);
  if (update->nsteps == update->ntimestep) {
    printf("processor %d: time(mvm) = %f\n",me,time_mvm);
    printf("processor %d: time(inv) = %f\n",me,time_inv);
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairDPDJung::allocate()
{
  int i,j;
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(a0,n+1,n+1,"pair:a0");
  memory->create(gamma,n+1,n+1,"pair:gamma");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  for (i = 0; i <= atom->ntypes; i++)
    for (j = 0; j <= atom->ntypes; j++)
      sigma[i][j] = gamma[i][j] = 0.0;
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairDPDJung::settings(int narg, char **arg)
{
  if (narg != 3) error->all(FLERR,"Illegal pair_style command");

  temperature = force->numeric(FLERR,arg[0]);
  cut_global = force->numeric(FLERR,arg[1]);
  seed = force->inumeric(FLERR,arg[2]);

  // initialize Marsaglia RNG with processor-unique seed

  if (seed <= 0) error->all(FLERR,"Illegal pair_style command");
  delete random;
  random = new RanMars(lmp,seed + comm->me);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairDPDJung::coeff(int narg, char **arg)
{
  if (narg < 4 || narg > 5)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double a0_one = force->numeric(FLERR,arg[2]);
  double gamma_one = force->numeric(FLERR,arg[3]);

  double cut_one = cut_global;
  if (narg == 5) cut_one = force->numeric(FLERR,arg[4]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      a0[i][j] = a0_one;
      gamma[i][j] = gamma_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairDPDJung::init_style()
{
  if (comm->ghost_velocity == 0)
    error->all(FLERR,"Pair dpd requires ghost atoms store velocity");

  // if newton off, forces between atoms ij will be double computed
  // using different random numbers

  if (force->newton_pair == 0 && comm->me == 0) error->warning(FLERR,
      "Pair dpd needs newton pair on for momentum conservation");

  irequest = neighbor->request(this);
  neighbor->requests[irequest]->pair = 1;
  neighbor->requests[irequest]->fix = 0;
  neighbor->requests[irequest]->half = 1;
  neighbor->requests[irequest]->full = 0;
  
  fn = new double[3*atom->nlocal];
  drn = new double[3*atom->nlocal];
  fcn = new double[3*atom->nlocal];
  fcn1 = new double[3*atom->nlocal];
  frn = new double[3*atom->nlocal];
  frn1 = new double[3*atom->nlocal];
  fdn = new double[3*atom->nlocal];
  fdn1 = new double[3*atom->nlocal];
  
  for (int i=0; i<3*atom->nlocal; i++) {
    drn[i] = 0.0;
    fn[i] = 0.0;
    fcn[i] = 0.0;
    fcn1[i] = 0.0;
    frn[i] = 0.0;
    frn1[i] = 0.0;
    fdn[i] = 0.0;
    fdn1[i] = 0.0;
  }
  
  initialized = 0;

}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairDPDJung::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  sigma[i][j] = sqrt(2.0*force->boltz*temperature*gamma[i][j]);

  cut[j][i] = cut[i][j];
  a0[j][i] = a0[i][j];
  gamma[j][i] = gamma[i][j];
  sigma[j][i] = sigma[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairDPDJung::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&a0[i][j],sizeof(double),1,fp);
        fwrite(&gamma[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairDPDJung::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&a0[i][j],sizeof(double),1,fp);
          fread(&gamma[i][j],sizeof(double),1,fp);
          fread(&cut[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&a0[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&gamma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairDPDJung::write_restart_settings(FILE *fp)
{
  fwrite(&temperature,sizeof(double),1,fp);
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&seed,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairDPDJung::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&temperature,sizeof(double),1,fp);
    fread(&cut_global,sizeof(double),1,fp);
    fread(&seed,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&temperature,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&seed,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);

  // initialize Marsaglia RNG with processor-unique seed
  // same seed that pair_style command initially specified

  if (random) delete random;
  random = new RanMars(lmp,seed + comm->me);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairDPDJung::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g\n",i,a0[i][i],gamma[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairDPDJung::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g\n",i,j,a0[i][j],gamma[i][j],cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairDPDJung::single(int i, int j, int itype, int jtype, double rsq,
                       double factor_coul, double factor_dpd, double &fforce)
{
  double r,rinv,wd,phi;

  r = sqrt(rsq);
  if (r < EPSILON) {
    fforce = 0.0;
    return 0.0;
  }

  rinv = 1.0/r;
  wd = 1.0 - r/cut[itype][jtype];
  fforce = a0[itype][jtype]*wd * factor_dpd*rinv;

  phi = 0.5*a0[itype][jtype]*cut[itype][jtype] * wd*wd;
  return factor_dpd*phi;
}

/* ----------------------------------------------------------------------
   multiplies an input vector with interaction matrix
------------------------------------------------------------------------- */

void PairDPDJung::compute_step(double* input, double* output)
{
  int i,j,ii,jj,inum,jnum,itype,jtype,itag,jtag;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double vxtmp,vytmp,vztmp,delvx,delvy,delvz;
  double rsq,r,rinv,dot,wd,randnum,factor_dpd;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int *tag = atom->tag;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double dtinvsqrt = 1.0/sqrt(update->dt);
  
  double pre = -update->dt / 2.0;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    itag = tag[i] -1;
    vxtmp = input[3*itag];
    vytmp = input[3*itag+1];
    vztmp = input[3*itag+2];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    
    //add unity matrix
    output[3*itag] += input[3*itag];
    output[3*itag+1] += input[3*itag+1];
    output[3*itag+2] += input[3*itag+2];
    //pre = 1.0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_dpd = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];
      jtag = tag[j] -1;

      if (rsq < cutsq[itype][jtype]) {
        r = sqrt(rsq);
        if (r < EPSILON) continue;     // r can be 0.0 in DPD systems
        rinv = 1.0/r;
        delvx = vxtmp - input[3*jtag];
        delvy = vytmp - input[3*jtag+1];
        delvz = vztmp - input[3*jtag+2];
        dot = delx*delvx + dely*delvy + delz*delvz;
        wd = 1.0 - r/cut[itype][jtype];
        fpair = -gamma[itype][jtype]*wd*wd*dot*rinv;
	//printf("wd %f\n",wd*wd);
        fpair *= pre*factor_dpd*rinv;

	//printf("%f\n",delx*fpair);
	
        output[3*itag] += delx*fpair;
        output[3*itag+1] += dely*fpair;
        output[3*itag+2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          output[3*jtag] -= delx*fpair;
          output[3*jtag+1] -= dely*fpair;
          output[3*jtag+2] -= delz*fpair;
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   computer inverse matrix for precise integration
------------------------------------------------------------------------- */
void PairDPDJung::compute_inverse(double* input, double* output)
{
  t1 = MPI_Wtime();
  int i,j;
  int mLanczos = 50;
  double tolLanczos = 0.0000001;

  // main Lanczos loop, determine krylov subspace
  int size = atom->nlocal*3; 
  Eigen::MatrixXd Vn;
  Vn.resize(size,1);
  for (i=0; i< size; i++) {
    Vn.col(0)(i) = input[i];
  }
  //cout << Vn << endl;
  VectorXd rk = VectorXd::Zero(size);
  VectorXd xk_save;
  double *alpha = new double[mLanczos+1];
  double *beta = new double[mLanczos+1];
  for (int k=0; k<mLanczos+1; k++) {
    alpha[k] = 0.0;
    beta[k] = 0.0;
  }
  // input vector is the FFT of the (uncorrelated) noise vector
  double norm = 0.0;
  norm = Vn.col(0).norm();
  Vn.col(0) = Vn.col(0).normalized();
  compute_step(&Vn.col(0)(0),&rk(0));
  //rk = res_matrix_eigen*Vn.col(0);
  //cout << rk << endl;
  //printf("\n");
  alpha[1] = (Vn.col(0).adjoint()*rk).value();
  // stop if already terminated
  if ( (alpha[1] -1.0)*(alpha[1] -1.0) < 0.0000000001 ) {
    xk_save = rk*norm;
    mLanczos = 1;
  }
  
  // main laczos loop
  for (int k=2; k<=mLanczos; k++) {
    rk = rk - alpha[k-1]*Vn.col(k-2);
    //cout << rk << endl;
    if (k>2) rk -= beta[k-2]*Vn.col(k-3);
    beta[k-1] = rk.norm();
    // set new v
    Vn.conservativeResize(size,k);
    Vn.col(k-1) = rk.normalized();
    compute_step(&Vn.col(k-1)(0),&rk(0));
    //rk = res_matrix_eigen*Vn.col(k-1);
    alpha[k] = (Vn.col(k-1).adjoint()*rk).value();

    //generate result vector by contructing Hessenberg-Matrix
    Eigen::MatrixXd Hk = Eigen::MatrixXd::Zero(k,k);
    for (i=0; i<k; i++) {
      for (j=0; j<k; j++) {
	if (i==j) Hk(i,j) = alpha[i+1];
	if (i==j+1||i==j-1) {
	  int kprime = j;
	  if (i>j) kprime = i;
	  Hk(i,j) = beta[kprime];
	}
      }
    }
    //cout << Hk << endl;

    // determine inverse-matrix (only on the small Hessenberg-Matrix)
      /*Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> Hk_eigen(Hk);
	Eigen::MatrixXd Hk_eigenvector = Hk_eigen.eigenvectors().real();
	if (Hk_eigen.info()!=0) {
	  //cout << A_dist << endl;
	  //cout << A_FT << endl;
	  cout << Hk << endl;
	  error->all(FLERR,"Hk is not positive-definite in pair/gle\n");
	}  
	Eigen::MatrixXd Hk_diag = Eigen::MatrixXd::Zero(k,k);
	for (int i=0; i<k; i++) {
	  Hk_diag(i,i) = sqrt(Hk_eigen.eigenvalues().real()(i));
	}
	MatrixXd f_Hk = Hk_eigenvector * Hk_diag * Hk_eigenvector.transpose();*/
    MatrixXd f_Hk = Hk.inverse();
    //cout << Hk << endl;
    // determine result vector
    VectorXd e1 = VectorXd::Zero(k);
    e1(0) = 1.0;
    VectorXd f_Hk1 = f_Hk * e1;
    VectorXd xk = Vn*f_Hk1*norm; 
    if (k==2) xk_save = xk;
    else {
      VectorXd diff = (xk_save - xk);
      double diff_norm = diff.norm();
      xk_save = xk;
      //cout << xk_save << endl;
      //printf("\n");
      // check for convergence
      if (diff_norm < tolLanczos) {
	//printf("%d\n",k);
	break;
      }
    }
  }
  
  //cout << xk_save << endl;
  for (i=0; i< size; i++) {
    output[i] = xk_save(i);
  }
  
  // test
  double* test_out = new double[size];
  for (i=0; i< size; i++) {
    test_out[i] = 0.0;
  }
  compute_step(output,test_out);
  for (i=0; i< size; i++) {
    //printf("%f %f %f\n",input[i],output[i],test_out[i]);
  }
  
  delete [] alpha;
  delete [] beta;
  t2 = MPI_Wtime();
  time_inv += t2-t1;
}

/* ----------------------------------------------------------------------
   create matrix
------------------------------------------------------------------------- */
void PairDPDJung::create_matrix(double** output){
  
  int i,j,ii,jj,inum,jnum,itype,jtype,itag,jtag;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double vxtmp,vytmp,vztmp,delvx,delvy,delvz;
  double rsq,r,rinv,dot,wd,randnum,factor_dpd;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int *tag = atom->tag;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double dtinvsqrt = 1.0/sqrt(update->dt);
  
  double pre = -update->dt / 2.0;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  double* dr = new double[3];

  // loop over neighbors of my atoms
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    itag = tag[i] -1;
    jlist = firstneigh[i];
    jnum = numneigh[i];
    
    //add unity matrix
    output[3*itag][3*itag] += 1.0;
    output[3*itag+1][3*itag+1] += 1.0;
    output[3*itag+2][3*itag+2] += 1.0;
    //pre = 1.0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_dpd = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      dr[0] = delx = xtmp - x[j][0];
      dr[1] = dely = ytmp - x[j][1];
      dr[2] = delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];
      jtag = tag[j] -1;

      if (rsq < cutsq[itype][jtype]) {
        r = sqrt(rsq);
        if (r < EPSILON) continue;     // r can be 0.0 in DPD systems
        rinv = 1.0/r;
        wd = 1.0 - r/cut[itype][jtype];
        fpair = -gamma[itype][jtype]*wd*wd*rinv;
        fpair *= pre*factor_dpd*rinv;

	for (int dim1=0; dim1<3; dim1++) {
	  for (int dim2=0; dim2<3; dim2++) {
	    output[3*itag+dim1][3*jtag+dim2] -= dr[dim1]*dr[dim2]*fpair;
	    output[3*itag+dim1][3*itag+dim2] += dr[dim1]*dr[dim2]*fpair;
	    if (newton_pair || j < nlocal) {
	      output[3*jtag+dim1][3*itag+dim2] -= dr[dim1]*dr[dim2]*fpair;
	      output[3*jtag+dim1][3*jtag+dim2] += dr[dim1]*dr[dim2]*fpair;
	    }
	  }
	}
      }
    }
  }
  
}
