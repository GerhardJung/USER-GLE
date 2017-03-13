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
   Contributing authors: Stephen Bond (SNL) and
                         Andrew Baczewski (Michigan State/SNL)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include "fix_gle_pair_aux.h"
#include "math_extra.h"
#include "atom.h"
#include "force.h"
#include "update.h"
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

/* ----------------------------------------------------------------------
   Parses parameters passed to the method, allocates some memory
------------------------------------------------------------------------- */

FixGLEPairAux::FixGLEPairAux(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  npair=1;
  
  global_freq = 1;
  nevery = 1;
  peratom_freq = 1;
  vector_flag = 1;
  size_vector = 8*npair;

  int narg_min = 5;
  if (narg < narg_min) error->all(FLERR,"Illegal fix gle/pair/aux command");

  t_target = force->numeric(FLERR,arg[3]);
  
  int seed = force->inumeric(FLERR,arg[4]);
 
  
  // Error checking for the first set of required input arguments
  if (seed <= 0) error->all(FLERR,"Illegal fix gle/pair/aux command");
  if (t_target < 0)
    error->all(FLERR,"Fix gle/pair/aux temperature must be >= 0");
  
  // allocate memory for Prony series extended variables
  memory->create(pair_list_part, atom->nlocal,atom->nlocal, "gle/pair/aux:pair_list_part");
  memory->create(pair_list_coef, atom->nlocal,atom->nlocal, "gle/pair/aux:pair_list_coef");
  pair_list_part[0][0]=1;
  pair_list_part[0][1]=-1;
  pair_list_part[1][0]=0;
  pair_list_part[1][1]=-1;
  pair_list_coef[0][0]=0;
  pair_list_coef[0][1]=-1;
  pair_list_coef[1][0]=0;
  pair_list_coef[1][1]=-1;
  
  q_aux = NULL;
  read_coef_aux();
  
  // initialize Marsaglia RNG with processor-unique seed
  random = new RanMars(lmp,seed + comm->me);
  
  memory->create(v_step, atom->nlocal, "gle/pair/aux:v_step");
  memory->create(f_step, atom->nlocal, "gle/pair/aux:v_step");

  // initialize the extended variables
  init_q_aux();

}

/* ----------------------------------------------------------------------
   Destroys memory allocated by the method
------------------------------------------------------------------------- */

FixGLEPairAux::~FixGLEPairAux()
{
  delete random;
  memory->destroy(q_aux);
  memory->destroy(pair_list_part);
  memory->destroy(pair_list_coef);
  memory->destroy(q_ran);
  memory->destroy(q_save);
  memory->destroy(q_B);
  memory->destroy(q_ps);

}

/* ----------------------------------------------------------------------
   Specifies when the fix is called during the timestep
------------------------------------------------------------------------- */

int FixGLEPairAux::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ----------------------------------------------------------------------
   Initialize the method parameters before a run
------------------------------------------------------------------------- */

void FixGLEPairAux::init()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}

/* ----------------------------------------------------------------------
   First half of a timestep (V^{n} -> V^{n+1/2}; X^{n} -> X^{n+1})
------------------------------------------------------------------------- */

void FixGLEPairAux::initial_integrate(int vflag)
{
  double dtfm;
  double ftm2v = force->ftm2v;

  double v_save[atom->nlocal];
  double meff;
  double theta_qss, theta_qsc, theta_qcs, theta_qcc11, theta_qcc12;
  double theta_qsps, theta_qspc;
  int ind_coef, ind_q=0;
  int j,s,c,n,m;

  // update v and x of atoms in group
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // set kT to the temperature in mvvv units
  double kT = (force->boltz)*t_target/(force->mvv2e);
  
  // save velocity for cross time integration
  for ( int i=0; i< nlocal; i++) v_save[i] = v[i][0];
  for ( int i=0; i< nlocal; i++) v_step[i] = v[i][0];

  // Advance V by dt/2
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      // calculate integration constants
      meff = mass[type[i]];   
      dtfm = dtf / mass[type[i]];
      //v[i][0] += dtfm * f_step[i];
      j = 0;
      while (pair_list_part[i][j]!=-1) {
	ind_coef = pair_list_coef[i][j];
	ind_q = ( i < pair_list_part[i][j]) ? 0 : 4;
	//printf("coef %d, q %d\n",ind_coef, ind_q);
	
	for (int k = 0; k < aux_terms; k++) {
	  v[i][0] -= q_ps[ind_coef][4*k]*dtfm *q_aux[ind_coef][8*k+ind_q];
	  v[i][0] -= q_ps[ind_coef][4*k+1]*dtfm *q_aux[ind_coef][8*k+ind_q+2];
	  v[i][0] -= q_ps[ind_coef][4*k+2]*dtfm *q_aux[ind_coef][8*k+4-ind_q];
	  v[i][0] -= q_ps[ind_coef][4*k+3]*dtfm *q_aux[ind_coef][8*k+6-ind_q];
	}
	j++;
      }
    }
  }

  // Advance X by dt
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      //x[i][0] += dtv * v[i][0];
    }
  }

  for (int i = 0; i < npair; i++) {
    for (int k = 0; k < aux_terms; k++) {
      for (n=0; n<8; n++) {
	q_ran[i][8*k+n] = random->gaussian();
	q_save[i][8*k+n] = q_aux[i][8*k+n];
      }
    } 
  }
  
  // Advance Q by dt
  for (int i = 0; i < npair; i++) {
    
    for (int k = 0; k < aux_terms; k++) {
      
      // update aux_var, self (s) and cross (c)
      for (int s=0; s<8; s+=4) {
	// update a
	q_aux[i][8*k+s] *= q_int[i][4*k];  
	q_aux[i][8*k+s] -= q_int[i][4*k+1]*q_save[i][8*k+s+1];  
	// update b
	q_aux[i][8*k+s+1] *= q_int[i][4*k];  
	q_aux[i][8*k+s+1] += q_int[i][4*k+1]*q_save[i][8*k+s];  
      }
      for (int c=2; c<8; c+=4) {
	//update a
	q_aux[i][8*k+c] *= q_int[i][4*k+2];  
	q_aux[i][8*k+c] -= q_int[i][4*k+3]*q_save[i][8*k+c+1];  
	//update b
	q_aux[i][8*k+c+1] *= q_int[i][4*k+2];  
	q_aux[i][8*k+c+1] += q_int[i][4*k+3]*q_save[i][8*k+c];  
      }
    
    
      // update momenta contribution
    
      // update noise
      for (n=0; n<4; n++) {
	for (m=0; m<4; m++) {
	  q_aux[i][8*k+n] += sqrt(update->dt)*q_B[i][16*k+4*n+m]*q_ran[i][8*k+m];
	  q_aux[i][8*k+n+4] +=  sqrt(update->dt)*q_B[i][16*k+4*n+m]*q_ran[i][8*k+m+4];
	}
      }

    }
    /*for (n=0; n<8; n++) {
      printf("%f ",q_aux[i][n]);
    }
    printf("\n");*/
  }
  
  
}

/* ----------------------------------------------------------------------
   Second half of a timestep (V^{n+1/2} -> V^{n+1})
------------------------------------------------------------------------- */

void FixGLEPairAux::final_integrate()
{

  double dtfm;
  double ftm2v = force->ftm2v;

  double v_save[atom->nlocal];
  double meff;
  double theta_vs, alpha_vs, theta_vc, alpha_vc;
  int ind_coef, ind_q;
  int j;


  // update v and x of atoms in group
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // set kT to the temperature in mvvv units
  double kT = (force->boltz)*t_target/(force->mvv2e);
  
  // save velocity for cross time integration
  for ( int i=0; i< nlocal; i++) f_step[i] = f[i][0];

  // Advance V by dt/2
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      // calculate integration constants
      meff = mass[type[i]];   
      dtfm = dtf / mass[type[i]];
      //v[i][0] += dtfm * f_step[i];
      j = 0;
      while (pair_list_part[i][j]!=-1) {
	ind_coef = pair_list_coef[i][j];
	ind_q = ( i < pair_list_part[i][j]) ? 0 : 4;
	//printf("coef %d, q %d\n",ind_coef, ind_q);
	for (int k = 0; k < aux_terms; k++) {
	  v[i][0] -= q_ps[ind_coef][4*k]*dtfm *q_aux[ind_coef][8*k+ind_q];
	  v[i][0] -= q_ps[ind_coef][4*k+1]*dtfm *q_aux[ind_coef][8*k+ind_q+2];
	  v[i][0] -= q_ps[ind_coef][4*k+2]*dtfm *q_aux[ind_coef][8*k+4-ind_q];
	  v[i][0] -= q_ps[ind_coef][4*k+3]*dtfm *q_aux[ind_coef][8*k+6-ind_q];
	}
	j++;
      }
    }
  }
 
  
  for (int i = 0; i < nlocal; i++) {
    f[i][0]=(v[i][0]-v_step[i])/update->dt*mass[type[i]];
  }

}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixGLEPairAux::compute_vector(int n)
{
  int j = n%8;
  int i = (n - j)/8;
  
  return q_aux[i][j];
}


/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixGLEPairAux::memory_usage()
{
  double bytes = atom->nlocal*atom->nlocal*aux_terms*sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixGLEPairAux::grow_arrays(int nmax)
{
  memory->grow(q_aux, nmax,aux_terms*8,"gle/pair/aux:q_aux");
}

/* ----------------------------------------------------------------------
   read in coefficients
------------------------------------------------------------------------- */
void FixGLEPairAux::read_coef_aux()
{
  FILE * input;
  input = fopen("aux_coef.dat","r");
  fscanf(input,"%d\n\n",&aux_terms);
  //printf("%d\n",aux_terms);

  if (aux_terms < 0)
    error->all(FLERR,"Fix gle/pair/aux terms must be > 0");
  
  // allocate memory
  memory->create(q_s, npair,aux_terms*4, "gle/pair/aux:q_s");
  memory->create(q_ps, npair,aux_terms*4, "gle/pair/aux:q_ps");
  memory->create(q_B, npair,aux_terms*4*4, "gle/pair/aux:q_B");
  
  for (int k=0; k<aux_terms; k++) fscanf(input,"%lf %lf %lf %lf\n",&q_s[0][4*k],&q_s[0][4*k+1],&q_s[0][4*k+2],&q_s[0][4*k+3]);
  //for (int k=0; k<aux_terms; k++) printf("%lf %lf %lf %lf\n",q_s[0][4*k],q_s[0][4*k+1],q_s[0][4*k+2],q_s[0][4*k+3]);
  
  fscanf(input,"\n");
  
  for (int k=0; k<aux_terms; k++) fscanf(input,"%lf %lf %lf %lf\n",&q_ps[0][4*k],&q_ps[0][4*k+1],&q_ps[0][4*k+2],&q_ps[0][4*k+3]);
  //for (int k=0; k<aux_terms; k++) printf("%lf %lf %lf %lf\n",q_ps[0][4*k],q_ps[0][4*k+1],q_ps[0][4*k+2],q_ps[0][4*k+3]);
  
  fscanf(input,"\n");
  
  for (int k=0; k<aux_terms; k++)
    for (int n=0; n<4; n++)
      fscanf(input,"%lf %lf %lf %lf\n",&q_B[0][16*k+4*n],&q_B[0][16*k+4*n+1],&q_B[0][16*k+4*n+2],&q_B[0][16*k+4*n+3]);
  
  /*for (int k=0; k<aux_terms; k++)
    for (int n=0; n<4; n++)
      printf("%lf %lf %lf %lf\n",q_B[0][16*k+4*n],q_B[0][16*k+4*n+1],q_B[0][16*k+4*n+2],q_B[0][16*k+4*n+3]);*/
  
}

/* ----------------------------------------------------------------------
   Initializes the extended variables to equilibrium distribution
   at t_start.
------------------------------------------------------------------------- */

void FixGLEPairAux::init_q_aux()
{
  int icoeff;
  double eq_sdev=0.0;
  int n;
  
  grow_arrays(npair);
  memory->create(q_ran, npair,aux_terms*8, "gle/pair/aux:q_ran");
  memory->create(q_save, npair,aux_terms*8, "gle/pair/aux:q_save");
  memory->create(q_int, npair,aux_terms*4, "gle/pair/aux:q_int");

  // set kT to the temperature in mvvv units
  double kT = (force->boltz)*t_target/(force->mvv2e);
  double scale = sqrt(kT)/(force->ftm2v);

  for (int i = 0; i < npair; i++) {
    for (int k = 0; k < aux_terms; k++) {
      for (int n = 0; n < 8; n++) {
	q_aux[i][8*k+n] = 0.0;
      }
    }
  }
  
  for (int i = 0; i < npair; i++) {
    for (int k = 0; k < aux_terms; k++) {
      q_int[i][4*k]=exp(-update->dt*q_s[i][4*k])*cos(update->dt*q_s[i][4*k+1]);
      q_int[i][4*k+1]=exp(-update->dt*q_s[i][4*k])*sin(update->dt*q_s[i][4*k+1]);
      q_int[i][4*k+2]=exp(-update->dt*q_s[i][4*k+2])*cos(update->dt*q_s[i][4*k+3]);
      q_int[i][4*k+3]=exp(-update->dt*q_s[i][4*k+2])*sin(update->dt*q_s[i][4*k+3]);
    }
  }
  
  

  return;
}
