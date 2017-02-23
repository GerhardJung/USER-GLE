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
#include "fix_gle_noise.h"
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

FixGLENoise::FixGLENoise(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{

  int narg_min = 6;
  if (narg < narg_min) error->all(FLERR,"Illegal fix gle/pair/aux command");

  t_target = force->numeric(FLERR,arg[3]);
  
  int seed = force->inumeric(FLERR,arg[4]);
  
  aux_terms = force->inumeric(FLERR,arg[5]);

  // allocate memory and read-in auxiliary series coefficients
  if (aux_terms < 0)
    error->all(FLERR,"Fix gle/pair/aux terms must be > 0");
  if (narg - narg_min < 6*(aux_terms)+2 )
    error->all(FLERR,"Fix gle/pair/aux needs more auxiliary variable coefficients");
  memory->create(aux_c11, aux_terms, "gle/pair/aux:aux_c11");
  memory->create(aux_lam1, aux_terms, "gle/pair/aux:aux_lam1");
  memory->create(aux_c21, aux_terms, "gle/pair/aux:aux_c21");
  memory->create(aux_c22, aux_terms, "gle/pair/aux:aux_c22");
  memory->create(aux_c24, aux_terms, "gle/pair/aux:aux_c24");
  memory->create(aux_lam2, aux_terms, "gle/pair/aux:aux_lam2");
  
  global_freq = 1;
  nevery = 1;
  peratom_freq = 1;
  vector_flag = 1;
  size_vector = atom->nlocal*aux_terms*atom->nlocal;
    
  int iarg = narg_min;
  aux_a_self = force->numeric(FLERR,arg[iarg]);
  aux_a_cross = force->numeric(FLERR,arg[iarg+1]);
  iarg +=2;
  int icoeff = 0;
  while (iarg < narg && icoeff < aux_terms) {
    double c11 = force->numeric(FLERR,arg[iarg]);
    double lam1 = force->numeric(FLERR,arg[iarg+1]);
    double c21 = force->numeric(FLERR,arg[iarg+2]);
    double c22 = force->numeric(FLERR,arg[iarg+3]);
    double c24 = force->numeric(FLERR,arg[iarg+4]);
    double lam2 = force->numeric(FLERR,arg[iarg+5]);

    if (lam1  <= 0 || lam2 <= 0)
      error->all(FLERR,"Fix gle/pair/aux lam coefficients must be > 0");

    // All atom types to have the same Prony series
    aux_c11[icoeff] = c11;
    aux_lam1[icoeff] = lam1;
    aux_c21[icoeff] = c21;
    aux_c22[icoeff] = c22;
    aux_c24[icoeff] = c24;
    aux_lam2[icoeff] = lam2;

    icoeff += 1;
    iarg += 6;
  }
  
  // Error checking for the first set of required input arguments
  if (seed <= 0) error->all(FLERR,"Illegal fix gle/pair/aux command");
  if (t_target < 0)
    error->all(FLERR,"Fix gle/pair/aux temperature must be >= 0");
  
  // allocate memory for Prony series extended variables
  q_aux = NULL;
  grow_arrays(atom->nlocal);
  for (int i=0; i<atom->nlocal; i++) {
    for (int k=0; k<aux_terms; k++) {
      for (int j=0; j<atom->nlocal; j++) {
	q_aux[i][atom->nlocal*k+j]=0;
      }
    }
  }

  // initialize Marsaglia RNG with processor-unique seed
  random = new RanMars(lmp,seed + comm->me);
  
  // allocate and create self and cross random numbers
  memory->create(ran_self, atom->nlocal, atom->nlocal*aux_terms,"gle/pair/aux:ran_self");
  memory->create(ran_cross, atom->nlocal, atom->nlocal*aux_terms,"gle/pair/aux:ran_cross");
  for (int i=0; i<atom->nlocal; i++) {
    for (int k=0; k<aux_terms; k++) {
      for (int j=0; j<atom->nlocal; j++) {
	ran_self[i][atom->nlocal*k+j]=random->gaussian();
	ran_cross[i][atom->nlocal*k+j]=random->gaussian();
      }
    }
  }
  
  // create cholesky decomposition
  memory->create(chol_decomp, atom->nlocal, atom->nlocal*aux_terms,"gle/pair/aux:chol_decomp");
  chol_decomp[0][0] = 1.0;
  chol_decomp[0][1] = 0.5;
  chol_decomp[0][2] = 0.4;
  chol_decomp[1][0] = 0.0;
  chol_decomp[1][1] = 0.866025;
  chol_decomp[1][2] = 0.11547;
  chol_decomp[2][0] = 0.0;
  chol_decomp[2][1] = 0.0;
  chol_decomp[2][2] = 0.909212;
  
  memory->create(v_step, atom->nlocal, "gle/pair/aux:v_step");
  memory->create(f_step, atom->nlocal, "gle/pair/aux:v_step");


}

/* ----------------------------------------------------------------------
   Destroys memory allocated by the method
------------------------------------------------------------------------- */

FixGLENoise::~FixGLENoise()
{
  delete random;
  memory->destroy(aux_c11);
  memory->destroy(aux_lam1);
  memory->destroy(aux_c21);
  memory->destroy(aux_c22);
  memory->destroy(aux_c24);
  memory->destroy(aux_lam2);
  memory->destroy(q_aux);
  memory->destroy(ran_self);
  memory->destroy(ran_cross);

}

/* ----------------------------------------------------------------------
   Specifies when the fix is called during the timestep
------------------------------------------------------------------------- */

int FixGLENoise::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ----------------------------------------------------------------------
   Initialize the method parameters before a run
------------------------------------------------------------------------- */

void FixGLENoise::init()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}

/* ----------------------------------------------------------------------
   First half of a timestep (V^{n} -> V^{n+1/2}; X^{n} -> X^{n+1})
------------------------------------------------------------------------- */

void FixGLENoise::initial_integrate(int vflag)
{
  double dtfm;
  double ftm2v = force->ftm2v;

  double v_save[atom->nlocal];
  double meff;
  double theta_vs, alpha_vs, theta_vc, alpha_vc;
  double theta_q1, theta_q2, alpha_q11, alpha_q21, alpha_q22, alpha_q24;

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

  // update random numbers
  for (int i=0; i<atom->nlocal; i++) {
    for (int k=0; k<aux_terms; k++) {
      for (int j=0; j<atom->nlocal; j++) {
	ran_self[i][atom->nlocal*k+j]=random->gaussian();
	ran_cross[i][atom->nlocal*k+j]=random->gaussian();
      }
    }
  }
  
  // Advance Q by dt
  for (int i = 0; i < nlocal; i++) {
    meff = mass[type[i]];
    if (mask[i] & groupbit) {
      for (int k = 0; k < aux_terms; k++) {
	theta_q1 = 1-update->dt/aux_lam1[k];
	q_aux[i][k*atom->nlocal+i] *= theta_q1;
	
	double sum = 0.0;
	for (int l=0; l<nlocal; l++) {
	  sum += ran_self[0][l]*chol_decomp[l][i];
	}
	q_aux[i][k*atom->nlocal+i] += sqrt(2*update->dt)*sum;
	
      }
    }
  }
  
  
}

/* ----------------------------------------------------------------------
   Second half of a timestep (V^{n+1/2} -> V^{n+1})
------------------------------------------------------------------------- */

void FixGLENoise::final_integrate()
{

}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixGLENoise::compute_vector(int n)
{
  int nlocal = atom->nlocal;
  int aux = n%nlocal;
  int atom = (n-aux)/nlocal;
  return q_aux[atom][aux];
}


/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixGLENoise::memory_usage()
{
  double bytes = atom->nlocal*atom->nlocal*aux_terms*sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixGLENoise::grow_arrays(int nmax)
{
  memory->grow(q_aux, nmax, atom->nlocal*aux_terms,"gld:q_aux");
}

