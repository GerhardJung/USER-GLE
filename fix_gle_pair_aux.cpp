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

  int narg_min = 6;
  if (narg < narg_min) error->all(FLERR,"Illegal fix gle/pair/aux command");

  t_target = force->numeric(FLERR,arg[3]);
  
  int seed = force->inumeric(FLERR,arg[4]);
  
  aux_terms = force->inumeric(FLERR,arg[5]);

  // allocate memory and read-in auxiliary series coefficients
  if (aux_terms < 0)
    error->all(FLERR,"Fix gle/pair/aux terms must be > 0");
  if (narg - narg_min < 3*(aux_terms)+2 )
    error->all(FLERR,"Fix gle/pair/aux needs more auxiliary variable coefficients");
  memory->create(aux_c_self, aux_terms, "gle/pair/aux:aux_c_self");
  memory->create(aux_c_cross, aux_terms, "gle/pair/aux:aux_c_cross");
  memory->create(aux_lam, aux_terms, "gle/pair/aux:aux_lam");
  
  global_freq = 1;
  nevery = 1;
  peratom_freq = 1;
  vector_flag = 1;
  size_vector = 2+aux_terms*atom->nlocal+atom->nlocal;
    
  int iarg = narg_min;
  aux_a_self = force->numeric(FLERR,arg[iarg]);
  aux_a_cross = force->numeric(FLERR,arg[iarg+1]);
  iarg +=2;
  int icoeff = 0;
  while (iarg < narg && icoeff < aux_terms) {
    double c_self = force->numeric(FLERR,arg[iarg]);
    double c_cross = force->numeric(FLERR,arg[iarg+1]);
    double lam = force->numeric(FLERR,arg[iarg+2]);

    if (c_self < 0 || c_cross < 0)
      error->all(FLERR,"Fix gle/pair/aux c coefficients must be >= 0");
    if (lam  <= 0)
      error->all(FLERR,"Fix gle/pair/aux lam coefficients must be > 0");

    // All atom types to have the same Prony series
    aux_c_self[icoeff] = c_self;
    aux_c_cross[icoeff] = c_cross;
    aux_lam[icoeff] = lam;

    icoeff += 1;
    iarg += 3;
  }
  
  // Error checking for the first set of required input arguments
  if (seed <= 0) error->all(FLERR,"Illegal fix gle/pair/aux command");
  if (t_target < 0)
    error->all(FLERR,"Fix gle/pair/aux temperature must be >= 0");
  
  // allocate memory for Prony series extended variables
  q_aux = NULL;
  grow_arrays(atom->nlocal);
  memory->create(q_ran, atom->nlocal,atom->nlocal*aux_terms, "gle/pair/aux:q_ran");
  memory->create(q_save, atom->nlocal,atom->nlocal*aux_terms, "gle/pair/aux:q_save");
  memory->create(q_B, atom->nlocal*atom->nlocal,atom->nlocal*atom->nlocal*aux_terms, "gle/pair/aux:q_B");
  
  q_B[0][0]=2.44949;
  q_B[0][1]=0.0;
  q_B[0][2]=0.0;
  q_B[0][3]=0.0;
  
  q_B[1][0]=1.22474;
  q_B[1][1]=1.58114;
  q_B[1][2]=0;
  q_B[1][3]=0;
  
  q_B[2][0]=0.0;
  q_B[2][1]=-1.26491;
  q_B[2][2]=1.54919;
  q_B[2][3]=0.0;
  
  q_B[3][0]=0.0;
  q_B[3][1]=0.0;
  q_B[3][2]=1.93649;
  q_B[3][3]=1.5;
  
  /*q_B[0][0]=2.44949;
  q_B[0][1]=0.0;
  q_B[0][2]=0.0;
  q_B[0][3]=0.0;
  
  q_B[1][0]=1.22474;
  q_B[1][1]=1.58114;
  q_B[1][2]=0.0;
  q_B[1][3]=0.0;
  
  q_B[2][0]=0.0;
  q_B[2][1]=0.0;
  q_B[2][2]=2.0;
  q_B[2][3]=0.0;
  
  q_B[3][0]=0.0;
  q_B[3][1]=0.0;
  q_B[3][2]=1.5;
  q_B[3][3]=1.93649;*/

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
  memory->destroy(aux_c_self);
  memory->destroy(aux_lam);
  memory->destroy(aux_c_cross);
  memory->destroy(q_aux);

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
      
      for (int k = 0; k < aux_terms; k++) {
	v[i][0] -= (+1)*dtfm *q_aux[i][i]; //ps11,ps22
	for (int j=0; j< nlocal; j++) {
	  if (j!=i) {
	    v[i][0] -= (-1)*dtfm *q_aux[i][j]; //ps21,ps12
	  }
	}
      }
    }
  }

  // Advance X by dt
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      x[i][0] += dtv * v[i][0];
    }
  }

  for (int i = 0; i < nlocal; i++) {
    for (int k = 0; k < aux_terms; k++) {
      for (int j = 0; j < nlocal; j++) {
	q_ran[i][j] = random->gaussian();
	q_save[i][j] = q_aux[i][j];
      }
    } 
  }
  
  // Advance Q by dt
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      for (int k = 0; k < aux_terms; k++) {
	
	//integrate self auxilliary variable
	theta_qss = 1-update->dt*(1);
	q_aux[i][i] *= theta_qss;	//a
	for (int j = 0; j < nlocal; j++) {
	  if (j!=i) {
	    theta_qsc = 1-update->dt*(2);
	    q_aux[i][j] *= (theta_qsc); //b
	  }
	}
	
	//include momenta
	theta_qsps = 1-update->dt*(-2.0);
	q_aux[i][i] -= (1-theta_qsps)*v[i][0];
	for (int j = 0; j < nlocal; j++) {
	  if (j!=i) {
	    theta_qspc = 1-update->dt*(-0.5);
	    q_aux[i][j] -= (1-theta_qspc)*v[j][0];
	  }
	}
	
	// include noise
	for (int j = 0; j < nlocal; j++) {
	  for (int i1 = 0; i1 < nlocal; i1++) {
	    for (int j1 = 0; j1 < nlocal; j1++) {
	      q_aux[i][j] += sqrt(update->dt)*q_B[i*nlocal+j][i1*nlocal+j1]*q_ran[i1][j1];
	    }
	  }
	}
	
      }
    }
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
      
      for (int k = 0; k < aux_terms; k++) {
	v[i][0] -= (+1)*dtfm *q_aux[i][i]; //ps11,ps22
	for (int j=0; j< nlocal; j++) {
	  if (j!=i) {
	    v[i][0] -= (-1)*dtfm *q_aux[i][j]; //ps21,ps12
	  }
	}
      }
    }
  }
 
  
  for (int i = 0; i < nlocal; i++) {
    f[i][0]=(v[i][0]-v_step[i])/update->dt;
  }

}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixGLEPairAux::compute_vector(int n)
{
  int nlocal = atom->nlocal;
  int j = n%nlocal;
  int i = (n - j)/nlocal;
  
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
  memory->grow(q_aux, atom->nlocal,atom->nlocal* aux_terms,"gld:q_aux");
}

/* ----------------------------------------------------------------------
   Initializes the extended variables to equilibrium distribution
   at t_start.
------------------------------------------------------------------------- */

void FixGLEPairAux::init_q_aux()
{
  int icoeff;
  double eq_sdev=0.0;

  // set kT to the temperature in mvvv units
  double kT = (force->boltz)*t_target/(force->mvv2e);
  double scale = sqrt(kT)/(force->ftm2v);

  for (int i = 0; i < atom->nlocal; i++) {
    if (atom->mask[i] & groupbit) {
      icoeff = 0;
      for (int k = 0; k < aux_terms; k++) {
        for (int j = 0; j < atom->nlocal; j++) {
	  q_aux[i][k*atom->nlocal+j] = 0;

	}
      }
    }
  }

  return;
}
