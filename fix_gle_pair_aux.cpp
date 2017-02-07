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
  if (aux_terms <= 0)
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
  size_vector = aux_terms*atom->nlocal;
    
  int iarg = narg_min;
  aux_a_self = force->numeric(FLERR,arg[iarg]);
  aux_a_cross = force->numeric(FLERR,arg[iarg+1]);
  aux_b_self = force->numeric(FLERR,arg[iarg+2]);
  iarg +=3;
  int icoeff = 0;
  while (iarg < narg && icoeff < aux_terms) {
    double c_self = force->numeric(FLERR,arg[iarg]);
    double c_cross = force->numeric(FLERR,arg[iarg+1]);
    double lam = force->numeric(FLERR,arg[iarg+2]);

    if (c_self < 0)
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
  
  //printf("a_s %f a_c %f b %f\n",aux_a_self, aux_a_cross, aux_b_self );
  
  // Error checking for the first set of required input arguments
  if (seed <= 0) error->all(FLERR,"Illegal fix gle/pair/aux command");
  if (t_target < 0)
    error->all(FLERR,"Fix gle/pair/aux temperature must be >= 0");
  
  // allocate memory for Prony series extended variables
  z_aux = NULL;
  f_aux = NULL;
  grow_arrays(atom->nlocal);

  // initialize Marsaglia RNG with processor-unique seed
  random = new RanMars(lmp,seed + comm->me);

  // initialize the extended variables
  init_zf_aux();
  
  //initialize memory to save random numbers
  memory->create(ran_save, atom->nlocal, "gle/pair/aux:ran_self");
  for (int i=0; i<atom->nlocal; i++) {
    ran_save[i] = random->gaussian();
  }

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
  memory->destroy(z_aux);
  memory->destroy(f_aux);
  memory->destroy(ran_save);

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
  double dtfm,dtfmeff;
  double ftm2v = force->ftm2v;

  double v_save[atom->nlocal];
  double meff;
  double theta_vs, alpha_vs, theta_vc, alpha_vc;
  double theta_q, alpha_q, alpha_qc;

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

  // Advance V by dt/2
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      // calculate integration constants
      meff = mass[type[i]]+aux_b_self;
      theta_vs = (1 - update->dt*aux_a_self/meff/2.0);
      alpha_vs = sqrt(update->dt*aux_a_self/2.0)/meff;
      theta_vc = (1 - update->dt*aux_a_cross/meff/2.0);
      
      dtfm = dtf / mass[type[i]];
      dtfmeff = dtf / meff;
      
      //if (i==0) printf ("start: %f\n",v[i][0]);
      
      v[i][0] *= theta_vs;
      
      //if (i==0) printf ("inst. friction: %f\n",v[i][0]);
      
      v[i][0] += alpha_vs * sqrt(kT) * ran_save[i];
      
      //if (i==0) printf ("noise: %f\n",v[i][0]);
      
      for (int j=0; j< nlocal; j++) {
	if (j!=i) {
	  v[i][0] -= (1-theta_vc)*v_save[j];
	  //if (i==0) printf ("neigh. fric.: %f\n",v[i][0]);
	}
      }
      v[i][0] += dtfmeff * f[i][0];
      //if (i==0) printf ("cons. force: %f\n",v[i][0]);
      for (int k = 0; k < aux_terms; k++) {
	v[i][0] += dtfm * aux_c_self[k]*(f_aux[i][k]-z_aux[i][k]);
	//if (i==0) printf ("q fric.: %f\n",v[i][0]);
	for (int j=0; j< nlocal; j++) {
	  if (j!=i) {
	    v[i][0] -= dtfm * aux_c_cross[k]*z_aux[j][k];
	    //if (i==0) printf ("neigh. q fric.: %f\n",v[i][0]);
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

  // Advance Q by dt
  for (int i = 0; i < nlocal; i++) {
    meff = mass[type[i]]+aux_b_self;
    if (mask[i] & groupbit) {
      for (int k = 0; k < aux_terms; k++) {
	theta_q = 1-update->dt/aux_lam[k];
        alpha_q = sqrt(update->dt/aux_lam[k]/aux_c_self[k]*2.0);
	
	// update Z(t)
        z_aux[i][k] *= theta_q;
        z_aux[i][k] += (1-theta_q)*mass[type[i]]/meff*aux_lam[k]*v[i][0];
	
	// update F(t)
	f_aux[i][k] *= theta_q;
        z_aux[i][k] += alpha_q*sqrt(kT) *mass[type[i]]/meff* random->gaussian();

      }
    }
  }
  
  
}

/* ----------------------------------------------------------------------
   Second half of a timestep (V^{n+1/2} -> V^{n+1})
------------------------------------------------------------------------- */

void FixGLEPairAux::final_integrate()
{
  double dtfm,dtfmeff;
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
  for ( int i=0; i< nlocal; i++) v_save[i] = v[i][0];

  // Advance V by dt/2
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      // calculate integration constants
      meff = mass[type[i]]+aux_b_self;
      theta_vs = (1 - update->dt*aux_a_self/meff/2.0);
      alpha_vs = sqrt(update->dt*aux_a_self/2.0)/meff;
      theta_vc = (1 - update->dt*aux_a_cross/meff/2.0);
      
      dtfm = dtf / mass[type[i]];
      dtfmeff = dtf / meff;
      
      //if (i==0) printf ("start: %f\n",v[i][0]);
      
      v[i][0] *= theta_vs;
      
      //if (i==0) printf ("inst. friction: %f\n",v[i][0]);
      ran_save[i] = random->gaussian();
      v[i][0] += alpha_vs * sqrt(kT) * ran_save[i];
      
      //if (i==0) printf ("noise: %f\n",v[i][0]);
      
      for (int j=0; j< nlocal; j++) {
	if (j!=i) {
	  v[i][0] -= (1-theta_vc)*v_save[j];
	  //if (i==0) printf ("neigh. fric.: %f\n",v[i][0]);
	}
      }
      v[i][0] += dtfmeff * f[i][0];
      //if (i==0) printf ("cons. force: %f\n",v[i][0]);
      for (int k = 0; k < aux_terms; k++) {
	v[i][0] += dtfm * aux_c_self[k]*(f_aux[i][k]-z_aux[i][k]);
	//if (i==0) printf ("q fric.: %f\n",v[i][0]);
	for (int j=0; j< nlocal; j++) {
	  if (j!=i) {
	    v[i][0] -= dtfm * aux_c_cross[k]*z_aux[j][k];
	    //if (i==0) printf ("neigh. q fric.: %f\n",v[i][0]);
	  }
	}
      }
    }
  }

}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixGLEPairAux::compute_vector(int n)
{
  int nlocal = atom->nlocal;
  int aux = n%nlocal;
  int atom = (n-aux)/nlocal;
  return z_aux[atom][aux];
}


/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixGLEPairAux::memory_usage()
{
  double bytes = 2*atom->nlocal*aux_terms*sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixGLEPairAux::grow_arrays(int nmax)
{
  memory->grow(z_aux, atom->nlocal, aux_terms,"gld:z_aux");
  memory->grow(f_aux, atom->nlocal, aux_terms,"gld:f_aux");
}

/* ----------------------------------------------------------------------
   Initializes the extended variables to equilibrium distribution
   at t_start.
------------------------------------------------------------------------- */

void FixGLEPairAux::init_zf_aux()
{

  for (int i = 0; i < atom->nlocal; i++) {
    if (atom->mask[i] & groupbit) {
      for (int k = 0; k < aux_terms; k++) {
        z_aux[i][k] = 0;
	f_aux[i][k] = 0;
      }
    }
  }

  return;
}
