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

  // initialize Marsaglia RNG with processor-unique seed
  random = new RanMars(lmp,seed + comm->me);

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
      meff = 1.5*mass[type[i]];
      theta_vs = (1 - update->dt*aux_a_self/meff/2.0);
      alpha_vs = sqrt(update->dt)/mass[type[i]]/2.0*(sqrt(aux_a_self-aux_a_cross)+sqrt(aux_a_self+aux_a_cross));
      theta_vc = (1 - update->dt*aux_a_cross/meff/2.0);
      alpha_vc = sqrt(update->dt)/mass[type[i]]/2.0*(sqrt(aux_a_self-aux_a_cross)-sqrt(aux_a_self+aux_a_cross));
      
      dtfm = dtf / mass[type[i]];
      
      //if (i==0) printf ("start: %f\n",v[i][0]);
      
      v[i][0] *= theta_vs;
      
      //if (i==0) printf ("inst. friction: %f\n",v[i][0]);
      
      v[i][0] += alpha_vs * sqrt(kT/2.0) * random->gaussian();
      
      //if (i==0) printf ("noise: %f\n",v[i][0]);
      
      for (int j=0; j< nlocal; j++) {
	if (j!=i) {
	  v[i][0] -= (1-theta_vc)*v_save[j];
	  //if (i==0) printf ("neigh. fric.: %f\n",v[i][0]);
	  v[i][0] += alpha_vc*sqrt(kT/2.0) * random->gaussian();
	  //if (i==0) printf ("neigh. noise.: %f\n",v[i][0]);
	}
      }
      v[i][0] += dtfm * f[i][0];
      //if (i==0) printf ("cons. force: %f\n",v[i][0]);
      for (int k = 0; k < aux_terms; k++) {
	v[i][0] -= dtfm * aux_c_self[k]*q_aux[i][k];
	//if (i==0) printf ("q fric.: %f\n",v[i][0]);
	for (int j=0; j< nlocal; j++) {
	  if (j!=i) {
	    v[i][0] += dtfm * aux_c_cross[k]*q_aux[j][k];
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
    meff = mass[type[i]];
    if (mask[i] & groupbit) {
      for (int k = 0; k < aux_terms; k++) {
	theta_q = 1-update->dt/aux_lam[k];
        alpha_q = sqrt(update->dt/aux_lam[k])*(1.0/sqrt(aux_c_self[k]-aux_c_cross[k])+1.0/sqrt(aux_c_self[k]+aux_c_cross[k]));
	alpha_qc = sqrt(update->dt/aux_lam[k])*(1.0/sqrt(aux_c_self[k]-aux_c_cross[k])-1.0/sqrt(aux_c_self[k]+aux_c_cross[k]));

        q_aux[i][k] *= theta_q;
        q_aux[i][k] += (1-theta_q)*mass[type[i]]/meff*aux_lam[k]*v[i][0];
        q_aux[i][k] += alpha_q*sqrt(kT/2.0) * random->gaussian();
	for (int j=0; j< nlocal; j++) {
	  if (j!=i) {
	    q_aux[i][k] += alpha_qc * sqrt(kT/2.0) * random->gaussian();
	  }
	}
	//if (i==0) printf ("v_q^2: %f\n",q_aux[i][k]*q_aux[i][k]);
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
  for ( int i=0; i< nlocal; i++) v_save[i] = v[i][0];

  // Advance V by dt/2
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      // calculate integration constants
      meff = mass[type[i]];
      theta_vs = (1 - update->dt*aux_a_self/meff/2.0);
      alpha_vs = sqrt(update->dt)/mass[type[i]]/2.0*(sqrt(aux_a_self-aux_a_cross)+sqrt(aux_a_self+aux_a_cross));
      theta_vc = (1 - update->dt*aux_a_cross/meff/2.0);
      alpha_vc = sqrt(update->dt)/mass[type[i]]/2.0*(sqrt(aux_a_self-aux_a_cross)-sqrt(aux_a_self+aux_a_cross));
      
      dtfm = dtf / mass[type[i]];
      
      v[i][0] *= theta_vs;
      v[i][0] += alpha_vs * sqrt(kT/2.0) * random->gaussian();
      for (int j=0; j< nlocal; j++) {
	if (j!=i) {
	  v[i][0] -= (1-theta_vc)*v_save[j];
	  v[i][0] += alpha_vc*sqrt(kT/2.0) * random->gaussian();
	}
      }
      v[i][0] += dtfm * f[i][0];
      for (int k = 0; k < aux_terms; k++) {
	v[i][0] -= dtfm * aux_c_self[k]*q_aux[i][k];
	for (int j=0; j< nlocal; j++) {
	  if (j!=i) {
	    v[i][0] += dtfm * aux_c_cross[k]*q_aux[j][k];
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
  return q_aux[atom][aux];
}


/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixGLEPairAux::memory_usage()
{
  double bytes = atom->nlocal*aux_terms*sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixGLEPairAux::grow_arrays(int nmax)
{
  memory->grow(q_aux, atom->nlocal, aux_terms,"gld:q_aux");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixGLEPairAux::copy_arrays(int i, int j, int delflag)
{
  for (int k = 0; k < aux_terms; k++) {
    q_aux[j][k] = q_aux[i][k];
  }
}

/* ----------------------------------------------------------------------
   Pack extended variables assoc. w/ atom i into buffer for exchange
   with another processor
------------------------------------------------------------------------- */

int FixGLEPairAux::pack_exchange(int i, double *buf)
{
  int m = 0;
  for (int k = 0; k < aux_terms; k++) {
    buf[m++] = q_aux[i][k];
  }
  return m;
}

/* ----------------------------------------------------------------------
   Unpack extended variables from exchange with another processor
------------------------------------------------------------------------- */

int FixGLEPairAux::unpack_exchange(int nlocal, double *buf)
{
  int m = 0;
  for (int k = 0; k < aux_terms; k++) {
    q_aux[nlocal][k] = buf[m++];
  }
  return m;
}


/* ----------------------------------------------------------------------
   Pack extended variables assoc. w/ atom i into buffer for
   writing to a restart file
------------------------------------------------------------------------- */

int FixGLEPairAux::pack_restart(int i, double *buf)
{
  int m = 0;
  buf[m++] = aux_terms + 1;
  for (int k = 0; k < aux_terms; k++)
  {
    buf[m++] = q_aux[i][k];
  }
  return m;
}

/* ----------------------------------------------------------------------
   Unpack extended variables to restart the fix from a restart file
------------------------------------------------------------------------- */

void FixGLEPairAux::unpack_restart(int nlocal, int nth)
{
  double **extra = atom->extra;

  // skip to the nth set of extended variables

  int m = 0;
  for (int i = 0; i< nth; i++) m += static_cast<int> (extra[nlocal][m]);
  m++;

  for (int k = 0; k < aux_terms; k=k+3)
  {
    q_aux[nlocal][k] = extra[nlocal][m++];
  }
}

/* ----------------------------------------------------------------------
   Returns the number of items in atomic restart data associated with
   local atom nlocal.  Used in determining the total extra data stored by
   fixes on a given processor.
------------------------------------------------------------------------- */

int FixGLEPairAux::size_restart(int nlocal)
{
  return aux_terms+1;
}

/* ----------------------------------------------------------------------
   Returns the maximum number of items in atomic restart data
   Called in Modify::restart for peratom restart.
------------------------------------------------------------------------- */

int FixGLEPairAux::maxsize_restart()
{
  return aux_terms+1;
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
        eq_sdev = scale*sqrt(aux_c_self[icoeff]/aux_lam[icoeff]);
        q_aux[i][k] = 0;

        icoeff += 1;
      }
    }
  }

  return;
}
