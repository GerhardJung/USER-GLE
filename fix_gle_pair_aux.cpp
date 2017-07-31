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
#include <iostream>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;
using namespace Eigen;


/* ----------------------------------------------------------------------
   Parses parameters passed to the method, allocates some memory
------------------------------------------------------------------------- */

FixGLEPairAux::FixGLEPairAux(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  
  global_freq = 1;
  nevery = 1;
  peratom_freq = 1;
  vector_flag = 1;

  int narg_min = 5;
  if (narg < narg_min) error->all(FLERR,"Illegal fix gle/pair/aux command");

  t_target = force->numeric(FLERR,arg[3]);
  
  int seed = force->inumeric(FLERR,arg[4]);
  
  // read number of auxilliary variables
  input = fopen(arg[5],"r");
  if (input == NULL) {
    char str[128];
    sprintf(str,"Cannot open fix gle/pair/aux file %s",arg[5]);
    error->one(FLERR,str);
  }
  
  // Error checking for the first set of required input arguments
  if (seed <= 0) error->all(FLERR,"Illegal fix gle/pair/aux command");
  if (t_target < 0)
    error->all(FLERR,"Fix gle/pair/aux temperature must be >= 0");
  
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
  memory->destroy(q_ran);
  memory->destroy(q_save);
  memory->destroy(q_B);
  memory->destroy(q_As);
  memory->destroy(q_Ac);
  memory->destroy(q_ints);
  memory->destroy(q_intv);

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
  int s,c,n,m;

  // update v and x of atoms in group
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  
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
      for (int j = 0; j < nlocal; j++) {
	for (int k = 0; k < aux_terms; k++) {
	  v[i][0] -= lltA[k](i,j)*dtfm *q_aux[j][2*k+1];
	}
      }

    }
  }

  // Advance X by dt
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      //x[i][0] += dtv * v[i][0];
    }
  }

  for (int i = 0; i < nlocal; i++) {
    for (int k = 0; k < aux_terms; k++) {
      q_ran[i][2*k] = random->gaussian();
      q_save[i][2*k] = q_aux[i][2*k];
      q_ran[i][2*k+1] = random->gaussian();
      q_save[i][2*k+1] = q_aux[i][2*k+1];
    } 
  }
  
  // Advance Q by dt
  for (int i = 0; i < nlocal; i++) {
    for (int k = 0; k < aux_terms; k++) {
      // update aux_var, self and cross
      q_aux[i][2*k] *= q_ints[2*k];  
      q_aux[i][2*k] += q_ints[2*k+1]*q_save[i][2*k+1];  
      q_aux[i][2*k+1] *= q_ints[2*k];  
      q_aux[i][2*k+1] -= q_ints[2*k+1]*q_save[i][2*k];  
      
      // update momenta contribution (for memory)
      for (int j = 0; j < nlocal; j++) {
	if (mask[j] & groupbit) {
	  q_aux[i][2*k] += q_intv[2*k]*lltA[k](j,i)*v[j][0]; 
	  q_aux[i][2*k+1] += q_intv[2*k+1]*lltA[k](j,i)*v[j][0]; 
	}
      }
    
      // update noise
      q_aux[i][2*k] += q_B[k]*q_ran[i][2*k];
      q_aux[i][2*k+1] += q_B[k]*q_ran[i][2*k+1];
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

  // update v and x of atoms in group
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  
  // save velocity for cross time integration
  for ( int i=0; i< nlocal; i++) f_step[i] = f[i][0];

  // Advance V by dt/2
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      // calculate integration constants
      meff = mass[type[i]];   
      dtfm = dtf / mass[type[i]];
      //v[i][0] += dtfm * f_step[i];
      for (int j = 0; j < nlocal; j++) {
	for (int k = 0; k < aux_terms; k++) {
	  v[i][0] -= lltA[k](i,j)*dtfm *q_aux[j][2*k+1];
	}
      }

    }
  }
 
  for (int i = 0; i < nlocal; i++) {
    f[i][0]=0.0;
    for (int j = 0; j < nlocal; j++) {
      for (int k = 0; k < aux_terms; k++) {
	f[i][0]+=lltA[k](i,j) *q_aux[j][2*k+1];
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
  int j = n%(2*nlocal);
  int i = (n - j)/(2*nlocal);
  
  return q_aux[i][j];
}


/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixGLEPairAux::memory_usage()
{
  double bytes = atom->nlocal*2*aux_terms*sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixGLEPairAux::grow_arrays(int nmax)
{
  memory->grow(q_aux, nmax,aux_terms*2,"gle/pair/aux:q_aux");
}

/* ----------------------------------------------------------------------
   read in coefficients
------------------------------------------------------------------------- */
void FixGLEPairAux::read_coef_aux()
{

  fscanf(input,"Auxilliary variables\n");
  fscanf(input,"N\n%d\n",&aux_terms);
  //printf("%d\n",aux_terms);

  if (aux_terms < 0)
    error->all(FLERR,"Fix gle/pair/aux terms must be > 0");
  size_vector = 2*aux_terms*atom->nlocal;
  
  // allocate memory
  memory->create(q_s, aux_terms*2, "gle/pair/aux:q_s");
  memory->create(q_As, aux_terms, "gle/pair/aux:q_As");
  memory->create(q_Ac, aux_terms, "gle/pair/aux:q_Ac");
  memory->create(q_B, aux_terms, "gle/pair/aux:q_B");
  
  // read time constants
  fscanf(input,"Time constants\n");
  fscanf(input,"q r\n");
  for (int k=0; k<aux_terms; k++) fscanf(input,"%lf %lf\n",&q_s[2*k],&q_s[2*k+1]);
  //for (int k=0; k<aux_terms; k++) printf("%lf %lf\n",q_s[2*k],q_s[2*k+1]);
  
  // read self amplitudes
  fscanf(input,"Self amplitudes\n");
  for (int k=0; k<aux_terms; k++) fscanf(input,"%lf ",&q_As[k]);
  //for (int k=0; k<aux_terms; k++) printf("%lf ",q_As[k]);
  //printf("\n");
  
  // read cross amplitudes
  fscanf(input,"\nCross amplitudes\n");
  for (int k=0; k<aux_terms; k++) fscanf(input,"%lf ",&q_Ac[k]);
  //for (int k=0; k<aux_terms; k++) printf("%lf ",q_Ac[k]);
  //printf("\n");

  fclose(input);
  
  // write out input function
  FILE * output;
  output = fopen("memory_output.cor","w");
  double ts;
  double sum_self,sum_cross;
  for (int t=0; t<10000; t++) {
    sum_self = 0.0;
    sum_cross = 0.0;
    ts = t*update->dt;
    for (int k=0; k<aux_terms; k++) {
      sum_self += q_As[k]*exp(-q_s[2*k]*ts)*cos(q_s[2*k+1]*ts);
      sum_cross += q_Ac[k]*exp(-q_s[2*k]*ts)*cos(q_s[2*k+1]*ts);
    }
    fprintf(output,"%f %f %f\n",ts,sum_self,sum_cross);
  }
  fclose(output);
  
}

/* ----------------------------------------------------------------------
   Initializes the extended variables to equilibrium distribution
   at t_start.
------------------------------------------------------------------------- */

void FixGLEPairAux::init_q_aux()
{
  
  int nlocal = atom->nlocal;
  
  // allocate memory
  grow_arrays(atom->nlocal);
  memory->create(q_ran, atom->nlocal,aux_terms*2, "gle/pair/aux:q_ran");
  memory->create(q_save, atom->nlocal,aux_terms*2, "gle/pair/aux:q_save");
  memory->create(q_ints, aux_terms*2, "gle/pair/aux:q_ints");
  memory->create(q_intv, aux_terms*2, "gle/pair/aux:q_intv");

  // initialize auxilliary variables
  for (int i = 0; i < atom->nlocal; i++) {
    for (int k = 0; k < aux_terms; k++) {
      for (int n=0; n<2; n++) {
	q_aux[i][2*k+n] = 0.0;
      }
    }
  }
  
  // calculate integration constants
  // exp(-dt*Ass)
  for (int k = 0; k < aux_terms; k++) {
    q_ints[2*k]=exp(-update->dt*q_s[2*k])*cos(update->dt*q_s[2*k+1]);
    q_ints[2*k+1]=exp(-update->dt*q_s[2*k])*sin(update->dt*q_s[2*k+1]);
  }
  
  // calculate integration constants
  // Ass^(-1)*(1-exp(-dt*Ass))
  for (int k = 0; k < aux_terms; k++) {
    q_intv[2*k]= -q_s[2*k+1]*(1.0-q_ints[2*k])/(q_s[2*k]*q_s[2*k]+q_s[2*k+1]*q_s[2*k+1])+q_s[2*k]*q_ints[2*k+1]/(q_s[2*k]*q_s[2*k]+q_s[2*k+1]*q_s[2*k+1]);
    q_intv[2*k+1]=q_s[2*k]*(1.0-q_ints[2*k])/(q_s[2*k]*q_s[2*k]+q_s[2*k+1]*q_s[2*k+1])+q_s[2*k+1]*q_ints[2*k+1]/(q_s[2*k]*q_s[2*k]+q_s[2*k+1]*q_s[2*k+1]);
    printf("%f %f\n",q_intv[2*k],q_intv[2*k+1]);
  }
  
  // CholDecomp[Css-exp(-dt*Ass)*Css*exp(-dt*Ass^T)] (hard coded)
  for (int k = 0; k < aux_terms; k++) {
    q_B[k]=sqrt(1-exp(-2.0*update->dt*q_s[2*k]));
  }
  
  // CholDecomp to determine Aps (on the fly)
  // initialize matrices
  A = new MatrixXd[aux_terms];
  lltA = new MatrixXd[aux_terms];
  for (int k = 0; k < aux_terms; k++) {
    A[k] = MatrixXd::Zero(nlocal,nlocal);
  }
  for (int k = 0; k < aux_terms; k++) {
    for (int i = 0; i < nlocal; i++) {
      for (int j = 0; j < nlocal; j++) {
	if (i==j) {
	  A[k](i,j)=q_As[k];
	} else {
	  A[k](i,j)=q_Ac[k];
	}
      }
    }
  }
  
  // perform cholesky decomposition
  for (int k = 0; k < aux_terms; k++) {
    //cout << "The matrix A is" << endl << A[k] << endl;
    LLT<MatrixXd> lltA_comp(A[k]); // compute the Cholesky decomposition of A
    lltA[k] = lltA_comp.matrixL(); // retrieve factor L  in the decomposition
    // The previous two lines can also be written as "L = A.llt().matrixL()"
    //cout << "The Cholesky factor L is" << endl << lltA[k] << endl;
    //cout << "To check this, let us compute L * L.transpose()" << endl;
    //cout << lltA[k] * lltA[k].transpose() << endl;
    //cout << "This should equal the matrix A" << endl;
    //printf("%f\n",lltA[k](0,0));
  }

  
  
}
