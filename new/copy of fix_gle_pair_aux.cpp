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
#include "comm.h"
#include "input.h"
#include "variable.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"
#include "group.h"
#include "domain.h"

#include "mpfit.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;
using namespace Eigen;

#define MAXLINE 1024

enum{TIME,FIT};

// definitions for fitting procedure
struct fit_struct {
  double *x;
  double *y;
  double *ey;
};
int expfunc(int m, int n, double *p, double *dy, double **dvec, void *fit);
void printresult(double *x, mp_result *result) ;


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
  
  MPI_Comm_rank(world,&me);

  int narg_min = 5;
  if (narg < narg_min) error->all(FLERR,"Illegal fix gle/pair/aux command");

  t_target = force->numeric(FLERR,arg[3]);
  
  int seed = force->inumeric(FLERR,arg[4]);
  
  // read input file
  input = fopen(arg[5],"r");
  if (input == NULL) {
    char str[128];
    sprintf(str,"Cannot open fix gle/pair/aux file %s",arg[5]);
    error->one(FLERR,str);
  }
  keyword = arg[6];
  
  // Error checking for the first set of required input arguments
  if (seed <= 0) error->all(FLERR,"Illegal fix gle/pair/aux command");
  if (t_target < 0)
    error->all(FLERR,"Fix gle/pair/aux temperature must be >= 0");
  
  // Timing
  time_read = 0.0;
  time_init = 0.0;
  time_int_rel1 = 0.0;
  time_dist_update = 0.0;
  time_int_aux = 0.0;
  time_int_rel2 = 0.0;
  
  // read input file
  t1 = MPI_Wtime();
  q_aux1 = NULL;
  q_aux2 = NULL;
  read_input();
  t2 = MPI_Wtime();
  time_read += t2 -t1;
  
  // initialize Marsaglia RNG with processor-unique seed
  random = new RanMars(lmp,seed + comm->me);
  
  memory->create(r_save, atom->nlocal, atom->nlocal, "gle/pair/aux:r_save");
  memory->create(r_step, atom->nlocal, 4*atom->nlocal, "gle/pair/aux:r_step");
  memory->create(f_step, atom->nlocal, 3, "gle/pair/aux:f_step");
  distance_update();

  // initialize the amplitude matrix, noise matrix and integration constants
  t1 = MPI_Wtime();
  init_aux();
  t2 = MPI_Wtime();
  time_init += t2 -t1;
}

/* ----------------------------------------------------------------------
   Destroys memory allocated by the method
------------------------------------------------------------------------- */

FixGLEPairAux::~FixGLEPairAux()
{

  delete random;
  memory->destroy(q_aux1);
  memory->destroy(q_ran1);
  memory->destroy(q_save1);
  memory->destroy(q_aux2);
  memory->destroy(q_ran2);
  memory->destroy(q_save2);
  memory->destroy(q_B);
  memory->destroy(q_ints);
  memory->destroy(q_intv);
  delete A;
  delete Aps;
  delete Asp;
  memory->destroy(r_save);
  memory->destroy(r_step);
  memory->destroy(f_step);

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
  
  t1 = MPI_Wtime();

  // Advance V by dt/2
  for (int k = 0; k < Naux; k++) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
	// Calculate integration constants
	meff = mass[type[i]];   
	dtfm = dtf / meff;
	v[i][0] += dtfm * f_step[i][0];
	v[i][1] += dtfm * f_step[i][1];
	v[i][2] += dtfm * f_step[i][2];
	for (int j = 0; j < nlocal; j++) {  
	  double matrix = *(Asp[k].data() + i*nlocal+j);
	  if (i==j) {
	    v[i][0] -= matrix*dtfm *q_aux2[k][3*j];
	    v[i][1] -= matrix*dtfm *q_aux2[k][3*j+1];
	    v[i][2] -= matrix*dtfm *q_aux2[k][3*j+2];
	    if (k==0) {
	      //printf ("%d %lf %lf\n",i,matrix*dtfm *q_aux2[k][3*j+1],matrix*dtfm *q_aux2[k][3*j+2]);
	    }
	  } else {
	    double qr = q_aux2[k][3*j]*r_step[i][4*j] + q_aux2[k][3*j+1]*r_step[i][4*j+1] +q_aux2[k][3*j+2]*r_step[i][4*j+2];
	    v[i][0] -= matrix*dtfm *qr*r_step[i][4*j];
	    v[i][1] -= matrix*dtfm *qr*r_step[i][4*j+1];
	    v[i][2] -= matrix*dtfm *qr*r_step[i][4*j+2];
	  }

	}
      }
    }
  }

  // Advance X by dt
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      //x[i][0] += dtv * v[i][0];
      //x[i][1] += dtv * v[i][1];
      //x[i][2] += dtv * v[i][2];
    }
  }
  t2 = MPI_Wtime();
  time_int_rel1 += t2 -t1;
  
  // Update cholesky if necessary
  t1 = MPI_Wtime();
  double dr_max = 0.0;
  double dr;
  for (int i = 0; i < nlocal; i++) {
    for (int j = 0; j < nlocal; j++) {
      if (i!=j) {
	dr = (r_save[i][j] - r_step[i][4*j+3]);
	dr = dr*dr;
	if (dr > dr_max) dr_max = dr;
      }
    }
  }
  //printf("dr_max %f\n",dr_max);
  if (dr_max > dStep*dStep) {
    //printf("Updating amplitudes\n");
    update_cholesky();
    
    // update r_save
    for (int i = 0; i < nlocal; i++) {
      for (int j = 0; j < nlocal; j++) {
	r_save[i][j]=r_step[i][4*j+3];
      }
    }
  }
  t2 = MPI_Wtime();
  time_dist_update += t2 -t1;
  
  // Determine normalized distance vectors
  distance_update();
  
  t1 = MPI_Wtime();
  for (int k = 0; k < Naux; k++) {
    for (int i = 0; i < nlocal; i++) {
      for (int d=0; d<3; d++) {
	q_ran1[k][3*i+d] = random->gaussian();
	q_ran2[k][3*i+d] = random->gaussian();
	q_save1[k][3*i+d] = q_aux1[k][3*i+d];
	q_save2[k][3*i+d] = q_aux2[k][3*i+d];
      }
    } 
  }
  
  // Advance Q by dt
  for (int k = 0; k < Naux; k++) {
    for (int i = 0; i < nlocal; i++) {
      
      // update aux_var, self and cross
      for (int d=0; d<3; d++) {
	q_aux1[k][3*i+d] *= q_ints[2*k];  
	q_aux1[k][3*i+d] += q_ints[2*k+1]*q_save2[k][3*i+d];  
	q_aux2[k][3*i+d] *= q_ints[2*k];  
	q_aux2[k][3*i+d] -= q_ints[2*k+1]*q_save1[k][3*i+d];  
      }
      
      
      // update momenta contribution (for memory)
      double tmp10 = q_aux1[k][3*i];
      double tmp11 = q_aux1[k][3*i+1];
      double tmp12 = q_aux1[k][3*i+2];
      double tmp20 = q_aux2[k][3*i];
      double tmp21 = q_aux2[k][3*i+1];
      double tmp22 = q_aux2[k][3*i+2];
      //double const1 = q_intv[2*k];
      //double const2 = q_intv[2*k+1];
      double const1 = 0.0;
      double const2 = 0.001;
      //printf("%f %f\n",const1,const2);
      for (int j = 0; j < nlocal; j++) {
	double matrix = *(Aps[k].data() + i*nlocal+j);
	if (k==0) {
	  //printf ("%d %d %lf \n",i,j,matrix);
	 }
	if (i==j) {
	  tmp10 += const1* matrix*v[j][0]; 
	  tmp11 += const1* matrix*v[j][1]; 
	  tmp12 += const1* matrix*v[j][2]; 
	  tmp20 += const2* matrix*v[j][0]; 
	  tmp21 += const2* matrix*v[j][1]; 
	  tmp22 += const2* matrix*v[j][2]; 
	  if (k==0) {
	    //printf ("%d %lf %lf %lf %lf\n",i,const1* matrix*v[j][1],const1* matrix*v[j][2],const2* matrix*v[j][1],const2* matrix*v[j][2]);
	   }
	} else {
	  double vr = v[j][0]*r_step[i][4*j] + v[j][1]*r_step[i][4*j+1] +v[j][2]*r_step[i][4*j+2];
	  tmp10 += const1* matrix*vr*r_step[i][4*j]; 
	  tmp11 += const1* matrix*vr*r_step[i][4*j+1]; 
	  tmp12 += const1* matrix*vr*r_step[i][4*j+2]; 
	  tmp20 += const2* matrix*vr*r_step[i][4*j]; 
	  tmp21 += const2* matrix*vr*r_step[i][4*j+1]; 
	  tmp22 += const2* matrix*vr*r_step[i][4*j+2]; 
	}
      }
      //q_aux1[k][3*i] = tmp10;
      //q_aux1[k][3*i+1]=tmp11;
      //q_aux1[k][3*i+2]=tmp12;
      //q_aux2[k][3*i] = tmp20;
      //q_aux2[k][3*i+1]=tmp21;
      //q_aux2[k][3*i+2]=tmp22;
    
      // update noise
      for (int d=0; d<3; d++) {
	q_aux1[k][3*i+d] += q_B[k]*q_ran1[k][3*i+d];
	q_aux2[k][3*i+d] += q_B[k]*q_ran2[k][3*i+d];
      }
    }
  }
  t2 = MPI_Wtime();
  time_int_aux += t2 -t1;
  
  
}

/* ----------------------------------------------------------------------
   Second half of a timestep (V^{n+1/2} -> V^{n+1})
------------------------------------------------------------------------- */

void FixGLEPairAux::final_integrate()
{

  double dtfm;
  double ftm2v = force->ftm2v;

  double meff;
  double theta_vs, alpha_vs, theta_vc, alpha_vc;
  int ind_coef, ind_q;

  t1 = MPI_Wtime();
  // update v and x of atoms in group
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  
  // save velocity for cross time integration
  for ( int i=0; i< nlocal; i++) {
    f_step[i][0] = f[i][0];
    f_step[i][1] = f[i][1];
    f_step[i][2] = f[i][2];
  }

  // Advance V by dt/2
  for (int k = 0; k < Naux; k++) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
	// Calculate integration constants
	meff = mass[type[i]];   
	dtfm = dtf / meff;
	v[i][0] += dtfm * f_step[i][0];
	v[i][1] += dtfm * f_step[i][1];
	v[i][2] += dtfm * f_step[i][2];
	for (int j = 0; j < nlocal; j++) {  
	  double matrix = *(Asp[k].data() + i*nlocal+j);
	  if (i==j) {
	    v[i][0] -= matrix*dtfm *q_aux2[k][3*j];
	    v[i][1] -= matrix*dtfm *q_aux2[k][3*j+1];
	    v[i][2] -= matrix*dtfm *q_aux2[k][3*j+2];
	    if (k==0) {
	      //printf ("%d %f %f\n",matrix*dtfm *q_aux2[k][3*j+1],i,matrix*dtfm *q_aux2[k][3*j+2]);
	    }
	  } else {
	    double qr = q_aux2[k][3*j]*r_step[i][4*j] + q_aux2[k][3*j+1]*r_step[i][4*j+1] +q_aux2[k][3*j+2]*r_step[i][4*j+2];
	    v[i][0] -= matrix*dtfm *qr*r_step[i][4*j];
	    v[i][1] -= matrix*dtfm *qr*r_step[i][4*j+1];
	    v[i][2] -= matrix*dtfm *qr*r_step[i][4*j+2];
	  }
	}
      }
    }
  }
 
  for (int k = 0; k < Naux; k++) {
    for (int i = 0; i < nlocal; i++) {
      for (int j = 0; j < nlocal; j++) {  
	double matrix = *(Asp[k].data() + i*nlocal+j);
	  if (i==j) {
	    f[i][0] -= matrix*q_aux2[k][3*j];
	    f[i][1] -= matrix*q_aux2[k][3*j+1];
	    f[i][2] -= matrix*q_aux2[k][3*j+2];
	    if (k==0) {
	      //printf ("%d %f %f\n",matrix*dtfm *q_aux2[k][3*j+1],i,matrix*dtfm *q_aux2[k][3*j+2]);
	    }
	  } else {
	    double qr = q_aux2[k][3*j]*r_step[i][4*j] + q_aux2[k][3*j+1]*r_step[i][4*j+1] +q_aux2[k][3*j+2]*r_step[i][4*j+2];
	    f[i][0] -= matrix*qr*r_step[i][4*j];
	    f[i][1] -= matrix*qr*r_step[i][4*j+1];
	    f[i][2] -= matrix*qr*r_step[i][4*j+2];
	  }
      }
    }
  }
  t2 = MPI_Wtime();
  time_int_rel2 += t2 -t1;
  
      // print timing
  if (update->nsteps == update->ntimestep) {
    printf("processor %d: time(read) = %f\n",me,time_read);
    printf("processor %d: time(init) = %f\n",me,time_init);
    printf("processor %d: time(int_rel1) = %f\n",me,time_int_rel1);
    printf("processor %d: time(dist_update) = %f\n",me,time_dist_update);
    printf("processor %d: time(int_aux) = %f\n",me,time_int_aux);
    printf("processor %d: time(int_rel2) = %f\n",me,time_int_rel2);
  }
  

}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixGLEPairAux::compute_vector(int n)
{
  int nlocal = atom->nlocal;
  int loc = n%(Naux);
  int k = (n - loc)/(3*nlocal);
  int d = loc%3;
  int i = (loc-d)/3;
  
  //printf("%d %d %d\n",k,i,d);
  
  return q_aux2[k][3*i+d];
}


/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixGLEPairAux::memory_usage()
{
  double bytes = atom->nlocal*atom->nlocal*3*Naux*sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixGLEPairAux::grow_arrays(int nmax)
{
  memory->grow(q_aux1, Naux, 3*nmax,"gle/pair/aux:q_aux1");
  memory->grow(q_aux2, Naux, 3*nmax,"gle/pair/aux:q_aux2");
}

/* ----------------------------------------------------------------------
   read input coefficients
------------------------------------------------------------------------- */
void FixGLEPairAux::read_input()
{
  char line[MAXLINE];
  
  // loop until section found with matching keyword

  while (1) {
    if (fgets(line,MAXLINE,input) == NULL)
      error->one(FLERR,"Did not find keyword in table file");
    if (strspn(line," \t\n\r") == strlen(line)) continue;  // blank line
    if (line[0] == '#') continue;                          // comment
    char *word = strtok(line," \t\n\r");
    if (strcmp(word,keyword) == 0) break;           // matching keyword
  }
  
  fgets(line,MAXLINE,input);
  char *word = strtok(line," \t\n\r\f");
  
  if (strcmp(word,"TIME") == 0) {
    memory_flag = TIME;
  } else if (strcmp(word,"FIT") == 0) {
    memory_flag = FIT;
  } else {
    printf("WORD: %s\n",word);
    error->one(FLERR,"Missing memory keyword in pair gle parameters");
  }
  word = strtok(NULL," \t\n\r\f");
  
  // default values
  Niter = 500;
  tStart= 0.0;
  tStep = 0.05;
  tStop = 5.0;
  
  while (word) {
    if (strcmp(word,"dStart") == 0) {
      word = strtok(NULL," \t\n\r\f");
      dStart = atof(word);
      printf("dStart %f\n",dStart);
    } else if (strcmp(word,"dStep") == 0) {
      word = strtok(NULL," \t\n\r\f");
      dStep = atof(word);
      printf("dStep %f\n",dStep);
    } else if (strcmp(word,"dStop") == 0) {
      word = strtok(NULL," \t\n\r\f");
      dStop = atof(word);
      printf("dStop %f\n",dStop);
    } else if (strcmp(word,"tStart") == 0) {
      word = strtok(NULL," \t\n\r\f");
      tStart = atof(word);
      printf("tStart %f\n",tStart);
    } else if (strcmp(word,"tStep") == 0) {
      word = strtok(NULL," \t\n\r\f");
      tStep = atof(word);
      printf("tStep %f\n",tStep);
    } else if (strcmp(word,"tStop") == 0) {
      word = strtok(NULL," \t\n\r\f");
      tStop = atof(word);
      printf("tStop %f\n",tStop);
    } else if (strcmp(word,"Naux") == 0) {
      word = strtok(NULL," \t\n\r\f");
      Naux = atof(word);
      printf("Naux %d\n",Naux);
    } else if (strcmp(word,"Niter") == 0) {
      word = strtok(NULL," \t\n\r\f");
      Niter = atoi(word);
      printf("Niter %d\n",Niter);
    } else {
      printf("WORD: %s\n",word);
      error->one(FLERR,"Invalid keyword in pair table parameters");
    }
    word = strtok(NULL," \t\n\r\f");
  }
  
  if (dStop < dStart)
    error->all(FLERR,"Fix gle/pair/aux dStop must be > dStart");
  Nd = (dStop - dStart) / dStep + 1.5;
  
  if (Naux < 0)
    error->all(FLERR,"Fix gle/pair/aux terms must be > 0");
  
  if (memory_flag == TIME) {
    if (tStop < tStart)
      error->all(FLERR,"Fix gle/pair/aux tStop must be > tStart");
    Nt = (tStop - tStart) / tStep + 1.5;
  }
  
  self_coeff = new double[3*Naux];    
  cross_coeff = new double[3*Naux*Nd];
    
  // initilize simulations for either TIME input (fitting necessary) or FIT input (only reading necessary)
  if (memory_flag == TIME) {
    read_input_time(input);
  } else { //memory_flag == FIT
    read_input_fit(input);
  }
  fclose(input);
  
  // write out input function
  FILE * output;
  output = fopen("memory.output","w");
  fprintf(output,"# input file to perform non-markovian simulation in lammps using fix_gle_pair_aux\n\n");
  fprintf(output,"%s\n",keyword);
  // print time coefficients
  fprintf(output,"TIME dStart %f dStep %f dStop %f tStart %f tStep %f tStop %f Naux %d\n",dStart,dStep,dStop,tStart,tStep,tStop,Naux);
  double t;
  int n,d;
  double sum_self,sum_cross;
  for (t=tStart; t<=tStop; t+=tStep) {
    sum_self = 0.0;
    for (n=0; n<Naux; n++) {
      sum_self += self_coeff[3*n]*exp(-self_coeff[3*n+1]*t)*cos(self_coeff[3*n+2]*t);
    }
    fprintf(output,"%f %f %f\n",0.0,t,sum_self);
  }
  for (d=0; d<Nd; d++) {
    for (t=tStart; t<=tStop; t+=tStep) {
      sum_cross = 0.0;
      for (n=0; n<Naux; n++) {
	sum_cross += cross_coeff[3*Naux*d+3*n]*exp(-cross_coeff[3*Naux*d+3*n+1]*t)*cos(cross_coeff[3*Naux*d+3*n+2]*t);
      }
      fprintf(output,"%f %f %f\n",dStart+d*dStep,t,sum_cross);
    }
  }
  fclose(output);
  
}

/* ----------------------------------------------------------------------
   read input coefficients - TIME keyword
------------------------------------------------------------------------- */
void FixGLEPairAux::read_input_time(FILE * input) {
  
  int d,t,i,n;
  double dummy_d, dummy_t,mem;
  
  double *self_data = new double[Nt];
  double *cross_data = new double[Nt*Nd];
  double *time = new double[Nt];
  
  //read self_memory
  for (t=0; t<Nt; t++) {
    fscanf(input,"%lf %lf %lf\n",&dummy_d, &dummy_t, &mem);
    self_data[t] = mem;
    time[t] = dummy_t;
    //printf("%f\n",mem);
  }
  
  // read cross_memory
  for (d=0; d< Nd; d++) {
    for (t=0; t<Nt; t++) {
      fscanf(input,"%lf %lf %lf\n",&dummy_d, &dummy_t, &mem);
      cross_data[Nt*d+t] = mem;
      //printf("%f %f %f\n",dummy_d,dummy_t,mem);
    }
  }
  
  // fit data
  // fit self_memory
  for (n=0; n<Naux; n++) {		 /* Initial conditions */  
    self_coeff[3*n] = self_data[0]/Naux;
    self_coeff[3*n+1] = 0.1*(n+1);
    self_coeff[3*n+2] = (n+1);
  }
  double *perror = new double[3*Naux];  /* Returned parameter errors */  
  mp_par pars[3*Naux];			/* Parameter constraints */   
  struct fit_struct fit;
  int status;
  mp_result result;
  memset(&result,0,sizeof(result));      /* Zero results structure */
  result.xerror = perror;
  memset(pars,0,sizeof(pars));		 /* Initialize constraint structure */
  for (n=0; n<Naux; n++) {
    //Amplitude
    pars[3*n].limited[0] = 1;    
    pars[3*n].limits[0] = 0.0;
    
    //exp-time constant
    pars[3*n+1].limited[0] = 1;    
    pars[3*n+1].limits[0] = 0.1;
    
    //fluc-time constant
    pars[3*n+2].limited[0] = 1;    
    pars[3*n+2].limits[0] = 0.0;
  }
  mp_config config;
  memset(&config, 0, sizeof(config));
  config.maxiter = Niter;
  double ey[Nt];
  for (i=0; i<Nt; i++) ey[i] = 0.5;
  fit.x = time;
  fit.y = self_data;
  fit.ey = ey;
  printf("Start fitting self-memory!\n");
  status = mpfit(expfunc, Nt, 3*Naux, self_coeff, pars, &config, (void *) &fit, &result);
  printf("*** testgaussfit status = %d\n", status);
  printresult(self_coeff, &result);
  
  // fit cross_memory
  for (d=0; d< Nd; d++) {
    fit.y = &cross_data[d*Nt];
    for (n=0; n<Naux; n++) {
      //amplitude -> restrained by self_amplitude 
      if (self_coeff[3*n]!=0.0) {
	pars[3*n].limited[0] = 1;    
	pars[3*n].limits[0] = 0.0*self_coeff[3*n];
	pars[3*n].limited[1] = 1;    
	pars[3*n].limits[1] = 0.5*self_coeff[3*n];
      } else {
	pars[3*n].fixed = 1;
      }
      //exp-time constant -> fixed
      pars[3*n+1].fixed = 1;    
      //fluc-time constant -> fixed
      pars[3*n+2].fixed = 1;    
    }
    double sign = 1.0;
    for (n=0; n<Naux; n++) {
      sign*=1.0;
      cross_coeff[d*3*Naux+3*n] = sign*0.1*self_coeff[3*n];
      cross_coeff[d*3*Naux+3*n+1] = self_coeff[3*n+1];
      cross_coeff[d*3*Naux+3*n+2] = self_coeff[3*n+2];
    }
    printf("Start fitting cross-memory!\n");
    status = mpfit(expfunc, Nt, 3*Naux, &cross_coeff[3*Naux*d], pars, &config, (void *) &fit, &result);
    printf("*** testgaussfit status = %d\n", status);
    printresult(&cross_coeff[3*Naux*d], &result);
  }
  
  // print coefficients
  FILE * output;
  output = fopen("memory_coefficients.output","w");
  
  fprintf(output,"# input file to perform non-markovian simulation in lammps using fix_gle_pair_aux\n\n");
  fprintf(output,"%s\n",keyword);
  
  // print time coefficients
  fprintf(output,"FIT dStart %f dStep %f dStop %f Naux %d\n",dStart,dStep,dStop,Naux);
  for (n=0; n<Naux; n++) {
    fprintf(output,"TIMECOEFFICIENT %d %f %f\n",n,self_coeff[3*n+1],self_coeff[3*n+2]);
  }
  
  // print self coefficients
  for (n=0; n<Naux; n++) {
    fprintf(output,"%f %d %f\n",0.0,n,self_coeff[3*n]);
  }
  
  // print cross coefficients
  for (d=0; d< Nd; d++) {
    for (n=0; n<Naux; n++) {
      fprintf(output,"%f %d %f\n",dStart+d*dStep,n,cross_coeff,cross_coeff[3*Naux*d+3*n]);
    }
  }
  
  fclose(output);
}

/* ----------------------------------------------------------------------
   read input coefficients - FIT keyword
------------------------------------------------------------------------- */
void FixGLEPairAux::read_input_fit(FILE * input) {
  
  double exp_time, fluc_time, amp, dummy_d;
  int dummy_n,d,n;
  // read coefficients
  for (n=0; n<Naux; n++) {
    fscanf(input,"TIMECOEFFICIENT %d %lf %lf\n",&dummy_n,&exp_time,&fluc_time);
    self_coeff[3*n+1]=exp_time;
    self_coeff[3*n+2]=fluc_time;
    for (d=0; d< Nd; d++) {
      cross_coeff[3*Naux*d+3*n+1]=exp_time;
      cross_coeff[3*Naux*d+3*n+2]=fluc_time;
    }
  }
  
  // read self coefficients
  for (n=0; n<Naux; n++) {
    fscanf(input,"%lf %d %lf\n",&dummy_d,&dummy_n,&amp);
    self_coeff[3*n]=amp;
  }
  
  // read cross coefficients
  for (d=0; d< Nd; d++) {
    for (n=0; n<Naux; n++) {
      fscanf(input,"%lf %d %lf\n",&dummy_d,&dummy_n,&amp);
      cross_coeff[3*Naux*d+3*n]=amp;
    }
  }
  
}


/* ----------------------------------------------------------------------
   Initializes the extended variables to equilibrium distribution
   at t_start.
------------------------------------------------------------------------- */
void FixGLEPairAux::init_aux()
{
  
  int nlocal = atom->nlocal;
  double **x = atom->x;
  double **f = atom->f;
  int k,d,i,j,n;
  
  // allocate memory
  grow_arrays(atom->nlocal);
  memory->create(q_ran1, Naux, 3*atom->nlocal, "gle/pair/aux:q_ran1");
  memory->create(q_ran2, Naux, 3*atom->nlocal, "gle/pair/aux:q_ran2");
  memory->create(q_save1,Naux,3*atom->nlocal, "gle/pair/aux:q_save1");
  memory->create(q_save2,Naux,3*atom->nlocal, "gle/pair/aux:q_save2");
  memory->create(q_ints, Naux*2, "gle/pair/aux:q_ints");
  memory->create(q_intv, Naux*2, "gle/pair/aux:q_intv");
  memory->create(q_B, Naux, "gle/pair/aux:q_B");
  size_vector = 3*Naux*atom->nlocal;

  // initialize auxilliary variables
  for (k = 0; k < Naux; k++) {
    for (i = 0; i < atom->nlocal; i++) {
      q_aux1[k][3*i] = 0.0;
      q_aux1[k][3*i+1] = 0.0;
      q_aux1[k][3*i+2] = 0.0;
      q_aux2[k][3*i] = 0.0;
      q_aux2[k][3*i+1] = 0.0;
      q_aux2[k][3*i+2] = 0.0;
    }
  }
  
  // initialize forces
  for ( i=0; i< nlocal; i++) {
    f_step[i][0] = 0.0;
    f_step[i][1] = 0.0;
    f_step[i][2] = 0.0;
  }
  
  // initialize distance saver (to check when cholesky-update is necessary)
  for (i = 0; i < nlocal; i++) {
    for (j = 0; j < nlocal; j++) {
      r_save[i][j] = r_step[i][4*j+3];
    }
  }
  
  // calculate integration constants
  // exp(-dt*Ass)
  for (k = 0; k < Naux; k++) {
    q_ints[2*k]=exp(-update->dt*self_coeff[3*k+1])*cos(update->dt*self_coeff[3*k+2]);
    q_ints[2*k+1]=exp(-update->dt*self_coeff[3*k+1])*sin(update->dt*self_coeff[3*k+2]);
  }
  
  // calculate integration constants
  // Ass^(-1)*(1-exp(-dt*Ass))
  for (int k = 0; k < Naux; k++) {
    q_intv[2*k]= -self_coeff[3*k+2]*(1.0-q_ints[2*k])/(self_coeff[3*k+1]*self_coeff[3*k+1]+self_coeff[3*k+2]*self_coeff[3*k+2])+self_coeff[3*k+1]*q_ints[2*k+1]/(self_coeff[3*k+1]*self_coeff[3*k+1]+self_coeff[3*k+2]*self_coeff[3*k+2]);
    q_intv[2*k+1]=self_coeff[3*k+1]*(1.0-q_ints[2*k])/(self_coeff[3*k+1]*self_coeff[3*k+1]+self_coeff[3*k+2]*self_coeff[3*k+2])+self_coeff[3*k+2]*q_ints[2*k+1]/(self_coeff[3*k+1]*self_coeff[3*k+1]+self_coeff[3*k+2]*self_coeff[3*k+2]);
    //printf("%f %f\n",q_intv[2*k],q_intv[2*k+1]);
  }
  
  // CholDecomp[Css-exp(-dt*Ass)*Css*exp(-dt*Ass^T)] (hard coded)
  for (int k = 0; k < Naux; k++) {
    q_B[k]=sqrt(1-exp(-2.0*update->dt*self_coeff[3*k+1]));
  }
  
  // initialize interaction matrices
  // CholDecomp to determine Aps (on the fly)
  // initialize matrices
  A = new MatrixXd[Naux];
  Aps = new MatrixXd[Naux];
  Asp = new MatrixXd[Naux];
  for (int k = 0; k < Naux; k++) {
    A[k] = MatrixXd::Zero(nlocal,nlocal);
  }

  // update cholesky
  update_cholesky();

  
}

/* ----------------------------------------------------------------------
   updates the interaction amplitudes by cholesky decomposition
------------------------------------------------------------------------- */
void FixGLEPairAux::update_cholesky() 
{
  // check distances
  int k,d,i,j,n;
  int nlocal = atom->nlocal;
  double **x = atom->x;
  
  for (i = 0; i < nlocal; i++) {
    for (j = 0; j < nlocal; j++) {
      if (i==j) {
	for (k = 0; k < Naux; k++) {
	  A[k](i,j)=self_coeff[3*k];
	}
      } else {
	  
	d = (r_step[i][4*j+3]- dStart)/dStep;
	  
	if (d < 0) {
	  printf("dist %f lower cut %f\n",r_step[i][4*j+3],dStart);
	  error->one(FLERR,"Particles closer than lower cutoff in fix/pair/gle\n"); 
	}
	if (d >= Nd) {
	  for (k = 0; k < Naux; k++) {
	    A[k](i,j) = 0.0;
	  }
	}
	else {
	  for (k = 0; k < Naux; k++) {
	    // correct
	    //A[k](i,j) = cross_coeff[3*Naux*d+3*k];
	    //test oseen
	    A[k](i,j) = self_coeff[3*k]*18.0/4.0 / r_step[i][4*j+3] * (1-2*9.0/3.0/(r_step[i][4*j+3]*r_step[i][4*j+3]));
	    //construct
	    /*if (j==i-1 || j==i+1 || (j==0&&i==nlocal-1) || (j==nlocal-1&&i==0)) {
	     A[k](i,j) = 0.6* self_coeff[3*k];
	    }
	    if (j==i-2 || j==i+2 || (j==0&&i==nlocal-2) || (j==nlocal-2&&i==0) || (j==1&&i==nlocal-1) || (j==nlocal-1&&i==1)) {
	      A[k](i,j) = 0.3* self_coeff[3*k];
	    }*/
	  }
	}
	  
      }
    }
  }
  
   // perform cholesky decomposition
  for (k = 0; k < Naux; k++) {
    //cout << "The matrix A is" << endl << A[k] << endl;
    LLT<MatrixXd> Aps_comp(A[k]); // compute the Cholesky decomposition of A
    Aps[k] = Aps_comp.matrixL(); // retrieve factor L  in the decomposition
    Asp[k] = Aps[k].transpose();
    // The previous two lines can also be written as "L = A.llt().matrixL()"
    //cout << "The Cholesky factor L is" << endl << Aps[k] << endl;
    //cout << "To check this, let us compute L * L.transpose()" << endl;
    if (k==0) {
      //cout << Aps[k] * Aps[k].transpose() << endl;
      //cout << A[k] << endl;
    }
    if (Aps_comp.info()!=0) {
      cout << Aps[k] * Aps[k].transpose() << endl;
      cout << A[k] << endl;
      ComplexEigenSolver<MatrixXd> A_eigen(A[k]);
      cout << A_eigen.eigenvalues() << endl;
      error->all(FLERR,"LLT Cholesky not possible!\n");
    }
    //cout << "This should equal the matrix A" << endl;
    //printf("%f\n",Aps[k](0,0));
  }
}

/* ----------------------------------------------------------------------
   updates the list of distance vectors between particles
------------------------------------------------------------------------- */
void FixGLEPairAux::distance_update() 
{
  int i,j;
  int nlocal = atom->nlocal;
  double **x = atom->x;
  double xtmp, ytmp, ztmp, delx,dely,delz, dist,idist;
  double min_dist = 20.0;
  
  for (i = 0; i < nlocal; i++) {
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    for (j = 0; j < nlocal; j++) {
      if (j!= i) {
	delx = xtmp - x[j][0];
	dely = ytmp - x[j][1];
	delz = ztmp - x[j][2];

	domain->minimum_image(delx,dely,delz);
	dist = sqrt(delx*delx + dely*dely + delz*delz);
	if (dist < min_dist) min_dist = dist;
	idist = 1.0/dist;
      
	r_step[i][4*j] = delx*idist;
	r_step[i][4*j+1] = dely*idist;
	r_step[i][4*j+2] = delz*idist;
	r_step[i][4*j+3] = dist;
	
      }
    }
  }
  
  if (update->ntimestep%10==0) {
    //printf("min_dist %f\n",min_dist);
  }
}

/* ----------------------------------------------------------------------
   help function for fitting proceduce - implements an exponential fit
------------------------------------------------------------------------- */
int expfunc(int m, int n, double *p, double *dy, double **dvec, void *fit)
{
  int i,j;
  struct fit_struct *v = (struct fit_struct *) fit;
  double *x, *y, *ey;
  double xc, sig2;

  x = v->x;
  y = v->y;
  ey = v->ey;

  for (i=0; i<m; i++) {
    dy[i] = y[i];
    for (j=0; j<n/3; j++) {
      dy[i] -=  p[3*j]*exp(-x[i]*p[3*j+1])*cos(x[i]*p[3*j+2]);
    }
    dy[i] /= ey[i];
  }

  return 0;
}

/* ----------------------------------------------------------------------
   help function to print fit results
------------------------------------------------------------------------- */
void printresult(double *x, mp_result *result) 
{
  int i;
  if ((x == 0) || (result == 0)) return;
  printf("  CHI-SQUARE = %f    (%d DOF)\n", 
	 result->bestnorm, result->nfunc-result->nfree);
  printf("        NPAR = %d\n", result->npar);
  printf("       NFREE = %d\n", result->nfree);
  printf("     NPEGGED = %d\n", result->npegged);
  printf("     NITER = %d\n", result->niter);
  printf("      NFEV = %d\n", result->nfev);
  printf("\n");
  for (i=0; i<result->npar; i++) {
    printf("  P[%d] = %f +/- %f\n", 
	   i, x[i], result->xerror[i]);
  }
}
