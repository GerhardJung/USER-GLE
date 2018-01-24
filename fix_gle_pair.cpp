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


/*
Careful:
-fix changes neighbor skin!
-> neighbor update (int Neighbor::check_distance()) does not make sence due to wrong skin
-> update every step necessary (not such a big problem since building the neighbour list is not the bottleneck in this kind of simulations)
*/

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "fix_gle_pair.h"
#include "math_extra.h"
#include "atom.h"
#include "force.h"
#include "pair.h"
#include "update.h"
#include "comm.h"
#include "input.h"
#include "variable.h"
#include "random_mars.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include "group.h"
#include "domain.h"
#include <iostream>
#include <vector>
#include "kiss_fft.h"
#include "kiss_fftr.h"
#include "eigenvalues_tridiagonal.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

#define MAXLINE 1024
#define PI 3.14159265359


/* ----------------------------------------------------------------------
   Parses parameters passed to the method, allocates some memory
------------------------------------------------------------------------- */

FixGLEPair::FixGLEPair(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  
  global_freq = 1;
  nevery = 1;
  peratom_freq = 1;
  peratom_flag = 1;
  restart_global = 1;
  
  MPI_Comm_rank(world,&me);

  int narg_min = 7;
  if (narg < narg_min) error->all(FLERR,"Illegal fix gle/pair command");

  // temperature
  t_target = force->numeric(FLERR,arg[3]);
  
  // seed for random numbers
  int seed = force->inumeric(FLERR,arg[4]);
  
  // read input file
  input = fopen(arg[5],"r");
  if (input == NULL) {
    char str[128];
    sprintf(str,"Cannot open fix gle/pair file %s",arg[5]);
    error->one(FLERR,str);
  }
  keyword = arg[6];
  
  // set precision of sqrt computation
  // default
  mLanczos = 50;
  tolLanczos = 0.0001;
  
  mLanczos = force->inumeric(FLERR,arg[7]);
  tolLanczos = force->numeric(FLERR,arg[8]);
  
  // error checking for the first set of required input arguments
  if (seed <= 0) error->all(FLERR,"Illegal fix gle/pair command");
  if (t_target < 0)
    error->all(FLERR,"Fix gle/pair temperature must be >= 0");
  
  // set number of dimensions
  d=3;
  
  // Timing
  time_read = 0.0;
  time_init = 0.0;
  time_int_rel1 = 0.0;
  time_noise = 0.0;
  time_matrix_create = 0.0;
  time_forwardft = 0.0;
  time_sqrt = 0.0;
  time_backwardft = 0.0;
  time_dist_update = 0.0;
  time_int_rel2 = 0.0;
  
  // read input file
  t1 = MPI_Wtime();
  read_input();
  t2 = MPI_Wtime();
  time_read += t2 -t1;
  
  // initialize Marsaglia RNG with processor-unique seed
  random = new RanMars(lmp,seed + comm->me);
  
  // initialize
  t1 = MPI_Wtime();
  int *type = atom->type;
  double *mass = atom->mass;
  dtf = 0.5 * update->dt * force->ftm2v;
  int_b = 1.0/(1.0+self_data[0]*update->dt/4.0/mass[type[0]]); // 4.0 because K_0 = 0.5*K(0)
  int_a = (1.0-self_data[0]*update->dt/4.0/mass[type[0]])*int_b; // 4.0 because K_0 = 0.5*K(0)
  lastindexN = 0,lastindexn=0;
  
  // allocate memory
  int nlocal = atom->nlocal;
  double **x = atom->x;
  double **f = atom->f;
  int k,i,j,n,t;
  int N = 2*Nt-2;
  memory->create(x_save, 3*atom->nlocal, Nt,  "gle/pair:x_save");
  memory->create(ran, N, atom->nlocal*d, "gle/pair:ran");
  memory->create(fd, atom->nlocal,3, "gle/pair:fd");
  memory->create(fc, atom->nlocal,3, "gle/pair:fc");
  memory->create(fr, atom->nlocal,3, "gle/pair:fr");
  memory->create(array, atom->nlocal,9, "gle/pair:array");
  size_peratom_cols = 9;
  array_atom = array;
  
  // initialize forces
  for ( i=0; i< nlocal; i++) {
    for (int dim1=0; dim1<d; dim1++) { 
      fc[i][dim1] = 0.0;
      fd[i][dim1] = 0.0;
      fr[i][dim1] = 0.0;
      array[i][dim1] = 0.0;
      array[i][3+dim1] = 0.0;
      array[i][6+dim1] = 0.0;
    }
  }
  
  // initiliaze position storage
  imageint *image = atom->image;
  tagint *tag = atom->tag;
  int itag;
  double unwrap[3];
  for (int i = 0; i < nlocal; i++) {
    itag = tag[i] - 1;
    domain->unmap(x[itag],image[itag],unwrap);
    for (int dim1=0; dim1<d; dim1++) { 
      for (int t = 0; t < Nt; t++) {
	x_save[3*itag+dim1][t] = unwrap[dim1];
      }
    }
  }
  
  // initilize (uncorrelated) random numbers
  for (int t = 0; t < N; t++) {
    for (int i = 0; i < nlocal; i++) {
      itag = tag[i] - 1;
      for (int dim1=0; dim1<d; dim1++) { 
	ran[t][d*itag+dim1] = random->gaussian();
      }
    }
  }
  t2 = MPI_Wtime();
  time_init += t2 -t1;

}


/* ----------------------------------------------------------------------
   Destroys memory allocated by the method
------------------------------------------------------------------------- */

FixGLEPair::~FixGLEPair()
{

  delete random;
  memory->destroy(ran);
  memory->destroy(x_save);
  
  memory->destroy(fc);
  memory->destroy(fd);
  memory->destroy(fr);
  
  delete [] cross_data;
  delete [] self_data;
  delete [] cross_data_ft;
  delete [] self_data_ft;

}


/* ----------------------------------------------------------------------
   Specifies when the fix is called during the timestep
------------------------------------------------------------------------- */

int FixGLEPair::setmask()
{
  
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  return mask;
  
}


/* ----------------------------------------------------------------------
   Initialize the method parameters before a run
------------------------------------------------------------------------- */

void FixGLEPair::init()
{
  
  // need a full neighbor list, built whenever re-neighboring occurs
  irequest = neighbor->request(this);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  
  //increase pair/cutoff to the target values
  if (!force->pair) {
    error->all(FLERR,"We need a pair potential to build neighbor-list! TODO: If this error appears, just create pair potential with zero amplitude\n");
  } else {
    int my_type = atom->type[0];
    double cutsq = dStop*dStop - force->pair->cutsq[my_type][my_type];
    if (cutsq > 0) {
      //increase skin
      neighbor->skin = dStop - sqrt(force->pair->cutsq[my_type][my_type]) + 0.3;
    }
    // since skin is increased neighbor needs to be updated every step
    char **c = (char**)&*(const char* const []){ "delay", "0", "every","1", "check", "no" };
    neighbor->modify_params(6,c);
  }
  
  // FFT memory kernel for later processing
  int i,t,l;
  int N = 2*Nt -2;
  kiss_fft_scalar * buf;
  kiss_fft_cpx * bufout;
  buf=(kiss_fft_scalar*)KISS_FFT_MALLOC(sizeof(kiss_fft_scalar)*N*(1+Nd));
  bufout=(kiss_fft_cpx*)KISS_FFT_MALLOC(sizeof(kiss_fft_cpx)*N*(1+Nd));
  memset(bufout,0,sizeof(kiss_fft_cpx)*N*(1+Nd));
  for (t=0; t<Nt; t++) {
    buf[t] = self_data[t];
    if (t==0 || t==Nt-1){ }
    else {
      buf[N-t] = self_data[t];
    }
  }
  for (l=0; l< Nd; l++) {
    for (t=0; t<Nt; t++) {
      buf[N*(l+1)+t] = cross_data[Nt*l+t];
      if (t==0 || t==Nt-1){ }
      else {
	buf[N*(l+1)+N-t] = cross_data[Nt*l+t];
      }
    }
  }
  
  kiss_fftr_cfg st = kiss_fftr_alloc( N ,0 ,0,0);
  for (i=0; i<1+Nd;i++) {
    kiss_fftr( st ,&buf[i*N],&bufout[i*N] );
  }
  
  for (t=0; t<Nt; t++) {
    self_data_ft[t] = bufout[t].r;
  }
  for (l=0; l< Nd; l++) {
    for (t=0; t<Nt; t++) {
      cross_data_ft[Nt*l+t] = bufout[N*(l+1)+t].r;
      if (cross_data_ft[Nt*l+t]*cross_data_ft[Nt*l+t] < 0.000000001) cross_data_ft[Nt*l+t] = sqrt(0.000000001);
    }
  }
  free(st);
  free(buf);
  free(bufout);
  
}


/* ----------------------------------------------------------------------
   First half of a timestep (V^{n} -> V^{n+1/2}; X^{n} -> X^{n+1})
------------------------------------------------------------------------- */

void FixGLEPair::initial_integrate(int vflag)
{
  
  double dtfm, meff;
  int i,dim1,t;
  int n,m;
  int itag;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  const int * _noalias const type = atom->type;
  const int * _noalias const tag = atom->tag;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  
  // update (uncorrelated) noise
  for (i = 0; i < nlocal; i++) {
    itag = tag[i]-1;
    for (dim1=0; dim1<d; dim1++) { 
      ran[lastindexN][d*itag+dim1] = random->gaussian();
      fr[itag][dim1] = 0.0;
      fd[itag][dim1] = 0.0;
    }
  }
  
  // Determine random force contribution
  t1 = MPI_Wtime();
  list = neighbor->lists[irequest];
  update_noise();
  t2 = MPI_Wtime();
  time_noise += t2 -t1;
  
  // Determine dissipative force contribution
  t1 = MPI_Wtime();
  domain->pbc();
  comm->exchange();
  comm->borders();
  neighbor->build();

  const int nthreads = comm->nthreads;
  const int inum = list->inum;
  
  #if defined(_OPENMP)
  #pragma omp parallel private (dim1,t,n,m) default(none) shared(x)
  #endif
  {
    const int * _noalias const type = atom->type;
    const double * _noalias const special_lj = force->special_lj;
    const int * _noalias const ilist = list->ilist;
    const int * _noalias const numneigh = list->numneigh;
    const int * const * const firstneigh = list->firstneigh;

    double xtmp,ytmp,ztmp,delx,dely,delz,fxtmp,fytmp,fztmp;
    double rsq,rsqi,r2inv,r6inv,forcelj,factor_lj,evdwl,fpair,dot;

    const int nlocal = atom->nlocal;
    int ii,j,jj,jnum,jtype,jtag;
    double *dr = new double[3];
    int ifrom, ito, tid;
    loop_setup_thr(ifrom, ito, tid, inum, nthreads);
    for (ii = ifrom; ii < ito; ii++) {
      const int i = ilist[ii];
      const int itype = type[i];
      const int itag = tag[i]-1;
      const int * _noalias const jlist = firstneigh[i];
    
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      jnum = numneigh[i];
      // self-correlation contribution
      for (dim1=0; dim1<d;dim1++) {
	n = lastindexn;
	m = lastindexn-1;
	if (m==-1) m=Nt-1;
	for (t = 1; t < Nt; t++) {
	  fd[itag][dim1] += self_data[t]*(x_save[3*itag+dim1][n]-x_save[3*itag+dim1][m]);
	  n--;
	  m--;
	  if (n==-1) n=Nt-1;
	  if (m==-1) m=Nt-1;
	}
      }
      // cross-correlation contribution
      for (jj = 0; jj < jnum; jj++) {
	j = jlist[jj];
	j &= NEIGHMASK;
	jtype = type[j];
	jtag = tag[j]-1;
	  
	dr[0] = xtmp - x[j][0];
	dr[1] = ytmp - x[j][1];
	dr[2] = ztmp - x[j][2];

	rsq = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
	rsqi = 1/rsq;
	int dist = (sqrt(rsq) - dStart)/dStep;
	      
	if (dist < 0) {
	  printf("dist: %f, lower cutoff: %f\n",sqrt(rsq),dStart);
	  error->all(FLERR,"Particles closer than lower cutoff in fix/pair\n");
	} else if (dist < Nd) {
	  n = lastindexn;
	  m = lastindexn-1;
	  if (m==-1) m=Nt-1;
	  for (t = 1; t < Nt; t++) {
	    dot = (dr[0]*(x_save[3*jtag][n]-x_save[3*jtag][m]) + dr[1]*(x_save[3*jtag+1][n]-x_save[3*jtag+1][m])+dr[2]*(x_save[3*jtag+2][n]-x_save[3*jtag+2][m]))*rsqi;
	    for (dim1=0; dim1<d; dim1++) {
	      fd[itag][dim1] += cross_data[dist*Nt+t]*dot*dr[dim1];
	    }
	    n--;
	    m--;
	    if (n==-1) n=Nt-1;
	    if (m==-1) m=Nt-1;
	  }
	}
      }
    }
    delete [] dr;
  }
  
  
  // Advance X by dt
  for (i = 0; i < nlocal; i++) {
    itag = tag[i] -1;
    if (mask[i] & groupbit) {
      meff = mass[type[i]];   
      //if ( update->ntimestep %10 == 0) { printf("x: %f fc: %f fd: %f fr: %f\n",x[i][0],fc[itag][0],fd[itag][0],fr[itag][0]);}
      for (dim1=0; dim1<d; dim1++) { 
	x[i][dim1] += int_b * update->dt * v[i][dim1] 
	  + int_b * update->dt * update->dt / 2.0 / meff * fc[itag][dim1] 
	  - int_b * update->dt / meff/ 2.0 * fd[itag][dim1]
	  + int_b*update->dt/ 2.0 / meff * fr[itag][dim1]; // convection, conservative, dissipative, random
      }
    }
  }
  t2 = MPI_Wtime();
  time_int_rel1 += t2 -t1;
  
  // Update time/positions
  t1 = MPI_Wtime();
  lastindexN++;
  if (lastindexN == 2*Nt-2) lastindexN = 0;
  lastindexn++;
  if (lastindexn == Nt) lastindexn = 0;
  imageint *image = atom->image;
  double unwrap[3];
  for (i = 0; i < nlocal; i++) {
    itag = tag[i]-1;
    domain->unmap(x[i],image[i],unwrap);
    x_save[3*itag][lastindexn] = unwrap[0];
    x_save[3*itag+1][lastindexn] = unwrap[1];
    x_save[3*itag+2][lastindexn] = unwrap[2];
  }
  t2 = MPI_Wtime();
  time_dist_update += t2 -t1;
  
}

/* ----------------------------------------------------------------------
   Second half of a timestep (V^{n+1/2} -> V^{n+1})
------------------------------------------------------------------------- */

void FixGLEPair::final_integrate()
{

  double dtfm,meff;
  int i,dim1;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  tagint *tag = atom->tag;
  int itag;

  // Advance V by dt
  t1 = MPI_Wtime();
  for (i = 0; i < nlocal; i++) {
    itag = tag[i]-1;
    if (mask[i] & groupbit) {
      meff = mass[type[i]];   
      dtfm = dtf / meff;
      for (dim1=0; dim1<d; dim1++) { 
	v[i][dim1] = int_a * v[i][dim1] 
	  + update->dt/2.0/meff * (int_a*fc[itag][dim1] + f[i][dim1]) 
	  - int_b * fd[itag][dim1]/meff 
	  + int_b*fr[itag][dim1]/meff;
      }
    }
  }
  
  // save conservative force for integration
  for ( i=0; i< nlocal; i++) {
    itag = tag[i]-1;
    fc[itag][0] = f[i][0];
    fc[itag][1] = f[i][1];
    fc[itag][2] = f[i][2];
  }

  // force equals .... (not yet implemented)
  for ( i=0; i< nlocal; i++) {
    itag = tag[i]-1;
    for (dim1=0; dim1<d; dim1++) { 
      f[i][dim1] = fr[itag][dim1];
      array[i][dim1] = fc[itag][dim1];
      array[i][3+dim1] = fd[itag][dim1];
      array[i][6+dim1] = fr[itag][dim1];
    }
  }
  t2 = MPI_Wtime();
  time_int_rel2 += t2 -t1;
  
  // print timing in the last timestep
  if (update->nsteps == update->ntimestep) {
    printf("processor %d: time(read) = %f\n",me,time_read);
    printf("processor %d: time(init) = %f\n",me,time_init);
    printf("processor %d: time(int_rel1) = %f\n",me,time_int_rel1);
    printf("processor %d: time(noise) = %f\n",me,time_noise);
    printf("processor %d: time(matrix_create) = %f\n",me,time_matrix_create);
    printf("processor %d: time(forwardft) = %f\n",me,time_forwardft);
    printf("processor %d: time(sqrt) = %f\n",me,time_sqrt);
    printf("processor %d: time(backwardft) = %f\n",me,time_backwardft);
    printf("processor %d: time(int_rel2) = %f\n",me,time_int_rel2);
  }

}


/* ----------------------------------------------------------------------
   print random force contribution
------------------------------------------------------------------------- */

double FixGLEPair::compute_vector(int n)
{
  
  return fr[n][0];
  
}


/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixGLEPair::memory_usage()
{
  // array to store the velocities
  int N = 2*Nt-2;
  double bytes = (3*atom->nlocal*Nt+3*atom->nlocal*N)*sizeof(double);
  return bytes;
}


/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixGLEPair::grow_arrays(int nmax)
{
  
}

/* ----------------------------------------------------------------------
   write data into restart file:
   - correlation
------------------------------------------------------------------------- */
void FixGLEPair::write_restart(FILE *fp){
  // calculate size of array
  int N = 2*Nt-2;
  int n = 3*atom->nlocal*Nt+3*atom->nlocal*N+2;
  double lastindexn_d = lastindexn;
  double lastindexN_d = lastindexN;

  //write data
  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(&lastindexn_d,sizeof(double),1,fp);
    fwrite(&lastindexN_d,sizeof(double),1,fp);
    fwrite(&x_save[0][0],sizeof(double),3*atom->nlocal*Nt,fp);
    fwrite(&ran[0][0],sizeof(double),3*atom->nlocal*N,fp);
  }
}


/* ----------------------------------------------------------------------
   read data from restart file:
   - correlation
------------------------------------------------------------------------- */
void FixGLEPair::restart(char *buf){
  printf("restart\n");
  double *dbuf = (double *) buf;
  int dcount = 0;
  int N = 2*Nt-2;
  int i,dim1,t;
  
  lastindexn = (int) dbuf[dcount++];
  lastindexN = (int) dbuf[dcount++];

  for (i = 0; i < atom->nlocal; i++) {
    for (dim1 = 0; dim1< d; dim1++) {
      for (t = 0; t < Nt; t++) {
	x_save[3*i+dim1][t] = dbuf[dcount++];
      }
    }
  }

  for (t = 0; t < N; t++)
    for (i = 0; i < atom->nlocal; i++) {
      ran[t][3*i] = dbuf[dcount++];
      ran[t][3*i+1] = dbuf[dcount++];
      ran[t][3*i+2] = dbuf[dcount++];
    }
}


/* ----------------------------------------------------------------------
   read input coefficients
------------------------------------------------------------------------- */
void FixGLEPair::read_input()
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
  
  // default values
  tStart= 0.0;
  tStep = 0.05;
  tStop = 5.0;
  dStart= 6.0;
  dStep = 0.05;
  dStop = 19.95;
  
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
    } else {
      error->one(FLERR,"Invalid keyword in pair table parameters");
    }
    word = strtok(NULL," \t\n\r\f");
  }
  
  if (dStop < dStart)
    error->all(FLERR,"Fix gle/pair/aux dStop must be > dStart");
  Nd = (dStop - dStart) / dStep + 1.5;
  
  if (tStop < tStart)
    error->all(FLERR,"Fix gle/pair/aux tStop must be > tStart");
  Nt = (tStop - tStart) / tStep + 1.5;
    
  // initilize simulations for either TIME input (fitting necessary) or FIT input (only reading necessary)
  int l,t,i;
  double dummy_d, dummy_t,mem;
  
  self_data = new double[Nt];
  cross_data = new double[Nt*Nd];
  self_data_ft = new double[Nt];
  cross_data_ft = new double[Nt*Nd];
  double *time = new double[Nt];
  
  //read self_memory
  for (t=0; t<Nt; t++) {
    fscanf(input,"%lf %lf %lf\n",&dummy_d, &dummy_t, &mem);
    self_data[t] = mem*update->dt;
    time[t] = dummy_t;
  }
  
  // read cross_memory
  for (l=0; l< Nd; l++) {
    for (t=0; t<Nt; t++) {
      fscanf(input,"%lf %lf %lf\n",&dummy_d, &dummy_t, &mem);
      cross_data[Nt*l+t] = mem*update->dt;
    }
  }
  delete [] time;
  fclose(input);
  
}


/* ----------------------------------------------------------------------
   Updates the correlated noise using the Lanczos method
------------------------------------------------------------------------- */

void FixGLEPair::update_noise() 
{
  // initialize input matrix
  int dist,t, counter;
  int k,s;
  int *type = atom->type;
  const int nlocal = atom->nlocal;
  int *tag = atom->tag;
  double **x = atom->x;
  const int N = 2*Nt-2;
  const int size = d*nlocal;
  int i,dim1,j,dim2,ii,jj,inum,jnum,itype,jtype,itag,jtag;
  double xtmp,ytmp,ztmp,rsq,r,ri;
  int *dist_pair_list;
  double **dr_pair_list;
  int neighbours=0;
  int dist_counter=0;
  int *ilist,*jlist,*numneigh,**firstneigh;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  
  // determine number of neighbours
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];
    itag = tag[i]-1;
    jlist = firstneigh[i];
    jnum = numneigh[i];
    for (jj = 0; jj < jnum; jj++) {
      neighbours++;
    }
  }
  
  // set dist/dr_pair_list
  double t1 = MPI_Wtime();
  dist_pair_list = new int[neighbours];
  memory->create(dr_pair_list, neighbours, 3,"gle/pair:dr_pair_list");
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
      j &= NEIGHMASK;
      jtype = type[j];
      jtag = tag[j]-1;
	
      dr_pair_list[dist_counter][0] = xtmp - x[j][0];
      dr_pair_list[dist_counter][1] = ytmp - x[j][1];
      dr_pair_list[dist_counter][2] = ztmp - x[j][2];

      rsq = dr_pair_list[dist_counter][0]*dr_pair_list[dist_counter][0] + dr_pair_list[dist_counter][1]*dr_pair_list[dist_counter][1] + dr_pair_list[dist_counter][2]*dr_pair_list[dist_counter][2];
      r = sqrt(rsq);
      ri = 1.0/r;
      dist = (r - dStart)/dStep;
      dr_pair_list[dist_counter][0]*=ri;
      dr_pair_list[dist_counter][1]*=ri;
      dr_pair_list[dist_counter][2]*=ri;
      dist_pair_list[dist_counter] = dist;
      //if (r < 6.8) printf("itag %d jtag %d, r: %f idst: %d\n",itag,jtag,r,dist);
      dist_counter++;
    }
  }
  double t2 = MPI_Wtime();
  time_matrix_create += t2-t1;
  
  // step 2: determine FT noise vector
  t1 = MPI_Wtime();
  kiss_fft_scalar * buf;
  kiss_fft_cpx * bufout;
  buf=(kiss_fft_scalar*)KISS_FFT_MALLOC(sizeof(kiss_fft_scalar)*size*N);
  bufout=(kiss_fft_cpx*)KISS_FFT_MALLOC(sizeof(kiss_fft_cpx)*size*N);
  memset(bufout,0,sizeof(kiss_fft_cpx)*size*N);
  int n = lastindexN,ind;
  for (t = 0; t < N; t++) {
    ind = Nt-1+t;
    if (ind >= N) ind -= N;
    for (i=0; i<size;i++) {
      buf[i*N+ind]=ran[n][i];
    }
    n--;
    if (n==-1) n=2*Nt-3;
  }  
  // FFT evaluation
  
  #if defined (_OPENMP)
  #pragma omp parallel private(i) default(none) shared(buf,bufout) 
  #endif
  {
    int ifrom, ito, tid;
    loop_setup_thr(ifrom, ito, tid, size,comm->nthreads);
    kiss_fftr_cfg st = kiss_fftr_alloc( N ,0 ,0,0);
    for (i=ifrom; i<ito;i++) {
      kiss_fftr( st ,&buf[i*N],&bufout[i*N] );
    }
    free(st);
  }
  t2 = MPI_Wtime();
  time_forwardft += t2-t1;
  
  // step 3: use lanczos method to compute sqrt-Matrix
  t1 = MPI_Wtime();

  // main Lanczos loop, determine krylov subspace
  std::vector<double *> FT_w;
  for (t=0; t<Nt; t++) {
    double* FT_w_loc = new double[size]; 
    FT_w.push_back(FT_w_loc);
  }
  int *work = new int[8];
  for (i=0; i<8; i++) {
    work[i]=0;
  }
#if defined (_OPENMP)
#pragma omp parallel for private(t,i,j) default(none) shared(bufout,dr_pair_list,dist_pair_list,FT_w,work) schedule(dynamic)
#endif
  for (t=0; t<Nt; t++) {
    std::vector<double *> Vn;
    double* Vn0 = new double[size]; 
    Vn.push_back(Vn0);
    double* rk = new double[size]; 
    for (i=0; i< size; i++) {
      rk[i] = 0.0;
    }
    double *alpha = new double[mLanczos+1];
    double *beta = new double[mLanczos+1];
    for (int k=0; k<mLanczos+1; k++) {
      alpha[k] = 0.0;
      beta[k] = 0.0;
    }
    // input vector is the FFT of the (uncorrelated) noise vector
    double norm = 0.0;
    for (i=0; i< size; i++) {
      Vn[0][i] = bufout[i*N+t].r;
      norm += bufout[i*N+t].r*bufout[i*N+t].r;
    }
    norm = sqrt(norm);
    double normi = 1.0/norm;
    for (i=0; i< size; i++) {
      Vn[0][i] *= normi;
    }
    //rk = A_FT * Vn.col(0);
    compute_step(t,dist_pair_list,dr_pair_list,Vn[0],rk);
    for (i=0; i< size; i++) {
      alpha[1] += Vn[0][i]*rk[i];
    }

    int warn = 0;
    // main laczos loop
    for (int k=2; k<=mLanczos; k++) {
      double norm2 = 0.0;
      for (i=0; i< size; i++) {
        rk[i] = rk[i] - alpha[k-1]*Vn[k-2][i];
        if (k>2) rk[i] -= beta[k-2]*Vn[k-3][i];
	norm2 += rk[i]*rk[i];
      }
      norm2 = sqrt(norm2);
      normi = 1.0/norm2;
      beta[k-1] = norm2;
      // set new v
      double* Vnk = new double[size]; 
      Vn.push_back(Vnk);
      for (i=0; i< size; i++) {
	Vn[k-1][i] = normi*rk[i];
      }
      //rk = A_FT * Vn.col(k-1);
      compute_step(t,dist_pair_list,dr_pair_list,Vn[k-1],rk);
      for (i=0; i< size; i++) {
	alpha[k] += Vn[k-1][i]*rk[i];
      }

      if (k>=2) {
	//generate result vector by contructing Hessenberg-Matrix (and do cholesky-decomposition)
	/*double * f_Hk = new double[k*k];
	for (i=0; i<k*k; i++) {
	  f_Hk[i] = 0.0;
	}
	// determine cholesky factors
	f_Hk[0]=sqrt(alpha[1]);
	f_Hk[1]=beta[1]/f_Hk[0];
	// determine cholesky
	for (i=1; i<k; i++) {
	  f_Hk[i*k] = sqrt(alpha[i+1]-f_Hk[(i-1)*k+1]*f_Hk[(i-1)*k+1]);
	  if (i<k-1) f_Hk[i*k+1] = beta[i+1]/f_Hk[i*k];
	}*/
	// calculate eigenvalue decompostion of hessenberg matrix
	double *d = new double[k+1];
	double *e = new double[k+1];
	double **z;
	memory->create(z,k+1,k+1,"gle/pair:z");
	for (i=0; i<= k; i++) {
	  d[i] = alpha[i];
	  e[i] = beta[i];
	  //printf("%f %f\n",alpha[i],beta[i]);
	  for (j=0; j<= k; j++) {
	    if (i==j) z[i][j] = 1.0;
	    else z[i][j] = 0.0;
	  }
	}
	tqli(d, e, k, z);
	
	// calculate sqrt-matrix
	double **zT;
	memory->create(zT,k+1,k+1,"gle/pair:zT");
	for (i=0; i<= k; i++) {
	  for (j=0; j<= k; j++) {
	    if (d[i] < 0) {
	      if (warn == 0) {
		//printf("w %d, iteration %d, eigenvalue %f\n",t,k,d[i]);
		//error->all(FLERR,"Negative eigenvalue in fix gle/pair decomposition!\n");
		//error->warning(FLERR,"Negative eigenvalue in fix gle/pair decomposition! Set to zero!\n");
		warn = 1;
		d[i] = 0.0;
	      } else {
		d[i] = 0.0;
	      }
	    }
	    zT[i][j] = sqrt(d[i])*z[j][i];
	  }
	}
	double **f_Hk;
	memory->create(f_Hk,k+1,k+1,"gle/pair:f_Hk");
	for (i=0; i<= k; i++) {
	  for (int l=0; l<= k; l++) {
	    f_Hk[i][l] = 0.0;
	    for (j=0; j<= k; j++) {
	      f_Hk[i][l] += z[i][j]*zT[j][l];
	    }
	  }
	}
	
	/*printf("f_Hk\n");
	for (i=0; i<= k; i++) {
	  for (j=0; j<= k; j++) {
	    printf("%f ",f_Hk[i][j]);
	  }
	  printf("\n");
	}
	
	for (i=0; i<= k; i++) {
	  for (int l=0; l<= k; l++) {
	    z[i][l] = 0.0;
	    for (j=0; j<= k; j++) {
	      z[i][l] += f_Hk[i][j]*f_Hk[j][l];
	    }
	  }
	}
	
	printf("f_Hk*f_Hk\n");
	for (i=0; i<= k; i++) {
	  for (j=0; j<= k; j++) {
	    printf("%f ",z[i][j]);
	  }
	  printf("\n");
	}*/
	delete [] d;
	delete [] e;
	memory->destroy(z);
	memory->destroy(zT);
	  
	// determine result vector
	double *res = new double[size];
	for (i=0; i< size; i++) {
	  res[i] = 0.0;
	  for (j=0; j<k; j++) {
	    res[i] += Vn[j][i]*f_Hk[1][j+1]*norm;
	  }
	}
	memory->destroy(f_Hk);
	if (k==2) {
	  for (i=0; i< size; i++) {
	    FT_w[t][i]=res[i];
	  }
	  delete [] res;
	}
	else {
	  double diff = 0.0;
	  for (i=0; i< size; i++) {
	    diff += (FT_w[t][i] - res[i])*(FT_w[t][i] - res[i]);
	  }
	  diff = sqrt(diff);
	  for (i=0; i< size; i++) {
	    FT_w[t][i] = res[i];
	  }
	  delete [] res;
	  // check for convergence
	  if (diff < tolLanczos) {
	    //printf("proc: %d k: %d\n",omp_get_thread_num(),k);
	    //work[omp_get_thread_num()]+=k;
	    break;
	  }
	}
      }
    }
    delete [] alpha;
    delete [] beta;
    delete [] rk;
    for (i=0; i<Vn.size(); i++) {
      delete [] Vn[i];
    }
    Vn.clear();
  }
  
  t2 = MPI_Wtime();
  time_sqrt += t2-t1;
  /*for (i=0; i<8; i++) {
    printf("full work proc %d: %d\n",i,work[i]);
  }*/
  // transform result vector back to time space
  t1 = MPI_Wtime();


  for (int i=0; i<nlocal;i++) {
    itag = tag[i]-1;
    for (int t = 0; t < Nt; t++) {
      if (t==0 || t==Nt-1) {
	fr[itag][0]+= FT_w[t][3*itag+0]/N*sqrt(update->dt);
	fr[itag][1]+= FT_w[t][3*itag+1]/N*sqrt(update->dt);
	fr[itag][2]+= FT_w[t][3*itag+2]/N*sqrt(update->dt);
      } else {
	fr[itag][0]+= 2*FT_w[t][3*itag+0]/N*sqrt(update->dt);
	fr[itag][1]+= 2*FT_w[t][3*itag+1]/N*sqrt(update->dt);
	fr[itag][2]+= 2*FT_w[t][3*itag+2]/N*sqrt(update->dt);
      }
    }
    //printf(" fr : itag %d  %f %f %f \n",itag,fr[itag][0],fr[itag][1],fr[itag][2]);
  }
  
  
  t2 = MPI_Wtime();
  time_backwardft += t2-t1;
  free(buf); free(bufout);
  for (t=0; t<Nt; t++) {
    delete [] FT_w[t];
  }
  FT_w.clear();
  memory->destroy(dr_pair_list);
  delete [] dist_pair_list;
}

/* ----------------------------------------------------------------------
   multiplies an input vector with interaction matrix
------------------------------------------------------------------------- */

void FixGLEPair::compute_step(int w, int* dist_pair_list, double **dr_pair_list, double* input, double* output)
{
  int i,j,ii,jj,inum,jnum,itype,jtype,itag,jtag;
  int *ilist,*jlist,*numneigh,**firstneigh;
  
  int dim1,dist;
  double dot;
  double* dr;

  double **x = atom->x;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int *tag = atom->tag;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  
  int dist_counter = 0;

  // loop over neighbors of my atoms
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];
    itag = tag[i]-1;

    jlist = firstneigh[i];
    jnum = numneigh[i];
      
    // set self-correlation
    for (dim1=0; dim1<d;dim1++) {
      output[itag*d+dim1]+= self_data_ft[w]*input[itag*d+dim1];
    }
      
    //set cross-correlation
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];
      jtag = tag[j]-1;
	
      dist = dist_pair_list[dist_counter];
      dr = dr_pair_list[dist_counter++];
	    
      if (dist < Nd) {
	dot = dr[0]*input[jtag*d]+dr[1]*input[jtag*d+1]+dr[2]*input[jtag*d+2];
	for (dim1=0; dim1<d;dim1++) {
	  output[itag*d+dim1] += cross_data_ft[dist*Nt+w]* dot*dr[dim1];
	}
      }
    }
  }
}

