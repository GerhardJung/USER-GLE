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
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include "kiss_fft.h"
#include "kiss_fftr.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;
using namespace Eigen;
typedef Eigen::Triplet<double> Td;
typedef Eigen::Triplet<int> Ti;

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
  vector_flag = 1;
  
  MPI_Comm_rank(world,&me);

  int narg_min = 5;
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
  
  // error checking for the first set of required input arguments
  if (seed <= 0) error->all(FLERR,"Illegal fix gle/pair command");
  if (t_target < 0)
    error->all(FLERR,"Fix gle/pair temperature must be >= 0");
  
  // set number of dimensions
  d=3;
  d2=d*d;
  
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
  memory->create(x_save, Nt, 3*atom->nlocal, "gle/pair:x_save");
  memory->create(ran, N, atom->nlocal*d, "gle/pair:ran");
  memory->create(fd, atom->nlocal,3, "gle/pair:fd");
  memory->create(fc, atom->nlocal,3, "gle/pair:fc");
  memory->create(fr, atom->nlocal,3, "gle/pair:fr");
  size_vector = atom->nlocal;
  
  // initialize forces
  for ( i=0; i< nlocal; i++) {
    for (int dim1=0; dim1<d; dim1++) { 
      fc[i][dim1] = 0.0;
      fd[i][dim1] = 0.0;
      fr[i][dim1] = 0.0;
    }
  }
  
  // initiliaze position storage
  imageint *image = atom->image;
  tagint *tag = atom->tag;
  int itag;
  double unwrap[3];
  for (int t = 0; t < Nt; t++) {
    for (int i = 0; i < nlocal; i++) {
      itag = tag[i] - 1;
      domain->unmap(x[itag],image[itag],unwrap);
      for (int dim1=0; dim1<d; dim1++) { 
	x_save[t][3*itag+dim1] = unwrap[dim1];
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
  self_data_ft = new double[Nt];
  cross_data_ft = new double[Nt*Nd];
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
      //printf("%f\n",cross_data_ft[Nt*l+t]);
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
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  tagint *tag = atom->tag;
  
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
  int j,dim2,ii,jj,inum,jnum,itype,jtype,jtag;
  double xtmp,ytmp,ztmp,rsq,rsqi;
  int *ilist,*jlist,*numneigh,**firstneigh;
  neighbor->build_one(list);
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  double *dr = new double[3];
  double proj;
  int dist;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    itag = tag[i]-1;
    jlist = firstneigh[i];
    jnum = numneigh[i];
      
    // self-correlation contribution
    for (dim1=0; dim1<d;dim1++) {
      n = lastindexn;
      m = lastindexn-1;
      if (m==-1) m=Nt-1;
      for (t = 1; t < Nt; t++) {
	fd[itag][dim1] += self_data[t]*(x_save[n][3*itag+dim1]-x_save[m][3*itag+dim1]);
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
      dist = (sqrt(rsq) - dStart)/dStep;
	    
      if (dist < 0) {
	printf("dist: %f, lower cutoff: %f\n",sqrt(rsq),dStart);
	error->all(FLERR,"Particles closer than lower cutoff in fix/pair\n");
      } else if (dist < Nd) {
	for (dim1=0; dim1<d;dim1++) {
	  for (dim2=0; dim2<d;dim2++) {
	    proj = dr[dim1]*dr[dim2]*rsqi;
	    if (proj != 0.0) {
	      n = lastindexn;
	      m = lastindexn-1;
	      if (m==-1) m=Nt-1;
	      for (t = 1; t < Nt; t++) {
		fd[itag][dim1] += proj*cross_data[dist*Nt+t]*(x_save[n][3*jtag+dim2]-x_save[m][3*jtag+dim2]);
		n--;
		m--;
		if (n==-1) n=Nt-1;
		if (m==-1) m=Nt-1;
	      }
	    }
	  }
	}
      }
    }
  }
  delete [] dr;
  
  //printf("new fr %f %f %f\n",fr[0][0],fr[0][1],fr[0][2]);
  //printf("new fd %f %f %f\n",fd[0][0],fd[0][1],fd[0][2]);
  
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
    x_save[lastindexn][3*itag] = unwrap[0];
    x_save[lastindexn][3*itag+1] = unwrap[1];
    x_save[lastindexn][3*itag+2] = unwrap[2];
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
    f[i][0] = fr[itag][0];
    f[i][1] = fr[itag][1];
    f[i][2] = fr[itag][2];
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
  double bytes = atom->nlocal*atom->nlocal*d2*sizeof(double);
  return bytes;
}


/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixGLEPair::grow_arrays(int nmax)
{
  
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
  int nlocal = atom->nlocal;
  int *tag = atom->tag;
  double **x = atom->x;
  int N = 2*Nt-2;
  int size = d*nlocal;
  int i,dim1,j,dim2,ii,jj,inum,jnum,itype,jtype,itag,jtag;
  double xtmp,ytmp,ztmp,rsq,rsqi;
  double *dr = new double[3];
  double proj;
  int *ilist,*jlist,*numneigh,**firstneigh;
  neighbor->build_one(list);
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  
  // step 1: create matrix A_FT (for t=0)
  double t1 = MPI_Wtime();
  std::vector<Td> tripletListd;
  std::vector<Ti> tripletListi;
  Eigen::SparseMatrix<double > A_FT(d*nlocal,d*nlocal);
  Eigen::SparseMatrix<int > A_dist(d*nlocal,d*nlocal);

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    itag = tag[i]-1;

    jlist = firstneigh[i];
    jnum = numneigh[i];
      
    // set self-correlation
    for (dim1=0; dim1<d;dim1++) {
      tripletListd.push_back(Td(itag*d+dim1,itag*d+dim1,self_data_ft[0]));
      tripletListi.push_back(Ti(itag*d+dim1,itag*d+dim1,-1));
    }
      
    //set cross-correlation
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
      dist = (sqrt(rsq) - dStart)/dStep;
	    
      if (dist < 0) {
	printf("dist: %f, lower cutoff: %f\n",sqrt(rsq),dStart);
	error->all(FLERR,"Particles closer than lower cutoff in fix/pair\n");
      } else if (dist < Nd) {
	for (dim1=0; dim1<d;dim1++) {
	  for (dim2=0; dim2<d;dim2++) {
	    proj = dr[dim1]*dr[dim2]*rsqi;
	    if (proj != 0.0) {
	      tripletListd.push_back(Td(itag*d+dim1,jtag*d+dim2,cross_data_ft[dist*Nt]*proj));
	      tripletListi.push_back(Ti(itag*d+dim1,jtag*d+dim2,dist));
	    }
	  }
	}
      }
    }
  }
  A_FT.setFromTriplets(tripletListd.begin(), tripletListd.end());
  A_dist.setFromTriplets(tripletListi.begin(), tripletListi.end());
  delete [] dr;
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
  kiss_fftr_cfg st = kiss_fftr_alloc( N ,0 ,0,0);
  for (i=0; i<size;i++) {
    kiss_fftr( st ,&buf[i*N],&bufout[i*N] );
  }
  free(st);
  t2 = MPI_Wtime();
  time_forwardft += t2-t1;
  
  // step 3: use lanczos method to compute sqrt-Matrix
  t1 = MPI_Wtime();
  int mLanczos = 50;
  double tolLanczos = 0.00001;

  // main Lanczos loop, determine krylov subspace
  std::vector<VectorXd> FT_w;
  
  for (t=0; t<Nt; t++) {
    // update correlation matrix
    if (t>0) {
      counter = 0;
      for (k=0; k<A_FT.outerSize(); ++k) {
	for (SparseMatrix<double>::InnerIterator it(A_FT,k); it; ++it) {
	  dist = A_dist.valuePtr()[counter];
	  if (dist == -1) {
	    it.valueRef() *= self_data_ft[t]/self_data_ft[t-1];
	  }
	  else it.valueRef() *= cross_data_ft[dist*Nt+t]/cross_data_ft[dist*Nt+t-1];
	  counter++;
	}
      }
    }
    Eigen::MatrixXd Vn;
    Vn.resize(size,1);
    VectorXd rk = VectorXd::Zero(size);
    double *alpha = new double[mLanczos+1];
    double *beta = new double[mLanczos+1];
    for (int k=0; k<mLanczos+1; k++) {
      alpha[k] = 0.0;
      beta[k] = 0.0;
    }
    // input vector is the FFT of the (uncorrelated) noise vector
    double norm = 0.0;
    Vn.col(0) = VectorXd::Zero(size);
    for (i=0; i< size; i++) {
      Vn(i,0) = bufout[i*N+t].r;
    }
    norm = Vn.col(0).norm();
    Vn.col(0) = Vn.col(0).normalized();
    double t1 = MPI_Wtime();
    rk = A_FT * Vn.col(0);
    double t2 = MPI_Wtime();
    alpha[1] = (Vn.col(0).adjoint()*rk).value();

    // main laczos loop
    for (int k=2; k<=mLanczos; k++) {
      rk = rk - alpha[k-1]*Vn.col(k-2);
      if (k>2) rk -= beta[k-2]*Vn.col(k-3);
      beta[k-1] = rk.norm();
      // set new v
      Vn.conservativeResize(size,k);
      Vn.col(k-1) = rk.normalized();
      rk = A_FT * Vn.col(k-1);
      alpha[k] = (Vn.col(k-1).adjoint()*rk).value();

      if (k>=2) {
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

	// determine sqrt-matrix (only on the small Hessenberg-Matrix)
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> Hk_eigen(Hk);
	Eigen::MatrixXd Hk_eigenvector = Hk_eigen.eigenvectors().real();
	if (Hk_eigen.info()!=0) {
	  cout << A_dist << endl;
	  cout << A_FT << endl;
	  cout << Hk << endl;
	  error->all(FLERR,"Hk is not positive-definite in pair/gle\n");
	}  
	Eigen::MatrixXd Hk_diag = Eigen::MatrixXd::Zero(k,k);
	for (int i=0; i<k; i++) {
	  Hk_diag(i,i) = sqrt(Hk_eigen.eigenvalues().real()(i));
	}
	MatrixXd f_Hk = Hk_eigenvector * Hk_diag * Hk_eigenvector.transpose();
	  
	// determine result vector
	VectorXd e1 = VectorXd::Zero(k);
	e1(0) = 1.0;
	VectorXd f_Hk1 = f_Hk * e1;
	VectorXd xk = Vn*f_Hk1*norm; 
	if (k==2) FT_w.push_back(xk);
	else {
	  VectorXd diff = (FT_w[t] - xk);
	  double diff_norm = diff.norm();
	  FT_w[t] = xk;
	  // check for convergence
	  if (diff_norm < tolLanczos) {
	    //printf("%d\n",k);
	    break;
	  }
	}
      }
    }
    delete [] alpha;
    delete [] beta;
  }
  t2 = MPI_Wtime();
  time_sqrt += t2-t1;
  
  // transform result vector back to time space
  t1 = MPI_Wtime();


  for (int i=0; i<nlocal;i++) {
    itag = tag[i]-1;
    for (int t = 0; t < Nt; t++) {
      if (t==0 || t==Nt-1) {
	fr[itag][0]+= FT_w[t](3*itag+0)/N*sqrt(update->dt);
	fr[itag][1]+= FT_w[t](3*itag+1)/N*sqrt(update->dt);
	fr[itag][2]+= FT_w[t](3*itag+2)/N*sqrt(update->dt);
      } else {
	fr[itag][0]+= 2*FT_w[t](3*itag+0)/N*sqrt(update->dt);
	fr[itag][1]+= 2*FT_w[t](3*itag+1)/N*sqrt(update->dt);
	fr[itag][2]+= 2*FT_w[t](3*itag+2)/N*sqrt(update->dt);
      }
    }
  }
  
  
  t2 = MPI_Wtime();
  time_backwardft += t2-t1;
  free(buf); free(bufout);
  
}

