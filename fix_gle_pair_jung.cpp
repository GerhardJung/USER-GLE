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
#include "fix_gle_pair_jung.h"
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
#include "kiss_fft.h"
#include "kiss_fftr.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;
using namespace Eigen;

#define MAXLINE 1024

enum{TIME,FIT};


/* ----------------------------------------------------------------------
   Parses parameters passed to the method, allocates some memory
------------------------------------------------------------------------- */

FixGLEPairJung::FixGLEPairJung(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  
  global_freq = 1;
  nevery = 1;
  peratom_freq = 1;
  vector_flag = 1;
  
  MPI_Comm_rank(world,&me);

  int narg_min = 5;
  if (narg < narg_min) error->all(FLERR,"Illegal fix gle/pair/jung command");

  t_target = force->numeric(FLERR,arg[3]);
  
  int seed = force->inumeric(FLERR,arg[4]);
  
  // read input file
  input = fopen(arg[5],"r");
  if (input == NULL) {
    char str[128];
    sprintf(str,"Cannot open fix gle/pair/jung file %s",arg[5]);
    error->one(FLERR,str);
  }
  keyword = arg[6];
  
  // Error checking for the first set of required input arguments
  if (seed <= 0) error->all(FLERR,"Illegal fix gle/pair/jung command");
  if (t_target < 0)
    error->all(FLERR,"Fix gle/pair/jung temperature must be >= 0");
  
  // Set number of dimensions
  d=3;
  d2=d*d;
  
  // Timing
  time_read = 0.0;
  time_init = 0.0;
  time_int_rel1 = 0.0;
  time_dist_update = 0.0;
  time_matrix_update = 0.0;
  time_forwardft= 0.0;
  time_chol= 0.0;
  time_backwardft= 0.0;
  time_int_rel2 = 0.0;
  
  // read input file
  t1 = MPI_Wtime();
  read_input();
  t2 = MPI_Wtime();
  time_read += t2 -t1;
  
  // initialize Marsaglia RNG with processor-unique seed
  random = new RanMars(lmp,seed + comm->me);
  
  t1 = MPI_Wtime();
  memory->create(x_save, Nt, 3*atom->nlocal, "gle/pair/aux:x_save");
  memory->create(r_save, atom->nlocal, atom->nlocal, "gle/pair/aux:r_save");
  memory->create(r_step, atom->nlocal, 4*atom->nlocal, "gle/pair/aux:r_step");
  memory->create(f_step, atom->nlocal, 3, "gle/pair/aux:f_step");
  distance_update();
  
  int *type = atom->type;
  double *mass = atom->mass;
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  int_b = 1.0/(1.0+self_data[0]*update->dt/4.0/mass[type[0]]); // 4.0 because K_0 = 0.5*K(0)
  int_a = (1.0-self_data[0]*update->dt/4.0/mass[type[0]])*int_b; // 4.0 because K_0 = 0.5*K(0)
  printf("integration: int_a %f, int_b %f mem %f\n",int_a,int_b,self_data[0]);
  lastindexN = 0,lastindexn=0;
  Nupdate = 0;
  
    
  int nlocal = atom->nlocal;
  double **x = atom->x;
  double **f = atom->f;
  int k,i,j,n,t;
  
  // allocate memory
  int N = 2*Nt-2;
  memory->create(ran, N, atom->nlocal*d, "gle/pair/jung:ran");
  memory->create(fd, atom->nlocal,3, "gle/pair/jung:fd");
  memory->create(fr, atom->nlocal*d,3, "gle/pair/jung:fr");
  size_vector = atom->nlocal;
  
  for (int i = 0; i < nlocal; i++) {
    for (int j = 0; j < nlocal; j++) {
      r_save[i][j]=r_step[i][4*j+3];
    }
  }
  
  // initialize forces
  for ( i=0; i< nlocal; i++) {
    for (int dim1=0; dim1<d; dim1++) { 
      f_step[i][dim1] = 0.0;
      fd[i][dim1] = 0.0;
      fr[i][dim1] = 0.0;
    }
  }
  
  imageint *image = atom->image;
  tagint *tag = atom->tag;
  double unwrap[3];
  for (int t = 0; t < Nt; t++) {
    for (int i = 0; i < nlocal; i++) {
      domain->unmap(x[i],image[i],unwrap);
      for (int dim1=0; dim1<d; dim1++) { 
	x_save[t][3*i+dim1] = unwrap[dim1];
      }
    }
  }
  
  for (int t = 0; t < N; t++) {
    for (int i = 0; i < nlocal; i++) {
      for (int dim1=0; dim1<d; dim1++) { 
	ran[t][d*i+dim1] = random->gaussian();
      }
    }
  }
  
  // initialize interaction matrices
  // CholDecomp to determine Aps (on the fly)
  // update cholesky
  update_cholesky();
  
  t2 = MPI_Wtime();
  time_init += t2 -t1;

}

/* ----------------------------------------------------------------------
   Destroys memory allocated by the method
------------------------------------------------------------------------- */

FixGLEPairJung::~FixGLEPairJung()
{

  delete random;
  memory->destroy(ran);

  memory->destroy(x_save);
  memory->destroy(r_save);
  memory->destroy(r_step);
  memory->destroy(f_step);
  
  memory->destroy(fd);
  memory->destroy(fr);
  
  delete [] cross_data;
  delete [] self_data;

}

/* ----------------------------------------------------------------------
   Specifies when the fix is called during the timestep
------------------------------------------------------------------------- */

int FixGLEPairJung::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ----------------------------------------------------------------------
   Initialize the method parameters before a run
------------------------------------------------------------------------- */

void FixGLEPairJung::init()
{
  
}

/* ----------------------------------------------------------------------
   First half of a timestep (V^{n} -> V^{n+1/2}; X^{n} -> X^{n+1})
------------------------------------------------------------------------- */

void FixGLEPairJung::initial_integrate(int vflag)
{
  double dtfm;
  double ftm2v = force->ftm2v;

  double meff;
  double theta_qss, theta_qsc, theta_qcs, theta_qcc11, theta_qcc12;
  double theta_qsps, theta_qspc;
  int ind_coef, ind_q=0;
  int s,c,m;
  int indi,indd,indj;

  // update v and x of atoms in group
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  tagint *tag = atom->tag;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  
  t1 = MPI_Wtime();
  
  // update noise
  for (int i = 0; i < nlocal; i++) {
    for (int dim1=0; dim1<d; dim1++) { 
      ran[lastindexN][d*i+dim1] = random->gaussian();
      fr[i][dim1] = 0.0;
      fd[i][dim1] = 0.0;
    }
  }
  
  // determine random contribution
  int n = lastindexN;
  int N = 2*Nt -2;
  double sqrt_dt=sqrt(update->dt);
  for (int t = 0; t < N; t++) {
    for (int i = 0; i < nlocal; i++) {
      indi = i*nlocal*d2;
      for (int dim1=0; dim1<d; dim1++) { 
	indd = dim1*d*nlocal;
	for (int j = 0; j < nlocal; j++) {
	  indj = d*j;
	  for (int dim2=0; dim2<d; dim2++) { 
	    fr[i][dim1] += *(a[t].data() +indi+indd+indj+dim2) * ran[n][indj+dim2]*sqrt_dt;
	  }
	}
      }
    }
    n--;
    if (n==-1) n=2*Nt-3;
  }
  
  // determine dissipative contribution
  n = lastindexn;
  m = lastindexn-1;
  if (m==-1) m=Nt-1;
  for (int t = 1; t < Nt; t++) {
    for (int i = 0; i < nlocal; i++) {
      indi = i*nlocal*d2;
      for (int dim1=0; dim1<d; dim1++) { 
	indd = dim1*d*nlocal;
	for (int j = 0; j < nlocal; j++) {
	  indj = d*j;
	  for (int dim2=0; dim2<d; dim2++) { 
	    fd[i][dim1] += *(A[t].data() +indi+indd+indj+dim2) * (x_save[n][3*j+dim2]-x_save[m][3*j+dim2]);
	  }
	}
      }
    }
    n--;
    m--;
    if (n==-1) n=Nt-1;
    if (m==-1) m=Nt-1;
  }
  t2 = MPI_Wtime();
  time_int_rel1 += t2 -t1;
  
  // Advance X by dt
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      meff = mass[type[i]];   
      //printf("x: %f f: %f fd: %f fr: %f\n",x[i][0],f_step[i][0],fd[i],fr[i]);
      for (int dim1=0; dim1<d; dim1++) { 
	x[i][dim1] += int_b * update->dt * v[i][dim1] 
	  + int_b * update->dt * update->dt / 2.0 / meff * f_step[i][dim1] 
	  - int_b * update->dt / meff/ 2.0 * fd[i][dim1]
	  + int_b*update->dt/ 2.0 / meff * fr[i][dim1]; // convection, conservative, dissipative, random
      }
      //printf("x: %f\n",x[i][0]);
      //x[i][1] += dtv * v[i][1];
      //x[i][2] += dtv * v[i][2];
    }
  }
  
  lastindexN++;
  if (lastindexN == 2*Nt-2) lastindexN = 0;
  lastindexn++;
  if (lastindexn == Nt) lastindexn = 0;


  
  // Update cholesky if necessary

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
    Nupdate++;
    //printf("Updating amplitudes\n");
    update_cholesky();
    
    // update r_save
    for (int i = 0; i < nlocal; i++) {
      for (int j = 0; j < nlocal; j++) {
	r_save[i][j]=r_step[i][4*j+3];
      }
    }
  }
  
  t1 = MPI_Wtime();
  // Determine normalized distance vectors
  distance_update();
  
  // Update positions
  imageint *image = atom->image;
  double unwrap[3];
  for (int i = 0; i < nlocal; i++) {
    domain->unmap(x[i],image[i],unwrap);
    x_save[lastindexn][3*i] = unwrap[0];
    x_save[lastindexn][3*i+1] = unwrap[1];
    x_save[lastindexn][3*i+2] = unwrap[2];
    //printf("x: %f %f %f uw: %f %f %f mass %f tag %d\n",x[i][0],x[i][1],x[i][2],unwrap[0],unwrap[1],unwrap[2],mass[type[i]],tag[i]);
    /*x_save[lastindex][3*i] = x[i][0];
    x_save[lastindex][3*i+1] = x[i][1];
    x_save[lastindex][3*i+2] = x[i][2];*/
  }
  
  t2 = MPI_Wtime();
  time_dist_update += t2 -t1;
  
  
}

/* ----------------------------------------------------------------------
   Second half of a timestep (V^{n+1/2} -> V^{n+1})
------------------------------------------------------------------------- */

void FixGLEPairJung::final_integrate()
{

  double dtfm;
  double ftm2v = force->ftm2v;

  double meff;
  double theta_vs, alpha_vs, theta_vc, alpha_vc;
  int ind_coef, ind_q;

  t1 = MPI_Wtime();
  // update v and x of atoms in group
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // Advance V by dt
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      // Calculate integration constants
      meff = mass[type[i]];   
      dtfm = dtf / meff;
      for (int dim1=0; dim1<d; dim1++) { 
	v[i][dim1] = int_a * v[i][dim1] 
	  + update->dt/2.0/meff * (int_a*f_step[i][dim1] + f[i][dim1]) 
	  - int_b * fd[i][dim1]/meff 
	  + int_b*fr[i][dim1]/meff;
      }
      //v[i][1] += dtfm * f_step[i][1];
      //v[i][2] += dtfm * f_step[i][2];
      //printf("v: %f %f %f\n",v[i][0],v[i][1],v[i][2]);
    }
  }
  
  // save conservative force for integration
  for ( int i=0; i< nlocal; i++) {
    f_step[i][0] = f[i][0];
    f_step[i][1] = f[i][1];
    f_step[i][2] = f[i][2];
  }

  // force equals .... (not yet implemented)
  for ( int i=0; i< nlocal; i++) {
    f[i][0] = fr[i][0];
    f[i][1] = fr[i][1];
    f[i][2] = fr[i][2];
  }

  t2 = MPI_Wtime();
  time_int_rel2 += t2 -t1;
  
      // print timing
  if (update->nsteps == update->ntimestep || update->ntimestep % 100000 == 0) {
    printf("Update %d times\n",Nupdate);
    printf("processor %d: time(read) = %f\n",me,time_read);
    printf("processor %d: time(init) = %f\n",me,time_init);
    printf("processor %d: time(int_rel1) = %f\n",me,time_int_rel1);
    printf("processor %d: time(matrix_update) = %f\n",me,time_matrix_update);
    printf("processor %d: time(forward_ft) = %f\n",me,time_forwardft);
    printf("processor %d: time(cholesky) = %f\n",me,time_chol);
    printf("processor %d: time(backward_ft) = %f\n",me,time_backwardft);
    printf("processor %d: time(dist_update) = %f\n",me,time_dist_update);
    printf("processor %d: time(int_rel2) = %f\n",me,time_int_rel2);
  }
  

}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixGLEPairJung::compute_vector(int n)
{
  tagint *tag = atom->tag;
  
  //printf("%d %d\n",t,i);
  
  return fr[n][0];
}


/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixGLEPairJung::memory_usage()
{
  double bytes = atom->nlocal*atom->nlocal*Nt*Nt*sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixGLEPairJung::grow_arrays(int nmax)
{
  
}

/* ----------------------------------------------------------------------
   read input coefficients
------------------------------------------------------------------------- */
void FixGLEPairJung::read_input()
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
  
  if (memory_flag == TIME) {
    if (tStop < tStart)
      error->all(FLERR,"Fix gle/pair/aux tStop must be > tStart");
    Nt = (tStop - tStart) / tStep + 1.5;
  }
  
    
  // initilize simulations for either TIME input (fitting necessary) or FIT input (only reading necessary)
  int d,t,i,n;
  double dummy_d, dummy_t,mem;
  
  self_data = new double[Nt];
  cross_data = new double[Nt*Nd];
  double *time = new double[Nt];
  
  //read self_memory
  for (t=0; t<Nt; t++) {
    fscanf(input,"%lf %lf %lf\n",&dummy_d, &dummy_t, &mem);
    self_data[t] = mem*update->dt;
    time[t] = dummy_t;
    //printf("%f\n",mem);
  }
  
  // read cross_memory
  for (d=0; d< Nd; d++) {
    for (t=0; t<Nt; t++) {
      fscanf(input,"%lf %lf %lf\n",&dummy_d, &dummy_t, &mem);
      cross_data[Nt*d+t] = mem*update->dt;
      //printf("%f %f %f\n",dummy_d,dummy_t,mem);
    }
  }
  delete [] time;
  fclose(input);
  
}

/* ----------------------------------------------------------------------
   updates the interaction amplitudes by cholesky decomposition
------------------------------------------------------------------------- */
void FixGLEPairJung::update_cholesky() 
{
  // initialize input matrix
  int k,dist,i,j,n;
  int nlocal = atom->nlocal;
  double **x = atom->x;
  tagint *tag = atom->tag;
  int indi,indd,indj;
  int indiFFT,inddFFT,indjFFT;
  double t1 = MPI_Wtime();
  A.clear();
  for (int t = 0; t < Nt; t++) {
    Eigen::MatrixXd A_loc = Eigen::MatrixXd::Zero(d*nlocal,d*nlocal);
    for (int i = 0; i < nlocal; i++) {
      indi = i*nlocal*d2;
      for (int dim1=0; dim1<d;dim1++) {
	indd = dim1*d*nlocal;
	for (int j = 0; j < nlocal; j++) {
	  indj = d*j;
	  if (i==j) {
	    double data = self_data[t];
	    for (int dim2=0; dim2<d;dim2++) {
	      if (dim1==dim2) *(A_loc.data() +i*nlocal*d2+dim1*nlocal*d+j*d+dim2) = data;
	      else *(A_loc.data() +i*nlocal*d2+dim1*nlocal*d+j*d+dim2) = 0.0;
	    }
	  }
	  else {
	    //printf("%f \n",r_step[i][4*(j)+3]);
	    dist = (r_step[i][4*(j)+3]- dStart)/dStep;
	    if (dist <= 0) {
	      error->all(FLERR,"Particles closer than lower cutoff in fix/pair/gle\n");
	    } else if (dist >= Nd) {
	      for (int dim2=0; dim2<d;dim2++) {
		*(A_loc.data() +indi+indd+indj+dim2) = 0.0;
	      }
	    } else {
	      double data = cross_data[Nt*dist+t];
	      for (int dim2=0; dim2<d;dim2++) {
		*(A_loc.data() +i*nlocal*d2+dim1*nlocal*d+j*d+dim2) = data*r_step[i][4*j+dim1]*r_step[i][4*j+dim2];
	      }
	    }
	  }
	}
      }
    }
    A.push_back(A_loc);
    //cout << A_loc << endl;
  }
  double t2 = MPI_Wtime();
  time_matrix_update += t2 -t1;
  vector<Eigen::MatrixXd> A_FT;
  vector<Eigen::MatrixXd> a_FT;
  
  // step 1: perform FT for every entry of A
  int N = 2*Nt-2;
  kiss_fft_scalar * buf;
  kiss_fft_cpx * bufout;
  buf=(kiss_fft_scalar*)KISS_FFT_MALLOC(sizeof(kiss_fft_scalar)*N*nlocal*nlocal*d2);
  bufout=(kiss_fft_cpx*)KISS_FFT_MALLOC(sizeof(kiss_fft_cpx)*N*nlocal*nlocal*d2);
  t1 = MPI_Wtime();
  for (int i=0; i<nlocal;i++) {
    indi = i*nlocal*d2;
    indiFFT = i*nlocal*d2*N;
    for (int dim1=0; dim1<d;dim1++) {
      indd = dim1*nlocal*d;
      inddFFT = dim1*nlocal*d*N;
      for (int j=0; j<nlocal;j++) {
	indj = d*j;
	indjFFT = d*N*j; 
	for (int dim2=0; dim2<d;dim2++) {
	  for (int t=0; t<Nt; t++) {
	    //data[i*nlocal*Nt+j*Nt+t] = A[t](i,j);
	    buf[indiFFT+inddFFT+indjFFT+N*dim2+t] = *(A[t].data() +indi+indd+indj+dim2);
	    if (t==0 || t==Nt-1){
	    //  buf[i*nlocal*N+j*N+N-t].r = 0;
	    } else buf[indiFFT+inddFFT+indjFFT+N*dim2+N-t] = *(A[t].data() +indi+indd+indj+dim2);
	  }
	}
      }
    }
  }
  t2 = MPI_Wtime();
  time_matrix_update += t2 -t1;
  // do the fft with kiss_fft
  kiss_fftr_cfg st = kiss_fftr_alloc( N ,0 ,0,0);
  t1 = MPI_Wtime();
  for (int i=0; i<nlocal;i++) {
    indiFFT = i*nlocal*d2*N;
    for (int dim1=0; dim1<d;dim1++) {
      inddFFT = dim1*nlocal*d*N;
      for (int j=0; j<nlocal;j++) {
	indjFFT = d*N*j; 
	for (int dim2=0; dim2<d;dim2++) {
	  kiss_fftr( st ,&buf[indiFFT+inddFFT+indjFFT+N*dim2],&bufout[indiFFT+inddFFT+indjFFT+N*dim2] );
	}
      }
    }
  }
  t2 = MPI_Wtime();
  time_forwardft += t2 -t1;
  t1 = MPI_Wtime();
  for (int t=0; t<Nt; t++) {
    //printf("FT(A)[%d] ",t);
    Eigen::MatrixXd A_FT0(d*nlocal,d*nlocal);
    for (int i=0; i<nlocal;i++) {
      indi = i*nlocal*d2;
      indiFFT = i*nlocal*d2*N;
      for (int dim1=0; dim1<d;dim1++) {
	indd = dim1*nlocal*d;
	inddFFT = dim1*nlocal*d*N;
	for (int j=0; j<nlocal;j++) {
	  indj = d*j;
	  indjFFT = d*N*j; 
	  for (int dim2=0; dim2<d;dim2++) {
	    //printf("(%d,%d): %f ",i,j,/*FT_data[i*nlocal*N+j*N+t].real(),FT_data[i*nlocal*N+j*N+t].imag(),*/bufout[i*nlocal*N+j*N+t].r);
	    *(A_FT0.data() +indi+indd+indj+dim2) = bufout[indiFFT+inddFFT+indjFFT+N*dim2+t].r;
	  }
	}
      }
    } 
    //printf("\n");
    A_FT.push_back(A_FT0);
  }
  t2 = MPI_Wtime();
  time_matrix_update += t2 -t1;
  //printf("-------------------------------\n");
  
  // step 2: perform cholesky decomposition for every Aw
  t1 = MPI_Wtime();
  for (int t=0; t<Nt; t++) {
    //cout << A_FT[t] << endl;
    Eigen::LLT<Eigen::MatrixXd> A_comp(A_FT[t]); // compute the Cholesky decomposition of A
    if (A_comp.info()!=0) {
      error->all(FLERR,"LLT Cholesky not possible!\n");
    }
    Eigen::MatrixXd a_FT0 = A_comp.matrixL().transpose();
    a_FT.push_back(a_FT0);
  }
  t2 = MPI_Wtime();
  time_chol += t2 -t1;
  t1 = MPI_Wtime();

  //printf("FT(a)[%d] ",t);
  for (int i=0; i<nlocal;i++) {
    indi = i*nlocal*d2;
    indiFFT = i*nlocal*d2*N;
    for (int dim1=0; dim1<d;dim1++) {
      indd = dim1*nlocal*d;
      inddFFT = dim1*nlocal*d*N;
      for (int j=0; j<nlocal;j++) {
	indj = d*j;
	indjFFT = d*N*j; 
	for (int dim2=0; dim2<d;dim2++) {
	  for (int t=0; t<Nt; t++) {
	    //printf("(%d,%d): %f ",i,j,a_FT[t](i,j));
	    bufout[indiFFT+inddFFT+indjFFT+N*dim2+t].r = *(a_FT[t].data() +indi+indd+indj+dim2);
	    bufout[indiFFT+inddFFT+indjFFT+N*dim2+t].i = 0.0;
	    if (t==0 || t==Nt-1){ }
	    else {
	      bufout[indiFFT+inddFFT+indjFFT+N*dim2+N-t].r = *(a_FT[t].data() +indi+indd+indj+dim2);
	      bufout[indiFFT+inddFFT+indjFFT+N*dim2+N-t].i = 0.0;
	    }
	  }
	}
      }
    } 
    //printf("\n");
  }
  t2 = MPI_Wtime();
  time_matrix_update += t2 -t1;
  
  A_FT.clear();
  a_FT.clear();

  kiss_fftr_cfg sti = kiss_fftr_alloc( N ,1 ,0,0);
  t1 = MPI_Wtime();
  for (int i=0; i<nlocal;i++) {
    indiFFT = i*nlocal*d2*N;
    for (int dim1=0; dim1<d;dim1++) {
      inddFFT = dim1*nlocal*d*N;
      for (int j=0; j<nlocal;j++) {
	indjFFT = d*N*j; 
	for (int dim2=0; dim2<d;dim2++) {
	  kiss_fftri( sti ,&bufout[indiFFT+inddFFT+indjFFT+N*dim2],&buf[indiFFT+inddFFT+indjFFT+N*dim2] );
	}
      }
    }
  }
  t2 = MPI_Wtime();
  time_backwardft += t2 -t1;
  free(st); free(sti);
  kiss_fft_cleanup();
  t1 = MPI_Wtime();
  a.clear();
  for (int t=0; t<N; t++) {
    //printf("a[%d] ",t);
    Eigen::MatrixXd a0(d*nlocal,d*nlocal);
    int ind = t+Nt-1;
    if (ind >= N) ind -= N;
    for (int i=0; i<nlocal;i++) {
      indi = i*nlocal*d2;
      indiFFT = i*nlocal*d2*N;
      for (int dim1=0; dim1<d;dim1++) {
	indd = dim1*nlocal*d;
	inddFFT = dim1*nlocal*d*N;
	for (int j=0; j<nlocal;j++) {
	  indj = d*j;
	  indjFFT = d*N*j; 
	  //printf("(%d,%d): %f ",i,j,buf[i*nlocal*N+j*N+t]/N);
	  // resort indices
	  for (int dim2=0; dim2<d;dim2++) {
	    *(a0.data() +indi+indd+indj+dim2) = buf[indiFFT+inddFFT+indjFFT+N*dim2+ind]/N;
	  }
	}
      }
    } 
    //printf("\n");
    a.push_back(a0);
  }
  t2 = MPI_Wtime();
  time_matrix_update += t2 -t1;
  //printf("-------------------------------\n");
  free(buf); free(bufout);

  
  // step 4: test the method
  /*for(int t=0;t<Nt;t++){
    Eigen::MatrixXd A_res = Eigen::MatrixXd::Zero(nlocal,nlocal);
    Eigen::MatrixXd A_loc;
    for(int s=0;s<N;s++){
      int ind = t+s;
      if (t+s>=N) ind -= N;
      A_loc = a[ind].transpose()*a[s];
      A_res += A_loc;
    }
    printf("A[%d] ",t);
    for (int i=0; i<nlocal;i++) {
      for (int j=0; j<nlocal;j++) {
	printf("(%d,%d): %f==%f ",i,j,A_res(i,j),A[t](i,j));
      }
    }
    printf("\n");
  }*/
}

/* ----------------------------------------------------------------------
   updates the list of distance vectors between particles
------------------------------------------------------------------------- */
void FixGLEPairJung::distance_update() 
{
  int i,j;
  int nlocal = atom->nlocal;
  double **x = atom->x;
  double xtmp, ytmp, ztmp, delx,dely,delz, dist,idist;
  double min_dist = 20.0;
  tagint *tag = atom->tag;
  
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
      
	r_step[i][4*(j)] = delx*idist;
	r_step[i][4*(j)+1] = dely*idist;
	r_step[i][4*(j)+2] = delz*idist;
	r_step[i][4*(j)+3] = dist;
	
      }
    }
  }
  
  if (update->ntimestep%10==0) {
    //printf("min_dist %f\n",min_dist);
  }
}


