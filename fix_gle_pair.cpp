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
#include "kiss_fft.h"
#include "kiss_fftr.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;
using namespace Eigen;
typedef ConjugateGradient<SparseMatrix<double>,Lower, IncompleteCholesky<double> > ICCG;
typedef Eigen::Triplet<double> T;

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
  time_chol_analyze= 0.0;
  time_chol_factorize= 0.0;
  time_backwardft= 0.0;
  time_int_rel2 = 0.0;
  time_test = 0.0;
  
  // read input file
  t1 = MPI_Wtime();
  read_input();
  t2 = MPI_Wtime();
  time_read += t2 -t1;
  
  // initialize Marsaglia RNG with processor-unique seed
  random = new RanMars(lmp,seed + comm->me);
  
  t1 = MPI_Wtime();
  memory->create(x_save, Nt, 3*atom->nlocal, "gle/pair/aux:x_save");
  memory->create(x_save_update,atom->nlocal,3, "gle/pair/aux:x_save_update");
  memory->create(fc, atom->nlocal, 3, "gle/pair/aux:fc");
  
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
  
  // initialize forces
  for ( i=0; i< nlocal; i++) {
    for (int dim1=0; dim1<d; dim1++) { 
      fc[i][dim1] = 0.0;
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
	x_save_update[i][dim1] = x[i][dim1];
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
  memory->destroy(x_save_update);
  
  memory->destroy(fc);
  memory->destroy(fd);
  memory->destroy(fr);
  
  delete [] cross_data;
  delete [] self_data;

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
  isInitialized = 0;
}

/* ----------------------------------------------------------------------
   First half of a timestep (V^{n} -> V^{n+1/2}; X^{n} -> X^{n+1})
------------------------------------------------------------------------- */

void FixGLEPair::initial_integrate(int vflag)
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
  
  // initilize in the first step
  if (!isInitialized) {
    list = neighbor->lists[irequest];
    update_cholesky();
    isInitialized = 1;
  }
  
  // determine random contribution
  int n = lastindexN;
  int N = 2*Nt -2;
  double sqrt_dt=sqrt(update->dt);
  for (int t = 0; t < N; t++) {
    for (int k=0; k<a[t].outerSize(); ++k) {
      int dim = k%d;
      int i = (k-dim)/d;
      for (SparseMatrix<double>::InnerIterator it(a[t],k); it; ++it) {
	fr[i][dim] += it.value()*ran[n][it.row()]*sqrt_dt;
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
    for (int k=0; k<A[t].outerSize(); ++k) {
      int dim = k%d;
      int i = (k-dim)/d;
      for (SparseMatrix<double>::InnerIterator it(A[t],k); it; ++it) {
	int dimj = it.row()%d;
	int j = (it.row()-dimj)/d;
	fd[i][dim] += it.value()* (x_save[n][3*j+dimj]-x_save[m][3*j+dimj]);
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
	  + int_b * update->dt * update->dt / 2.0 / meff * fc[i][dim1] 
	  - int_b * update->dt / meff/ 2.0 * fd[i][dim1]
	  + int_b*update->dt/ 2.0 / meff * fr[i][dim1]; // convection, conservative, dissipative, random
      }
    }
  }
  
  lastindexN++;
  if (lastindexN == 2*Nt-2) lastindexN = 0;
  lastindexn++;
  if (lastindexn == Nt) lastindexn = 0;

  // Check whether Cholesky Update is necessary
  double dr_max = 0.0;
  double dx,dy,dz,rsq;
  for (int i = 0; i < nlocal; i++) {
    dx = x_save_update[i][0] - x[i][0];
    dy = x_save_update[i][1] - x[i][1];
    dz = x_save_update[i][2] - x[i][2];
    rsq = dx*dx+dy*dy+dz*dz;
    if (rsq > dr_max) dr_max = rsq;
  }
  //printf("dr_max %f\n",dr_max);
  //if (dr_max > dStep*dStep/4.0) 
  {
    Nupdate++;
    for (int i = 0; i < nlocal; i++) {
      x_save_update[i][0] = x[i][0];
      x_save_update[i][1] = x[i][1];
      x_save_update[i][2] = x[i][2];
    }
    update_cholesky();
  }
  
  t1 = MPI_Wtime();
  
  // Update positions
  imageint *image = atom->image;
  double unwrap[3];
  for (int i = 0; i < nlocal; i++) {
    domain->unmap(x[i],image[i],unwrap);
    x_save[lastindexn][3*i] = unwrap[0];
    x_save[lastindexn][3*i+1] = unwrap[1];
    x_save[lastindexn][3*i+2] = unwrap[2];
  }
  
  t2 = MPI_Wtime();
  time_dist_update += t2 -t1;
  
  
}

/* ----------------------------------------------------------------------
   Second half of a timestep (V^{n+1/2} -> V^{n+1})
------------------------------------------------------------------------- */

void FixGLEPair::final_integrate()
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
	  + update->dt/2.0/meff * (int_a*fc[i][dim1] + f[i][dim1]) 
	  - int_b * fd[i][dim1]/meff 
	  + int_b*fr[i][dim1]/meff;
      }
    }
  }
  
  // save conservative force for integration
  for ( int i=0; i< nlocal; i++) {
    fc[i][0] = f[i][0];
    fc[i][1] = f[i][1];
    fc[i][2] = f[i][2];
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
     printf("processor %d: time(cholesky_analyze) = %f\n",me,time_chol_analyze);
     printf("processor %d: time(cholesky_factorize) = %f\n",me,time_chol_factorize);
    printf("processor %d: time(backward_ft) = %f\n",me,time_backwardft);
    printf("processor %d: time(dist_update) = %f\n",me,time_dist_update);
    printf("processor %d: time(int_rel2) = %f\n",me,time_int_rel2);
    printf("processor %d: time(test) = %f\n",me,time_test);
  }
  

}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixGLEPair::compute_vector(int n)
{
  tagint *tag = atom->tag;
  
  //printf("%d %d\n",t,i);
  
  return fr[n][0];
}


/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixGLEPair::memory_usage()
{
  double bytes = atom->nlocal*atom->nlocal*Nt*Nt*sizeof(double);
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
  
  if (tStop < tStart)
    error->all(FLERR,"Fix gle/pair/aux tStop must be > tStart");
  Nt = (tStop - tStart) / tStep + 1.5;
  
    
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
void FixGLEPair::update_cholesky() 
{
  // initialize input matrix
  int dist,t;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int *tag = atom->tag;
  double **x = atom->x;
  int i,j,ii,jj,inum,jnum,itype,jtype,itag,jtag;
  double xtmp,ytmp,ztmp,rsq,rsqi;
  double *dr = new double[3];
  int *ilist,*jlist,*numneigh,**firstneigh;
  neighbor->build_one(list);
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  int non_zero=0;
  int *row;
  int *col;
  double t1 = MPI_Wtime();
  std::vector<T> tripletList;
  A.clear();
  a.clear();
  
  for (int t = 0; t < Nt; t++) {
    Eigen::SparseMatrix<double> A0_sparse(nlocal*d,nlocal*d);
    // access the neighbor lists
    for (int ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      itype = type[i];
      itag = tag[i]-1;
      jlist = firstneigh[i];
      jnum = numneigh[i];
      
      // set self-correlation
      for (int dim1=0; dim1<d;dim1++) {
	tripletList.push_back(T(d*itag+dim1,d*itag+dim1,self_data[t]));
	if (t==0) non_zero++;
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
	  error->all(FLERR,"Particles closer than lower cutoff in fix/pair/jung\n");
	} else if (dist < Nd) {
	  double data = cross_data[Nt*dist+t];
	  for (int dim1=0; dim1<d;dim1++) {
	    for (int dim2=0; dim2<d;dim2++) {
	      if (data*dr[dim1]*dr[dim2]*rsqi != 0) {
		tripletList.push_back(T(d*itag+dim1,d*jtag+dim2,data*dr[dim1]*dr[dim2]*rsqi));
		if (t==0) non_zero++;
	      }
	    }
	  }
	}
      }
    }
    A0_sparse.setFromTriplets(tripletList.begin(), tripletList.end());
    tripletList.clear();
    A.push_back(A0_sparse);
    //if (t==10) cout << A0_sparse << endl;
  }
  //printf("non_zero %d\n",non_zero);
  delete [] dr;
  double t2 = MPI_Wtime();
  time_matrix_update += t2 -t1;
  
  // init column and row indices
  row = new int[non_zero];
  col = new int[non_zero];
  int counter = 0;
  for (int k=0; k<A[0].outerSize(); ++k) {
    for (SparseMatrix<double>::InnerIterator it(A[0],k); it; ++it) {
      col[counter] = it.col();
      row[counter] = it.row();
      counter++;
    }
  }
  
  // step 1: perform FT for every entry of A
  int N = 2*Nt-2;
  kiss_fft_scalar * buf;
  kiss_fft_cpx * bufout;
  buf=(kiss_fft_scalar*)KISS_FFT_MALLOC(sizeof(kiss_fft_scalar)*N*non_zero);
  bufout=(kiss_fft_cpx*)KISS_FFT_MALLOC(sizeof(kiss_fft_cpx)*N*non_zero);
  t1 = MPI_Wtime();
  for (int t=0; t<Nt; t++) {
    int counter = 0;
    for (int k=0; k<A[t].outerSize(); ++k) {
      for (SparseMatrix<double>::InnerIterator it(A[t],k); it; ++it) {
	buf[counter*N+t] = it.value();
	if (t==0 || t==Nt-1){ }
	else {
	  buf[counter*N+N-t] = it.value();
	}
	counter++;
      }
    }
  }
  t2 = MPI_Wtime();
  time_matrix_update += t2 -t1;
  // do the fft with kiss_fft
  kiss_fftr_cfg st = kiss_fftr_alloc( N ,0 ,0,0);
  t1 = MPI_Wtime();
  for (int i=0; i<non_zero;i++) {
    kiss_fftr( st ,&buf[i*N],&bufout[i*N] );
  }
  t2 = MPI_Wtime();
  time_forwardft += t2 -t1;
  t1 = MPI_Wtime();
  std::vector<Eigen::SparseMatrix<double> > A_FT;
  //printf("FT_A\n");
  for (int t=0; t<Nt; t++) {
    Eigen::SparseMatrix<double> A_FT0_sparse(nlocal*d,nlocal*d);
    for (int i=0; i<non_zero;i++) {
      tripletList.push_back(T(row[i],col[i],bufout[i*N+t].r));
    }
    A_FT0_sparse.setFromTriplets(tripletList.begin(), tripletList.end());
    tripletList.clear();
    A_FT.push_back(A_FT0_sparse);
    //cout << A_FT0_sparse;
  }
  t2 = MPI_Wtime();
  time_matrix_update += t2 -t1;
  //printf("-------------------------------\n");
  
  // step 2: determine eigenvalue bounds of A_FT matrices
  int mLanczos = 20;
  int size = d*nlocal;
  double tolLanczos = 0.000001;
  std::vector<Eigen::VectorXd> vk;
  VectorXd wk = VectorXd::Zero(size);
  double *alpha = new double[mLanczos];
  double *beta = new double[mLanczos];
  double evMax=0.0, evMin=0.0, evMaxm1=0.0, evMinm1= 0.0;
  // generate random vector v and normalize
  double norm = 0.0, random=0.0;
  for (int k=0; k<mLanczos; k++) {
    VectorXd vk_loc = VectorXd::Zero(size);
    vk.push_back(vk_loc);
  }
  for (int i=0; i< size; i++) {
    vk[0](i) = rand();
  }
  vk[0] = vk[0].normalized();
  //printf("\n");
  for (int k=0; k<mLanczos; k++) {
    alpha[k] = 0.0;
    beta[k] = 0.0;
  }
  int k;
  // main safeguarded Lanczos loop, determine eigenvalue bounds
  for (k=1; k<mLanczos; k++) {
    wk = A_FT[0]*vk[k-1];
    if (k>=2) wk -= beta[k-1]*vk[k-2];
    alpha[k-1] = (wk.adjoint()*vk[k-1]).value();
    // rescale w
    wk -= alpha[k-1]*vk[k-1];
    beta[k] = wk.norm();
    // set new v
    vk[k] = wk.normalized(); 

    //printf("Lanczos Step %d: alpha %f, beta %f evMin %f, evMax %f\n",k,alpha[k-1],beta[k],evMin,evMax);
    if (k>=2) {
      //generate eigenvalues/eigenvectors and check whether they fullfill the tolerance
      Eigen::MatrixXd Tk = Eigen::MatrixXd::Zero(k,k);
      for (int i=0; i<k; i++) {
	for (int j=0; j<k; j++) {
	  if (i==j) Tk(i,j) = alpha[i];
	  if (i==j+1||i==j-1) {
	    int kprime = j;
	    if (i>j) kprime = i;
	    Tk(i,j) = beta[kprime];
	  }
	}
      }
      //printf("Tk-Matrix\n");
      //cout << Tk << endl;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> Tk_eigen(Tk);
      Eigen::MatrixXd Tk_eigenvector = Tk_eigen.eigenvectors().real();
      /*double norm = 0.0;
      for (int i=0; i<size; i++){
	double norm_loc = 0.0;
	for (int kloc=0; kloc<k; kloc++) {
	  norm_loc += Tk_eigenvector(kloc,0)*vk[kloc*size+i];
	  //printf("%f ",norm_loc);
	}
	//printf("\n");
	norm += norm_loc*norm_loc;
      }
      printf("Test Qk %f\n",norm);*/
      //Test Matrix
      /*for (int i=0; i<size;i++) {
	for (int kloc=0; kloc<k; kloc++) {
	  //left side
	  double result_left = 0.0;
	  for (int j=0; j<size;j++) {
	    result_left += A(i,j) * vk[kloc*size+j];
	  }
	  //right side
	  double result_right = 0.0;
	  for (int kloc2=0; kloc2<k;kloc2++) {
	    result_right += vk[kloc2*size+i]*Tk(kloc2,kloc);
	  }
	  if (kloc == k-1)
	    result_right += wk[i];
	  printf("Test Eq: %f = %f\n",result_left,result_right);
	}
      }*/
      //printf("Tk-Eigenmatrix\n");
      //cout << Tk_eigenvector << endl;
      double mu_min = 999999.0;
      double mu_max = 0.0;
      int i_min=0, i_max=0;
      for (int i=0; i<k; i++){
	double ev_loc = Tk_eigen.eigenvalues().real()(i);
	//printf("%f\n",Tk_eigen.eigenvalues().real()(i));
	if (ev_loc < mu_min) {
	  mu_min = ev_loc;
	  i_min = i;
	}
	if (ev_loc > mu_max) {
	  mu_max = ev_loc;
	  i_max = i;
	}
      }
      evMaxm1 = evMax;
      evMinm1 = evMin;
      //printf("Lanczos Step %d: evMax %f, evMin %f \n",k,mu_max,mu_min);
      evMax = mu_max + sqrt(Tk_eigenvector(k-1,i_max)*Tk_eigenvector(k-1,i_max))*beta[k];
      evMin = mu_min - sqrt(Tk_eigenvector(k-1,i_min)*Tk_eigenvector(k-1,i_min))*beta[k];
      //printf("Lanczos Step %d: evMax %f, evMin %f \n",k,evMax,evMin);
      if( sqrt((evMax-evMaxm1)*(evMax-evMaxm1))/evMaxm1 < tolLanczos ) break;
    }
  }
  printf("Lanczos EV-bounds %d: evMin = %f, evMax = %f\n",k,evMin,evMax);
  
  // step 3: determine Chebyshev coefficients by approximating the squareroot function
  evMin = evMin/2;
  evMax = evMax*2;
  int n;
  int nmax = 10;
  int ntest = 100;
  double epsilon = 0.001;
  double epsilon_loc = 0.0;
  double *x_test = new double[ntest];
  for (int i=0; i< ntest; i++) {
    double step = (evMax-evMin)/((double) ntest);
    x_test[i] = evMin + step*i;
  }
  double *x0 = new double[nmax+1];
  double ta = 2.0/(evMax-evMin);
  double tb = - (evMax+evMin)/(evMax-evMin);
  double *c = new double[nmax+1];
  double *T = new double[nmax+1];
  // check for the dgeree n of the polynomial
  for (n=1; n<= nmax; n++) {
    for (int l=0; l<=n; l++) {
      x0[l] = (evMax+evMin)/2.0 + (evMax-evMin)/2.0*cos((((double)2.0*l)+1.0)*PI/(((double) 2.0*n)+2.0));
    }
    // determine ck
    for (int k=0; k<=n; k++) {
      c[k] = 0.0;
      for (int l=0; l<=n; l++) {
	// determine Tk(xl)
	T[0] = 1.0;
	T[1] = ta*x0[l] + tb;
	for (int kloc = 2; kloc<=k; kloc++ ) {
	  T[kloc] = 2*(ta*x0[l] + tb)*T[kloc-1]-T[kloc-2];
	}
	c[k] += 2.0/(((double) n)+1)*sqrt(x0[l])*T[k];
      }
    }
    // check accuricy
    //printf("n %d: ",n);
    epsilon_loc = 0.0;
    for (int i=0; i< ntest; i++) {
      double pnx = 0.0;
      for (int k=0; k<=n; k++) {
	T[0] = 1.0;
	T[1] = ta*x_test[i] + tb;
	for (int kloc = 2; kloc<=k; kloc++ ) {
	  T[kloc] = 2*(ta*x_test[i] + tb)*T[kloc-1]-T[kloc-2];
	}
	pnx += c[k]*T[k];
	// Test
	if (k==n) {
	  double TN = 2*(ta*x_test[i] + tb)*T[k]-T[k-1];
	  //printf("TN %f = 0\n",TN);
	}
      }
      pnx -= 0.5*c[0];
      if ( sqrt((sqrt(x_test[i])-pnx)*(sqrt(x_test[i])-pnx)) > epsilon_loc ) epsilon_loc = sqrt((sqrt(x_test[i])-pnx)*(sqrt(x_test[i])-pnx));
      //printf("%f/%f ",sqrt(x_test[i]),pnx);
    }
    //printf("\n");
    //printf("Chebyshev Step %d: Acc %f\n",n,epsilon_loc);
    if (epsilon_loc < epsilon) break;
  }
  printf("Final degree: %d, Accuracy: %f\n",n,epsilon_loc);
  nmax = n+1;
  
  // step 4a: (debugging) determine FT noise matrix FT_k
  std::vector<Eigen::SparseMatrix<double> > a_FT;
  Eigen::SparseMatrix<double> a_FT_loc0(size,size);
  a_FT_loc0.setIdentity();
  a_FT.push_back(a_FT_loc0);
  //printf("test0\n");
  Eigen::SparseMatrix<double> a_FT_loc1 = ta*A_FT[0] + tb*a_FT[0];
  a_FT.push_back(a_FT_loc1);
  //printf("test\n");
  for (int n=2; n<nmax; n++) {
     Eigen::SparseMatrix<double> a_FT_locn = 2*ta*A_FT[0]*a_FT[n-1]+2*tb*a_FT[n-1]-a_FT[n-2];
    a_FT.push_back(a_FT_locn);
  }
  // determine sqrt-matrix
  Eigen::SparseMatrix<double> a_FT_final(size,size);
  a_FT_final.setZero();
  a_FT_final += 0.5*c[0]*a_FT[0];
  for (int n=1; n<nmax; n++) {
    a_FT_final += c[n]*a_FT[n];
    printf("sparse: %d\n",a_FT_final.nonZeros());
  }
  //printf("Sqrt-FT_a_final:\n");
  //cout << a_FT_final << endl;
  //cout << a_FT_final*a_FT_final << endl;
  //cout << A_FT[0] << endl;
  
  // step 4b: (debugging) determine noise matrix k
  
  
}


