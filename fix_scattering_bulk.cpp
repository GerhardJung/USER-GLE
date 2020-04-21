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

#include "stdlib.h"
#include "string.h"
#include "unistd.h"
#include "fix_scattering_bulk.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "group.h"
#include "fix_store.h"
#include "input.h"
#include "variable.h"
#include "domain.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "atom.h"
#include "comm.h"
#include <sstream>

using namespace LAMMPS_NS;
using namespace FixConst;

FixScatteringBulk::FixScatteringBulk(LAMMPS * lmp, int narg, char **arg):
  Fix (lmp, narg, arg)
{
  if (narg < 10) error->all(FLERR,"Illegal fix scattering/bulk command");
  nevery = force->inumeric(FLERR,arg[3]);
  nrelax = force->inumeric(FLERR,arg[4]);
  N_blocks = force->inumeric(FLERR,arg[5]);
  N_count = force->inumeric(FLERR,arg[6]);
  N_levels = force->inumeric(FLERR,arg[7]);
  N_levels_msd = force->inumeric(FLERR,arg[8]);
   N_cor = force->inumeric(FLERR,arg[9]);
    nFunCorr = force->inumeric(FLERR,arg[10]);
}

/* ---------------------------------------------------------------------- */

FixScatteringBulk::~FixScatteringBulk()
{
  
}

/* ---------------------------------------------------------------------- */

int FixScatteringBulk::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixScatteringBulk::init() {

  AllocArrays();
  
  if (N_blocks % N_count != 0) error->all(FLERR,"FixScatteringBulk N_blocks mod N_count must be 0");
  dmin = N_blocks/N_count;
  
  if (nrelax % nevery != 0) error->all(FLERR,"FixScatteringBulk nrelax mod nevery must be 0");

  for (int j=0; j<nFunCorr; j++) {
    strucFac[j]=0.0;
  }
  
  t_loc = 0;
  kmax = 0;
  count = 0;
  
  ZeroSpacetimeCorr ();
  ZeroSpacetimeCorrIn ();
  ZeroSpacetimeCorrIn2 ();
}

/* ---------------------------------------------------------------------- */
  
void FixScatteringBulk::setup(int vflag) {

}

/* ---------------------------------------------------------------------- */

void FixScatteringBulk::end_of_step() {
  // Do every timestep (or every tickerstep)
  //printf("Ticker, EvalSlit\n");
  if (update->ntimestep % nevery == 0)
    EvalSpacetimeCorr ();
  
  if (update->nsteps == update->ntimestep && t_loc != 0) {
   AccumSpacetimeCorr();
  }
}

/* ---------------------------------------------------------------------- */
  
/* Help functions to calculate Structure factor and coherent scattering function */
void FixScatteringBulk::AllocArrays(){
  // has to be adapted for parallelization
  int N=atom->nlocal;

  int nb,k;
  AllocMem (valST, 6 * nFunCorr, real);
  
  AllocMem (valVEL, 3 * N, real);
  
  AllocMem (strucFac, nFunCorr, real);
  
  AllocMem2 (blocking_sum, N_blocks*N_levels_msd, 3*N, real);
  AllocMem2 (correlationIn, N_blocks*N_levels_msd, nFunCorr, real);
  AllocMem2 (MSD, N_blocks*N_levels_msd,3,real);
    AllocMem2 (NGP, N_blocks*N_levels_msd,4,real);
  AllocMem2 (countIn, N_blocks*N_levels_msd, nFunCorr, int);
  AllocMem (countMSD, N_blocks*N_levels_msd, int);
    AllocMem (countNGP, N_blocks*N_levels_msd, int);
  
  AllocMem2 (shift, N_blocks*N_levels, 6 *  nFunCorr, real);
  AllocMem2 (accumulator, N_levels, 6  * nFunCorr, real);
  AllocMem (naccumulator, N_levels, int);
  AllocMem2 (correlation, N_blocks*N_levels, nFunCorr, real);
  AllocMem (countcor, N_blocks*N_levels,int);
  AllocMem (insertindex, N_levels,int);
  
  AllocMem2 (shiftVACF, N_blocks*N_levels, 3 *N, real);
  AllocMem2 (accumulatorVACF, N_levels, 3*N, real);
  AllocMem (naccumulatorVACF, N_levels, int);
  AllocMem2 (correlationVACF, N_blocks*N_levels, 3, real);
  AllocMem (countcorVACF, N_blocks*N_levels,int);
  AllocMem (insertindexVACF, N_levels,int);
  
  AllocMem (pos_save, 3*N,real);
  AllocMem2 (pos_save2, 9*N,N_cor,real);
  
  AllocMem2 (correlationIn2, N_cor, 9*nFunCorr, real);
  AllocMem (countIn2, N_cor, int);
  
  lastindex = 0;
}
  
/***************************************************************************************/
  
  void FixScatteringBulk::EvalSpacetimeCorr (){
    real b, c, c0, c1, c2, kVal, s, s1, s2;
    real cc, sc, QVal;
    int j,  k, m, nb,  nv;
    
    int N=atom->nlocal;
    int nlocal = atom->nlocal;
    int *mask= atom->mask;
    double **x = atom->x;
    double **v = atom->v;
    double **f = atom->f;

    for (j = 0; j < 6 * nFunCorr; j ++) valST[j] = 0.;
    for (j = 0; j < 3*N; j ++) valVEL[j] = 0.;
    count++;
    
    // calculate FT for coherent scattering fct
    kVal = 2. * M_PI / domain->xprd;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        j = 0;
        for (k = 0; k < 3; k ++) {
          for (m = 0; m < nFunCorr; m ++) {
            if (m == 0) {
              b = kVal * x[i][k];

              c = cos (b);
              s = sin (b);
              c0 = c;
            } else if (m == 1) {
              c1 = c;
              s1 = s;
              c = 2. * c0 * c1 - 1.;
              s = 2. * c0 * s1;
            } else {
              c2 = c1;
              s2 = s1;
              c1 = c;
              s1 = s;
              c = 2. * c0 * c1 - c2;
              s = 2. * c0 * s1 - s2;
            }
            valST[j ++] += c; //second element of SF -real part
            valST[j ++] += s; //imaginary 
          }
        }
      }
    }
    // acumualte and calculate logarithmic correlation function
    add(valST,0);
    
    
    // add values to VACF
    j = 0;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        for (k = 0; k < 3; k ++) {
          valVEL[j ++] = v[i][k];
        }
      }
      //if (i==1) printf("input v[i][0]=%f\n",v[i][0]);
    }
    // acumualte and calculate logarithmic velocity correlation function
    addVACF(valVEL,0,N);
	
    // calc structure factor
    for (k = 0; k < 3; k ++) {
      for (m = 0; m < nFunCorr; m ++) {
        int ind_j = 2*k*nFunCorr + 2*m;
        double c = valST[ind_j];
        double s = valST[ind_j+1];

        strucFac[m] += c*c + s*s;
      }
    }
    
    // calculate \Delta r in blocking algorithm for self-intermediate scattering function
    int t = t_loc;
    int np=0;
    int max_levels = (int) (log(t)/log(N_blocks));
    if (max_levels >= N_levels) max_levels = N_levels-1;
    
    tagint *tag = atom->tag;
      imageint *image = atom->image;
    int itag;
    double unwrap[3];
    //printf("max_l %d\n",max_levels);
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        double delx,dely,delz;
        int j0; // = i % N_blocks;
        //if (np==0) printf("checkin0\n");
        if (t==0) {
          itag = tag[i] - 1;
          domain->unmap(x[i],image[i],unwrap);
          pos_save[3*itag] = unwrap[0];
          pos_save[3*itag+1] = unwrap[1];
          pos_save[3*itag+2] = unwrap[2];
        }
        for (int k=0; k<(max_levels+1); k++) {
            
          if (k==0) {
            itag = tag[i] - 1;
            domain->unmap(x[i],image[i],unwrap);
            delx = unwrap[0]-pos_save[3*itag];
            dely = unwrap[1]-pos_save[3*itag+1];
            delz = unwrap[2]-pos_save[3*itag+2];
            pos_save[3*itag] = unwrap[0];
            pos_save[3*itag+1] = unwrap[1];
            pos_save[3*itag+2] = unwrap[2];
          } else {
            delx = blocking_sum[N_blocks-1+(k-1)*N_blocks][np];
            dely = blocking_sum[N_blocks-1+(k-1)*N_blocks][np+N];
            delz = blocking_sum[N_blocks-1+(k-1)*N_blocks][np+2*N];
          }
            
          int nblocks_to_k = 1 ; // (int) round(pow(N_blocks,k));

          for (int kk=0; kk<k; kk++) nblocks_to_k *= N_blocks;
          //if (np==0) printf("checkin1\n");
          if (t % nblocks_to_k == 0) {
              
            j0 =  ((t) / nblocks_to_k-1) % N_blocks;  // was -1

            if (j0==0) {
              blocking_sum[j0+k*N_blocks][np] = delx; //blocking_sum[N_blocks-1][k-1][n] ;
              blocking_sum[j0+k*N_blocks][np+N] = dely; //blocking_sum[N_blocks-1][k-1][n] ;
              blocking_sum[j0+k*N_blocks][np+2*N] = delz; //blocking_sum[N_blocks-1][k-1][n] ;
            }
            else {
              blocking_sum[j0+k*N_blocks][np] = blocking_sum[j0-1+k*N_blocks][np] + delx; 
              blocking_sum[j0+k*N_blocks][np+N] = blocking_sum[j0-1+k*N_blocks][np+N] + dely; 
              blocking_sum[j0+k*N_blocks][np+2*N] = blocking_sum[j0-1+k*N_blocks][np+2*N] + delz; 
            }
            
            
            // now calculate incoherent scattering functions
            for (int km = 0; km < 3; km ++) {
              for (m = 0; m < nFunCorr; m ++) {
                if (m == 0) {
                  b = kVal * blocking_sum[j0+k*N_blocks][km*N+np] ;
                  //if (k==0) printf("x coord %f\n",Sim->particles[p1].getPosition()[k]);
                  c = cos (b);
                  s = sin (b);
                  c0 = c;
                } else if (m == 1) {
                  c1 = c;
                  s1 = s;
                  c = 2. * c0 * c1 - 1.;
                  s = 2. * c0 * s1;
                } else {
                  c2 = c1;
                  s2 = s1;
                  c1 = c;
                  s1 = s;
                  c = 2. * c0 * c1 - c2;
                  s = 2. * c0 * s1 - s2;
                }
            
                correlationIn[j0+k*N_blocks][m] += c;
                countIn[j0+k*N_blocks][m] += 1;
              }
            }
            
            
            // calc MSD
            MSD[j0+k*N_blocks][0]+=blocking_sum[j0+k*N_blocks][np]*blocking_sum[j0+k*N_blocks][np];
            MSD[j0+k*N_blocks][1]+=blocking_sum[j0+k*N_blocks][np+N]*blocking_sum[j0+k*N_blocks][np+N];
            MSD[j0+k*N_blocks][2]+=blocking_sum[j0+k*N_blocks][np+2*N]*blocking_sum[j0+k*N_blocks][np+2*N];
            countMSD[j0+k*N_blocks] += 1;
            
            NGP[j0+k*N_blocks][0]+=blocking_sum[j0+k*N_blocks][np]*blocking_sum[j0+k*N_blocks][np]*blocking_sum[j0+k*N_blocks][np]*blocking_sum[j0+k*N_blocks][np];
            NGP[j0+k*N_blocks][1]+=blocking_sum[j0+k*N_blocks][np+N]*blocking_sum[j0+k*N_blocks][np+N]*blocking_sum[j0+k*N_blocks][np+N]*blocking_sum[j0+k*N_blocks][np+N];
            NGP[j0+k*N_blocks][2]+=blocking_sum[j0+k*N_blocks][np+2*N]*blocking_sum[j0+k*N_blocks][np+2*N]*blocking_sum[j0+k*N_blocks][np+2*N]*blocking_sum[j0+k*N_blocks][np+2*N];
            NGP[j0+k*N_blocks][3]+=(blocking_sum[j0+k*N_blocks][np]*blocking_sum[j0+k*N_blocks][np] + blocking_sum[j0+k*N_blocks][np+N]*blocking_sum[j0+k*N_blocks][np+N] + blocking_sum[j0+k*N_blocks][np+2*N]*blocking_sum[j0+k*N_blocks][np+2*N])*(blocking_sum[j0+k*N_blocks][np]*blocking_sum[j0+k*N_blocks][np] + blocking_sum[j0+k*N_blocks][np+N]*blocking_sum[j0+k*N_blocks][np+N] + blocking_sum[j0+k*N_blocks][np+2*N]*blocking_sum[j0+k*N_blocks][np+2*N]);
            countNGP[j0+k*N_blocks] += 1;
          }
        }
        np++;
      }
    }
    
    // calculate incoherent scattering function and derivatives (expensive!!)
    real bx, cx, c0x, c1x, c2x, sx, s1x, s2x;
    real by, cy, c0y, c1y, c2y, sy, s1y, s2y;
    real bz, cz, c0z, c1z, c2z, sz, s1z, s2z;
    int tcor_max = N_cor;
    if (t_loc<N_cor) tcor_max=t_loc;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        // save new positions and velocities
        itag = tag[i] - 1;
        domain->unmap(x[i],image[i],unwrap);
        pos_save2[9*itag][lastindex] = unwrap[0];
        pos_save2[9*itag+1][lastindex] = unwrap[1];
        pos_save2[9*itag+2][lastindex] = unwrap[2];
        pos_save2[9*itag+3][lastindex] = v[i][0];
        pos_save2[9*itag+4][lastindex] = v[i][1];
        pos_save2[9*itag+5][lastindex] = v[i][2];
        pos_save2[9*itag+6][lastindex] = f[i][0];
        pos_save2[9*itag+7][lastindex] = f[i][1];
        pos_save2[9*itag+8][lastindex] = f[i][2];
      }
    }
    
    if (update->ntimestep % nrelax == 0) {
      for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
          // save new positions and velocities
          itag = tag[i] - 1;
          // calculate correlation function
          int ind1 = lastindex;
          int ind2 = ind1;
          for (int tcor=0; tcor < tcor_max; tcor++) {
            double dx = pos_save2[9*itag][ind2]-pos_save2[9*itag][ind1];
            double dy = pos_save2[9*itag+1][ind2]-pos_save2[9*itag+1][ind1];
            double dz = pos_save2[9*itag+2][ind2]-pos_save2[9*itag+2][ind1];
            
            double vx0 = pos_save2[9*itag+3][ind1];
            double vy0 = pos_save2[9*itag+4][ind1];
            double vz0 = pos_save2[9*itag+5][ind1];
            double vxt = pos_save2[9*itag+3][ind2];
            double vyt = pos_save2[9*itag+4][ind2];
            double vzt = pos_save2[9*itag+5][ind2];
            
            double Fx0 = pos_save2[9*itag+6][ind1];
            double Fy0 = pos_save2[9*itag+7][ind1];
            double Fz0 = pos_save2[9*itag+8][ind1];
            double Fxt = pos_save2[9*itag+6][ind2];
            double Fyt = pos_save2[9*itag+7][ind2];
            double Fzt = pos_save2[9*itag+8][ind2];
            
            for (m = 0; m < nFunCorr; m ++) {
              if (m == 0) {
                bx = kVal * dx ;
                by = kVal * dy ;
                bz = kVal * dz ;
                //if (k==0) printf("x coord %f\n",Sim->particles[p1].getPosition()[k]);
                cx = cos (bx);
                sx = sin (bx);
                c0x = cx;
                cy = cos (by);
                sy = sin (by);
                c0y = cy;
                cz = cos (bz);
                sz = sin (bz);
                c0z = cz;
              } else if (m == 1) {
                c1x = cx;
                s1x = sx;
                cx = 2. * c0x * c1x - 1.;
                sx = 2. * c0x * s1x;
                c1y = cy;
                s1y = sy;
                cy = 2. * c0y * c1y - 1.;
                sy = 2. * c0y * s1y;
                c1z = cz;
                s1z = sz;
                cz = 2. * c0z * c1z - 1.;
                sz = 2. * c0z * s1z;
              } else {
                c2x = c1x;
                s2x = s1x;
                c1x = cx;
                s1x = sx;
                cx = 2. * c0x * c1x - c2x;
                sx = 2. * c0x * s1x - s2x;
                c2y = c1y;
                s2y = s1y;
                c1y = cy;
                s1y = sy;
                cy = 2. * c0y * c1y - c2y;
                sy = 2. * c0y * s1y - s2y;
                c2z = c1z;
                s2z = s1z;
                c1z = cz;
                s1z = sz;
                cz = 2. * c0z * c1z - c2z;
                sz = 2. * c0z * s1z - s2z;
              }
              double qVal = (m+1)*kVal;
              double q2Val = qVal*qVal;
              double q3Val = q2Val*qVal;
              double q4Val = q2Val*q2Val;
          
              // S
              correlationIn2[tcor][9*m] += (cx + cy + cz)/3.0;
              // dS/dt
              correlationIn2[tcor][9*m+1] -= qVal*(vxt*sx+vyt*sz+vzt*sz)/3.0;
              // dS^2/dt^2
              correlationIn2[tcor][9*m+2] += q2Val*(vxt*vx0*cx+vyt*vy0*cy+vzt*vz0*cz)/3.0;
              // dS^3/dt^3
              correlationIn2[tcor][9*m+3] -= q3Val*(vxt*vxt*vx0*sx+vyt*vyt*vy0*sy+vzt*vzt*vz0*sz)/3.0;
              correlationIn2[tcor][9*m+4] += q2Val*(Fxt*vx0*cx+Fyt*vy0*cy+Fzt*vz0*cz)/3.0;
              // dS^4/dt^4
              correlationIn2[tcor][9*m+5] -= q4Val*(vxt*vxt*vx0*vx0*cx+vyt*vyt*vy0*vy0*cy+vzt*vzt*vz0*vz0*cz)/3.0;
              correlationIn2[tcor][9*m+6] -= q3Val*(Fx0*vxt*vxt*sx+Fy0*vyt*vyt*sy+Fz0*vzt*vzt*sz)/3.0;
              correlationIn2[tcor][9*m+7] -= q3Val*(vx0*vx0*Fxt*sx+vy0*vy0*Fyt*sy+vz0*vz0*Fzt*sz)/3.0;
              correlationIn2[tcor][9*m+8] += q2Val*(Fxt*Fx0*cx+Fyt*Fy0*cy+Fzt*Fz0*cz)/3.0;
              
              
              if (m==0) countIn2[tcor] += 1;
            }
                
            
            ind2 --;
            if (ind2 < 0) ind2 += N_cor;
          }
          
        }
      }
    }
        
     
    lastindex++;
    if (lastindex >N_cor-1) lastindex -= N_cor;
    
    int t_tot = (int) pow(N_blocks,N_levels_msd);
    if (t_loc == t_tot-1) {
      AccumSpacetimeCorr ();
      t_loc = 0;
      kmax = 0;
    } else t_loc++;
  }
  
  /***************************************************************************************/
  
  void FixScatteringBulk::add(real * val, int k){
    // If we exceed the correlator side, the value is discarded
    if (k == N_levels) return;
    if (k > kmax) kmax=k;

    // Insert new value in shift array
    for (int i = 0; i < 6 * nFunCorr; i ++)
      shift[k*N_blocks+insertindex[k]][i] = val[i];

    // Add to accumulator and, if needed, add to next correlator
    for (int i = 0; i < 6 * nFunCorr ; i ++)
      accumulator[k][i] += val[i];
    ++naccumulator[k];
    if (naccumulator[k]==N_count) {
      for (int i = 0; i < 6 * nFunCorr ; i ++) accumulator[k][i] /= N_count;
      add(&accumulator[k][0], k+1);
      for (int i = 0; i < 6 * nFunCorr ; i ++) accumulator[k][i]=0.0;
      naccumulator[k]=0;
    }

    // Calculate correlation function
    unsigned int ind1=insertindex[k];
    if (k==0) { // First correlator is different
      int ind2=ind1;
      for (int j=0;j<N_blocks;++j) {
	for (int kp = 0; kp < 3; kp ++) {
	  for (int m = 0; m < nFunCorr; m ++) {
	    int nv = m;
	    int ind_j = kp*2*nFunCorr + 2*m;
	    double c = shift[k*N_blocks+ind2][ind_j];
	    double s = shift[k*N_blocks+ind2][ind_j+1];
	    double c2 = shift[k*N_blocks+ind1][ind_j];
	    double s2 = shift[k*N_blocks+ind1][ind_j+1];
	    
	    if (c > -1e10) {

	      correlation[k*N_blocks+j][nv]  += c*c2 + s*s2;

	      if (nv==0) ++countcor[k*N_blocks+j];
	    }
	  }
	}
	--ind2;
	if (ind2<0) ind2+=N_blocks;
      }
    } else {
      int ind2=ind1-dmin;
      for (int j=dmin;j<N_blocks;++j) {
	if (ind2<0) ind2+=N_blocks;
	for (int kp = 0; kp < 3; kp ++) {
	  for (int m = 0; m < nFunCorr; m ++) {
	    int nv = m ;
	    int ind_j = kp*2*nFunCorr + 2*m;
	    double c = shift[k*N_blocks+ind2][ind_j];
	    double s = shift[k*N_blocks+ind2][ind_j+1];
	    double c2 = shift[k*N_blocks+ind1][ind_j];
	    double s2 = shift[k*N_blocks+ind1][ind_j+1];
	    
	    if (c > -1e10) {
	      correlation[k*N_blocks+j][nv]  += c*c2 + s*s2;

	      if (nv==0) ++countcor[k*N_blocks+j];
	    }

	  }
	}
	--ind2;
      }
    }

    ++insertindex[k];
    if (insertindex[k]==N_blocks) insertindex[k]=0;


  }
  
   /***************************************************************************************/
  
  void FixScatteringBulk::addVACF(real * val, int k, int N){
    // If we exceed the correlator side, the value is discarded
    if (k == N_levels) return;
    if (k > kmax) kmax=k;

    // Insert new value in shift array
    for (int i = 0; i < 3 * N; i ++)
      shiftVACF[k*N_blocks+insertindexVACF[k]][i] = val[i];

    // Add to accumulator and, if needed, add to next correlator
    for (int i = 0; i < 3 * N ; i ++)
      accumulatorVACF[k][i] += val[i];
    ++naccumulatorVACF[k];
    if (naccumulatorVACF[k]==N_count) {
      for (int i = 0; i < 3 * N ; i ++) accumulatorVACF[k][i] /= N_count;
      addVACF(&accumulatorVACF[k][0], k+1,N);
      for (int i = 0; i < 3 * N ; i ++) accumulatorVACF[k][i]=0.0;
      naccumulatorVACF[k]=0;
    }

    // Calculate correlation function
    unsigned int ind1=insertindexVACF[k];
    if (k==0) { // First correlator is different
      int ind2=ind1;
      for (int j=0;j<N_blocks;++j) {
	for (int m = 0; m < N; m ++) {
      
	  for (int kp = 0; kp < 3; kp ++) {
	    int ind_j = 3*m+kp;
	    double t0 = shiftVACF[k*N_blocks+ind2][ind_j];
	    double t1 = shiftVACF[k*N_blocks+ind1][ind_j];
	    
	    if (t0 > -1e10) {

	      correlationVACF[k*N_blocks+j][kp]  += t0*t1;
	      
	      if (kp==0) ++countcorVACF[k*N_blocks+j];

	    }
	  }
	}
	--ind2;
	if (ind2<0) ind2+=N_blocks;
      }
    } else {
      int ind2=ind1-dmin;
      for (int j=dmin;j<N_blocks;++j) {
	if (ind2<0) ind2+=N_blocks;
	for (int m = 0; m < N; m ++) {

	  for (int kp = 0; kp < 3; kp ++) {
	    int ind_j = 3*m+kp;
	    double t0 = shiftVACF[k*N_blocks+ind2][ind_j];
	    double t1 = shiftVACF[k*N_blocks+ind1][ind_j];
	    
	    if (t0 > -1e10) {
	      correlationVACF[k*N_blocks+j][kp]  += t0*t1;
	      if (kp==0) ++countcorVACF[k*N_blocks+j];
	    }

	  }
	}
	--ind2;
      }
    }

    ++insertindexVACF[k];
    if (insertindexVACF[k]==N_blocks) insertindexVACF[k]=0;

  }
  
/***************************************************************************************/

  void FixScatteringBulk::AccumSpacetimeCorr (){
    
    // print coherent
        long double sysTime = update->ntimestep*nevery*update->dt;;
    //printf("systemTime %Lf\n",sysTime);
    std::string out_string;
    std::stringstream ss;
    ss << "S_t" << sysTime << ".dat";
    out_string = ss.str();
    printf("EvalBulk: Coherent scattering function output written to %s \n",(ss.str()).c_str());
    FILE * out = fopen(out_string.c_str(),"w");
    PrintSpacetimeCorr (out);
    fclose(out);
    
    // incoherent scattering fct
    std::string out_string2;
    std::stringstream ss2;
    ss2 << "SIn_t" << sysTime << ".dat";
    out_string2 = ss2.str();
    printf("EvalBulk: Incoherent scattering function output written to %s \n",(ss2.str()).c_str());
    out = fopen(out_string2.c_str(),"w");
    PrintSpacetimeCorrIn (out);
    fclose(out);
    
            std::stringstream ss4;
    ss4 << "SIn_Der_t" << sysTime << ".dat";
    printf("EvalBulk: Incoherent scattering function (and derivatives) output written to %s \n",(ss4.str()).c_str());
    std::string out_string4 = ss4.str();
    out = fopen(out_string4.c_str(),"w");
    PrintSpacetimeCorrIn2 (out);
    fclose(out);
    
    std::stringstream ss3;
    ss3 << "Struc_Fac_t" << sysTime << ".dat";
    printf("EvalBulk: Structure factor output written to %s \n",(ss3.str()).c_str());
    std::string out_string3 = ss3.str();
    out = fopen(out_string3.c_str(),"w");
    PrintStrucFac (out);
    fclose(out);
    

    
    count = 0;
    for (int m = 0; m < nFunCorr; m ++) {
      strucFac[m]=0.0;
    }
    
    
    ZeroSpacetimeCorr ();
    ZeroSpacetimeCorrIn ();
    ZeroSpacetimeCorrIn2 ();
  }
  
/***************************************************************************************/

  void FixScatteringBulk::ZeroSpacetimeCorr () 
  {
    int N=atom->nlocal;
    
    for (int kp = 0; kp < N_blocks*N_levels; kp ++) {
      for (int j = 0; j < 6* nFunCorr; j ++) { 
	shift[kp][j] = -2E10;
      }
      for (int j = 0; j < nFunCorr; j ++) { 
	correlation[kp][j] = 0.;
      }
      countcor[kp] = 0;
    }
    for (int k = 0; k < N_levels; k ++) {
      for (int j = 0; j < 6 * nFunCorr; j ++) { 
	accumulator[k][j] = 0.;
      }
      naccumulator[k] = 0;
      insertindex[k] = 0;
    }
    
        for (int kp = 0; kp < N_blocks*N_levels; kp ++) {
      for (int j = 0; j < 3*N; j ++) { 
	shiftVACF[kp][j] = -2E10;
      }
      for (int j = 0; j < 3; j ++) { 
	correlationVACF[kp][j] = 0.;
      }
      countcorVACF[kp] = 0;
    }
    for (int k = 0; k < N_levels; k ++) {
      for (int j = 0; j < 3*N; j ++) { 
	accumulatorVACF[k][j] = 0.;
      }
      naccumulatorVACF[k] = 0;
      insertindexVACF[k] = 0;
    }
  }

/***************************************************************************************/
  
  void FixScatteringBulk::ZeroSpacetimeCorrIn () 
  {
    int N=atom->nlocal;
    
    for (int kp = 0; kp < N_blocks*N_levels_msd; kp ++) {
      for (int i = 0; i < N; i ++) { 
	blocking_sum[kp][i] = 0.;
	blocking_sum[kp][i+N] = 0.;
	blocking_sum[kp][i+2*N] = 0.;
      }
      for (int j = 0; j < nFunCorr; j ++) { 
	correlationIn[kp][j] = 0.;
	countIn[kp][j] = 0;
      }
      MSD[kp][0] = 0.;
      MSD[kp][1] = 0.;
      MSD[kp][2] = 0.;
      countMSD[kp]=0;
            NGP[kp][0] = 0.;
      NGP[kp][1] = 0.;
      NGP[kp][2] = 0.;
      NGP[kp][3] = 0.;
      countNGP[kp]=0;
      
    }
 
  }
  
  /***************************************************************************************/
  
  void FixScatteringBulk::ZeroSpacetimeCorrIn2 () 
  {
    
    for (int tloc = 0; tloc < N_cor; tloc ++) {
      for (int j = 0; j < 9*nFunCorr; j ++) { 
        correlationIn2[tloc][j] = 0.;
      }
      countIn2[tloc] = 0;
    }
 
  }
  
/***************************************************************************************/
  
  void FixScatteringBulk::PrintStrucFac  (FILE *fp){

        int N=atom->nlocal;
    
    fprintf(fp,"#qval S(q)\n");
  
    // print structure factor
    double kval = 2. * M_PI / domain->xprd;
    for (int m = 0; m < nFunCorr; m ++) {
      fprintf(fp,"%f ",(m+1)*kval);
	fprintf(fp,"%f ",strucFac[m]/((double) count)/3. / ((double) N));
      
      fprintf(fp,"\n");
    }
  }
  
/***************************************************************************************/

  void FixScatteringBulk::PrintSpacetimeCorr (FILE *fp){
    real tVal;
    int j, n;
    
    int N=atom->nlocal;
    const double dt = nevery*update->dt;

    double kVal = 2. * M_PI / domain->xprd;
      fprintf (fp, "(1):t ");
      for (int m = 0; m < nFunCorr; m ++) {
          fprintf (fp, "(%d):%f ",(m+1)*kVal,m+2);
      }
      for (int m = 0; m < 3; m ++) {
          fprintf (fp, "(%d):VACFdim(%d) ",m+202,m);
      }
      fprintf (fp, "\n");
      
      for (int j=0;j<N_blocks;++j) {
	if (countcor[j] > 0) {
	  double t = j*dt;
	  fprintf (fp, "%8.4f", t);
	  for (int m = 0; m < nFunCorr; m ++) {
	    int nv = m ;
	    fprintf (fp, " %8.4f", correlation[j][nv]/countcor[j]/N);
	  }
	  	  if (countcorVACF[j]>0) {
	   fprintf (fp, " %10.6f %10.6f %10.6f", correlationVACF[j][0]/countcorVACF[j], correlationVACF[j][1]/countcorVACF[j], correlationVACF[j][2]/countcorVACF[j]);
	  }
	  fprintf (fp, "\n");
	}
      }
      
      for (int k=1;k<=kmax;++k) {
	for (int j=dmin;j<N_blocks;++j) {
	  if (countcor[k*N_blocks+j]>0) {
	    double t = j * pow((double)N_count, k)*nevery*update->dt;
	    fprintf (fp, "%8.4f", t);
	    for (int m = 0; m < nFunCorr; m ++) {
	      int nv = m ;
	      fprintf (fp, " %8.4f", correlation[k*N_blocks+j][nv]/countcor[k*N_blocks+j]/N);
	    }
	    	  if (countcorVACF[k*N_blocks+j]>0) {
	   fprintf (fp, " %10.6f %10.6f %10.6f", correlationVACF[k*N_blocks+j][0]/countcorVACF[k*N_blocks+j], correlationVACF[k*N_blocks+j][1]/countcorVACF[k*N_blocks+j], correlationVACF[k*N_blocks+j][2]/countcorVACF[k*N_blocks+j]);
	  }
	    fprintf (fp, "\n");
	  }
	}
      }
      
      fprintf (fp, "\n\n");
  }
  
/***************************************************************************************/

  void FixScatteringBulk::PrintSpacetimeCorrIn (FILE *fp){
       double kVal = 2. * M_PI / domain->xprd;
      fprintf (fp, "(1):t ");
      for (int m = 0; m < nFunCorr; m ++) {
          fprintf (fp, "(%d):%f ",(m+1)*kVal,m+2);
      }
      for (int m = 0; m < 3; m ++) {
          fprintf (fp, "(%d):MSDdim(%d) ",m+202,m);
      }
      for (int m = 0; m < 3; m ++) {
          fprintf (fp, "(%d):M4Ddim(%d) ",m+205,m);
      }
      fprintf (fp, "(208):NGP");
      fprintf (fp, "\n");
      
      const double dt = nevery*update->dt;
    
      for (int k=0; k<N_levels_msd; k++)
	for (int j=0; j<N_blocks; j++) {
	  double time_here = (j+1) * (pow(N_blocks,k))*dt;
	  //printf("count %d %d %d\n",k,j,countIn[k*N_blocks+j][0]);
	  fprintf (fp, "%8.4f", time_here);
	  for (int m = 0; m < nFunCorr; m ++) {
	    int nv = m ;
	    if (countIn[k*N_blocks+j][nv] > 0) {
            fprintf (fp, " %8.4f", correlationIn[k*N_blocks+j][nv]/countIn[k*N_blocks+j][nv]);
        }
	  }
	  if (countMSD[k*N_blocks+j] > 0) {
	    double MSDmean = (MSD[k*N_blocks+j][0]/countMSD[k*N_blocks+j]+MSD[k*N_blocks+j][1]/countMSD[k*N_blocks+j]+MSD[k*N_blocks+j][2]/countMSD[k*N_blocks+j]);
          double NGPmean = 3.0/5.0*(NGP[k*N_blocks+j][3]/countNGP[k*N_blocks+j])/(MSDmean*MSDmean) - 1.0;
	    fprintf (fp, " %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f", MSD[k*N_blocks+j][0]/countMSD[k*N_blocks+j], MSD[k*N_blocks+j][1]/countMSD[k*N_blocks+j], MSD[k*N_blocks+j][2]/countMSD[k*N_blocks+j], NGP[k*N_blocks+j][0]/countNGP[k*N_blocks+j], NGP[k*N_blocks+j][1]/countNGP[k*N_blocks+j], NGP[k*N_blocks+j][2]/countNGP[k*N_blocks+j],NGPmean);
	  }

	  
	  
	  fprintf (fp, "\n");
	}
      
      fprintf (fp, "\n\n");
  }
    
/***************************************************************************************/

  void FixScatteringBulk::PrintSpacetimeCorrIn2 (FILE *fp){
       double kVal = 2. * M_PI / domain->xprd;
      fprintf (fp, "(1):t ");
      for (int m = 0; m < nFunCorr; m ++) {
          fprintf (fp, "(%d):%f ",(m+1)*kVal,m+2);
      }
      fprintf (fp, "\n");
      
      const double dt = nevery*update->dt;
    

	for (int tloc=0; tloc<N_cor; tloc++) {
	  double time_here = tloc*dt;
	  //printf("count %d %d %d\n",k,j,countIn[k*N_blocks+j][0]);
	  fprintf (fp, "%8.4f", time_here);
	  for (int m = 0; m < nFunCorr; m ++) {
	    if (countIn2[tloc] > 0) {
        for (int i=0; i<9; i++) fprintf (fp, " %10.8f", correlationIn2[tloc][9*m+i]/((double) countIn2[tloc]));
      }
	  }
	  
	  fprintf (fp, "\n");
	}
      
 
  }
 
