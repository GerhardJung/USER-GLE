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
#include "fix_scattering_log.h"
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

using namespace LAMMPS_NS;
using namespace FixConst;

FixScatteringLog::FixScatteringLog(LAMMPS * lmp, int narg, char **arg):
  Fix (lmp, narg, arg)
{
  if (narg < 7) error->all(FLERR,"Illegal fix scattering/log command");

  
}

/* ---------------------------------------------------------------------- */

FixScatteringLog::~FixScatteringLog()
{
  
}

/* ---------------------------------------------------------------------- */

int FixScatteringLog::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixScatteringLog::init() {
    profileCount = 0;
    AllocArrays();
    //printf("pb %d\n",profileBins);
    for (int i=0; i<profileBins; i++) {
      densityProfile[i]=0.0;
    }
    for (int j=0; j<nFunCorr; j++) {
      for (int n=0; n<(2*nModes-1)*(2*nModes-1); n++) {
	strucFac[j][n]=0.0;
      }
    }
    
    t_loc = 0;
    kmax = 0;
    ZeroSpacetimeCorr ();
    ZeroSpacetimeCorrIn ();
}

/* ---------------------------------------------------------------------- */
  
void FixScatteringLog::setup(int vflag) {
  end_of_step();
}

/* ---------------------------------------------------------------------- */

void FixScatteringLog::end_of_step() {
  // Do every timestep (or every tickerstep)
  //printf("Ticker, EvalSlit\n");
  EvalSpacetimeCorr ();
  
  if (timestep == update->ntimestep) {
    output();
  }
}

/* ---------------------------------------------------------------------- */

void FixScatteringLog::output() {
    long double sysTime = Sim->systemTime/Sim->units.unitTime();
    std::stringstream ss2;
    ss2 << "density_profile_t" << sysTime << ".dat";
    printf("EvalSlit: Density-Profile output written to %s \n",(ss2.str()).c_str());
    std::string out_string2 = ss2.str();
    FILE * out = fopen(out_string2.c_str(),"w");
    PrintDensityProfile (out);
    fclose(out);
    
    std::stringstream ss3;
    ss3 << "Struc_Fac_t" << sysTime << ".dat";
    printf("EvalSlit: Structure factor output written to %s \n",(ss3.str()).c_str());
    std::string out_string3 = ss3.str();
    out = fopen(out_string3.c_str(),"w");
    PrintStrucFac (out);
    fclose(out);
    
    profileCount = 0;
    
    std::string out_string;
    std::stringstream ss;
    ss << "S_t" << sysTime << ".dat";
    out_string = ss.str();
    printf("EvalSlit: Coherent scattering function output written to %s \n",(ss.str()).c_str());
    out = fopen(out_string.c_str(),"w");
    PrintSpacetimeCorr (out);
    ZeroSpacetimeCorr ();
    fclose(out);
    
    std::string out_string4;
    std::stringstream ss4;
    ss4 << "SIn_t" << sysTime << ".dat";
    out_string4 = ss4.str();
    printf("EvalSlit: Incoherent scattering function output written to %s \n",(ss4.str()).c_str());
    out = fopen(out_string4.c_str(),"w");
    PrintSpacetimeCorrIn (out);
    ZeroSpacetimeCorrIn ();
    fclose(out);
}
  
  /* Help functions to calculate Structure factor and coherent scattering function */
  void FixScatteringLog::AllocArrays(){
    int N=0;
    for (const shared_ptr<Species>& sp1 : Sim->species) {
      for (const size_t& p1 : *sp1->getRange()) {
	N++;
      }
    }
    //printf("particles %d\n",N);
    int nb,k;
    AllocMem (valST, 8 * nModes * nFunCorr, real);
   
    AllocMem (densityProfile, profileBins, real);
    AllocMem2 (strucFac, nFunCorr, (2*nModes-1)*(2*nModes-1), real);
    
    AllocMem2 (blocking_sum, N_blocks*N_levels, 2*N, real);
    AllocMem2 (blocking_sumz, N_blocks*N_levels, N, real);
    AllocMem2 (correlationIn, N_blocks*N_levels,nModes * nFunCorr, real);
    AllocMem2 (countIn, N_blocks*N_levels,nModes * nFunCorr, int);
    AllocMem2 (shift, N_blocks*N_levels, 8 * nModes * nFunCorr, real);
    AllocMem2 (accumulator, N_levels, 8 * nModes * nFunCorr, real);
    AllocMem (naccumulator, N_levels, int);
    AllocMem2 (correlation, N_blocks*N_levels,nModes * nFunCorr, real);
    AllocMem (countcor, N_blocks*N_levels,int);
    AllocMem (insertindex, N_levels,int);
    AllocMem (pos_save, 3*N,real);
  }
  
/***************************************************************************************/
  
  void FixScatteringLog::EvalSpacetimeCorr (){
    real b, c, c0, c1, c2, kVal, s, s1, s2;
    real cc, sc, QVal;
    int j,  k, m, nb,  nv;
    
    const double dt = dynamic_cast<const SysTicker&>(*Sim->systems["SystemTicker"]).getPeriod() / Sim->units.unitTime();
    
    int N=0;
    for (const shared_ptr<Species>& sp1 : Sim->species) {
      for (const size_t& p1 : *sp1->getRange()) {
	N++;
      }
    }
    for (j = 0; j < 8 * nModes * nFunCorr; j ++) valST[j] = 0.;
    
    // calculate the density profile
    profileCount++;
    for (const shared_ptr<Species>& sp1 : Sim->species) {
      for (const size_t& p1 : *sp1->getRange()) {
	double pos = Sim->particles[p1].getPosition()[2];
	pos /= channel_w;
	if (pos < 1) densityProfile[int(pos*profileBins)]++;
      }
    }
    kVal = 2. * M_PI / Sim->primaryCellSize[1];
    QVal = 2. * M_PI /(channel_w-1.0);
    
    //printf("check00\n");
    
    // calculate FT for coherent scattering fct
    for (const shared_ptr<Species>& sp1 : Sim->species) {
      for (const size_t& p1 : *sp1->getRange()) {
	j = 0;
	//printf("pos %f\n",(Sim->particles[p1].getPosition()[2]-channel_w/2.0));
	for (int n=0; n<nModes; n++) {
	  cc = cos (n*QVal*(Sim->particles[p1].getPosition()[2]-channel_w/2.0));
	  sc = sin (n*QVal*(Sim->particles[p1].getPosition()[2]-channel_w/2.0));

	  for (k = 0; k < 2; k ++) {
	    for (m = 0; m < nFunCorr; m ++) {
	      if (m == 0) {
		b = kVal * Sim->particles[p1].getPosition()[k];
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
	      valST[j ++] += c*cc; //second element of SF -real part
	      valST[j ++] += c*sc; //imaginary 
	      valST[j ++] += s*cc; //second element of SF -real part
	      valST[j ++] += s*sc; //imaginary 
	    }
	  }
	}
      }
    }
    // acumualte and calculate logarithmic correlation function
    add(valST,0);
    
    //printf("check0\n");
    
    // calc structure factor
    for (int n=-nModes+1; n<nModes; n++) {
      for (int n2=-nModes+1; n2<nModes; n2++) {
	for (k = 0; k < 2; k ++) {
	  for (m = 0; m < nFunCorr; m ++) {
	    int ind_j = abs(n)*8*nFunCorr + k*4*nFunCorr + 4*m;
	    int ind_j2 = abs(n2)*8*nFunCorr + k*4*nFunCorr + 4*m;
	    double c_cc = valST[ind_j];
	    double c_sc = valST[ind_j+1];
	    double s_cc = valST[ind_j+2];
	    double s_sc = -valST[ind_j+3];
	    double c_cc2 = valST[ind_j2];
	    double c_sc2 = valST[ind_j2+1];
	    double s_cc2 = valST[ind_j2+2];
	    double s_sc2 = -valST[ind_j2+3];
	    if (n<0) {
	      s_sc = - s_sc;
	      c_sc = - c_sc;
	    }
	    if (n2<0) {
	      s_sc2 = - s_sc2;
	      c_sc2 = - c_sc2;
	    }
	    strucFac[m][(n+nModes-1)*(2*nModes-1)+n2+nModes-1] += (c_cc + s_sc)*(c_cc2 + s_sc2) + (s_cc+c_sc)*(s_cc2+c_sc2);
	    //strucFac[m][n*(2*nModes-1)+n2] += 1.0;
	  }
	}
      }
    }
    
    //printf("check1\n");
    
    // calculate \Delta r in blocking algorithm for self-intermediate scattering function
    int i = t_loc;
    int np=0;
    int max_levels = (int) (log(i)/log(N_blocks));
    if (max_levels >= N_levels) max_levels = N_levels-1;
    //printf("max_l %d\n",max_levels);
    for (const shared_ptr<Species>& sp1 : Sim->species) {
      for (const size_t& p1 : *sp1->getRange()) {
	double delx,dely,delz;
	int j0; // = i % N_blocks;
	//if (np==0) printf("checkin0\n");
	if (i==0) {
	  pos_save[3*np] = Sim->particles[p1].getPosition()[0];
	  pos_save[3*np+1] = Sim->particles[p1].getPosition()[1];
	  pos_save[3*np+2] = Sim->particles[p1].getPosition()[2];
	}
	for (int k=0; k<(max_levels+1); k++) {
	    
	  if (k==0) {
	    delx = Sim->particles[p1].getPosition()[0]-pos_save[3*np];
	    dely = Sim->particles[p1].getPosition()[1]-pos_save[3*np+1];
	    delz = Sim->particles[p1].getPosition()[2]-pos_save[3*np+2];
	    pos_save[3*np] = Sim->particles[p1].getPosition()[0];
	    pos_save[3*np+1] = Sim->particles[p1].getPosition()[1];
	    pos_save[3*np+2] = Sim->particles[p1].getPosition()[2];
	  } else {
	    delx = blocking_sum[N_blocks-1+(k-1)*N_blocks][np];
	    dely = blocking_sum[N_blocks-1+(k-1)*N_blocks][np+N];
	    delz = blocking_sumz[N_blocks-1+(k-1)*N_blocks][np];
	  }
	    
	  int nblocks_to_k = 1 ; // (int) round(pow(N_blocks,k));

	  for (int kk=0; kk<k; kk++) nblocks_to_k *= N_blocks;
	  //if (np==0) printf("checkin1\n");
	  if (i % nblocks_to_k == 0) {
	      
	    j0 =  ((i) / nblocks_to_k-1) % N_blocks;  // was -1

	    if (j0==0) {
	      blocking_sum[j0+k*N_blocks][np] = delx; //blocking_sum[N_blocks-1][k-1][n] ;
	      blocking_sum[j0+k*N_blocks][np+N] = dely; //blocking_sum[N_blocks-1][k-1][n] ;
	      blocking_sumz[j0+k*N_blocks][np] = delz; //blocking_sum[N_blocks-1][k-1][n] ;
	    }
	    else {
	      blocking_sum[j0+k*N_blocks][np] = blocking_sum[j0-1+k*N_blocks][np] + delx; 
	      blocking_sum[j0+k*N_blocks][np+N] = blocking_sum[j0-1+k*N_blocks][np+N] + dely; 
	      blocking_sumz[j0+k*N_blocks][np] = blocking_sumz[j0-1+k*N_blocks][np] + delz; 
	    }
	    
	    //if (np==0) printf("checkin2\n");

	    // now calculate incoherent scattering functions
	    for (int n=0; n<nModes; n++) {
	      cc = cos (n*QVal*blocking_sumz[j0+k*N_blocks][np]);
	      sc = sin (n*QVal*blocking_sumz[j0+k*N_blocks][np]);

	      for (int km = 0; km < 2; km ++) {
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
		  
		  nv = m + n*nFunCorr;
		  correlationIn[j0+k*N_blocks][nv] += c*cc-s*sc;
		  countIn[j0+k*N_blocks][nv] += 1;
		}
	      }
	    }
	    //if (np==0) printf("checkin3\n");
	  }
	}
	np++;
      }
    }
    
    //printf("check2\n");
    
    int t_tot = (int) pow(N_blocks,N_levels);
    if (t_loc == t_tot) {
      AccumSpacetimeCorr ();
      t_loc = 0;
      kmax = 0;
    } else t_loc++;
  }
  
  /***************************************************************************************/
  
  void FixScatteringLog::add(real * val, int k){
    // If we exceed the correlator side, the value is discarded
    if (k == N_levels) return;
    if (k > kmax) kmax=k;

    // Insert new value in shift array
    for (int i = 0; i < 8 * nFunCorr * nModes; i ++)
      shift[k*N_blocks+insertindex[k]][i] = val[i];

    // Add to accumulator and, if needed, add to next correlator
    for (int i = 0; i < 8 * nFunCorr * nModes; i ++)
      accumulator[k][i] += val[i];
    ++naccumulator[k];
    if (naccumulator[k]==N_blocks) {
      for (int i = 0; i < 8 * nFunCorr * nModes; i ++) accumulator[k][i] /= N_blocks;
      add(&accumulator[k][0], k+1);
      for (int i = 0; i < 8 * nFunCorr * nModes; i ++) accumulator[k][i]=0.0;
      naccumulator[k]=0;
    }

    // Calculate correlation function
    unsigned int ind1=insertindex[k];
    if (k==0) { // First correlator is different
      int ind2=ind1;
      for (int j=0;j<N_blocks;++j) {
	for (int n=0; n<nModes; n++) {
	  for (int kp = 0; kp < 2; kp ++) {
	    for (int m = 0; m < nFunCorr; m ++) {
	      int nv = m + n*nFunCorr;
	      int ind_j = n*8*nFunCorr + kp*4*nFunCorr + 4*m;
	      double c_cc = shift[k*N_blocks+ind2][ind_j];
	      double c_sc = shift[k*N_blocks+ind2][ind_j+1];
	      double s_cc = shift[k*N_blocks+ind2][ind_j+2];
	      double s_sc = -shift[k*N_blocks+ind2][ind_j+3];
	      double c_cc2 = shift[k*N_blocks+ind1][ind_j];
	      double c_sc2 = shift[k*N_blocks+ind1][ind_j+1];
	      double s_cc2 = shift[k*N_blocks+ind1][ind_j+2];
	      double s_sc2 = -shift[k*N_blocks+ind1][ind_j+3];
	      //printf("befor calc\n");
	      if (c_cc > -1e10) {
		correlation[k*N_blocks+j][nv]  += (c_cc + s_sc)*(c_cc2 + s_sc2) + (s_cc+c_sc)*(s_cc2+c_sc2);
	      //printf("after calc\n");
	      //if (n==1) printf("%f \n",tBuf[nb].acfST[nv][tBuf[nb].count]);
		if (nv==0) ++countcor[k*N_blocks+j];
	      }
	    }
	  }
	}
	--ind2;
	if (ind2<0) ind2+=N_blocks;
      }
    } else {
      int ind2=ind1-1;
      for (int j=1;j<N_blocks;++j) {
	if (ind2<0) ind2+=N_blocks;
	for (int n=0; n<nModes; n++) {
	  for (int kp = 0; kp < 2; kp ++) {
	    for (int m = 0; m < nFunCorr; m ++) {
	      int nv = m + n*nFunCorr;
	      int ind_j = n*8*nFunCorr + kp*4*nFunCorr + 4*m;
	      double c_cc = shift[k*N_blocks+ind2][ind_j];
	      double c_sc = shift[k*N_blocks+ind2][ind_j+1];
	      double s_cc = shift[k*N_blocks+ind2][ind_j+2];
	      double s_sc = -shift[k*N_blocks+ind2][ind_j+3];
	      double c_cc2 = shift[k*N_blocks+ind1][ind_j];
	      double c_sc2 = shift[k*N_blocks+ind1][ind_j+1];
	      double s_cc2 = shift[k*N_blocks+ind1][ind_j+2];
	      double s_sc2 = -shift[k*N_blocks+ind1][ind_j+3];
	      //printf("befor calc\n");
	      if (c_cc > -1e10) {
		correlation[k*N_blocks+j][nv]  += (c_cc + s_sc)*(c_cc2 + s_sc2) + (s_cc+c_sc)*(s_cc2+c_sc2);
		//printf("after calc\n");
		//if (n==1) printf("%f \n",tBuf[nb].acfST[nv][tBuf[nb].count]);
		if (nv==0) ++countcor[k*N_blocks+j];
	      }
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

  void FixScatteringLog::AccumSpacetimeCorr (){
    
    printf("accum1\n");

    // print coherent
    long double sysTime = Sim->systemTime/Sim->units.unitTime();
    //printf("systemTime %Lf\n",sysTime);
    std::string out_string;
    std::stringstream ss;
    ss << "S_t" << sysTime << ".dat";
    out_string = ss.str();
    printf("EvalSlit: Coherent scattering function output written to %s \n",(ss.str()).c_str());
    FILE * out = fopen(out_string.c_str(),"w");
    PrintSpacetimeCorr (out);
    ZeroSpacetimeCorr ();
    fclose(out);
    
    printf("accum2\n");
    
    // incoherent scattering fct
    std::string out_string2;
    std::stringstream ss2;
    ss2 << "SIn_t" << sysTime << ".dat";
    out_string2 = ss2.str();
    printf("EvalSlit: Incoherent scattering function output written to %s \n",(ss2.str()).c_str());
    out = fopen(out_string2.c_str(),"w");
    PrintSpacetimeCorrIn (out);
    ZeroSpacetimeCorrIn ();
    fclose(out);
    
    printf("accum3\n");
  }
  
/***************************************************************************************/

  void FixScatteringLog::ZeroSpacetimeCorr () 
  {
    for (int kp = 0; kp < N_blocks*N_levels; kp ++) {
      for (int j = 0; j < 8*nModes * nFunCorr; j ++) { 
	shift[kp][j] = -2E10;
      }
      for (int j = 0; j < nModes * nFunCorr; j ++) { 
	correlation[kp][j] = 0.;
      }
      countcor[kp] = 0;
    }
    for (int k = 0; k < N_levels; k ++) {
      for (int j = 0; j < 8*nModes * nFunCorr; j ++) { 
	accumulator[k][j] = 0.;
      }
      naccumulator[k] = 0;
      insertindex[k] = 0;
    }
  }

/***************************************************************************************/
  
  void FixScatteringLog::ZeroSpacetimeCorrIn () 
  {
    int N=0;
    for (const shared_ptr<Species>& sp1 : Sim->species) {
      for (const size_t& p1 : *sp1->getRange()) {
	N++;
      }
    }
    
    for (int kp = 0; kp < N_blocks*N_levels; kp ++) {
      for (int i = 0; i < N; i ++) { 
	blocking_sumz[kp][i] = 0.;
	blocking_sum[kp][i] = 0.;
	blocking_sum[kp][i+N] = 0.;
      }
      for (int j = 0; j < nModes * nFunCorr; j ++) { 
	correlationIn[kp][j] = 0.;
	countIn[kp][j] = 0;
      }
      
    }
 
  }
  
/***************************************************************************************/
  
  void FixScatteringLog::PrintDensityProfile (FILE *fp){
    real binsize = channel_w/((double) profileBins);
    real vol = Sim->primaryCellSize[0]*Sim->primaryCellSize[1]*binsize;
  
    double n0 = 0.0;
    int N=0;
    
    for (int i=0; i<profileBins; i++) {
      fprintf(fp,"%f %f\n",i*binsize-channel_w/2.0,densityProfile[i]/((double) profileCount)/vol);
      n0 += densityProfile[i]/((double) profileCount)/vol*binsize;
      N += densityProfile[i];
      densityProfile[i] = 0;
    }
    
    printf("EvalSlit: n=%f, n0=%f, N=%d\n",n0,n0*Sim->primaryCellSize[2]/4.0,N/profileCount);
  }
  
/***************************************************************************************/
  
  void FixScatteringLog::PrintStrucFac  (FILE *fp){

    fprintf(fp,"#qval ");
    for (int n=-nModes+1; n<nModes; n++) {
      for (int n2=-nModes+1; n2<nModes; n2++) {

	fprintf(fp,"%d/%d ",n,n2);
      }
    }
    fprintf(fp,"\n");
    
    // print structure factor
    real kval = 2. * M_PI / Sim->primaryCellSize[1];
    for (int m = 0; m < nFunCorr; m ++) {
      fprintf(fp,"%f ",(m+1)*kval);
      for (int n=-nModes+1; n<nModes; n++) {
	for (int n2=-nModes+1; n2<nModes; n2++) {
	  for (const shared_ptr<Species>& sp1 : Sim->species) {
	    fprintf(fp,"%f ",strucFac[m][(n+nModes-1)*(2*nModes-1)+n2+nModes-1]/((double) profileCount)/2. / sp1->getCount());
	  }
	}
      }
      fprintf(fp,"\n");
    }
  }
  
/***************************************************************************************/

  void FixScatteringLog::PrintSpacetimeCorr (FILE *fp){
    real tVal;
    int j, n;
    
    int N=0;
    for (const shared_ptr<Species>& sp1 : Sim->species) {
      for (const size_t& p1 : *sp1->getRange()) {
	N++;
      }
    }

    for (n=0; n<nModes; n++) {
      fprintf (fp, "n=%d\n",n);
      
      const double dt = dynamic_cast<const SysTicker&>(*Sim->systems["SystemTicker"]).getPeriod() / Sim->units.unitTime();
      
      for (int j=0;j<N_blocks;++j) {
	if (countcor[j] > 0) {
	  double t = j*dt;
	  fprintf (fp, "%8.4f", t);
	  for (int m = 0; m < nFunCorr; m ++) {
	    int nv = m + n*nFunCorr;
	    fprintf (fp, " %8.4f", correlation[j][nv]/countcor[j]/N);
	  }
	  fprintf (fp, "\n");
	}
      }
      
      for (int k=1;k<=kmax;++k) {
	for (int j=1;j<N_blocks;++j) {
	  if (countcor[k*N_blocks+j]>0) {
	    double t = j * pow((double)N_blocks, k)*dt;
	    fprintf (fp, "%8.4f", t);
	    for (int m = 0; m < nFunCorr; m ++) {
	      int nv = m + n*nFunCorr;
	      fprintf (fp, " %8.4f", correlation[k*N_blocks+j][nv]/countcor[k*N_blocks+j]/N);
	    }
	    fprintf (fp, "\n");
	  }
	}
      }
      
      fprintf (fp, "\n\n");
    }
  }
  
/***************************************************************************************/

  void FixScatteringLog::PrintSpacetimeCorrIn (FILE *fp){
    real tVal;
    int j, n;

    for (n=0; n<nModes; n++) {
      fprintf (fp, "n=%d\n",n);
      
      const double dt = dynamic_cast<const SysTicker&>(*Sim->systems["SystemTicker"]).getPeriod() / Sim->units.unitTime();
    
      for (int k=0; k<N_levels; k++)
	for (int j=0; j<N_blocks; j++) {
	  double time_here = (j+1) * (pow(N_blocks,k))*dt;
	  fprintf (fp, "%8.4f", time_here);
	  for (int m = 0; m < nFunCorr; m ++) {
	    int nv = m + n*nFunCorr;
	    if (countIn[k*N_blocks+j][nv] > 0) fprintf (fp, " %8.4f", correlationIn[k*N_blocks+j][nv]/countIn[k*N_blocks+j][nv]);
	  }
	  fprintf (fp, "\n");
	}
      
      fprintf (fp, "\n\n");
    }
  }
 
}