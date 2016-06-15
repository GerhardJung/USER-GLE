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
   Contributing authors:
     Benoit Leblanc, Dave Rigby, Paul Saxe (Materials Design)
     Reese Jones (Sandia)
------------------------------------------------------------------------- */

#include "stdlib.h"
#include "string.h"
#include "unistd.h"
#include "compute_memory_volterra.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "group.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "atom.h"
#include "comm.h"
#include "fix_ave_correlate_peratom.h"

using namespace LAMMPS_NS;

enum{ONE,RUNNING};
enum{PERATOM,GLOBAL};

/* ---------------------------------------------------------------------- */

ComputeMemoryVolterra::ComputeMemoryVolterra(LAMMPS * lmp, int narg, char **arg):
  Compute (lmp, narg, arg)
{
  if (narg < 6) error->all(FLERR,"Illegal fix memory/volterra command");
  
  nevery_corr = force->inumeric(FLERR,arg[3]);
  nrepeat = force->inumeric(FLERR,arg[4]);
  nfreq = force->inumeric(FLERR,arg[5]);
  
  // velocities are relevant variables, also needed are forces -> 21 correlations calculated but only 18 needed -> mapping in mask
  ncorr = 18;
  nmem = 6;
  
  // this compute produces a global array
  array_flag = 1;
  size_array_rows =  nrepeat;
  size_array_cols =  nmem;
  extarray = 0;

  MPI_Comm_rank(world,&me);
  
  // this compute is evaluated every nfreq steps
  nevery = nfreq;
  
  // read in optional parameter
  memory_switch = PERATOM;
  int iarg = 6;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"switch") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      if (strcmp(arg[iarg+1],"peratom") == 0) memory_switch = PERATOM;
      else if (strcmp(arg[iarg+1],"global") == 0) memory_switch = GLOBAL;
      else error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      iarg += 2;
    } else error->all(FLERR,"Illegal fix ave/correlate/peratom command");
  }

  // setup and error check
  // will be transferred to the fix/ave/correlate/peratom
  
  // init variables for forces and velocities
  char **newarg_v = new char*[18];
  newarg_v[0] = (char *) "vx";  newarg_v[1] = (char *) "atom"; newarg_v[2] = (char *) "vx"; 
  newarg_v[3] = (char *) "vy";  newarg_v[4] = (char *) "atom"; newarg_v[5] = (char *) "vy"; 
  newarg_v[6] = (char *) "vz";  newarg_v[7] = (char *) "atom"; newarg_v[8] = (char *) "vz"; 
  newarg_v[9] = (char *) "fx";  newarg_v[10] = (char *) "atom"; newarg_v[11] = (char *) "fx"; 
  newarg_v[12] = (char *) "fy";  newarg_v[13] = (char *) "atom"; newarg_v[14] = (char *) "fy"; 
  newarg_v[15] = (char *) "fz";  newarg_v[16] = (char *) "atom"; newarg_v[17] = (char *) "fz"; 
  for (int i=0; i<6; i++) {
    input->variable->set(3,&newarg_v[3*i]); 
  }
  delete [] newarg_v;
  
  // init ave/correlate fix
  // id = compute-ID + COMPUTE_CORRELATE, fix group = compute group
  int n = strlen(id) + strlen("_COMPUTE_CORRELATE") + 1;
  id_fix = new char[n];
  strcpy(id_fix,id);
  strcat(id_fix,"_COMPUTE_CORRELATE");
  char c_nevery[15];
  char c_nrepeat[15];
  char c_nfreq[15];
  sprintf(c_nevery, "%d", nevery_corr);
  sprintf(c_nrepeat, "%d", nrepeat);
  sprintf(c_nfreq, "%d", nfreq);
  
  int narg_corr = 17;
  if(memory_switch==GLOBAL) narg_corr += 2;
  char **newarg_f = new char*[narg_corr];
  newarg_f[0] = id_fix;
  newarg_f[1] = group->names[igroup];
  newarg_f[2] = (char *) "ave/correlate/peratom";
  newarg_f[3] = c_nevery;
  newarg_f[4] = c_nrepeat;
  newarg_f[5] = c_nfreq;
  newarg_f[6] = (char *) "v_vx"; newarg_f[7] = (char *) "v_vy"; newarg_f[8] = (char *) "v_vz"; 
  newarg_f[9] = (char *) "v_fx"; newarg_f[10] = (char *) "v_fy"; newarg_f[11] = (char *) "v_fz";
  newarg_f[12] = (char *) "type"; newarg_f[13] = (char *) "auto/upper";
  newarg_f[14] = (char *) "ave"; newarg_f[15] = (char *) "running";
  newarg_f[16] = (char *) "restart";
  if(memory_switch==GLOBAL) newarg_f[17] = (char *) "switch"; newarg_f[18] = (char *) "global";
  modify->add_fix(narg_corr,newarg_f);
  fix = (FixAveCorrelatePeratom *) modify->fix[modify->nfix-1];
  delete [] newarg_f;
  
  //determine amss of the particles
  int nlocal= atom->nlocal;
  int *mask= atom->mask; 
  int *type = atom->type;
  double *a_mass = atom->mass;
  int a;
  for (a= 0; a < nlocal; a++) {
    if(mask[a] & groupbit) {
      mass=a_mass[type[a]];
    }
  }
  
  // allocate memory
  memory->create(array,nrepeat,nmem,"memory/volterra:array");
  int i,j;
  for (i = 0; i<nrepeat; i++)
    for (j = 0; j<nmem; j++)
      array[i][j]=0;
}

/* ---------------------------------------------------------------------- */

ComputeMemoryVolterra::~ComputeMemoryVolterra()
{
  
  // check nfix in case all fixes have already been deleted
  if (modify->nfix) modify->delete_fix(id_fix);
  delete [] id_fix;
  memory->destroy(array);
  
}

/* ---------------------------------------------------------------------- */

void ComputeMemoryVolterra::init()
{
  // set fix which stores original atom velocities

  int ifix = modify->find_fix(id_fix);
  if (ifix < 0) error->all(FLERR,"Could not find compute memory/volterra fix ID");
  fix = (FixAveCorrelatePeratom *) modify->fix[ifix];
  
}

/* ----------------------------------------------------------------------
   compute array value
------------------------------------------------------------------------- */

void ComputeMemoryVolterra::compute_array()
{
  double **corr;
  int i,j;
  memory->create(corr,nrepeat,ncorr,"memory/volterra:corr");
  // calculate correlation first
  fix->end_of_step();
  //read in correlation function of the invoked fix
  //mask tells where to find the correct correlations
  double mask[] = {0,3,15,1,4,16,2,5,17,6,9,18,7,10,19,11,14,20};
  for (i = 0; i<nrepeat; i++)
    for (j = 0; j<ncorr; j++)
      corr[i][j]=fix->compute_array(i, mask[j]+2);
  // use correlation function to calculate memory
  for (j = 0; j<nmem; j++){
    array[0][j]=corr[0][3*j+2]/corr[0][3*j]/mass/mass;
    for(i = 1; i<nrepeat; i++){
      //denum = C(0)+dt*C'(i)
      double denum = mass*mass*corr[0][3*j]+0.5*mass*update->dt*corr[i][3*j+1];
      //num = C''(i)-dt*sum(C'(i-ip)*k(ip))
      double num = corr[i][3*j+2];
      num -= 0.5*mass*corr[i][3*j+1]*array[0][j]*update->dt;
      int ip;
      for(ip = 1; ip<i; ip++){
	num -= mass*update->dt*corr[i-ip][3*j+1]*array[ip][j];
      }
      array[i][j]=num/denum;
    }
  }
  for (j = 0; j<nmem; j++){
    for(i = 0; i<nrepeat; i++){
      array[i][j] *= mass*mass*corr[0][3*j];
    }
  }
    
  memory->destroy(corr);
}