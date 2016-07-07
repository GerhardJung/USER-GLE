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
#include "fix_ave_correlate_peratom.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "group.h"
#include "fix_store.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "atom.h"
#include "comm.h"
#include <algorithm>    // std::find
#include <math.h>	// fabs

using namespace LAMMPS_NS;
using namespace FixConst;

enum{COMPUTE,FIX,VARIABLE};
enum{ONE,RUNNING};
enum{NORMAL,ORTHOGONAL,ORTHOGONALSECOND};
enum{AUTO,CROSS,AUTOCROSS,AUTOUPPER};
enum{PERATOM,GLOBAL};
enum{NOT_DEPENDENED,VAR_DEPENDENED,DIST_DEPENDENED};

#define INVOKED_SCALAR 1
#define INVOKED_VECTOR 2
#define INVOKED_ARRAY 4
#define INVOKED_PERATOM 8

/* ---------------------------------------------------------------------- */

FixAveCorrelatePeratom::FixAveCorrelatePeratom(LAMMPS * lmp, int narg, char **arg):
  Fix (lmp, narg, arg)
{
  if (narg < 7) error->all(FLERR,"Illegal fix ave/correlate/peratom command");

  MPI_Comm_rank(world,&me);

  nevery = force->inumeric(FLERR,arg[3]);
  nrepeat = force->inumeric(FLERR,arg[4]);
  nfreq = force->inumeric(FLERR,arg[5]);
  

  global_freq = nfreq;
  // parse values until one isn't recognized

  which = new int[narg];
  argindex = new int[narg];
  ids = new char*[narg];
  value2index = new int[narg];
  nvalues = 0;

  int iarg = 6;
  while (iarg < narg) {
    if (strncmp(arg[iarg],"c_",2) == 0 ||
        strncmp(arg[iarg],"f_",2) == 0 ||
        strncmp(arg[iarg],"v_",2) == 0) {
      if (arg[iarg][0] == 'c') which[nvalues] = COMPUTE;
      else if (arg[iarg][0] == 'f') which[nvalues] = FIX;
      else if (arg[iarg][0] == 'v') which[nvalues] = VARIABLE;

      int n = strlen(arg[iarg]);
      char *suffix = new char[n];
      strcpy(suffix,&arg[iarg][2]);

      char *ptr = strchr(suffix,'[');
      if (ptr) {
        if (suffix[strlen(suffix)-1] != ']')
          error->all(FLERR,"Illegal fix ave/correlate/peratom command");
        argindex[nvalues] = atoi(ptr+1);
        *ptr = '\0';
      } else argindex[nvalues] = 0;

      n = strlen(suffix) + 1;
      ids[nvalues] = new char[n];
      strcpy(ids[nvalues],suffix);
      delete [] suffix;

      nvalues++;
      iarg++;
    } else break;
  }

  // optional args

  type = AUTO;
  ave = ONE;
  startstep = 0;
  prefactor = 1.0;
  fp = NULL;
  dynamics = NORMAL;
  memory_switch = PERATOM;
  include_orthogonal =  0;
  variable_flag = NOT_DEPENDENED;
  bins = 1;
  factor = 1;
  mean_file = NULL;
  mean_flag = 0;
  variable_nvalues = 0;
  overwrite = 0;
  char *title1 = NULL;
  char *title2 = NULL;
  char *title3 = NULL;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"type") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      if (strcmp(arg[iarg+1],"auto") == 0) type = AUTO;
      if (strcmp(arg[iarg+1],"cross") == 0) type = CROSS;
      else if (strcmp(arg[iarg+1],"auto/cross") == 0){ 
	type = AUTOCROSS;
	factor = 2;
      } else if (strcmp(arg[iarg+1],"auto/upper") == 0) type = AUTOUPPER;
      else error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"ave") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      if (strcmp(arg[iarg+1],"one") == 0) ave = ONE;
      else if (strcmp(arg[iarg+1],"running") == 0) ave = RUNNING;
      else error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"start") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      startstep = force->inumeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"prefactor") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      prefactor = force->numeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"file") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      if (me == 0) {
        fp = fopen(arg[iarg+1],"w");
        if (fp == NULL) {
          char str[128];
          sprintf(str,"Cannot open fix ave/correlate/peratom file %s",arg[iarg+1]);
          error->one(FLERR,str);
        }
      }
      iarg += 2;
    } else if (strcmp(arg[iarg],"variable") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      if (strncmp(arg[iarg+1],"v_",2) == 0) {
	variable_flag = VAR_DEPENDENED;
	variable_nvalues = 1;
	int n = strlen(arg[iarg+1]);
	char *suffix = new char[n];
	strcpy(suffix,&arg[iarg+1][2]);
	n = strlen(suffix) + 1;
	variable_id = new char[n];
	strcpy(variable_id,suffix);
	delete [] suffix;
      } else if (strcmp(arg[iarg+1],"distance") == 0) {
	factor = 2;
	variable_flag = DIST_DEPENDENED;
	variable_nvalues = 3;
      } else error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      range = force->inumeric(FLERR,arg[iarg+2]);
      bins = force->inumeric(FLERR,arg[iarg+3]);
      iarg += 4;
    } else if (strcmp(arg[iarg],"dynamics") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      if (strcmp(arg[iarg+1],"normal") == 0) dynamics = NORMAL;
      else if (strcmp(arg[iarg+1],"orthogonal") == 0) {
	dynamics = ORTHOGONAL;
	include_orthogonal = nvalues + 6;
      } else if (strcmp(arg[iarg+1],"orthogonal/second") == 0) {
	dynamics = ORTHOGONALSECOND;
	include_orthogonal = nvalues + 6;
      } else error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"switch") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      if (strcmp(arg[iarg+1],"peratom") == 0) memory_switch = PERATOM;
      else if (strcmp(arg[iarg+1],"global") == 0) memory_switch = GLOBAL;
      else error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"restart") == 0) {
      restart_global = 1;
      iarg += 1;
    } else if (strcmp(arg[iarg],"overwrite") == 0) {
      overwrite = 1;
      iarg += 1;
    } else if (strcmp(arg[iarg],"mean") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      if (me == 0) {
        mean_file = fopen(arg[iarg+1],"w");
        if (mean_file == NULL) {
          char str[128];
          sprintf(str,"Cannot open fix ave/correlate/peratom file %s",arg[iarg+1]);
          error->one(FLERR,str);
        }
      }
      mean_flag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"title1") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      delete [] title1;
      int n = strlen(arg[iarg+1]) + 1;
      title1 = new char[n];
      strcpy(title1,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"title2") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      delete [] title2;
      int n = strlen(arg[iarg+1]) + 1;
      title2 = new char[n];
      strcpy(title2,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"title3") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      delete [] title3;
      int n = strlen(arg[iarg+1]) + 1;
      title3 = new char[n];
      strcpy(title3,arg[iarg+1]);
      iarg += 2;
    } else error->all(FLERR,"Illegal fix ave/correlate/peratom command");
  }

  // setup and error check
  // for fix inputs, check that fix frequency is acceptable

  if (nevery <= 0 || nrepeat <= 0 || nfreq <= 0)
    error->all(FLERR,"Illegal fix ave/correlate/peratom command");
  if (nfreq % nevery)
    error->all(FLERR,"Illegal fix ave/correlate/peratom command");
  if (ave == ONE && nfreq < (nrepeat-1)*nevery)
    error->all(FLERR,"Illegal fix ave/correlate/peratom command");
  if (ave != RUNNING && overwrite)
    error->all(FLERR,"Illegal fix ave/correlate/peratom command");
  if (dynamics == ORTHOGONAL || dynamics == ORTHOGONALSECOND) {
    nav = nfreq/nevery - nrepeat;
    // to calculate orthogonal dynamics we first have to initialize the arrays with nav values. 
    // then we can calculate the correlation for the next nrepeat steps.
    if (nav <= 0)
      error->all(FLERR,"Illegal fix ave/correlate/peratom command");
    nsave = nav;
  } else {
    nsave = nrepeat;
  }
  // distance dependence only makes sence when we calculate cross correlation
  if (variable_flag == DIST_DEPENDENED && type != CROSS){
    error->all(FLERR,"Illegal fix ave/correlate/peratom command: distance dependence without cross correlation");
  }
  if (variable_flag == DIST_DEPENDENED && nvalues % 3) {
    error->all(FLERR,"Illegal fix ave/correlate/peratom command: distance dependence decomposes 3d-system into parallel and orthogonal component");
  }

  int i,j,o;
  int n_scalar = 0;
  for (i = 0; i < nvalues; i++) {
    if (which[i] == COMPUTE) {
      int icompute = modify->find_compute(ids[i]);
      //no such compute
      if (icompute < 0)
        error->all(FLERR,"Compute ID for fix ave/correlate/peratom does not exist");
      if (argindex[i] == 0) { //allegedly a scalar
	if (modify->compute[icompute]->peratom_flag != 1) {
	  error->all(FLERR, "Fix ave/correlate/peratom compute does not calculate a peratom scalar");
	}
      } else {
	if (modify->compute[icompute]->peratom_flag == 1) {
	  if (argindex[i] > modify->compute[icompute]->size_peratom_cols) {
	    error->all(FLERR,"Fix ave/correlate/peratom compute vector is accessed out-of-range");
	  }
	} else {
	  error->all(FLERR, "Fix ave/correlate/peratom compute does not calculate a peratom vector");
	}
      }
    } else if (which[i] == FIX) {
      int ifix = modify->find_fix(ids[i]);
      if (ifix < 0)
        error->all(FLERR,"Fix ID for fix ave/correlate/peratom does not exist");
      if (argindex[i] == 0) { //allegedly a scalar
	if (modify->fix[ifix]->peratom_flag != 1) {
	  error->all(FLERR, "Fix ave/correlate/peratom fix does not calculate a peratom scalar");
	}
      } else {
	if (modify->fix[ifix]->peratom_flag == 1) {
	  if (argindex[i] > modify->fix[ifix]->size_peratom_cols) {
	    error->all(FLERR,"Fix ave/correlate/peratom fix vector is accessed out-of-range");
	  }
	} else {
	  error->all(FLERR, "Fix ave/correlate/peratom fix does not calculate a peratom vector");
	}
      }
      if (nevery % modify->fix[ifix]->peratom_freq)
	error->all(FLERR,"Fix for fix ave/correlate/peratom "
                   "not computed at compatible time");
      else if (nevery % modify->fix[ifix]->global_freq)
        error->all(FLERR,"Fix for fix ave/correlate/peratom "
                   "not computed at compatible time");

    } else if (which[i] == VARIABLE) {
      int ivariable = input->variable->find(ids[i]);
      if (ivariable < 0)
        error->all(FLERR,"Variable name for fix ave/correlate/peratom does not exist");
      if (input->variable->atomstyle(ivariable) != 1) {
        error->all(FLERR, "Fix ave/correlate/peratom variable is not an equal- or atom-style variable");
      }
    }
  }
  
  // npair = # of correlation pairs to calculate
   if (type == AUTO || type == CROSS || type == AUTOCROSS) npair = nvalues;
   if (variable_flag == DIST_DEPENDENED) npair /= 3;
   if (type == AUTOUPPER) npair = nvalues*(nvalues+1)/2;
  // print file comment lines
  if (fp && me == 0) {
    if (title1) fprintf(fp,"%s\n",title1);
    else fprintf(fp,"# Time-correlated data for fix %s\n",id);
    if (title2) fprintf(fp,"%s\n",title2);
    else fprintf(fp,"# Timestep Number-of-time-windows\n");
    if (title3) fprintf(fp,"%s\n",title3);
    else {
      fprintf(fp,"# Index TimeDelta Ncount");
      if (variable_flag == DIST_DEPENDENED)
	for (i = 0; i < nvalues ; i+=3){
	  int n1 = strlen(arg[6+i])+strlen("_p");
	  int n2 = strlen(arg[6+i])+strlen("_o");
	  char str1[n1];
	  char str2[n2];
	  strcpy(str1, arg[6+i]);
	  strcpy(str2, arg[6+i]);
	  strcat(str1,"_p");
	  strcat(str2,"_o");
          fprintf(fp," %s*%s",str1,str1);
	  fprintf(fp," %s*%s",str2,str2);
	}
      else if (type == AUTO || type == AUTOCROSS || type == CROSS )
        for (i = 0; i < nvalues ; i++)
          fprintf(fp," %s*%s",arg[6+i],arg[6+i]);
      else if (type == AUTOUPPER)
        for (i = 0; i < nvalues; i++)
          for (j = i; j < nvalues; j++)
            fprintf(fp," %s*%s",arg[6+i],arg[6+j]);
      fprintf(fp,"\n");
    }
    filepos = ftell(fp);
  }

  delete [] title1;
  delete [] title2;
  delete [] title3;
  
 

  // allocate and initialize memory for averaging
  // set count and corr to zero since they accumulate
  // also set save versions to zero in case accessed via compute_array()
  corr_length = nrepeat*bins*factor;
  memory->create(count,corr_length,"ave/correlate/peratom:count");
  memory->create(save_count,corr_length,"ave/correlate/peratom:save_count");
  memory->create(corr,corr_length,npair,"ave/correlate/peratom:corr");
  memory->create(save_corr,corr_length,npair,"ave/correlate/peratom:save_corr"); 
  for (i = 0; i < corr_length; i++) {
    save_count[i] = count[i] =  0.0;
    for (j = 0; j < npair; j++) save_corr[i][j] = corr[i][j] = 0.0;
  }
  
  if (mean_flag) {
    // create file
    if (me == 0) {
      fprintf(mean_file,"# Time-averaged data for fix %s\n",id);
      fprintf(mean_file,"# Index Ncount");
      if (variable_flag == DIST_DEPENDENED)
	for (i = 0; i < nvalues ; i+=3){
	  int n1 = strlen(arg[6+i])+strlen("_p");
	  int n2 = strlen(arg[6+i])+strlen("_o");
	  char str1[n1];
	  char str2[n2];
	  strcpy(str1, arg[6+i]);
	  strcpy(str2, arg[6+i]);
	  strcat(str1,"_p");
	  strcat(str2,"_o");
          fprintf(fp," %s*%s",str1,str1);
	  fprintf(fp," %s*%s",str2,str2);
	}
      else for (i = 0; i < nvalues ; i++) fprintf(mean_file," %s*%s",arg[6+i],arg[6+i]);
      fprintf(mean_file,"\n");
      mean_filepos = ftell(mean_file);
    }
 
    // init memory
    memory->create(mean,nvalues*bins,"ave/correlate/peratom:mean");
    memory->create(mean_count,bins,"ave/correlate/peratom:mean_count");
    for(o = 0; o < bins; o++){
      for (i = 0; i < nvalues; i++) {
	mean[i+o*nvalues]=0.0;
      }
      mean_count[o]=0.0;
    }
  }
  

  // this fix produces a global array
  array_flag = 1;
  if (variable_flag == VAR_DEPENDENED || variable_flag == DIST_DEPENDENED) size_array_rows = nrepeat*bins;
  else size_array_rows = nrepeat;
  size_array_cols = npair+2;
  extarray = 0;

  // nvalid = next step on which end_of_step does something
  // add nvalid to all computes that store invocation times
  // since don't know a priori which are invoked by this fix
  // once in end_of_step() can set timestep for ones actually invoked

  firstindex = 0;
  lastindex = -1;
  nsample = 0;
  nvalid = nextvalid();
  modify->addstep_compute_all(nvalid);
  first = 1;
  
  // find the number of atoms in the relevant group
  int a;
  int nlocal= atom->nlocal;
  int *mask= atom->mask;  
  int ngroup_loc=0, ngroup_scan=0;
  int *indices_group;
  
  memory->grow(indices_group,ngroup_loc,"ave/correlate/peratomindices_group");
  for (a= 0; a < nlocal; a++) {
    if(mask[a] & groupbit) {
      ngroup_loc++;
      memory->grow(indices_group,ngroup_loc,"ave/correlate/peratomindices_group");
      indices_group[ngroup_loc-1]=a;
    }
  }
  MPI_Allreduce(&ngroup_loc, &ngroup_glo, 1, MPI_INT, MPI_SUM, world);
  MPI_Exscan(&ngroup_loc,&ngroup_scan,1,MPI_INT, MPI_SUM, world);
  //printf("local=%d, scan=%d, global=%d\n",ngroup_loc,ngroup_scan,ngroup_glo);
  if((type == AUTOCROSS || type == CROSS) && ngroup_glo < 2) error->all(FLERR,"Illegal fix ave/correlate/peratom command: Cross-correlation with only one particle");
  array= NULL;
  variable_store=NULL;
  if(nvalues > 0) {
    if(memory_switch == PERATOM){
      // need to grow array size
      comm->maxexchange_fix = MAX(comm->maxexchange_fix,(nvalues+include_orthogonal+variable_nvalues)*nsave);
      grow_arrays(atom->nmax);
      atom->add_callback(0);
    } else {
      // create global memorys
      grow_arrays(ngroup_glo);
      tagint *group_ids_loc;
      double *group_mass_loc;
      memory->create(group_ids,ngroup_glo,"ave/correlate/peratom:group_ids");
      memory->create(group_ids_loc,ngroup_glo,"ave/correlate/peratom:group_ids_loc");
      memory->create(group_mass,ngroup_glo,"ave/correlate/peratom:group_mass");
      memory->create(group_mass_loc,ngroup_glo,"ave/correlate/peratom:group_mass_loc");
      // find ids of groupmembers
      tagint *tag = atom->tag;
      int *type = atom->type;
      double *mass = atom->mass;
      for (a= 0; a < ngroup_glo; a++) {
	group_ids_loc[a]=group_ids[a]=group_mass_loc[a]=group_mass[a]=0;
      }
      for (a= 0; a < ngroup_loc; a++) {
	group_ids_loc[a+ngroup_scan]=tag[indices_group[a]];
	group_mass_loc[a+ngroup_scan]=mass[type[indices_group[a]]];
      }
      MPI_Allreduce(group_ids_loc, group_ids, ngroup_glo, MPI_INT, MPI_SUM, world);
      MPI_Allreduce(group_mass_loc, group_mass, ngroup_glo, MPI_DOUBLE, MPI_SUM, world);
      memory->destroy(group_ids_loc);
      memory->destroy(group_mass_loc);

      // create memory for data storage and distribute
      memory->create(group_data_loc,ngroup_glo,nvalues+include_orthogonal+variable_nvalues,"ave/correlate/peratom:group_data_loc");
      memory->create(group_data,ngroup_glo,nvalues+include_orthogonal+variable_nvalues,"ave/correlate/peratom:group_data");
    }
  }
  memory->destroy(indices_group);
  
  if (dynamics == ORTHOGONAL || dynamics == ORTHOGONALSECOND) {
    //create memory for orthogonal dynamics
    int i,a,r,t;
    memory->create(alpha,ngroup_glo,3*nvalues*nrepeat,"ave/correlate/peratom:alpha");
    memory->create(norm,ngroup_glo,3*nrepeat,"ave/correlate/peratom:norm");
    if (dynamics == ORTHOGONALSECOND){
      memory->create(epsilon,ngroup_glo,3*nvalues*nrepeat,"ave/correlate/peratom:epsilon");
      memory->create(kappa,ngroup_glo,nrepeat,"ave/correlate/peratom:kappa");
      memory->create(zeta,ngroup_glo,nrepeat,"ave/correlate/peratom:zeta"); 
    }
    // set memory to zero
    for (a= 0; a < ngroup_glo; a++) {
      for (t = 0; t < nrepeat; t++) {
	if (dynamics == ORTHOGONALSECOND){
	  kappa[a][t] = 0;
	  zeta[a][t] = 0;
	}
	for (r=0; r<3; r++){
	  norm[a][r+t*3]=0;
	  for (i = 0; i < nvalues; i++) {
	    alpha[a][i+r*nvalues+t*nvalues*3]=0;
	    if (dynamics == ORTHOGONALSECOND) {
	      epsilon[a][i+r*nvalues+t*nvalues*3]=0;
	    }
	  }
	}
      }
    }
  }
  //init timing
  /* time_init_compute=0;
  calc_write_nvalues=0;
  write_var=0;
  write_orthogonal=0;
  reduce_write_global=0;
  time_calc=0;
  time_calc_mean=0;
  time_total=0; */
}

/* ---------------------------------------------------------------------- */

FixAveCorrelatePeratom::~FixAveCorrelatePeratom()
{

  delete [] which;
  delete [] argindex;
  delete [] value2index;
  for (int i = 0; i < nvalues; i++) delete [] ids[i];
  delete [] ids;

  memory->destroy(array);
  memory->destroy(count);
  memory->destroy(save_count);
  memory->destroy(corr);
  memory->destroy(save_corr);
  if (variable_flag == VAR_DEPENDENED || variable_flag == DIST_DEPENDENED) memory->destroy(variable_store);
  
  if (dynamics == ORTHOGONAL || dynamics == ORTHOGONALSECOND) {
    memory->destroy(alpha);
    memory->destroy(norm);
    if (dynamics == ORTHOGONALSECOND){
      memory->destroy(epsilon);
      memory->destroy(kappa);
      memory->destroy(zeta); 
    }
  }
  
  memory->destroy(group_data_loc);
  memory->destroy(group_data);
  
  memory->destroy(group_ids);
  memory->destroy(group_mass);
  
  if (mean_flag) {
    memory->destroy(mean);
    memory->destroy(mean_count);
  }
  
  if (fp && me == 0) fclose(fp);
  
}

/* ---------------------------------------------------------------------- */

int FixAveCorrelatePeratom::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAveCorrelatePeratom::init()
{
  // set current indices for all computes,fixes,variables
  int i;
  for (i = 0; i < nvalues; i++) {
    if (which[i] == COMPUTE) {
      int icompute = modify->find_compute(ids[i]);
      if (icompute < 0)
        error->all(FLERR,"Compute ID for fix ave/correlate/peratom does not exist");
      value2index[i] = icompute;

    } else if (which[i] == FIX) {
      int ifix = modify->find_fix(ids[i]);
      if (ifix < 0)
        error->all(FLERR,"Fix ID for fix ave/correlate/peratom does not exist");
      value2index[i] = ifix;

    } else if (which[i] == VARIABLE) {
      int ivariable = input->variable->find(ids[i]);
      if (ivariable < 0)
        error->all(FLERR,"Variable name for fix ave/correlate/peratom does not exist");
      value2index[i] = ivariable;
    }
  }
  
  if (variable_flag == VAR_DEPENDENED){
    int ivariable = input->variable->find(variable_id);
    if (ivariable < 0)
      error->all(FLERR,"Variable name for fix ave/correlate/peratom does not exist");
    variable_value2index = ivariable;
  }

  // need to reset nvalid if nvalid < ntimestep b/c minimize was performed

  if (nvalid < update->ntimestep) {
    firstindex = 0;
    lastindex = -1;
    nsample = 0;
    nvalid = nextvalid();
    modify->addstep_compute_all(nvalid);
  }
  
}

/* ----------------------------------------------------------------------
   only does something if nvalid = current timestep
------------------------------------------------------------------------- */

void FixAveCorrelatePeratom::setup(int vflag)
{
  end_of_step();
}

/* ---------------------------------------------------------------------- */

void FixAveCorrelatePeratom::end_of_step()
{

  
  int a,i,j,o,v,r,ngroup_loc=0;
  double scalar;
  double *peratom_data;
  int *indices_group;
      
  int nlocal= atom->nlocal;
  int *mask= atom->mask;  
  double *mass = atom->mass;
  tagint *tag = atom->tag;
  
  //printf("nsample = %d, lastindex = %d, nvalid = %d\n",nsample,lastindex,nvalid);
  
  // skip if not step which requires doing something
  bigint ntimestep = update->ntimestep;
  if (ntimestep != nvalid) return;
  
  //double tt1 = MPI_Wtime();
  //t1 = MPI_Wtime();
  
  // find relevant particles // find group-member on each processor
  memory->grow(indices_group,ngroup_loc,"ave/correlate/peratomindices_group");
  for (a= 0; a < nlocal; a++) {
    if(mask[a] & groupbit) {
      ngroup_loc++;
      memory->grow(indices_group,ngroup_loc,"ave/correlate/peratomindices_group");
      indices_group[ngroup_loc-1]=a;
    }
  }
  
  // reset group_data
  if(memory_switch==GLOBAL){
    for (a = 0; a < ngroup_glo; a++) {
      for (i = 0; i < nvalues + include_orthogonal+variable_nvalues; i++) {
	group_data[a][i] = 0;
	group_data_loc[a][i] = 0;
      }
    }
  }

  // accumulate results of computes,fixes,variables to origin
  // compute/fix/variable may invoke computes so wrap with clear/add

  modify->clearstep_compute();

  //t2 = MPI_Wtime();
  //time_init_compute += t2 -t1;
  // lastindex = index in values ring of latest time sample

  lastindex++;
  if (lastindex == nsave) lastindex = 0;
  
  //t1 = MPI_Wtime();
  
  for (i = 0; i < nvalues; i++) {
    v = value2index[i];

    // invoke compute if not previously invoked

    if (which[i] == COMPUTE) {
      Compute *compute = modify->compute[v];

      if(!(compute->invoked_flag & INVOKED_PERATOM)) {
	compute->compute_peratom();
	compute->invoked_flag |= INVOKED_PERATOM;
      }
      if (argindex[i] == 0) {
	peratom_data= compute->vector_atom;
      } else {
	peratom_data= compute->array_atom[argindex[i]-1];
      }

    // access fix fields, guaranteed to be ready
    } else if (which[i] == FIX) {
      if (argindex[i] == 0) 
	peratom_data= modify->fix[v]->vector_atom;
      else
	peratom_data= modify->fix[v]->array_atom[i-1];

    // evaluate equal-style variable
    } else {
      memory->create(peratom_data, nlocal, "ave/correlation/peratom:peratom_data");
      input->variable->compute_atom(v, igroup, peratom_data, 1, 0);
    }
    
    for (a= 0; a < ngroup_loc; a++) {
      double data = peratom_data[indices_group[a]];
      if(memory_switch==PERATOM){
	int offset1= i*nsave + lastindex;
	array[indices_group[a]][offset1]= data;
	if ( (dynamics == ORTHOGONAL || dynamics == ORTHOGONALSECOND) && nsample < nsave) {
	  int offset2= (i+nvalues+6)*nsave + lastindex;
	  array[indices_group[a]][offset2]= data;
	}
      } else {
	tagint *ids_ptr;
	ids_ptr = std::find(group_ids,group_ids+ngroup_glo,tag[indices_group[a]]);
	int ind = (ids_ptr - group_ids);
	//printf("base %d, id %d ind %d\n",group_ids,ids_ptr,n);
	group_data_loc[ind][i] = data;
	if ( (dynamics == ORTHOGONAL || dynamics == ORTHOGONALSECOND) && nsample < nsave) {
	  group_data_loc[ind][i+nvalues+6] = data;
	}
      }
    }
    
    //if this was done by an atom-style variable, we need to free the mem we allocated
    if (which[i] == VARIABLE) {
      memory->destroy(peratom_data);
    }
  }
  
  //t2 = MPI_Wtime();
  //calc_write_nvalues += t2 - t1;
  
  //t1 = MPI_Wtime();
  
  //update variable dependency
  if (variable_flag == VAR_DEPENDENED){
    memory->create(peratom_data, nlocal, "ave/correlation/peratom:peratom_data");
    input->variable->compute_atom(variable_value2index, igroup, peratom_data, 1, 0);
    for (a= 0; a < ngroup_loc; a++) {
      if(memory_switch==PERATOM){
	variable_store[indices_group[a]][lastindex]= peratom_data[indices_group[a]];
      } else {
	tagint *ids_ptr;
	ids_ptr = std::find(group_ids,group_ids+ngroup_glo,tag[indices_group[a]]);
	int ind = (ids_ptr - group_ids);
	group_data_loc[ind][nvalues+include_orthogonal] = peratom_data[indices_group[a]];
      }
    }
    memory->destroy(peratom_data);
  } else if (variable_flag == DIST_DEPENDENED) {
    double **x = atom->x;
    for (a= 0; a < ngroup_loc; a++) {
      for (r = 0; r < 3 ; r++) {
	if(memory_switch==PERATOM){
	  variable_store[indices_group[a]][r*nsave+lastindex]= x[indices_group[a]][r];
	} else {
	  tagint *ids_ptr;
	  ids_ptr = std::find(group_ids,group_ids+ngroup_glo,tag[indices_group[a]]);
	  int ind = (ids_ptr - group_ids);
	  group_data_loc[ind][nvalues+include_orthogonal+r] = x[indices_group[a]][r];
	}
      } 
    }
  }
  
  //t2 = MPI_Wtime();
  //write_var += t2 - t1;
  
  //t1 = MPI_Wtime();
  
  if (dynamics == ORTHOGONAL || dynamics == ORTHOGONALSECOND) {
    double **v = atom->v;
    double **f = atom->f;
    // include velocities and forces to the array 
    for (a= 0; a < ngroup_loc; a++) {
      for (r = 0; r < 3 ; r++) {
	if(memory_switch==PERATOM){
	  int offset = (r+nvalues)*nsave + lastindex;
	  array[indices_group[a]][offset] = v[indices_group[a]][r];
	} else {
	  tagint *ids_ptr;
	  ids_ptr = std::find(group_ids,group_ids+ngroup_glo,tag[indices_group[a]]);
	  int ind = (ids_ptr - group_ids);
	  group_data_loc[ind][r+nvalues] = v[indices_group[a]][r];
	}
      }
    }
    for (a= 0; a < ngroup_loc; a++) {
      for (r = 0; r < 3 ; r++) {
	if(memory_switch==PERATOM){
	  int offset= (r+nvalues+3)*nsave + lastindex;
	  array[indices_group[a]][offset] = f[indices_group[a]][r];
	} else {
	  tagint *ids_ptr;
	  ids_ptr = std::find(group_ids,group_ids+ngroup_glo,tag[indices_group[a]]);
	  int ind = (ids_ptr - group_ids);
	  group_data_loc[ind][r+nvalues+3] = f[indices_group[a]][r];
	}
      }
    }
  }
  
  //t2 = MPI_Wtime();
  //write_orthogonal += t2 - t1;
  
  //t1 = MPI_Wtime();
  
  // include group_data into global array
  if(memory_switch==GLOBAL){
    // exclude the last nvalues while memory calculation is performed
    double exclude_memory = 0;
    if ((nsample >= nsave) && (dynamics == ORTHOGONAL || dynamics == ORTHOGONALSECOND)) exclude_memory = -nvalues;
    
    //transport data
    MPI_Win win;
    int block_length = nvalues+include_orthogonal+variable_nvalues;
    MPI_Win_create(&group_data[0][0],ngroup_glo*block_length*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);
    for (a= 0; a < ngroup_loc; a++) {
      tagint *ids_ptr;
      ids_ptr = std::find(group_ids,group_ids+ngroup_glo,tag[indices_group[a]]);
      int ind = (ids_ptr - group_ids);
      MPI_Put(&group_data_loc[ind][0],block_length,MPI_DOUBLE,0,ind*block_length,block_length,MPI_DOUBLE,win);
    }
    MPI_Win_fence(0, win);
    
    //scatter 
    MPI_Get(&group_data[0][0],ngroup_glo*block_length,MPI_DOUBLE,0,0,ngroup_glo*block_length,MPI_DOUBLE,win);
    MPI_Win_fence(0, win);
    
    MPI_Win_free(&win); 
    
    for (a= 0; a < ngroup_glo; a++) {
      for (i=0; i< nvalues+include_orthogonal+exclude_memory;i++) {
	int offset = i*nsave + lastindex;
	array[a][offset] = group_data[a][i];
      }
      if (variable_flag == VAR_DEPENDENED) {
	variable_store[a][lastindex] = group_data[a][nvalues+include_orthogonal];
      } else if (variable_flag == DIST_DEPENDENED) {
	for (r = 0; r < 3 ; r++) {
	  variable_store[a][lastindex+r*nsave] = group_data[a][nvalues+include_orthogonal+r];
	}
      }
    }
  }
  
  //t2 = MPI_Wtime();
  //reduce_write_global += t2 - t1;
  
  
  // fistindex = index in values ring of earliest time sample
  // nsample = number of time samples in values ring

  if( (dynamics==ORTHOGONAL || dynamics == ORTHOGONALSECOND) || nsample < nsave) nsample++;

  int t = nsample - nsave;
  
  nvalid += nevery;
  modify->addstep_compute(nvalid);

  if (me == 0) {
    // calculate all Cij() enabled by latest values
    if (dynamics==NORMAL || t == 0){
      //t1 = MPI_Wtime();
      accumulate(indices_group, ngroup_loc);
      //t2 = MPI_Wtime();
      //time_calc += t2 - t1;
    } else if (dynamics==ORTHOGONAL && t > 0 && t < nrepeat) {
    
      int k,m,r;
    
      //calculate alpha + norm
    
      for (a= 0; a < ngroup_glo; a++) {
	for (r=0; r<3; r++){
	  //go to A_1 -> data from the previos step is needed
	  m = lastindex - 1;
	  for (k = nsave-1; k > 0; k--) {
	    if (m < 0) m = nsave -1;
	    double pnm = array[a][(r+nvalues)*nsave+m];
	    norm[a][r+3*t] += pnm*pnm;
	    m--;
	  }
	  // reset + calculate + norm alpha
	  for (i=0; i<nvalues; i++) {
	    m = lastindex - 1;
	    for (k = nsave-1; k > 0; k--) {
	      if (m < 0) m = nsave-1;
	      double pnm = array[a][(r+nvalues)*nsave+m];
	      double anm = array[a][(i+nvalues+6)*nsave+k];
	      alpha[a][i+r*nvalues+t*nvalues*3] += anm*pnm;
	      m--;
	    }
	  }
	}
      }

      //calculate A(n+1)
      for (a= 0; a < ngroup_glo; a++) {
	for (r=0; r<3; r++){
	  for (i=0; i<nvalues; i++) {
	    double alpha_loc = alpha[a][i+r*nvalues+t*nvalues*3]/(norm[a][r+t*3]*group_mass[a]);
	    m = lastindex-1;
	    for (k = nsave-1; k > 0; k--) {
	      if (m < 0) m = nsave-1;
	      double fnm = array[a][(r+nvalues+3)*nsave+m];
	      array[a][(i+nvalues+6)*nsave+k] += alpha_loc*fnm*update->dt;
	      m--;
	    }
	  }
	}
      }
	
      //accumulate
      accumulate(indices_group, ngroup_loc);
    } else if (dynamics==ORTHOGONALSECOND && nsample > nsave && t < nrepeat) {
      int k,m,mm1,mp1,r;
    
      for (a= 0; a < ngroup_glo; a++) {
	for (r=0; r<3; r++){
	  // initialize the counter
	  m = lastindex-1; 
	  if (m < 0) m = nsave-1;
	  mm1 = m-1;
	  if (mm1 < 0) mm1 = nsave-1;
	  for (k = nsave-1; k > 1; k--) {
	    double pnm = array[a][(r+nvalues)*nsave+m];
	    double fnm = array[a][(r+nvalues+3)*nsave+m];
	    double fnmm1 = array[a][(r+nvalues+3)*nsave+mm1];
	    norm[a][r+3*t] += pnm*pnm;
	    kappa[a][t] += pnm*fnm;
	    zeta[a][t] += pnm*fnmm1;
	    m--; mm1--;
	    if (m < 0) m = nsave-1; if (mm1 < 0) mm1 = nsave-1;
	  }
	  //  calculate alpha
	  for (i=0; i<nvalues; i++) {
	    m = lastindex-1;
	    if (m < 0) m = nsave-1;
	    for (k = nsave-1; k > 1; k--) {
	      double pnm = array[a][(r+nvalues)*nsave+m];
	      double anm = array[a][(i+nvalues+6)*nsave+k];
	      double anmm1 = array[a][(i+nvalues+6)*nsave+k-1];
	      alpha[a][i+r*nvalues+t*3*nvalues] += anm*pnm;
	      epsilon[a][i+r*nvalues+t*3*nvalues] += anmm1*pnm;
	      m--;
	      if (m < 0) m = nsave-1;
	    }
	  }
	}
      }

      //calculate A(n+1)
      for (a= 0; a < ngroup_glo; a++) {
	for (r=0; r<3; r++){
	  double kappa_loc = kappa[a][t]/(norm[a][3*t]+norm[a][3*t+1]+norm[a][3*t+2]);
	  double zeta_loc = zeta[a][t]/(norm[a][3*t]+norm[a][3*t+1]+norm[a][3*t+2]);
	  for (i=0; i<nvalues; i++) {
	    double alpha_loc = alpha[a][i+r*nvalues+t*3*nvalues]/(norm[a][r+t*3]*group_mass[a]);
	    double epsilon_loc = epsilon[a][i+r*nvalues+t*3*nvalues]/(norm[a][r+t*3]*group_mass[a]);
	    m = lastindex-1; 
	    if (m < 0) m = nsave-1;
	    mp1 = lastindex;
	    for (k = nsave-1; k > 0; k--) {
	      double fnm = array[a][(r+nvalues+3)*nsave+m];
	      double fnmp1 = array[a][(r+nvalues+3)*nsave+mp1];
	      array[a][(i+nvalues+6)*nsave+k] += alpha_loc*fnm*update->dt/2.0
		+update->dt/2.0*fnmp1/(1-update->dt/2.0*kappa_loc)
		*(epsilon_loc+zeta_loc*alpha_loc*update->dt/2.0);
	      m--; mp1--;
	      if (m < 0) m = nsave-1; if (mp1 < 0) mp1 = nsave-1;
	    }
	  }
	}
      }

      //accumulate

      accumulate(indices_group, ngroup_loc);
    }
  
  
    //calculate mean
    if (mean_flag){ 
      calc_mean(indices_group, ngroup_loc);
    }
  
  }
  
  //double tt2 = MPI_Wtime();
  //time_total += tt2-tt1;
    
  if (ntimestep % nfreq || first) {
    first = 0;
    memory->destroy(indices_group);
    return;
  }
  
  if (me == 0) {
    // save results in save_count and save_corr
    for (i = 0; i < corr_length; i++) {
      save_count[i] = count[i];
      if (count[i]) {
	for (j = 0; j < npair; j++)
	  save_corr[i][j] = prefactor*corr[i][j]/count[i];
      } else {
	for (j = 0; j < npair; j++)
	  save_corr[i][j] = 0.0;
      }
    }
    
    // output result to file
    if (fp) {
      if (overwrite) fseek(fp,filepos,SEEK_SET);
      fprintf(fp,BIGINT_FORMAT " %d\n",ntimestep,nrepeat);
      for (i = 0; i < corr_length/factor; i++) {
	if (variable_flag == VAR_DEPENDENED || variable_flag == DIST_DEPENDENED) {
	  int loc_bin = i%nrepeat;
	  int loc_ind = (i - loc_bin)/nrepeat;
	  fprintf(fp,"%d %d %f %f",loc_ind+1,loc_ind*nevery,range/bins*loc_bin,count[i]);
	} else {
	  fprintf(fp,"%d %d %f",i+1,i*nevery,count[i]);
	}
	if (count[i]) {
	  for (j = 0; j < npair; j++)
	    fprintf(fp," %g",prefactor*corr[i][j]/count[i]);
	} else {
	  for (j = 0; j < npair; j++)
	    fprintf(fp," 0.0");
	}
	if (type == AUTOCROSS || variable_flag == DIST_DEPENDENED) {
	  int offset = i + corr_length/2;
	  if (type == AUTOCROSS)
	    fprintf(fp," %f",count[offset]);
	  if (count[offset]) {
	    for (j = 0; j < npair; j++)
	      fprintf(fp," %g",prefactor*corr[offset][j]/count[offset]);
	  } else {
	    for (j = 0; j < npair; j++)
	      fprintf(fp," 0.0");
	  }
	}
	fprintf(fp,"\n");
      }
      fflush(fp);
      if (overwrite) {
	long fileend = ftell(fp);
	ftruncate(fileno(fp),fileend);
      }
    }
    
        
    // output mean result to file
    if (mean_flag) {
      if (overwrite) fseek(mean_file,mean_filepos,SEEK_SET);
      for (o = 0; o < bins; o++) {
	if (variable_flag == VAR_DEPENDENED || variable_flag == DIST_DEPENDENED)
	  fprintf(mean_file,"%f %f",mean_count[o],range/bins*o);
	else 
	  fprintf(mean_file,"%f",mean_count[o]);
	if (count[o]){
	  for (j = 0; j < nvalues; j++)
	    fprintf(mean_file," %g",mean[j+o*nvalues]/mean_count[o]);
	} else {
	  for (j = 0; j < nvalues; j++)
	    fprintf(mean_file," 0.0");
	}
	fprintf(mean_file,"\n");
      }
      fflush(mean_file);
      if (overwrite) {
	long fileend = ftell(mean_file);
	ftruncate(fileno(mean_file),fileend);
      }
    }
    
    // zero accumulation if requested
    // recalculate Cij(0)
    if (ave == ONE) {
      for (i = 0; i < corr_length; i++) {
	count[i] = 0.0;
	for (j = 0; j < npair; j++)
	  corr[i][j] = 0.0;
      }
      if (mean_flag) {
	for(o = 0; o < bins; o++){
	  for (i = 0; i < nvalues; i++) {
	    mean[i+o*nvalues]=0.0;
	  }
	  mean_count[o]=0.0;
	}
      }
    }
  }
  
  nsample = 1;
  lastindex  = 0;
  if(ntimestep != update->nsteps && dynamics == NORMAL) accumulate(indices_group, ngroup_loc); 

  memory->destroy(indices_group);
  
  // print timing
  //printf("processor %d: time(init_compute) = %f\n",me,time_init_compute);
  //printf("processor %d: time(calc+write_nvalues) = %f\n",me,calc_write_nvalues);
  //printf("processor %d: time(write_var) = %f\n",me,write_var);
  //printf("processor %d: time(write_orthogonal) = %f\n",me,write_orthogonal);
  //printf("processor %d: time(reduce_write_global) = %f\n",me,reduce_write_global);
  //printf("processor %d: time(calc) = %f\n",me,time_calc);
  //printf("processor %d: time(total) = %f\n",me,time_total);
}

/* ----------------------------------------------------------------------
   accumulate correlation data using more recently added values
------------------------------------------------------------------------- */
void FixAveCorrelatePeratom::accumulate(int *indices_group, int ngroup_loc)
{
  int a,b,i,j,k,o,m,n,ipair;
  int t = nsample - nsave;
  int nlocal= atom->nlocal;
  tagint *tag = atom->tag;
  
  // create local memory for accumulation and reduction
  double *local_accum_corr;
  double *global_accum_corr;
  double *local_accum_count;
  double *global_accum_count;
  int accum_nsample = nsample;
  if (dynamics == ORTHOGONAL || dynamics == ORTHOGONALSECOND) accum_nsample = 1;
  int accum_length = accum_nsample*bins*factor;
  
  memory->create (local_accum_corr, accum_length,"ave/correlate/peratom:local_accum_corr");
  memory->create (global_accum_corr, accum_length,"ave/correlate/peratom:global_accum_corr");
  memory->create (local_accum_count, accum_length,"ave/correlate/peratom:local_accum_count");
  memory->create (global_accum_count, accum_length,"ave/correlate/peratom:global_accum_count");
  for (k = 0; k < accum_length; k++){
    local_accum_corr[k] = global_accum_corr[k] = 0.0;
    local_accum_count[k] = global_accum_count[k] = 0.0;
  }
  
  // accumulate
  n = lastindex;
  ipair = 0;
  int incr_nvalues = 1;
  if (variable_flag == DIST_DEPENDENED){ 
    incr_nvalues = 3;
  }
  
  for (i = 0; i < nvalues; i+=incr_nvalues) {
    //determine whether just autocorrelation or also mixed correlation (different observables)
    double nvalues_upper = i+1;
    if (type == AUTOUPPER) nvalues_upper = nvalues;
    for (j = i; j < nvalues_upper; j+=incr_nvalues) {
      
      for (a= 0; a < ngroup_glo; a++) {
	//determine whether just autocorrelation or also cross correlation (different atoms)
	double ngroup_lower = a;
	double ngroup_upper = a+1;
	if (type == CROSS || type == AUTOCROSS){
	  ngroup_lower = a;
	  ngroup_upper = ngroup_glo;
	}
	for (b = ngroup_lower; b < ngroup_upper; b++) {
	  if (type == CROSS && a==b) continue;
	  m = lastindex;
	  int inda,indb;
	  if(memory_switch==PERATOM){
	    inda=indices_group[a];
	    indb=indices_group[b];
	  } else {
	    inda = a;
	    indb = b;
	  }
	  if(dynamics==NORMAL){
	    for (k = 0; k < nsample; k++) {
	      if (variable_flag == VAR_DEPENDENED){
		double dV = variable_store[inda][m] - variable_store[indb][n];
		dV=fabs(dV);
		if(dV<range){
		  int ind = dV/range*bins;
		  int offset= k*bins+ind;
		  //count once
		  if (type == AUTOCROSS && b!=a) {
		    if(i==0&&j==0) local_accum_count[offset+accum_length/2]+=1.0;
		    local_accum_corr[offset+accum_length/2] += array[inda][i * nsave + m]*array[indb][j * nsave + n];
		  } else {
		    if(i==0&&j==0) local_accum_count[offset]+=1.0;
		    local_accum_corr[offset]+=array[inda][i * nsave + m]*array[indb][j * nsave + n];
		  }
		}
	      } else if (variable_flag == DIST_DEPENDENED) {
		double *dr = new double[3];
		dr[0] = variable_store[inda][m] - variable_store[indb][n];
		dr[1] = variable_store[inda][m+nsave] - variable_store[indb][n+nsave];
		dr[2] = variable_store[inda][m+2*nsave] - variable_store[indb][n+2*nsave];
		double dV = sqrt( dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]);
		//printf("dV=%f\n",dV);
		if(dV<range){
		  double *res_data = new double[8];
		  double *inp_data = new double[6];
		  int p;
		  for (p=0; p<3; p++) inp_data[p] = array[inda][ (i+p) * nsave + m];
		  for (p=0; p<3; p++) inp_data[p+3] = array[indb][ (j+p) * nsave + n];
		  decompose(res_data,dr,inp_data);
		  //for (int z=0; z<8; z++) printf("res_data[%d]=%f\n",z,res_data[z]);
		  // calculate correlation
		  int ind = dV/range*bins;
		  int offset= k*bins+ind;
		  if(i==0&&j==0) local_accum_count[offset]+=1.0;
		  local_accum_corr[offset] += res_data[0]*res_data[1];
		  if(i==0&&j==0) local_accum_count[offset+accum_length/2]+=1.0;
		  local_accum_corr[offset+accum_length/2] += res_data[2]*res_data[5]+res_data[3]*res_data[6]+res_data[4]*res_data[7];
		  delete[] res_data;
		  delete[] inp_data;
		}
		delete[] dr;
	      } else { //no variable dependency
		if (type == AUTOCROSS && b!=a) {
		  if(i==0&&j==0) local_accum_count[k+accum_length/2]+=1.0;
		  local_accum_corr[k+accum_length/2] += array[inda][i * nsave + m]*array[indb][j * nsave + n];
		} else {
		  if(i==0&&j==0) local_accum_count[k]+=1.0;
		  local_accum_corr[k]+= array[inda][i * nsave + m]*array[indb][j * nsave + n];
		}
	      }
	      m--;
	      if (m < 0) m = nsave-1;
	    }
	  } else { //dynamics ORTHOGONAL/ORTHOGONALSECOND
	    for (k = nsave-1; k > 0; k--) {
	      if (variable_flag == VAR_DEPENDENED){
		double dV = variable_store[inda][m] - variable_store[indb][n];
		dV=fabs(dV);
		if(dV<range){
		  int ind = dV/range*bins;
		  int offset= t*bins+ind;
		  //count once
		  if (type == AUTOCROSS && b!=a) {
		    if(i==0&&j==0) local_accum_count[ind+accum_length/2]+=1.0;
		    local_accum_corr[ind+accum_length/2] += array[inda][i*nsave+m]*array[indb][(j+nvalues+6)*nsave+k];
		  } else {
		    if(i==0&&j==0) local_accum_count[ind]+=1.0;
		    local_accum_corr[ind]+=array[inda][i * nsave + m]*array[indb][j * nsave + n];
		  }
		}
	      } else if (variable_flag == DIST_DEPENDENED) {
		double *dr = new double[3];
		dr[0] = variable_store[inda][m] - variable_store[indb][n];
		dr[1] = variable_store[inda][m+nsave] - variable_store[indb][n+nsave];
		dr[2] = variable_store[inda][m+2*nsave] - variable_store[indb][n+2*nsave];
		double dV = sqrt( dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]);
		//printf("dV=%f\n",dV);
		if(dV<range){
		  double *res_data = new double[8];
		  double *inp_data = new double[6];
		  int p;
		  for (p=0; p<3; p++) inp_data[p] = array[inda][(i+p)*nsave+m];
		  for (p=0; p<3; p++) inp_data[p+3] = array[indb][(j+p+nvalues+6)*nsave+k];
		  decompose(res_data,dr,inp_data);
		  //for (int z=0; z<8; z++) printf("res_data[%d]=%f\n",z,res_data[z]);
		  // calculate correlation
		  int ind = dV/range*bins;
		  int offset= t*bins+ind;
		  if(i==0&&j==0) local_accum_count[ind]+=1.0;
		  local_accum_corr[ind] += res_data[0]*res_data[1];
		  if(i==0&&j==0) local_accum_count[ind+accum_length/2]+=1.0;
		  local_accum_corr[ind+accum_length/2] += res_data[2]*res_data[5]+res_data[3]*res_data[6]+res_data[4]*res_data[7];
		  delete[] res_data;
		  delete[] inp_data;
		}
		delete[] dr;
	      } else { //no variable dependency
		if (type == AUTOCROSS && b!=a) {
		  if(i==0&&j==0) local_accum_count[1]+=1.0;
		  local_accum_corr[1] +=  array[inda][i*nsave+m]*array[indb][(j+nvalues+6)*nsave+k];
		} else {
		  if(i==0&&j==0) local_accum_count[0]+=1.0;
		  local_accum_corr[0]+= array[inda][i*nsave+m]*array[indb][(j+nvalues+6)*nsave+k];
		}
	      }
	      m--;
	      if (m < 0) m = nsave-1;
	    }
	  }
	}
      }

      // reduce the results from each proc to calculate the global correlation
      if (dynamics == NORMAL) {
	for (k = 0; k < accum_nsample; k++) {
	  for (o = 0; o < bins; o++) {
	    int offset = k*bins+o;
	    //printf("offset=%d, ipair=%d, nrepeat=%d, npair=%d\n",offset,ipair,nrepeat, npair);
	    corr[offset][ipair]+= local_accum_corr[offset];
	    count[offset]+= local_accum_count[offset];
	    local_accum_corr[offset] = global_accum_corr[offset] = 0.0;
	    local_accum_count[offset] = global_accum_count[offset] = 0.0;
	    if (type == AUTOCROSS || variable_flag == DIST_DEPENDENED) {
	      //printf("%d %d %f\n",offset+corr_length/2,ipair,corr[offset+corr_length/2][ipair]);
	      corr[offset+corr_length/2][ipair]+= local_accum_corr[offset+accum_length/2];
	      count[offset+corr_length/2]+= local_accum_count[offset+accum_length/2];
	      local_accum_corr[offset+accum_length/2] = global_accum_corr[offset+accum_length/2] = 0.0;
	      local_accum_count[offset+accum_length/2] = global_accum_count[offset+accum_length/2] = 0.0;
	    }
	  }
	}
      } else {
	for (o = 0; o < bins; o++) {
	  int offset = t*bins+o;
	  corr[offset][ipair] += global_accum_corr[o];
	  count[offset] += global_accum_count[o];
	  local_accum_corr[o] = global_accum_corr[o] = 0.0;
	  local_accum_count[o] = global_accum_count[o] = 0.0;
	  if (type == AUTOCROSS || variable_flag == DIST_DEPENDENED) {
	    corr[offset+corr_length/2][ipair] += global_accum_corr[o+accum_length/2];
	    count[offset+corr_length/2] += global_accum_count[o+accum_length/2];
	    local_accum_corr[o+accum_length/2] = global_accum_corr[o+accum_length/2] = 0.0;
	    local_accum_count[o+accum_length/2] = global_accum_count[o+accum_length/2] = 0.0;
	  }
	}
      }
      ipair++;
    }
  }
  
  memory->destroy(local_accum_corr);
  memory->destroy(global_accum_corr);
  memory->destroy(local_accum_count);
  memory->destroy(global_accum_count);

}

/* ----------------------------------------------------------------------
   decompose the variables into a parallel and an orthogonal component
------------------------------------------------------------------------- */
void FixAveCorrelatePeratom::decompose(double *res_data, double *dr, double *inp_data) {
  double proj1 = inp_data[0]*dr[0] + inp_data[1]*dr[1] + inp_data[2]*dr[2];
  double proj2 = inp_data[3]*dr[0] + inp_data[4]*dr[1] + inp_data[5]*dr[2];
  double dr2 = dr[0]*dr[0] + dr[1]*dr[1] +dr[2]*dr[2];
  proj1 /= dr2;
  proj2 /= dr2;
  double *F1_p = new double[3];
  double *F2_p = new double[3];
  int p;
  for (p=0; p<3; p++){
    F1_p[p] = proj1*dr[p];
    F2_p[p] = proj2*dr[p];
  }
  // parallel components
  res_data[0] = sgn(proj1)*sqrt( F1_p[0]*F1_p[0] + F1_p[1]*F1_p[1] + F1_p[2]*F1_p[2] );
  res_data[1] = sgn(-proj2)*sqrt( F2_p[0]*F2_p[0] + F2_p[1]*F2_p[1] + F2_p[2]*F2_p[2] );
  // orthogonal components
  for (p=0; p<3; p++){
    res_data[2+p] = inp_data[p] - F1_p[p];
    res_data[5+p] = inp_data[3+p] - F2_p[p];
  }
  
  delete[] F1_p;
  delete[] F2_p;

  //if(res_data[0]==0.0){
  //  printf("dr[0]=%f, dr[0]=%f, dr[0]=%f, inp_data[0]=%f, inp_data[1]=%f, inp_data[2]=%f\n",dr[0],dr[1],dr[2],inp_data[0],inp_data[1],inp_data[2]);
  //}
}

/* ----------------------------------------------------------------------
   calculate mean values using more recently added values
------------------------------------------------------------------------- */
void FixAveCorrelatePeratom::calc_mean(int *indices_group, int ngroup_loc){ 
  int a,b,i,o,k;
  tagint *tag = atom->tag;
  
  // create local memory for accumulation and reduction
  
  double *local_accum_mean;
  double *global_accum_mean;
  double *local_accum_count;
  double *global_accum_count;
  int accum_length = bins;
  
  memory->create(local_accum_mean, accum_length,"ave/correlate/peratom:local_accum_mean");
  memory->create(global_accum_mean, accum_length,"ave/correlate/peratom:global_accum_mean");
  memory->create(local_accum_count, accum_length,"ave/correlate/peratom:local_accum_count");
  memory->create(global_accum_count, accum_length,"ave/correlate/peratom:global_accum_count");
  for (o=0; o<accum_length; o++){
    local_accum_mean[o] = global_accum_mean[o] = 0.0;
    local_accum_count[o] = global_accum_count[o] = 0.0;
  }
  int incr_nvalues = 1;
  if (variable_flag == DIST_DEPENDENED){ 
    incr_nvalues = 3;
  }

  for (i = 0; i < nvalues; i+=incr_nvalues) {
    for (a= 0; a < ngroup_glo; a++) {
      //determine whether just autocorrelation or also cross correlation (different atoms)
      double ngroup_upper = a+1;
      if (type == CROSS || type == AUTOCROSS){
	ngroup_upper = ngroup_glo;
      }
      for (b = a; b < ngroup_upper; b++) {
	if (type == CROSS && a==b) continue;
	int inda,indb;
	if(memory_switch==PERATOM){
	  inda=indices_group[a];
	  indb=indices_group[b];
	} else {
	  inda = a;
	  indb = b;
	}
	if (variable_flag == VAR_DEPENDENED){
	  double dV = variable_store[inda][lastindex] - variable_store[indb][lastindex];
	  dV=fabs(dV);
	  if(dV<range){
	    int ind = dV/range*bins;
	    if (i==0) local_accum_count[ind] += 2.0;
	    local_accum_mean[ind] += array[inda][i * nsave + lastindex];
	    local_accum_mean[ind] += array[indb][i * nsave + lastindex];
	  }
	} else if (variable_flag == DIST_DEPENDENED) {
	  
	  double *dr = new double[3];
	  dr[0] = variable_store[inda][lastindex] - variable_store[indb][lastindex];
	  dr[1] = variable_store[inda][lastindex+nsave] - variable_store[indb][lastindex+nsave];
	  dr[2] = variable_store[inda][lastindex+2*nsave] - variable_store[indb][lastindex+2*nsave];
	  double dV = sqrt( dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]);

	  //printf("dV=%f\n",dV);
	  if(dV<range){
	    double *res_data = new double[8];
	    double *inp_data = new double[6];
	    int p;
	    for (p=0; p<3; p++) inp_data[p] = array[inda][ (i+p) * nsave + lastindex];
	    for (p=0; p<3; p++) inp_data[p+3] = array[indb][ (i+p) * nsave + lastindex];
	    decompose(res_data,dr,inp_data);
	    //for (int z=0; z<8; z++) printf("res_data[%d]=%f\n",z,res_data[z]);
	    // calculate correlation
	    int ind = dV/range*bins;
	    if(i==0) local_accum_count[ind]+=2.0;
	    local_accum_mean[ind] += res_data[0];
	    local_accum_mean[ind] += res_data[1];
	    delete[] res_data;
	    delete[] inp_data;
	  } 
	  delete[] dr;
	} else {
	  if(i==0) local_accum_count[0] += 1.0;
	  local_accum_mean[0] += array[inda][i * nsave + lastindex];
	}
      }
    }
    
    // reduce the results from each proc to calculate the global correlation
    for(o=0; o<bins; o++){
      mean[o*nvalues+i] += local_accum_mean[o];
      mean_count[o] += local_accum_count[o];
      local_accum_mean[o] = global_accum_mean[o] = 0.0;
      local_accum_count[o] = global_accum_count[o] = 0.0;
    }
  }
 
  memory->destroy (local_accum_mean);
  memory->destroy (global_accum_mean);
  memory->destroy (local_accum_count);
  memory->destroy (global_accum_count);
}

/* ----------------------------------------------------------------------
   return I,J array value
------------------------------------------------------------------------- */

double FixAveCorrelatePeratom::compute_array(int i, int j)
{
  if (j == 0) return 1.0*i*nevery;
  else if (j == 1) return 1.0*save_count[i];
  else if (save_count[i]) return save_corr[i][j-2];
  return 0.0;
}

/* ----------------------------------------------------------------------
   nvalid = next step on which end_of_step does something
   this step if multiple of nevery, else next multiple
   startstep is lower bound
------------------------------------------------------------------------- */

bigint FixAveCorrelatePeratom::nextvalid()
{
  bigint nvalid = update->ntimestep;
  if (startstep > nvalid) nvalid = startstep;
  if (nvalid % nevery) nvalid = (nvalid/nevery)*nevery + nevery;
  return nvalid;
}

/* ---------------------------------------------------------------------- */

void FixAveCorrelatePeratom::reset_timestep(bigint ntimestep)
{
  if (ntimestep > nvalid) error->all(FLERR,"Fix ave/correlate/peratom missed timestep");
}

/* --------------------------------------------------------------------- */

int FixAveCorrelatePeratom::pack_exchange(int i, double* buf) {
  int offset= 0;
  for (int m= 0; m < nvalues + include_orthogonal; m++) {
    for (int k= 0; k < nsample; k++) {
      buf[offset] = array[i][offset];
      offset++;
    }
    for (int k= nsample; k < nsave; k++) {
      buf[offset++]= 0.0;
    }
  }
  // add variable dependency
  if (variable_flag == VAR_DEPENDENED){
    for (int k= 0; k < nsample; k++) {
      buf[offset++] = variable_store[i][k];
    }
    for (int k= nsample; k < nsave; k++) {
      buf[offset++]= 0.0;
    }
  } else if (variable_flag == DIST_DEPENDENED){
    for(int r=0; r<3; r++){
      for (int k= 0; k < nsample; k++) {
	buf[offset++] = variable_store[i][r*nsave+k];
      }
      for (int k= nsample; k < nsave; k++) {
	buf[offset++]= 0.0;
      }
    }
  }
  return offset;
}

/* --------------------------------------------------------------------- */

int FixAveCorrelatePeratom::unpack_exchange(int nlocal, double* buf) {
  int offset= 0;
  for (int m= 0; m < nvalues + include_orthogonal; m++) {
    for (int k= 0; k < nsample; k++) {
      array[nlocal][offset]= buf[offset];
      offset++;
    }
    for (int k= nsample; k < nsave; k++) {
      array[nlocal][offset++]= 0.0;
    }
  }
  // add variable dependency
  if (variable_flag == VAR_DEPENDENED){
    for (int k= 0; k < nsample; k++) {
      variable_store[nlocal][k]=buf[offset++]; 
    }
    for (int k= nsample; k < nsave; k++) {
      variable_store[nlocal][k]=0.0;
    }
  } else if (variable_flag == DIST_DEPENDENED){
    for (int r=0; r<3; r++){
      for (int k= 0; k < nsample; k++) {
	variable_store[nlocal][k+r*nsave]=buf[offset++]; 
      }
      for (int k= nsample; k < nsave; k++) {
	variable_store[nlocal][k+r*nsave]=0.0;
      }
    }
  }
  return offset;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixAveCorrelatePeratom::memory_usage() {
  double bytes;
  int atoms;
  
  if(memory_switch==PERATOM) atoms = atom->nmax;
  else atoms = ngroup_glo;
  bytes = atoms * (nvalues + include_orthogonal+variable_nvalues) * nsave * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate atom-based array
------------------------------------------------------------------------- */

void FixAveCorrelatePeratom::grow_arrays(int nmax) {
  memory->grow(array,nmax,(nvalues + include_orthogonal)*nsave,"fix_ave/correlate/peratom:array");
  if (variable_flag == VAR_DEPENDENED || variable_flag == DIST_DEPENDENED) memory->grow(variable_store,ngroup_glo,nsave*variable_nvalues,"fix_ave/correlate/peratom:variable_store");
  array_atom = array;
  if (array) vector_atom = array[0];
  else vector_atom = NULL;
}

/* ----------------------------------------------------------------------
   copy values within local atom-based array
------------------------------------------------------------------------- */

void FixAveCorrelatePeratom::copy_arrays(int i, int j, int delflag) {
  int offset= 0;
  for (int m= 0; m < nvalues + include_orthogonal; m++) {
    for (int k= 0; k < nsample; k++) {
      array[j][offset] = array[i][offset];
      offset++;
    }
    for (int k= nsample; k < nsave; k++) {
      array[j][offset++]= 0.0;
    }
  }
  if (variable_flag == VAR_DEPENDENED){
    for (int k= 0; k < nsample; k++) {
      variable_store[j][k]=variable_store[i][k];
    }
    for (int k= nsample; k < nsave; k++) {
      variable_store[j][k]=0.0;
    }
  } else if (variable_flag == DIST_DEPENDENED){
    for (int r=0; r<3;r++) {
      for (int k= 0; k < nsample; k++) {
	variable_store[j][k+r*nsave]=variable_store[i][k+r*nsave];
      }
      for (int k= nsample; k < nsave; k++) {
	variable_store[j][k+r*nsave]=0.0;
      } 
    }
  }
}

/* ----------------------------------------------------------------------
   write data into restart file:
   - correlation
   - if orthogonal dynamics: alpha and norm (+ epsilon, zeta and kappa)
------------------------------------------------------------------------- */
void FixAveCorrelatePeratom::write_restart(FILE *fp){
  // calculate size of array
  int ncount = corr_length;
  int ncorr = corr_length*npair;
  // count and correlation
  int n = ncount+ ncorr;
  // orth. dynamics
  if (dynamics == ORTHOGONAL || dynamics == ORTHOGONALSECOND){
    n += 3*nrepeat * (1 + nvalues) * ngroup_glo;
    if (dynamics == ORTHOGONALSECOND){
      n += nrepeat * (3*nvalues + 2) * ngroup_glo;
    }
  }
  
  if (mean_flag) n += bins*nvalues + bins;
  
  //write data
  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    // count and correlation
    fwrite(count,sizeof(double),ncount,fp);
    fwrite(&corr[0][0],sizeof(double),ncorr,fp);
    // orth. dynamics
    if (dynamics == ORTHOGONAL || dynamics == ORTHOGONALSECOND){
      fwrite(&alpha[0][0],sizeof(double),3*nrepeat*nvalues*ngroup_glo,fp);
      fwrite(&norm[0][0],sizeof(double),3*nrepeat*ngroup_glo,fp);
      if (dynamics == ORTHOGONALSECOND){
	fwrite(&epsilon[0][0],sizeof(double),3*nrepeat*nvalues*ngroup_glo,fp);
	fwrite(&kappa[0][0],sizeof(double),nrepeat*ngroup_glo,fp);
	fwrite(&zeta[0][0],sizeof(double),nrepeat*ngroup_glo,fp);
      }
    }
    //mean
    if (mean_flag) {
      fwrite(mean_count,sizeof(double),bins,fp);
      fwrite(mean,sizeof(double),bins*nvalues,fp);
    }
  }
}


/* ----------------------------------------------------------------------
   read data from restart file:
   - correlation
   - if orthogonal dynamics: alpha and norm (+ epsilon, zeta and kappa)
------------------------------------------------------------------------- */
void FixAveCorrelatePeratom::restart(char *buf){
  double *dbuf = (double *) buf;
  int i,j,o, a, t, r;
  int dcount = 0;
  // count + correlation
 
  for (i = 0; i < corr_length; i++) count[i] = dbuf[dcount++];

  for (i = 0; i < corr_length; i++)
    for (j = 0; j < npair; j++) corr[i][j] = dbuf[dcount++];

  // orth. dynamics
  if (dynamics == ORTHOGONAL || dynamics == ORTHOGONALSECOND){
    for (a = 0; a < ngroup_glo; a++)
      for (t = 0; t < nrepeat; t++) 
	for (r=0; r<3; r++)
	  for (i = 0; i < nvalues; i++)
	    alpha[a][i+r*nvalues+t*nvalues*3] = dbuf[dcount++];
	    
    for (a = 0; a < ngroup_glo; a++)
      for (t = 0; t < nrepeat; t++) 
	for (r=0; r<3; r++)
	    norm[a][r+t*3] = dbuf[dcount++];
	    
    if (dynamics == ORTHOGONALSECOND){
      for (a = 0; a < ngroup_glo; a++)
	for (t = 0; t < nrepeat; t++) 
	  for (r=0; r<3; r++)
	    for (i = 0; i < nvalues; i++) 
	      epsilon[a][i+r*nvalues+t*nvalues*3] = dbuf[dcount++];
	      
      for (a = 0; a < ngroup_glo; a++)
	for (t = 0; t < nrepeat; t++) 
	  kappa[a][t] = dbuf[dcount++];
	
      for (a = 0; a < ngroup_glo; a++)
	for (t = 0; t < nrepeat; t++) 
	  zeta[a][t] = dbuf[dcount++];
    }
  }
  
  //mean
  if (mean_flag) {
    for (o=0; o<bins; o++) mean_count[o] = dbuf[dcount++];
      
    for (o=0; o<bins; o++)
      for (i=0; i<nvalues; i++) mean[i+o*nvalues] = dbuf[dcount++];
  }
}

/* sign-function */
template <typename T> int FixAveCorrelatePeratom::sgn(T val) {
    return (T(0) < val) - (val < T(0));
}