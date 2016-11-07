
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
#include "domain.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "atom.h"
#include "comm.h"
#include <algorithm>    // std::find
#include <math.h>    // fabs

using namespace LAMMPS_NS;
using namespace FixConst;

enum{COMPUTE,FIX,VARIABLE};
enum{ONE,RUNNING};
enum{NORMAL,ORTHOGONAL,ORTHOGONALSECOND};
enum{AUTO,CROSS,AUTOCROSS,AUTOUPPER, FULL};
enum{PERATOM,PERGROUP, GROUP};
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
  MPI_Comm_size(world,&nprocs);

  nevery = force->inumeric(FLERR,arg[3]);
  nrepeat = force->inumeric(FLERR,arg[4]);
  nfreq = force->inumeric(FLERR,arg[5]);


  global_freq = nfreq;
  // parse values until one isn't recognized

  which = new int[narg];
  argindex = new int[narg];
  ids = new char*[narg];
  value2index = new int[narg];
  for (int i=0; i< narg; i++) {
    which[i]=-1;
    argindex[i]=-1;
    ids[i]=NULL;
    value2index[i]=-1;
  }
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
  v_counter = 0;
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
      else if (strcmp(arg[iarg+1],"full") == 0) type = FULL;
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

      } else if (strcmp(arg[iarg+1],"orthogonal/second") == 0) {
    dynamics = ORTHOGONALSECOND;
      } else error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"switch") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      if (strcmp(arg[iarg+1],"peratom") == 0) memory_switch = PERATOM;
      else if (strcmp(arg[iarg+1],"pergroup") == 0) memory_switch = PERGROUP;
      else if (strcmp(arg[iarg+1],"group") == 0) {
	memory_switch = GROUP;
	if (iarg+3 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
	ngroup_glo = force->inumeric(FLERR,arg[iarg+2]);
	if (iarg+4+ngroup_glo > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
	cor_groupbit = new int[ngroup_glo];
	for (int i=0; i<ngroup_glo; i++) {
	  int igroup = group->find(arg[iarg+3+i]);
	  if (igroup == -1) error->all(FLERR,"Could not find fix group ID");
	  cor_groupbit[i] = group->bitmask[igroup];
	}
	nvalues = force->inumeric(FLERR,arg[iarg+3+ngroup_glo]);
	cor_valbit = new int[nvalues];
	if (iarg+4+ngroup_glo+nvalues > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
	for (int i=0; i<nvalues; i++) {
	  char *loc_arg = arg[iarg+4+ngroup_glo+i];
	  if (strcmp(loc_arg,"vx") == 0) cor_valbit[i] = 1;
	  else if (strcmp(loc_arg,"vy") == 0) cor_valbit[i] = 2;
	  else if (strcmp(loc_arg,"vz") == 0) cor_valbit[i] = 3;
	  else if (strcmp(loc_arg,"fx") == 0) cor_valbit[i] = 4;
	  else if (strcmp(loc_arg,"fy") == 0) cor_valbit[i] = 5;
	  else if (strcmp(loc_arg,"fz") == 0) cor_valbit[i] = 6;
	  else if (strncmp(loc_arg,"c_",2) == 0 ||
	    strncmp(loc_arg,"f_",2) == 0 ||
	    strncmp(loc_arg,"v_",2) == 0) {
	    if (loc_arg[0] == 'c') {
	      which[i] = COMPUTE;
	      cor_valbit[i] = 7;
	    }
	    else if (loc_arg[0] == 'f') {
	      cor_valbit[i] = 8;
	      which[i] = FIX;
	    }
	    else if (loc_arg[0] == 'v') {
	      printf("variable1\n");
	      cor_valbit[i] = 9;
	      which[i] = VARIABLE; 
	      v_counter++;
	    }
	    
	    int n = strlen(loc_arg);
	    char *suffix = new char[n];
	    strcpy(suffix,&loc_arg[2]);

	    char *ptr = strchr(suffix,'[');
	    if (ptr) {
	      if (suffix[strlen(suffix)-1] != ']')
		error->all(FLERR,"Illegal fix ave/correlate/peratom command");
	      argindex[i] = atoi(ptr+1);
	      *ptr = '\0';
	    } else argindex[i] = 0;

	    n = strlen(suffix) + 1;
	    ids[i] = new char[n];
	    strcpy(ids[i],suffix);
	    delete [] suffix;
	  }
	  else error->all(FLERR,"Illegal fix ave/correlate/peratom command");
	}
	iarg += 2 + ngroup_glo + nvalues;
      } else error->all(FLERR,"Illegal fix ave/correlate/peratom command");
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
  
  if ( dynamics == ORTHOGONAL || dynamics == ORTHOGONALSECOND ) include_orthogonal = nvalues + 6;

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

  // check variables (for peratom/pergroup)
  int i,j,o;
  if (memory_switch != GROUP) {
    for (i = 0; i < nvalues; i++) {
      if (which[i] == COMPUTE) {
	int icompute = modify->find_compute(ids[i]);
	//no such compute
	if (icompute < 0)
	  error->all(FLERR,"Compute ID for fix ave/correlate/peratom does not exist");
	if (argindex[i] == 0) { 
	  if (modify->compute[icompute]->peratom_flag != 1 ) {
	    error->all(FLERR, "Fix ave/correlate/peratom compute does not calculate a peratom scalar");
	  } 
	} else {
	  if (modify->compute[icompute]->peratom_flag == 1) {
	    if (argindex[i] > modify->compute[icompute]->size_peratom_cols) {
	      error->all(FLERR,"Fix ave/correlate/peratom compute vector is accessed out-of-range");
	    }
	  } else {
	    error->all(FLERR, "Fix ave/correlate/peratom compute does not calculate a peratom/global vector (depending on switch)");
	  }
	}
      } else if (which[i] == FIX) {
	int ifix = modify->find_fix(ids[i]);
	if (ifix < 0)
	  error->all(FLERR,"Fix ID for fix ave/correlate/peratom does not exist");
	if (argindex[i] == 0) { 
	  if (modify->fix[ifix]->peratom_flag != 1 ) {
	    error->all(FLERR, "Fix ave/correlate/peratom fix does not calculate a peratom scalar");
	  }
	} else {
	  if (modify->fix[ifix]->peratom_flag == 1) {
	    if (argindex[i] > modify->fix[ifix]->size_peratom_cols) {
	      error->all(FLERR,"Fix ave/correlate/peratom fix vector is accessed out-of-range");
	    }
	  } else {
	    error->all(FLERR, "Fix ave/correlate/peratom fix does not calculate a peratom/global vector (depending on switch)");
	  }
	}
	if (nevery % modify->fix[ifix]->peratom_freq)
	  error->all(FLERR,"Fix for fix ave/correlate/peratom not computed at compatible time");
	else if (nevery % modify->fix[ifix]->global_freq)
	  error->all(FLERR,"Fix for fix ave/correlate/peratom not computed at compatible time");
      } else if (which[i] == VARIABLE) {
	int ivariable = input->variable->find(ids[i]);
	if (ivariable < 0)
	  error->all(FLERR,"Variable name for fix ave/correlate/peratom does not exist");
	if (input->variable->atomstyle(ivariable) == 0 ) {
	  error->all(FLERR, "Fix ave/correlate/peratom variable is not an atom-style variable");
	}
      }
    }
  }

  // npair = # of correlation pairs to calculate
  if (type == AUTO || type == CROSS || type == AUTOCROSS) npair = nvalues;
  if (variable_flag == DIST_DEPENDENED) npair /= 3;
  if (type == AUTOUPPER) npair = nvalues*(nvalues+1)/2;
  if (type == FULL) npair = nvalues*nvalues;
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
      else if (type == FULL)
        for (i = 0; i < nvalues; i++)
          for (j = 0; j < nvalues; j++)
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
  memory->create(local_count,corr_length,"ave/correlate/peratom:local_count");
  memory->create(global_count,corr_length,"ave/correlate/peratom:local_count");
  memory->create(save_count,corr_length,"ave/correlate/peratom:save_count");
  memory->create(local_corr,corr_length,npair,"ave/correlate/peratom:local_corr");
  memory->create(global_corr,corr_length,npair,"ave/correlate/peratom:global_corr");
  memory->create(save_corr,corr_length,npair,"ave/correlate/peratom:save_corr");
  memory->create(local_corr_err,corr_length,npair,"ave/correlate/peratom:local_corr_err");
  memory->create(global_corr_err,corr_length,npair,"ave/correlate/peratom:global_corr_err");
  memory->create(save_corr_err,corr_length,npair,"ave/correlate/peratom:save_corr_err");
  for (i = 0; i < corr_length; i++) {
    save_count[i] = local_count[i] =  global_count[i] = 0.0;
    for (j = 0; j < npair; j++) save_corr[i][j] = local_corr[i][j] = global_corr[i][j] = save_corr_err[i][j] = local_corr_err[i][j] = global_corr_err[i][j] = 0.0;
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

  // determine group members
  if ( memory_switch != GROUP ) {
    memory->grow(indices_group,ngroup_loc,"ave/correlate/peratomindices_group");
    for (a= 0; a < nlocal; a++) {
      if(mask[a] & groupbit) {
	ngroup_loc++;
	memory->grow(indices_group,ngroup_loc,"ave/correlate/peratomindices_group");
	indices_group[ngroup_loc-1]=a;
      }
    }
    MPI_Allreduce(&ngroup_loc, &ngroup_glo, 1, MPI_INT, MPI_SUM, world);
    if ( ngroup_glo == 0 ) error->all(FLERR,"Illegal fix ave/correlate/peratom command: No group members");
    MPI_Exscan(&ngroup_loc,&ngroup_scan,1,MPI_INT, MPI_SUM, world);
    if((type == AUTOCROSS || type == CROSS) && ngroup_glo < 2) error->all(FLERR,"Illegal fix ave/correlate/peratom command: Cross-correlation with only one particle");
  }
  
  array= NULL;
  variable_store=NULL;
  if(nvalues > 0) {
    if(memory_switch == PERATOM){
      // need to grow array size
      comm->maxexchange_fix = MAX(comm->maxexchange_fix,(nvalues+include_orthogonal+variable_nvalues)*nsave);
      grow_arrays(atom->nmax);
      atom->add_callback(0);
      double *group_mass_loc;
      	int *type = atom->type;
	double *mass = atom->mass;
      memory->create(group_mass,ngroup_glo,"ave/correlate/peratom:group_mass");
      memory->create(group_mass_loc,ngroup_glo,"ave/correlate/peratom:group_mass_loc");
      for (a= 0; a < ngroup_glo; a++) {
	group_mass_loc[a]=group_mass[a]=0;
      }
      for (a= 0; a < ngroup_loc; a++) {
	group_mass_loc[a+ngroup_scan]=mass[type[indices_group[a]]];
      }
      MPI_Allreduce(group_mass_loc, group_mass, ngroup_glo, MPI_DOUBLE, MPI_SUM, world);
	memory->destroy(group_mass_loc);
      
    } else {
      // create global memorys
      grow_arrays(ngroup_glo);
      
      if (memory_switch == PERGROUP) {
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
      } else {
	int *type = atom->type;
	double *mass = atom->mass;
	double *group_mass_loc;
	memory->create(group_mass,ngroup_glo,"ave/correlate/peratom:group_mass");
	memory->create(group_mass_loc,ngroup_glo,"ave/correlate/peratom:group_mass_loc");
	for ( j = 0; j < ngroup_glo; j++) group_mass[j]=group_mass_loc[j]=0;
	for ( a= 0; a < nlocal; a++ ) {
	// valid saves the index of the group
	  for ( j = 0; j < ngroup_glo; j++) {
	    if(mask[a] & cor_groupbit[j]) {
	      group_mass_loc[j]+=mass[type[a]];
	    }
	  }	
	}
	MPI_Allreduce(group_mass_loc, group_mass, ngroup_glo, MPI_DOUBLE, MPI_SUM, world);
	memory->destroy(group_mass_loc);
      }

      // create memory for data storage and distribute
      memory->create(group_data_loc,ngroup_glo,nvalues+include_orthogonal+variable_nvalues,"ave/correlate/peratom:group_data_loc");
      memory->create(group_data,ngroup_glo,nvalues+include_orthogonal+variable_nvalues,"ave/correlate/peratom:group_data");
    }
  }
  if ( memory_switch != GROUP ) {
    memory->destroy(indices_group);
  }

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
  time_init_compute=0;
  calc_write_nvalues=0;
  write_var=0;
  write_orthogonal=0;
  reduce_write_global=0;
  time_calc=0;
  time_red_calc=0;
  time_calc_mean=0;
  time_total=0;
  
}

/* ---------------------------------------------------------------------- */

FixAveCorrelatePeratom::~FixAveCorrelatePeratom()
{

  delete [] which;
  delete [] argindex;
  delete [] value2index;
  for (int i = 0; i < nvalues; i++) {
    delete [] ids[i];
  }
  delete [] ids;

  memory->destroy(array);
  memory->destroy(local_count);
  memory->destroy(save_count);
  memory->destroy(global_count);
  memory->destroy(local_corr);
  memory->destroy(global_corr);
  memory->destroy(save_corr);
  memory->destroy(local_corr_err);
  memory->destroy(global_corr_err);
  memory->destroy(save_corr_err);
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

  if (memory_switch != GROUP) memory->destroy(group_ids);
  memory->destroy(group_mass);

  if (mean_flag) {
    memory->destroy(mean);
    memory->destroy(mean_count);
  }

  if (fp && me == 0) fclose(fp);
  
  atom->delete_callback(id,0);

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


  int a,i,j,o,v2i,r,ngroup_loc=0;
  double scalar;
  double *peratom_data;
  int *indices_group;
  int *counter, *counter_glo;

  int nlocal= atom->nlocal;
  int *mask= atom->mask;
  double *mass = atom->mass;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  tagint *tag = atom->tag;

  //printf("nsample = %d, lastindex = %d, nvalid = %d\n",nsample,lastindex,nvalid);

  // skip if not step which requires doing something
  bigint ntimestep = update->ntimestep;
  if (ntimestep != nvalid) return;

  double tt1 = MPI_Wtime();
  t1 = MPI_Wtime();

  // find relevant particles // find group-member on each processor
  if(memory_switch!=GROUP){
    memory->grow(indices_group,ngroup_loc,"ave/correlate/peratomindices_group");
    for (a= 0; a < nlocal; a++) {
      if(mask[a] & groupbit) {
	ngroup_loc++;
	memory->grow(indices_group,ngroup_loc,"ave/correlate/peratomindices_group");
	indices_group[ngroup_loc-1]=a;
      }
    }
  }

  // reset group_data
  if(memory_switch!=PERATOM){
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

  t2 = MPI_Wtime();
  time_init_compute += t2 -t1;

  lastindex++;
  if (lastindex == nsave) lastindex = 0;

  t1 = MPI_Wtime();

  if(memory_switch!=GROUP){
    for (i = 0; i < nvalues; i++) {
      v2i = value2index[i];
      // invoke compute if not previously invoked
      if (which[i] == COMPUTE) {
	Compute *compute = modify->compute[v2i];
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
	peratom_data= modify->fix[v2i]->vector_atom;
	else{
	  memory->create(peratom_data, nlocal, "ave/correlation/peratom:peratom_data");
	  for (a= 0; a < nlocal; a++) {
	    peratom_data[a] = modify->fix[v2i]->array_atom[a][argindex[i]-1];
	  }
	}
      // evaluate equal-style variable
      } else {
	memory->create(peratom_data, nlocal, "ave/correlation/peratom:peratom_data");
	input->variable->compute_atom(v2i, igroup, peratom_data, 1, 0);
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
      if (which[i] == VARIABLE || which[i] == FIX ) {
	memory->destroy(peratom_data);
      }
    }
  } else { //calculate group properties
    //evaluate all variable and compute all computes
    double v_store[v_counter*nlocal];
    int v_counter_loc = 0;
    for (i = 0; i < nvalues; i++) {
      switch ( cor_valbit[i] ) {
	case 7: {
	  Compute *compute = modify->compute[value2index[i]];
	  if(!(compute->invoked_flag & INVOKED_PERATOM)) {
	    compute->compute_peratom();
	    compute->invoked_flag |= INVOKED_PERATOM;
	  }
	}
	break;
	case 9: {
	  input->variable->compute_atom(value2index[i],igroup,&v_store[v_counter_loc],v_counter,0);
	  v_counter_loc++;
	}
	break;
      }
    }

    counter = new int[ngroup_glo];
    counter_glo = new int[ngroup_glo];
    for ( j = 0; j < ngroup_glo; j++) counter[j]=counter_glo[j]=0;
    for ( a= 0; a < nlocal; a++ ) {
      // valid saves the index of the group
      int valid = 0;
      for ( j = 1; j <= ngroup_glo; j++) {
	if(mask[a] & cor_groupbit[j-1]) {
	  valid = j;
	  counter[j-1]++;
	}
      }
      
      if(valid) {
	v_counter_loc = 0;
	for (i = 0; i < nvalues; i++) {
	  double data = 0.0;
	  switch ( cor_valbit[i] ) {
	    case 1:
	      data = v[a][0];
	    break;
	    case 2:
	      data = v[a][1];
	    break;
	    case 3:
	      data = v[a][2];
	    break;
	    case 4:
	      data = f[a][0];
	    break;
	    case 5:
	      data = f[a][1];
	    break;
	    case 6:
	      data = f[a][2];
	    break;
	    case 7: {
	      Compute *compute = modify->compute[value2index[i]];
	      if (argindex[i] == 0) data = compute->vector_atom[a];
	      else data= compute->array_atom[argindex[i]-1][a];
	    }
	    break;
	    case 8: {
	      if (argindex[i] == 0) data= modify->fix[value2index[i]]->vector_atom[a];
	      else data = modify->fix[value2index[i]]->array_atom[a][argindex[i]-1];
	    }
	    break;
	    case 9: {
	      data=v_store[a*v_counter+v_counter_loc];
	      v_counter_loc++;
	    }
	    break;
	  }
	  group_data_loc[valid-1][i] += data;
	  if ( (dynamics == ORTHOGONAL || dynamics == ORTHOGONALSECOND) && nsample < nsave) {
	    group_data_loc[valid-1][i+nvalues+6] += data;
	  }
	}
      }
    }
  }

  t2 = MPI_Wtime();
  calc_write_nvalues += t2 - t1;

  t1 = MPI_Wtime();

  //update variable dependency
  if(memory_switch!=GROUP){ 
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
  } else { //group properties
    if (variable_flag == DIST_DEPENDENED) {
      for ( a= 0; a < nlocal; a++ ) {
	// valid saves the index of the group
	int valid = 0;
	for ( j = 1; j <= ngroup_glo; j++) {
	  if(mask[a] & cor_groupbit[j-1]) {
	    valid = j;
	  }
	}
	if(valid) {
	  for (r = 0; r < 3; r++) {
	    double x_loc = x[a][0];
	    double y_loc = x[a][1];
	    double z_loc = x[a][2];
	    domain->minimum_image(x_loc,y_loc,z_loc);
	    group_data_loc[valid-1][nvalues+include_orthogonal] += x_loc;
	    group_data_loc[valid-1][nvalues+include_orthogonal+1] += y_loc;
	    group_data_loc[valid-1][nvalues+include_orthogonal+2] += z_loc; 
	  }
	}
      }
    }
  }

  t2 = MPI_Wtime();
  write_var += t2 - t1;

  t1 = MPI_Wtime();

  if ( memory_switch != GROUP ) { 
    if (dynamics == ORTHOGONAL || dynamics == ORTHOGONALSECOND) {
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
  } else {
    if (dynamics == ORTHOGONAL || dynamics == ORTHOGONALSECOND) {
      for ( a= 0; a < nlocal; a++ ) {
	// valid saves the index of the group
	int valid = 0;
	for ( j = 1; j <= ngroup_glo; j++) {
	  if(mask[a] & cor_groupbit[j-1]) {
	    valid = j;
	  }
	}
      
	if(valid) {
	  for (r = 0; r < 3; r++) {
	    group_data_loc[valid-1][r+nvalues] += v[a][r];
	    group_data_loc[valid-1][r+nvalues+3] += f[a][r];
	  }
	}
      }
    }
  }

  t2 = MPI_Wtime();
  write_orthogonal += t2 - t1;

  t1 = MPI_Wtime();

  // include pergroup data into global array
  if( memory_switch==PERGROUP || memory_switch==GROUP ){
    // exclude the last nvalues while memory calculation is performed
    double exclude_memory = 0;
    if ((nsample >= nsave) && (dynamics == ORTHOGONAL || dynamics == ORTHOGONALSECOND)) exclude_memory = -nvalues;
    MPI_Allreduce(&group_data_loc[0][0], &group_data[0][0], ngroup_glo*(nvalues+include_orthogonal+variable_nvalues), MPI_DOUBLE, MPI_SUM, world);
    if (memory_switch==GROUP) MPI_Allreduce(counter, counter_glo, ngroup_glo, MPI_INT, MPI_SUM, world);
    for (a= 0; a < ngroup_glo; a++) {
      for (i=0; i< nvalues+include_orthogonal+exclude_memory;i++) {
	int offset = i*nsave + lastindex;
	array[a][offset] = group_data[a][i];
      }
      if (memory_switch==GROUP) {
	for (i=0; i< nvalues;i++) {
	  int offset = i*nsave + lastindex;
	  if ( cor_valbit[i]==1 || cor_valbit[i]==2 || cor_valbit[i]==3 )  array[a][offset] /= counter_glo[a];
	}
	if ( dynamics == ORTHOGONAL || dynamics == ORTHOGONALSECOND ) {
	  for (i=nvalues; i< nvalues+3;i++) {
	    int offset = i*nsave + lastindex;
	    array[a][offset] /= counter_glo[a];
	  }
	  for (i=nvalues+6; i < nvalues+include_orthogonal+exclude_memory;i++) {
	    int offset = i*nsave + lastindex;
	    if ( cor_valbit[i-nvalues-6]==1 || cor_valbit[i-nvalues-6]==2 || cor_valbit[i-nvalues-6]==3 )  array[a][offset] /= counter_glo[a];
	  }
	}
      }
      	 
      if (variable_flag == VAR_DEPENDENED) {
	variable_store[a][lastindex] = group_data[a][nvalues+include_orthogonal];
      } else if (variable_flag == DIST_DEPENDENED) {
	for (r = 0; r < 3 ; r++) {
	  variable_store[a][lastindex+r*nsave] = group_data[a][nvalues+include_orthogonal+r];
	  variable_store[a][lastindex+r*nsave] /= counter_glo[a];
	}
      }
    }
  }

  t2 = MPI_Wtime();
  reduce_write_global += t2 - t1;


  // fistindex = index in values ring of earliest time sample
  // nsample = number of time samples in values ring

  if( (dynamics==ORTHOGONAL || dynamics == ORTHOGONALSECOND) || nsample < nsave) nsample++;

  int t = nsample - nsave;

  nvalid += nevery;
  modify->addstep_compute(nvalid);

  // calculate all Cij() enabled by latest values
  if (dynamics==NORMAL || t == 0){
    t1 = MPI_Wtime();
    accumulate(indices_group, ngroup_loc);
    t2 = MPI_Wtime();
    time_calc += t2 - t1;
  } else if (dynamics==ORTHOGONAL && t > 0 && t < nrepeat) {
    t1 = MPI_Wtime();
    int k,m,r;
    double dt = update->dt*nevery;

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
          array[a][(i+nvalues+6)*nsave+k] += alpha_loc*fnm*dt;
          m--;
        }
      }
    }
      }

      //accumulate
      accumulate(indices_group, ngroup_loc);
      t2 = MPI_Wtime();
      time_calc += t2 - t1;
    } else if (dynamics==ORTHOGONALSECOND && nsample > nsave && t < nrepeat) {
      t1 = MPI_Wtime();
      int k,m,mm1,mp1,r;
      double dt = update->dt*nevery;

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
          array[a][(i+nvalues+6)*nsave+k] += alpha_loc*fnm*dt/2.0
        +dt/2.0*fnmp1/(1-dt/2.0*kappa_loc)
        *(epsilon_loc+zeta_loc*alpha_loc*dt/2.0);
          m--; mp1--;
          if (m < 0) m = nsave-1; if (mp1 < 0) mp1 = nsave-1;
        }
      }
    }
      }

      //accumulate

      accumulate(indices_group, ngroup_loc);
      t2 = MPI_Wtime();
      time_calc += t2 - t1;
    }


    //calculate mean
    if (mean_flag){
      calc_mean(indices_group, ngroup_loc);
    }

  double tt2 = MPI_Wtime();
  time_total += tt2-tt1;

  if (ntimestep % nfreq || first) {
    first = 0;
    if(memory_switch!=GROUP) {
      memory->destroy(indices_group);
    } else {
      delete [] counter;
      delete [] counter_glo;
    }
    return;
  }

  //reduce the results from every proc
  MPI_Reduce(local_count, global_count, corr_length, MPI_DOUBLE, MPI_SUM, 0, world);
  MPI_Reduce(&local_corr[0][0], &global_corr[0][0], npair*corr_length, MPI_DOUBLE, MPI_SUM, 0, world);
  MPI_Reduce(&local_corr_err[0][0], &global_corr_err[0][0], npair*corr_length, MPI_DOUBLE, MPI_SUM, 0, world);
  //reset local arrays
  for (i = 0; i < corr_length; i++) {
    save_count[i] += global_count[i];
    local_count[i] =  global_count[i] = 0.0;
    for (j = 0; j < npair; j++){
      save_corr[i][j] += global_corr[i][j];
      save_corr_err[i][j] += global_corr_err[i][j];
      local_corr[i][j] = global_corr[i][j] = 0.0;
      local_corr_err[i][j] = global_corr_err[i][j] = 0.0;
    }
  }

  if (me == 0) {
    // output result to file
    if (fp) {
      if (overwrite) fseek(fp,filepos,SEEK_SET);
      fprintf(fp,BIGINT_FORMAT " %d\n",ntimestep,nrepeat);
      for (i = 0; i < corr_length/factor; i++) {
    if (variable_flag == VAR_DEPENDENED || variable_flag == DIST_DEPENDENED) {
      int loc_bin = i%bins;
      int loc_ind = (i - loc_bin)/bins;
      fprintf(fp,"%d %d %lf %lf",loc_ind+1,loc_ind*nevery,range/bins*loc_bin,save_count[i]);
    } else {
      fprintf(fp,"%d %d %lf",i+1,i*nevery,save_count[i]);
    }
    if (save_count[i]) {
      for (j = 0; j < npair; j++)
        fprintf(fp," %g %g",prefactor*save_corr[i][j]/save_count[i],prefactor*save_corr_err[i][j]/save_count[i]);
    } else {
      for (j = 0; j < npair; j++)
        fprintf(fp," 0.0 0.0");
    }
    if (type == AUTOCROSS || variable_flag == DIST_DEPENDENED) {
      int offset = i + corr_length/2;
      if (type == AUTOCROSS)
        fprintf(fp," %lf",save_count[offset]);
      if (save_count[offset]) {
        for (j = 0; j < npair; j++)
          fprintf(fp," %g %g",prefactor*save_corr[offset][j]/save_count[offset],prefactor*save_corr_err[offset][j]/save_count[offset]);
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
    if (mean_count[o]){
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
    save_count[i] = 0.0;
    for (j = 0; j < npair; j++)
      save_corr[i][j] = 0.0;
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

  if(memory_switch!=GROUP) {
    memory->destroy(indices_group);
  } else {
    delete [] counter;
    delete [] counter_glo;
  }
 

  // print timing
  //printf("processor %d: time(init_compute) = %f\n",me,time_init_compute);
  //printf("processor %d: time(calc+write_nvalues) = %f\n",me,calc_write_nvalues);
  //printf("processor %d: time(write_var) = %f\n",me,write_var);
  //printf("processor %d: time(write_orthogonal) = %f\n",me,write_orthogonal);
  //printf("processor %d: time(reduce_write_global) = %f\n",me,reduce_write_global);
  //printf("processor %d: time(calc) = %f\n",me,time_calc);
  //printf("processor %d: time(red_calc) = %f\n",me,time_red_calc);
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

  //calculate work distribution
  int sample_start = 0,
      sample_stop = 0;
  if (dynamics==NORMAL) {
    int work = nsample/nprocs;
    int rest = nsample - nprocs*work;
    //printf("work: %d, rest: %d\n",work,rest);
    if (me < rest) {
      sample_start = (work+1)*me;
      sample_stop = (work+1)*(me+1);
    } else {
      sample_start = (work+1)*rest + work*(me-rest);
      sample_stop = (work+1)*rest + work*(me-rest+1);
    }
    //sample_start = 0;
    //sample_stop = nsample;
  } else {
    int work = (nsave-1)/nprocs;
    int rest = nsave - 1 - nprocs*work;
    //printf("work: %d, rest: %d\n",work,rest);
    if (me < rest) {
      sample_start = nsave - 1 - (work+1)*me;
      sample_stop = nsave - 1 - (work+1)*(me+1);
    } else {
      sample_start = nsave - 1 - (work+1)*rest - work*(me-rest);
      sample_stop = nsave - 1 - (work+1)*rest - work*(me-rest+1);
    }
    //sample_start = nsave - 1 ;
    //sample_stop = 0;
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
    if (type == AUTOUPPER || type == FULL) nvalues_upper = nvalues;
    double nvalues_lower = i;
    if (type == FULL) nvalues_lower = 0;
    for (j = nvalues_lower; j < nvalues_upper; j+=incr_nvalues) {
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

	  //initialize counter for work distribution
	  if(dynamics==NORMAL) m = lastindex - sample_start;
	  else m = lastindex - (nsave - 1 - sample_start);
	  if (m < 0) m = nsave+m;
	  //printf("%d\n",m);
	  int inda,indb;
	  if(memory_switch==PERATOM){
	    inda=indices_group[a];
	    indb=indices_group[b];
	  } else {
	    inda = a;
	    indb = b;
	  }
	  if(dynamics==NORMAL){
	    for (k = sample_start; k < sample_stop; k++) {
	      if (variable_flag == VAR_DEPENDENED){
		double dV = variable_store[inda][m] - variable_store[indb][n];
		dV=fabs(dV);
		if(dV<range){
		  int ind = dV/range*bins;
		  int offset= k*bins+ind;
		  double val0 = array[indb][j * nsave + n];
		  double valt = array[inda][i * nsave + m];
		  double cor = val0*valt;
		  if (type == AUTOCROSS && b!=a) {
		    if(i==0&&j==0) local_count[offset+corr_length/2]+=1.0;
		    local_corr[offset+corr_length/2][ipair] += cor;
		    local_corr_err[offset+corr_length/2][ipair] += cor*cor;
		  } else {
		    if(i==0&&j==0) {
		      local_count[offset]+=1.0;
		    }
		    local_corr[offset][ipair]+=cor;
		    local_corr_err[offset][ipair]+=cor*cor;
		  }
		}
	      } else if (variable_flag == DIST_DEPENDENED) {
		double *dr = new double[3];
		dr[0] = variable_store[inda][m] - variable_store[indb][n];
		dr[1] = variable_store[inda][m+nsave] - variable_store[indb][n+nsave];
		dr[2] = variable_store[inda][m+2*nsave] - variable_store[indb][n+2*nsave];
		domain->minimum_image(dr[0],dr[1],dr[2]);
		double dV = sqrt( dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]);
		if(dV<range){
		  double *res_data = new double[8];
		  double *inp_data = new double[6];
		  int p;
		  for (p=0; p<3; p++) inp_data[p] = array[inda][ (i+p) * nsave + m];
		  for (p=0; p<3; p++) inp_data[p+3] = array[indb][ (j+p) * nsave + n];
		  // calculate radial component of the forces (mapped on the distance vector)
		  decompose(res_data,dr,inp_data);
		  int ind = dV/range*bins;
		  int offset= k*bins+ind;
		  double val0 = res_data[1];
		  double valt = res_data[0];
		  double cor = val0*valt;
		  if(i==0&&j==0) local_count[offset]+=1.0;
		  local_corr[offset][ipair] += cor;
		  local_corr_err[offset][ipair] += cor*cor;
		  // angular component
		  if(i==0&&j==0) local_count[offset+corr_length/2]+=1.0;
		  cor = res_data[2]*res_data[5]+res_data[3]*res_data[6]+res_data[4]*res_data[7];
		  local_corr[offset+corr_length/2][ipair] += cor;
		  local_corr_err[offset+corr_length/2][ipair] += cor*cor;
		  delete[] res_data;
		  delete[] inp_data;
		}
		delete[] dr;
	      } else { //no variable dependency
		double val0 = array[indb][j * nsave + n];
		double valt = array[inda][i * nsave + m];
		double cor = val0*valt;
		if (type == AUTOCROSS && b!=a) {
		  if(i==0&&j==0) local_count[k+corr_length/2]+=1.0;
		  local_corr[k+corr_length/2][ipair] += cor;
		  local_corr_err[k+corr_length/2][ipair] += cor*cor;
		} else {
		  if(i==0&&j==0) local_count[k]+=1.0;
		  local_corr[k][ipair]+= cor;
		  local_corr_err[k][ipair]+= cor*cor;
		}
	      }
	      m--;
	      if (m < 0) m = nsave-1;
	    }
	  } else { //dynamics ORTHOGONAL/ORTHOGONALSECOND
	    for (k = sample_start; k > sample_stop; k--) {
	      if (variable_flag == VAR_DEPENDENED){
		double dV = variable_store[inda][m] - variable_store[indb][n];
		dV=fabs(dV);
		if(dV<range){
		  int ind = dV/range*bins;
		  int offset= t*bins+ind;
		  double val0 = array[indb][(j+nvalues+6)*nsave+k];
		  double valt = array[inda][i*nsave+m];
		  double cor = val0*valt;
		  if (type == AUTOCROSS && b!=a) {
		    if(i==0&&j==0) local_count[offset+corr_length/2]+=1.0;
		    local_corr[offset+corr_length/2][ipair] += cor;
		    local_corr_err[offset+corr_length/2][ipair] += cor*cor;
		  } else {
		    if(i==0&&j==0) local_count[offset]+=1.0;
		    local_corr[offset][ipair]+= cor;
		    local_corr_err[offset][ipair]+= cor*cor;
		  }
		}
	      } else if (variable_flag == DIST_DEPENDENED) {
		double *dr = new double[3];
		dr[0] = variable_store[inda][m] - variable_store[indb][n];
		dr[1] = variable_store[inda][m+nsave] - variable_store[indb][n+nsave];
		dr[2] = variable_store[inda][m+2*nsave] - variable_store[indb][n+2*nsave];
		domain->minimum_image(dr[0],dr[1],dr[2]);
		double dV = sqrt( dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]);
		//printf("dV=%f\n",dV);
		if(dV<range){
		  double *res_data = new double[8];
		  double *inp_data = new double[6];
		  int p;
		  for (p=0; p<3; p++) inp_data[p] = array[inda][(i+p)*nsave+m];
		  for (p=0; p<3; p++) inp_data[p+3] = array[indb][(j+p+nvalues+6)*nsave+k];
		  decompose(res_data,dr,inp_data);
		  // calculate radial component of the forces (mapped on the distance vector)
		  int ind = dV/range*bins;
		  int offset= t*bins+ind;
		  double val0 = res_data[1];
		  double valt = res_data[0];
		  double cor = val0*valt;
		  if(i==0&&j==0) local_count[offset]+=1.0;
		  local_corr[offset][ipair] += cor;
		  local_corr_err[offset][ipair] += cor*cor;
		  cor = res_data[2]*res_data[5]+res_data[3]*res_data[6]+res_data[4]*res_data[7];
		  if(i==0&&j==0) local_count[offset+corr_length/2]+=1.0;
		  local_corr[offset+corr_length/2][ipair] += cor;
		  local_corr_err[offset+corr_length/2][ipair] += cor*cor;
		  delete[] res_data;
		  delete[] inp_data;
		}
		delete[] dr;
	      } else { //no variable dependency
		int offset= t;
		double val0 = array[indb][(j+nvalues+6)*nsave+k];
		double valt = array[inda][i*nsave+m];
		double cor = val0*valt;
		if (type == AUTOCROSS && b!=a) {
		  if(i==0&&j==0) local_count[offset+corr_length/2]+=1.0;
		  local_corr[offset+corr_length/2][ipair] += cor;
		  local_corr_err[offset+corr_length/2][ipair] += cor*cor;
		} else {
		  if(i==0&&j==0) local_count[offset]+=1.0;
		  local_corr[offset][ipair]+= cor;
		  local_corr_err[offset][ipair]+= cor*cor;
		}
	      }
	      m--;
	      if (m < 0) m = nsave-1;
	    }
	  }
	}
      }
      ipair++;
    }
  }
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
}

/* ----------------------------------------------------------------------
   calculate mean values using more recently added values
------------------------------------------------------------------------- */
void FixAveCorrelatePeratom::calc_mean(int *indices_group, int ngroup_loc){
  int a,b,i,o,k;
  tagint *tag = atom->tag;

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
	    if (i==0) mean_count[ind] += 2.0;
	    mean[ind*nvalues+i] += array[inda][i * nsave + lastindex];
	  }
	} else if (variable_flag == DIST_DEPENDENED) {

	  double *dr = new double[3];
	  dr[0] = variable_store[inda][lastindex] - variable_store[indb][lastindex];
	  dr[1] = variable_store[inda][lastindex+nsave] - variable_store[indb][lastindex+nsave];
	  dr[2] = variable_store[inda][lastindex+2*nsave] - variable_store[indb][lastindex+2*nsave];
	  domain->minimum_image(dr[0],dr[1],dr[2]);
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
	    if(i==0) mean_count[ind]+=2.0;
	    mean[ind*nvalues+i] += res_data[0];
	    mean[ind*nvalues+i] += res_data[1];
	    delete[] res_data;
	    delete[] inp_data;
	  }
	  delete[] dr;
	} else {
	  if(i==0) mean_count[0] += 1.0;
	  mean[i] += array[inda][i * nsave + lastindex];
	}
      }
    }
  }
}

/* ----------------------------------------------------------------------
   return I,J array value
------------------------------------------------------------------------- */

double FixAveCorrelatePeratom::compute_array(int i, int j)
{
  if (j == 0) return 1.0*i*nevery;
  else if (j == 1) return 1.0*save_count[i];
  else if (save_count[i]) return prefactor*save_corr[i][j-2]/save_count[i];
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
  printf("ngroup = %d, atom =%d\n", ngroup_glo,atom->nmax);
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
  int n = ncount+ 2*ncorr;
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
    fwrite(save_count,sizeof(double),ncount,fp);
    fwrite(&save_corr[0][0],sizeof(double),ncorr,fp);
    fwrite(&save_corr_err[0][0],sizeof(double),ncorr,fp);
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

  for (i = 0; i < corr_length; i++) save_count[i] = dbuf[dcount++];

  for (i = 0; i < corr_length; i++)
    for (j = 0; j < npair; j++) save_corr[i][j] = dbuf[dcount++];
    
  for (i = 0; i < corr_length; i++)
    for (j = 0; j < npair; j++) save_corr_err[i][j] = dbuf[dcount++];

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

