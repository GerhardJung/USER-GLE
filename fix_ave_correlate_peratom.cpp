
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
enum{AUTO,CROSS,AUTOCROSS,AUTOUPPER, UPPERCROSS, FULL};
enum{PERATOM,PERGROUP, PERPAIR, PERGROUP_PERPAIR, GROUP,ATOM};
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

  int i;
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
  memory_switch = PERATOM;
  variable_flag = NOT_DEPENDENED;
  bins = 1;
  factor = 1;
  mean_file = NULL;
  mean_flag = 0;
  fluc_flag = 0;
  variable_nvalues = 0;
  overwrite = 0;
  v_counter = 0;
  cross_flag = 0;
  char *title1 = NULL;
  char *title2 = NULL;
  char *title3 = NULL;
  body = NULL;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"type") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      if (strcmp(arg[iarg+1],"auto") == 0) type = AUTO;
      else if (strcmp(arg[iarg+1],"cross") == 0){
	if (iarg+3 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
	type = CROSS;
	cross_flag = force->inumeric(FLERR,arg[iarg+2]);
	iarg += 1;
      }
      else if (strcmp(arg[iarg+1],"auto/upper") == 0) type = AUTOUPPER;
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
      range = force->numeric(FLERR,arg[iarg+2]);
      bins = force->inumeric(FLERR,arg[iarg+3]);
      iarg += 4;
    } else if (strcmp(arg[iarg],"switch") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      if (strcmp(arg[iarg+1],"peratom") == 0) memory_switch = PERATOM;
      else if (strcmp(arg[iarg+1],"pergroup") == 0) memory_switch = PERGROUP;
      else if (strcmp(arg[iarg+1],"perpair") == 0) memory_switch = PERPAIR;
      else if (strcmp(arg[iarg+1],"pergroup/perpair") == 0) {
	if (iarg+4 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
	memory_switch = PERGROUP_PERPAIR;
	nvalues_pg = force->inumeric(FLERR,arg[iarg+2]);
	nvalues_pp = force->inumeric(FLERR,arg[iarg+3]);
	if ((nvalues_pg + nvalues_pp) != nvalues) {
	  error->all(FLERR,"Illegal fix ave/correlate/peratom command");
	}
	iarg += 2;
      }
      else if (strcmp(arg[iarg+1],"group") == 0) {
	memory_switch = GROUP;
	if (iarg+3 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
	ngroup_glo = force->inumeric(FLERR,arg[iarg+2]);
	if (iarg+4+ngroup_glo > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
	cor_groupbit = new int[ngroup_glo];
	cor_group = new int[ngroup_glo];
	for (int i=0; i<ngroup_glo; i++) {
	  int igroup = group->find(arg[iarg+3+i]);
	  if (igroup == -1) error->all(FLERR,"Could not find fix group ID");
	  cor_groupbit[i] = group->bitmask[igroup];
	  cor_group[i]=igroup;
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
      } else if (strcmp(arg[iarg+1],"atom") == 0) {
	memory_switch = ATOM;
	tagint *molecule;
	int *mask = atom->mask;
	int nlocal = atom->nlocal;
	
	memory->grow(body,atom->nmax,"rigid:body");
	atom->add_callback(0);

	if (iarg + 3 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");

	// determine whether atom-style variable or atom property is used.
	if (strstr(arg[iarg+2],"v_") == arg[iarg+2]) {
	  int ivariable = input->variable->find(arg[iarg+2]+2);
	  if (ivariable < 0)
	    error->all(FLERR,"Variable name for fix ave/correlate/peratom command atom does not exist");
	  if (input->variable->atomstyle(ivariable) == 0)
	    error->all(FLERR,"fix ave/correlate/peratom command atom variable is no atom-style variable");
	  double *value = new double[nlocal];
	  input->variable->compute_atom(ivariable,0,value,1,0);
	  int minval = INT_MAX;
	  for (i = 0; i < nlocal; i++)
	    if (mask[i] & groupbit) minval = MIN(minval,(int)value[i]);
	  int vmin = minval;
	  MPI_Allreduce(&vmin,&minval,1,MPI_INT,MPI_MIN,world);
	  molecule = new tagint[nlocal];
	  for (i = 0; i < nlocal; i++){
	    if (mask[i] & groupbit) molecule[i] = (tagint)((tagint)value[i] - minval + 1);
	    //printf("mol %d\n",molecule[i]);
	  }
	  delete[] value;
	} else error->all(FLERR,"Unsupported fix ave/correlate/peratom command atom property");
	iarg += 3;

	tagint maxmol_tag = -1;
	for (i = 0; i < nlocal; i++)
	  if (mask[i] & groupbit) maxmol_tag = MAX(maxmol_tag,molecule[i]);

	tagint itmp;
	MPI_Allreduce(&maxmol_tag,&itmp,1,MPI_LMP_TAGINT,MPI_MAX,world);
	if (itmp+1 > MAXSMALLINT)
	  error->all(FLERR,"Too many molecules for fix ave/correlate/peratom command");
	maxmol = (int) itmp;

	int *ncount;
	memory->create(ncount,maxmol+1,"ave/correlate/peratom:ncount");
	for (i = 0; i <= maxmol; i++) ncount[i] = 0;

	for (i = 0; i < nlocal; i++)
	  if (mask[i] & groupbit) ncount[molecule[i]]++;

	memory->create(mol2body,maxmol+1,"ave/correlate/peratom:mol2body");
	MPI_Allreduce(ncount,mol2body,maxmol+1,MPI_INT,MPI_SUM,world);

	nbody = 0;
	for (i = 0; i <= maxmol; i++)
	  if (mol2body[i]) mol2body[i] = nbody++;
	  else mol2body[i] = -1;

	memory->create(body2mol,nbody,"ave/correlate/peratom:body2mol");

	nbody = 0;
	for (i = 0; i <= maxmol; i++)
	  if (mol2body[i] >= 0) body2mol[nbody++] = i;

	for (i = 0; i < nlocal; i++) {
	  body[i] = -1;
	  if (mask[i] & groupbit) body[i] = mol2body[molecule[i]];
	}
	ngroup_glo = maxmol-1;
	//printf("%d\n",ngroup_glo);
	
	memory->destroy(ncount);
	delete [] molecule;
	
	// determine value to be calculated
	nvalues = force->inumeric(FLERR,arg[iarg]);
	cor_valbit = new int[nvalues];
	if (iarg+1+nvalues > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
	for (int i=0; i<nvalues; i++) {
	  char *loc_arg = arg[iarg+1+i];
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
	iarg += nvalues-1;
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
    } else if (strcmp(arg[iarg],"fluctuation") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix ave/correlate/peratom command");
      if (me == 0) {
        mean_file_fluc = fopen(arg[iarg+1],"r");
        if (mean_file_fluc == NULL) {
          char str[128];
          sprintf(str,"Cannot open fix ave/correlate/peratom file %s",arg[iarg+1]);
          error->one(FLERR,str);
        }
      }
      fluc_flag = 1;
      fluc_lower  = force->inumeric(FLERR,arg[iarg+2]);
      fluc_upper  = force->inumeric(FLERR,arg[iarg+3]);
      iarg += 4;
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
  
  nsave = nrepeat;

  // distance dependence only makes sence when we calculate cross correlation
  if (variable_flag == DIST_DEPENDENED && (type != CROSS && type != UPPERCROSS)){
    error->all(FLERR,"Illegal fix ave/correlate/peratom command: distance dependence without cross correlation");
  }
  if (variable_flag == PERPAIR && (type != CROSS && type != UPPERCROSS)){
    error->all(FLERR,"Illegal fix ave/correlate/peratom command: perpair switch without cross correlation");
  }
  if (variable_flag == DIST_DEPENDENED && nvalues % 3) {
    error->all(FLERR,"Illegal fix ave/correlate/peratom command: distance dependence decomposes 3d-system into parallel and orthogonal component");
  }
  if (fluc_flag && variable_flag != DIST_DEPENDENED ) {
    error->all(FLERR,"Illegal fix ave/correlate/peratom command: fluctuations only for distance dependence");
  }
  

  // check variables (for peratom/pergroup)
  int j,o;
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
  if (type == UPPERCROSS) npair = nvalues/3*(nvalues/3+1)/2;
  if (type == FULL) npair = nvalues*nvalues;
  //printf("npair %d\n",npair);
  // print file comment lines
  if (fp && me == 0) {
    if (title1) fprintf(fp,"%s\n",title1);
    else fprintf(fp,"# Time-correlated data for fix %s\n",id);
    if (title2) fprintf(fp,"%s\n",title2);
    else fprintf(fp,"# Timestep Number-of-time-windows\n");
    if (title3) fprintf(fp,"%s\n",title3);
    else {
      fprintf(fp,"# Index TimeDelta Ncount");
      if (variable_flag == DIST_DEPENDENED) {
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
      } else if (type == AUTO || type == AUTOCROSS || type == CROSS )
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
  
  if (fluc_flag) {
    // init memory
    memory->create(mean_fluc_data,bins,"ave/correlate/peratom:mean");
    // skip first lines starting with #
    char buf[0x1000];
    long filepos;
    filepos = ftell(mean_file_fluc);
    while (fgets(buf, sizeof(buf), mean_file_fluc) != NULL) {
      if (buf[0] != '#') {
	fseek(mean_file_fluc,filepos,SEEK_SET);
	break;
      }
      filepos = ftell(mean_file_fluc);
    } 
  
    //read potentials
    int r;
    double dist,pot;
    for(r=0; r<bins; r++){
      fscanf(mean_file_fluc,"%lf %lf\n",&dist,&pot);
      mean_fluc_data[r] = pot;
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
  if ( memory_switch != GROUP &&   memory_switch != ATOM) {
    memory->create(indices_group,ngroup_loc,"ave/correlate/peratomindices_group");
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
    if((type == AUTOCROSS || type == CROSS || type == UPPERCROSS) && ngroup_glo < 2) error->all(FLERR,"Illegal fix ave/correlate/peratom command: Cross-correlation with only one particle");
  }

  array= NULL;
  variable_store=NULL;
  if(nvalues > 0) {
    if(memory_switch == PERATOM){
      // need to grow array size
      comm->maxexchange_fix = MAX(comm->maxexchange_fix,(nvalues+variable_nvalues)*nsave);
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
      if (memory_switch == PERPAIR || memory_switch == PERGROUP_PERPAIR) grow_arrays(ngroup_glo*ngroup_glo);
      else grow_arrays(ngroup_glo);
      
      if (memory_switch == PERGROUP || memory_switch == PERPAIR || memory_switch == PERGROUP_PERPAIR) {
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
	  //printf("%d\n",body[a]);
	  for ( j = 0; j < ngroup_glo; j++) {
	    if(memory_switch == GROUP && mask[a] & cor_groupbit[j]) {
	      group_mass_loc[j]+=mass[type[a]];
	    }
	    if(memory_switch == ATOM && body[a] == j+1) {
	      group_mass_loc[j]+=mass[type[a]];
	    }
	  }	
	}
	MPI_Allreduce(group_mass_loc, group_mass, ngroup_glo, MPI_DOUBLE, MPI_SUM, world);
	memory->destroy(group_mass_loc);
      }

      // create memory for data storage and distribute
      if (memory_switch == PERPAIR || memory_switch == PERGROUP_PERPAIR) {
	memory->create(group_data_loc,ngroup_glo*ngroup_glo,nvalues+variable_nvalues,"ave/correlate/peratom:group_data_loc");
	memory->create(group_data,ngroup_glo*ngroup_glo,nvalues+variable_nvalues,"ave/correlate/peratom:group_data");
      } else {
	memory->create(group_data_loc,ngroup_glo,nvalues+variable_nvalues,"ave/correlate/peratom:group_data_loc");
	memory->create(group_data,ngroup_glo,nvalues+variable_nvalues,"ave/correlate/peratom:group_data");
      }
    }
  }

  if ( memory_switch != GROUP && memory_switch != ATOM ) {
    memory->destroy(indices_group);
  }

  //init timing
  time_init_compute=0;
  calc_write_nvalues=0;
  write_var=0;
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

  if (memory_switch != PERATOM) {
    memory->destroy(group_data_loc);
    memory->destroy(group_data);
  }

  if (memory_switch != GROUP && memory_switch != PERATOM && memory_switch != ATOM) memory->destroy(group_ids);
  memory->destroy(group_mass);

  if (mean_flag) {
    memory->destroy(mean);
    memory->destroy(mean_count);
  }
  
  if (fluc_flag) memory->destroy(mean_fluc_data);

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


  int a,b,i,j,o,v2i,r,ngroup_loc=0;
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
  if(memory_switch!=GROUP && memory_switch!=ATOM){
    memory->create(indices_group,ngroup_loc,"ave/correlate/peratomindices_group");
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
    if (memory_switch==PERPAIR || memory_switch == PERGROUP_PERPAIR){
      for (a = 0; a < ngroup_glo; a++) {
	for (b = 0; b < ngroup_glo; b++) {
	  for (i = 0; i < nvalues + variable_nvalues; i++) {
	    group_data[a*ngroup_glo+b][i] = 0;
	    group_data_loc[a*ngroup_glo+b][i] = 0;
	  }
	}
      }
    } else {
      for (a = 0; a < ngroup_glo; a++) {
	for (i = 0; i < nvalues + variable_nvalues; i++) {
	  group_data[a][i] = 0;
	  group_data_loc[a][i] = 0;
	}
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

  // care here, that sometimes the index changes -> use tag[a]
  if(memory_switch!=GROUP && memory_switch!=ATOM){
    for (i = 0; i < nvalues; i++) {
      v2i = value2index[i];
      // invoke compute if not previously invoked
      if (which[i] == COMPUTE) {
	Compute *compute = modify->compute[v2i];
	if(!(compute->invoked_flag & INVOKED_PERATOM)) {
	  compute->compute_peratom();
	  compute->invoked_flag |= INVOKED_PERATOM;
	}
	if (memory_switch==PERPAIR) {
	  peratom_data= compute->array_atom[(argindex[i]-1)*nlocal];
	} else {
	  if (argindex[i] == 0)
	    peratom_data= compute->vector_atom;
	  else{
	    memory->create(peratom_data, nlocal, "ave/correlation/peratom:peratom_data");
	    for (a= 0; a < nlocal; a++) {
	      peratom_data[tag[a]-1] = compute->array_atom[a][argindex[i]-1];
	    }
	  }
	}
      // access fix fields, guaranteed to be ready
      } else if (which[i] == FIX) {
	if (memory_switch==PERPAIR) {
	  memory->create(peratom_data, nlocal*nlocal, "ave/correlation/peratom:peratom_data");
	  for (a= 0; a < nlocal; a++) {
	    for (b= 0; b < nlocal; b++) {
	      peratom_data[a*nlocal+b] = modify->fix[v2i]->array_atom[a][(argindex[i]-1)*nlocal+b];
	      //if (a==0 && b==2) printf("corr input: %f\n",peratom_data[a*nlocal+b]);
	    }
	  }
	} else if (memory_switch==PERGROUP_PERPAIR) {
	  memory->create(peratom_data, nlocal*nlocal, "ave/correlation/peratom:peratom_data");
	  if (i < nvalues_pg) {
	    if (argindex[i] == 0) {
	      for (a= 0; a < nlocal; a++) {
		peratom_data[a]= modify->fix[v2i]->vector_atom[a];
	      }
	    } else {
	      for (a= 0; a < nlocal; a++) {
		peratom_data[a] = modify->fix[v2i]->array_atom[a][argindex[i]-1];
	      }
	    }
	  } else {
	    for (a= 0; a < nlocal; a++) {
	      for (b= 0; b < nlocal; b++) {
		peratom_data[a*nlocal+b] = modify->fix[v2i]->array_atom[a][(argindex[i]-1)*nlocal+b];
		//if (a==0 && b==2) printf("corr input: %f\n",peratom_data[a*nlocal+b]);
	      }
	    }
	  }
	} else {
	  if (argindex[i] == 0)
	    peratom_data= modify->fix[v2i]->vector_atom;
	  else{
	    memory->create(peratom_data, nlocal, "ave/correlation/peratom:peratom_data");
	    for (a= 0; a < nlocal; a++) {
	      peratom_data[a] = modify->fix[v2i]->array_atom[a][argindex[i]-1];
	    }
	  }
	}
      // evaluate equal-style variable
      } else {
	// variable with perpair not implemented
	if (memory_switch==PERGROUP_PERPAIR) memory->create(peratom_data, nlocal*nlocal, "ave/correlation/peratom:peratom_data");
	else memory->create(peratom_data, nlocal, "ave/correlation/peratom:peratom_data");
	input->variable->compute_atom(v2i, igroup, peratom_data, 1, 0);
	
      }
      if (memory_switch==PERPAIR || memory_switch==PERGROUP_PERPAIR) {
	for (a= 0; a < ngroup_loc; a++) {
	  tagint *ids_ptr;
	  ids_ptr = std::find(group_ids,group_ids+ngroup_glo,tag[indices_group[a]]);
	  int inda = (ids_ptr - group_ids);
	  if (memory_switch==PERGROUP_PERPAIR && i < nvalues_pg) group_data_loc[inda][i] = peratom_data[indices_group[a]];
	  else {
	    for (b= 0; b < ngroup_loc; b++) {
	      ids_ptr = std::find(group_ids,group_ids+ngroup_glo,tag[indices_group[b]]);
	      int indb = (ids_ptr - group_ids);
	      //printf("base %d, id %d ind %d\n",group_ids,ids_ptr,n);
	      group_data_loc[inda*ngroup_loc+indb][i] = peratom_data[indices_group[a]*ngroup_loc+indices_group[b]];
	    }
	  }
	  //if (a==0) printf("indices_grop %d, data %f\n",indices_group[a],group_data_loc[inda][i]);
	}
      } else {
	for (a= 0; a < ngroup_loc; a++) {
	  double data = peratom_data[indices_group[a]];
	  if(memory_switch==PERATOM){
	    int offset1= i*nsave + lastindex;
	    array[indices_group[a]][offset1]= data;
	  } else {
	    tagint *ids_ptr;
	    ids_ptr = std::find(group_ids,group_ids+ngroup_glo,tag[indices_group[a]]);
	    int ind = (ids_ptr - group_ids);
	    //printf("base %d, id %d ind %d\n",group_ids,ids_ptr,n);
	    group_data_loc[ind][i] = data;
	  }
	}
      }
      //if this was done by an atom-style variable, we need to free the mem we allocated
      if (which[i] == VARIABLE || which[i] == FIX ) {
	//printf("destroy %d\n",i);
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
	if(memory_switch == GROUP && mask[a] & cor_groupbit[j-1]) {
	  valid = j;
	  counter[j-1]++;
	} 
      }
      
      if (memory_switch == ATOM) {
	valid = body[a];
	if (body[a]!=0) counter[body[a]-1]++;
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
	}
      }
    }
  }

  t2 = MPI_Wtime();
  calc_write_nvalues += t2 - t1;

  t1 = MPI_Wtime();

  //update variable dependency
  if(memory_switch!=GROUP && memory_switch!=ATOM){ 
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
	  group_data_loc[ind][nvalues] = peratom_data[indices_group[a]];
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
	    group_data_loc[ind][nvalues+r] = x[indices_group[a]][r];
	  }
	}
      }
    }
  } 

  t2 = MPI_Wtime();
  write_var += t2 - t1;

  t1 = MPI_Wtime();

  // include pergroup data into global array
  if( memory_switch==PERGROUP || memory_switch==PERPAIR || memory_switch==PERGROUP_PERPAIR || memory_switch==GROUP || memory_switch ==ATOM){
    if (memory_switch==PERPAIR || memory_switch==PERGROUP_PERPAIR) MPI_Allreduce(&group_data_loc[0][0], &group_data[0][0], ngroup_glo*ngroup_glo*(nvalues+variable_nvalues), MPI_DOUBLE, MPI_SUM, world);
    else if (memory_switch!= GROUP && memory_switch!= ATOM) MPI_Allreduce(&group_data_loc[0][0], &group_data[0][0], ngroup_glo*(nvalues+variable_nvalues), MPI_DOUBLE, MPI_SUM, world);
    else MPI_Allreduce(&group_data_loc[0][0], &group_data[0][0], ngroup_glo*(nvalues), MPI_DOUBLE, MPI_SUM, world);
    if (memory_switch==GROUP || memory_switch==ATOM) MPI_Allreduce(counter, counter_glo, ngroup_glo, MPI_INT, MPI_SUM, world);
    for (a= 0; a < ngroup_glo; a++) {
      if (memory_switch==PERPAIR || memory_switch==PERGROUP_PERPAIR) {
	for (b= 0; b < ngroup_glo; b++) {
	  for (i=0; i< nvalues;i++) {
	    int offset = i*nsave + lastindex;
	    array[a*ngroup_glo+b][offset] = group_data[a*ngroup_glo+b][i];
	  }
	}
      } else {
	for (i=0; i< nvalues;i++) {
	  int offset = i*nsave + lastindex;
	  array[a][offset] = group_data[a][i];
	}
      }
      if (memory_switch==GROUP || memory_switch == ATOM) {
	for (i=0; i< nvalues;i++) {
	  int offset = i*nsave + lastindex;
	  if ( cor_valbit[i]==1 || cor_valbit[i]==2 || cor_valbit[i]==3 )  array[a][offset] /= counter_glo[a];
	}
      }
      	 
      if (variable_flag == VAR_DEPENDENED) {
	variable_store[a][lastindex] = group_data[a][nvalues];
      } else if (variable_flag == DIST_DEPENDENED) {
	if(memory_switch!=GROUP && memory_switch!=ATOM){ 
	  for (r = 0; r < 3 ; r++) {
	    variable_store[a][lastindex+r*nsave] = group_data[a][nvalues+r];
	  }
	} else {
	  double xcm[3];
	  if (memory_switch==GROUP) group->xcm(igroup,counter_glo[a],xcm);
	  else this->xcm(a+1,counter_glo[a],xcm);
	  variable_store[a][lastindex] = xcm[0];
	  variable_store[a][lastindex+nsave] = xcm[1];
	  variable_store[a][lastindex+2*nsave] = xcm[2];
	  //printf("%d %f %f %f\n",a,variable_store[a][lastindex+0*nsave],variable_store[a][lastindex+1*nsave],variable_store[a][lastindex+2*nsave]);
	}
      }
    }
  }

  t2 = MPI_Wtime();
  reduce_write_global += t2 - t1;


  // fistindex = index in values ring of earliest time sample
  // nsample = number of time samples in values ring

  if( nsample < nsave) nsample++;

  nvalid += nevery;
  modify->addstep_compute(nvalid);

  // calculate all Cij() enabled by latest values
  t1 = MPI_Wtime();
  accumulate(indices_group, ngroup_loc);
  t2 = MPI_Wtime();
  //time_calc += t2 - t1;

  t1 = MPI_Wtime();
  //calculate mean
  if (mean_flag){
    calc_mean(indices_group, ngroup_loc);
  }
  t2 = MPI_Wtime();
  time_calc_mean += t2 - t1;

  double tt2 = MPI_Wtime();
  //time_total += tt2-tt1;

  if (ntimestep % nfreq || first) {
    first = 0;
    if(memory_switch!=GROUP && memory_switch!=ATOM) {
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
	    fprintf(fp," %.15lg %g",prefactor*save_corr[i][j]/save_count[i],prefactor*save_corr_err[i][j]/save_count[i]);
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
	      fprintf(fp," %.15lg %g",prefactor*save_corr[offset][j]/save_count[offset],prefactor*save_corr_err[offset][j]/save_count[offset]);
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
  if(ntimestep != update->nsteps ) accumulate(indices_group, ngroup_loc);

  if(memory_switch!=GROUP && memory_switch!=ATOM) {
    memory->destroy(indices_group);
  } else {
    delete [] counter;
    delete [] counter_glo;
  }


  // print timing
 // printf("processor %d: time(init_compute) = %f\n",me,time_init_compute);
  //printf("processor %d: time(calc+write_nvalues) = %f\n",me,calc_write_nvalues);
  //printf("processor %d: time(write_var) = %f\n",me,write_var);
  //printf("processor %d: time(reduce_write_global) = %f\n",me,reduce_write_global);
  printf("processor %d: time(calc) = %f\n",me,time_calc);
  //printf("processor %d: time(red_calc) = %f\n",me,time_calc_mean);
  printf("processor %d: time(total) = %f\n",me,time_total);
}

/* ----------------------------------------------------------------------
   accumulate correlation data using more recently added values
------------------------------------------------------------------------- */
void FixAveCorrelatePeratom::accumulate(int *indices_group, int ngroup_loc)
{
  //printf("test\n");
  int a,b,i,j,k,o,m,n,ipair;
  int ind,offset, ind_t, ind_0;
  int t = nsample - nsave;
  int nlocal= atom->nlocal;
  tagint *tag = atom->tag;
  
  double delx, dely, delz, delx_t, dely_t,delz_t, delx_0, dely_0, delz_0;
  double fabx_t, faby_t, fabz_t,fabx_0, faby_0, fabz_0;
  double rsq, dist, dist_t, dist_0, fabr_t,fabr_0;

  //calculate work distribution
  int sample_start = 0,
      sample_stop = 0;
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

  // accumulate
  n = lastindex;
  int incr_nvalues = 1;
  if (variable_flag == DIST_DEPENDENED){
    incr_nvalues = 3;
  }

  double t1 = MPI_Wtime();
  #if defined (_OPENMP)
  #pragma omp parallel private(a,b,ipair,k,m,ind,offset,i,j,delx,dely,delz,rsq,dist,delx_t,dely_t,delz_t,dist_t,delx_0,dely_0,delz_0,dist_0,fabx_t,faby_t,fabz_t,fabx_0,faby_0,fabz_0,fabr_0,fabr_t,ind_t,ind_0) default(none) shared(n,sample_stop, sample_start,indices_group,incr_nvalues) 
  #endif
  {
    ipair = 0;
    int kfrom, kto, tid;
    loop_setup_thr(kfrom, kto, tid, sample_stop - sample_start,comm->nthreads);
    for (i = 0; i < nvalues; i+=incr_nvalues) {
      //determine whether just autocorrelation or also mixed correlation (different observables)
      double nvalues_upper = i+1;
      if (type == AUTOUPPER || type == UPPERCROSS || type == FULL) nvalues_upper = nvalues;
      double nvalues_lower = i;
      if (type == FULL) nvalues_lower = 0;
      for (j = nvalues_lower; j < nvalues_upper; j+=incr_nvalues) {
	//printf("i %d j %d ipair %d\n",i,j,ipair);
	for (a= 0; a < ngroup_glo; a++) {
	  //determine whether just autocorrelation or also cross correlation (different atoms)
	  double ngroup_lower = a;
	  double ngroup_upper = a+1;
	  if (type == CROSS || type == AUTOCROSS || type == UPPERCROSS){
	    ngroup_lower = a;
	    ngroup_upper = ngroup_glo;
	  }
	  for (b = ngroup_lower; b < ngroup_upper; b++) {
	    if ((type == CROSS || type == UPPERCROSS) && a==b) continue;

	    //initialize counter for work distribution
	    //printf("%d\n",m);
	    int inda,indb;
	    if(memory_switch==PERATOM){
	      inda=indices_group[a];
	      indb=indices_group[b];
	    } else {
	      inda = a;
	      indb = b;
	    }
	    
	    // jump if not in range
	    if (variable_flag == VAR_DEPENDENED){
	      double dV = variable_store[inda][n] - variable_store[indb][n];
	      dV=fabs(dV);
	      if(dV>range) continue;
	    } 
	    if (variable_flag == DIST_DEPENDENED){
	      delx = variable_store[inda][n] -variable_store[indb][n];
	      dely = variable_store[inda][n+nsave]-variable_store[indb][n+nsave]  ;
	      delz = variable_store[inda][n+2*nsave]- variable_store[indb][n+2*nsave];
	      domain->minimum_image(delx,dely,delz);
	      rsq = delx*delx + dely*dely + delz*delz;
	      dist = sqrt (rsq);
	      if(dist>range) continue;
	    }
	    
	    // calc correlation
	    m = lastindex - sample_start - kfrom;
	    if (m < 0) m = nsave+m;
	    double t1 = MPI_Wtime();
	    for (k = sample_start+kfrom; k < sample_start+kto; k++) {
	      if (variable_flag == VAR_DEPENDENED){
		double dV = variable_store[inda][m] - variable_store[indb][n];
		dV=fabs(dV);
		if(dV<range){
		  ind = dV/range*bins;
		  offset= k*bins+ind;
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
		delx = variable_store[inda][m] -variable_store[indb][n];
		dely = variable_store[inda][m+nsave]-variable_store[indb][n+nsave]  ;
		delz = variable_store[inda][m+2*nsave]- variable_store[indb][n+2*nsave];
		domain->minimum_image(delx,dely,delz);
		rsq = delx*delx + dely*dely + delz*delz;
		dist = sqrt (rsq);
		//if(dist < 6.0) {
		//  printf("dist %d %d: %f\n",a,b,dist);
		//}
		if(dist<range){
		  double disti = 1.0/dist;
		  delx_t = variable_store[inda][m] -variable_store[indb][m];
		  dely_t = variable_store[inda][m+nsave]-variable_store[indb][m+nsave]  ;
		  delz_t = variable_store[inda][m+2*nsave]- variable_store[indb][m+2*nsave];
		  domain->minimum_image(delx_t,dely_t,delz_t);
		  dist_t = sqrt(delx_t*delx_t + dely_t*dely_t + delz_t*delz_t);
		  delx_0 = variable_store[inda][n] -variable_store[indb][n];
		  dely_0 = variable_store[inda][n+nsave]-variable_store[indb][n+nsave]  ;
		  delz_0 = variable_store[inda][n+2*nsave]- variable_store[indb][n+2*nsave];
		  domain->minimum_image(delx_0,dely_0,delz_0);
		  dist_0 = sqrt(delx_0*delx_0 + dely_0*dely_0 + delz_0*delz_0);
		  if (cross_flag ==0) {
		    if (memory_switch==PERPAIR || (memory_switch == PERGROUP_PERPAIR && i >= nvalues_pg)) {
		      fabx_t = array[inda*ngroup_glo+indb][ i*nsave + m];
		      faby_t = array[inda*ngroup_glo+indb][ i*nsave + nsave + m];
		      fabz_t = array[inda*ngroup_glo+indb][ i*nsave + 2*nsave + m];
		    } else {
		      fabx_t = array[inda][ i*nsave + m];
		      faby_t = array[inda][ i*nsave + nsave + m];
		      fabz_t = array[inda][ i*nsave + 2*nsave + m];
		    }
		    if (memory_switch==PERPAIR || (memory_switch == PERGROUP_PERPAIR && j >= nvalues_pg)) {
		      fabx_0 = array[inda*ngroup_glo+indb][ j*nsave + n];
		      faby_0 = array[inda*ngroup_glo+indb][ j*nsave + nsave + n];
		      fabz_0 = array[inda*ngroup_glo+indb][ j*nsave + 2*nsave + n];
		    } else {
		      fabx_0 = array[indb][ j*nsave + n];
		      faby_0 = array[indb][ j*nsave + nsave + n];
		      fabz_0 = array[indb][ j*nsave + 2*nsave + n];
		    }
		    } else {
		    if (memory_switch==PERPAIR || (memory_switch == PERGROUP_PERPAIR && i >= nvalues_pg)) {
		      fabx_t = array[inda*ngroup_glo+indb][ i*nsave + m];
		      faby_t = array[inda*ngroup_glo+indb][ i*nsave + nsave + m];
		      fabz_t = array[inda*ngroup_glo+indb][ i*nsave + 2*nsave + m];
		    } else {
		      fabx_t = array[inda][ i*nsave + m] - array[indb][ i*nsave + m];
		      faby_t = array[inda][ i*nsave + nsave + m] - array[indb][ i*nsave + nsave + m];
		      fabz_t = array[inda][ i*nsave + 2*nsave + m] - array[indb][ i*nsave + 2*nsave + m];
		    }
		    if (memory_switch==PERPAIR || (memory_switch == PERGROUP_PERPAIR && j >= nvalues_pg)) {
		      fabx_0 = array[inda*ngroup_glo+indb][ j*nsave + n];
		      faby_0 = array[inda*ngroup_glo+indb][ j*nsave + nsave + n];
		      fabz_0 = array[inda*ngroup_glo+indb][ j*nsave + 2*nsave + n];
		    } else {
		      fabx_0 = array[inda][ j*nsave + n] - array[indb][ j*nsave + n];
		      faby_0 = array[inda][ j*nsave + nsave + n] - array[indb][ j*nsave + nsave + n];
		      fabz_0 = array[inda][ j*nsave + 2*nsave + n] - array[indb][ j*nsave + 2*nsave + n];
		    }
		  }
		  // calculate radial component of the forces/velocities (mapped on the distance vector)
		  fabr_t = (fabx_t*delx_t + faby_t*dely_t +fabz_t*delz_t) * disti;
		  fabr_0 = (fabx_0*delx_0 + faby_0*dely_0 +fabz_0*delz_0) * disti;
		  //delfx_p = frp * delx / rsq;
		  //delfy_p = frp * dely / rsq;
		  //delfz_p = frp * delz / rsq;
		  
		  //printf("dist %f fabr_t %f fabx_t %f delx_t %f\n",dist,fabr_t, fabx_t, delx_t);
		  
		  //if (m==n) printf("dist: %f, dist_t %f, dist_0 %f\n",dist,dist_t, dist_0);
		  
		  ind = dist/range*bins;
		  //ind_t = dist_t/range*bins;
		  //ind_0 = dist_0/range*bins;
		  offset= k*bins+ind;
		  if (fluc_flag) {
		    //printf ("fluc_lower %d, fluc_upper %d\n",fluc_lower, fluc_upper);
		    //if ( fluc_lower <= i +1 && i+1 <= fluc_upper && dist_t < range) fabr_t -= mean_fluc_data[ind_t];
		    //if ( fluc_lower <= j +1 && j+1 <= fluc_upper && dist_0 < range) fabr_0 -= mean_fluc_data[ind_0];
		  }
		  if(i==0&&j==0) local_count[offset]+=1.0;
		  local_corr[offset][ipair] += fabr_t*fabr_0;
		  local_corr_err[offset][ipair] += fabr_t*fabr_0*fabr_t*fabr_0;
		}
	      } else { //no variable dependency
		// since atoms are renamed, you have to access tag[i]
		double val0 = array[indb][j * nsave + n];
		double valt = array[inda][i * nsave + m];
		//if (a==0) printf("%f %f\n",val0,valt);
		double cor = val0*valt;
		//printf("%d %f %f\n",n-m,valt,val0);
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
	      double t2 = MPI_Wtime();
  time_calc += t2 -t1;
	  }	
	}
      }
      ipair++;
    }
  }
  
  double t2 = MPI_Wtime();
  time_total += t2 -t1;
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
  res_data[0] = sgn(-proj1)*sqrt( F1_p[0]*F1_p[0] + F1_p[1]*F1_p[1] + F1_p[2]*F1_p[2] );
  res_data[1] = sgn(proj2)*sqrt( F2_p[0]*F2_p[0] + F2_p[1]*F2_p[1] + F2_p[2]*F2_p[2] );
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
  
  int ind;
  
  double delx, dely, delz, fabx, faby, fabz;
  double rsq, dist, fabr;

  int incr_nvalues = 1;
  if (variable_flag == DIST_DEPENDENED){
    incr_nvalues = 3;
  }

  for (i = 0; i < nvalues; i+=incr_nvalues) {
    for (a= 0; a < ngroup_glo; a++) {
      //determine whether just autocorrelation or also cross correlation (different atoms)
      double ngroup_lower = a;
      double ngroup_upper = a+1;
      if (type == CROSS || type == AUTOCROSS || type == UPPERCROSS){
	ngroup_lower = a;
	ngroup_upper = ngroup_glo;
      }
      for (b = ngroup_lower; b < ngroup_upper; b++) {
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
	  delx = variable_store[inda][lastindex]-variable_store[indb][lastindex]  ;
	  dely = variable_store[inda][lastindex+nsave]-variable_store[indb][lastindex+nsave]  ;
	  delz =  variable_store[inda][lastindex+2*nsave]-variable_store[indb][lastindex+2*nsave];
	  domain->minimum_image(delx,dely,delz);
	  rsq = delx*delx + dely*dely + delz*delz;
	  dist = sqrt (rsq);

	  //if (dist < 2.0) printf("a %d b %d dist %f\n",a,b,dist);
	  
	  if(dist<range){
	    
	    if (cross_flag ==0) {
		   if (memory_switch==PERPAIR || (memory_switch == PERGROUP_PERPAIR && i >= nvalues_pg)) {
		    fabx = array[inda*ngroup_glo+indb][ i*nsave + lastindex];
		    faby = array[inda*ngroup_glo+indb][ i*nsave + nsave + lastindex];
		    fabz = array[inda*ngroup_glo+indb][ i*nsave + 2*nsave + lastindex];
		   } else {
		    fabx = array[inda][ i*nsave + lastindex];
		    faby = array[inda][ i*nsave + nsave + lastindex];
		    fabz = array[inda][ i*nsave + 2*nsave + lastindex];
		   }
		  } else {
		   if (memory_switch==PERPAIR || (memory_switch == PERGROUP_PERPAIR && i >= nvalues_pg)) {
		    fabx = array[inda*ngroup_glo+indb][ i*nsave + lastindex];
		    faby = array[inda*ngroup_glo+indb][ i*nsave + nsave + lastindex];
		    fabz = array[inda*ngroup_glo+indb][ i*nsave + 2*nsave + lastindex];
		   } else {
		    fabx = array[inda][ i*nsave + lastindex] - array[indb][ i*nsave + lastindex];
		    faby = array[inda][ i*nsave + nsave + lastindex] - array[indb][ i*nsave + nsave + lastindex];
		    fabz = array[inda][ i*nsave + 2*nsave + lastindex] - array[indb][ i*nsave + 2*nsave + lastindex];
		   }
		  }

	    //if (dist >3.4) printf("mean calc: %f %f %f %f\n",dist,fabx,faby,fabz);
	    // calculate radial component of the forces/velocities (mapped on the distance vector)
	    fabr = (fabx*delx + faby*dely + fabz*delz) / dist;
	    //delfx_p = frp * delx / rsq;
	    //delfy_p = frp * dely / rsq;
	    //delfz_p = frp * delz / rsq;
	    // calculate correlation
	    ind = dist/range*bins;
	    
	    if(i==0) mean_count[ind]+=1.0;
	    mean[ind*nvalues+i] += fabr;
	  }
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

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixAveCorrelatePeratom::copy_arrays(int i, int j, int delflag)
{
  body[j] = body[i];
  
  /*int offset= 0;
  for (int m= 0; m < nvalues; m++) {
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
  }*/
}

/* --------------------------------------------------------------------- */

int FixAveCorrelatePeratom::pack_exchange(int i, double* buf) {
  int offset= 0;
  if (memory_switch == ATOM) {
    buf[0] = ubuf(body[i]).d;
    offset++;
  }
  
  for (int m= 0; m < nvalues ; m++) {
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
  if (memory_switch == ATOM) {
    body[nlocal] = (int) ubuf(buf[0]).i;
    offset++;
  }
  
  for (int m= 0; m < nvalues ; m++) {
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
  bytes = atoms * (nvalues +variable_nvalues) * nsave * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate atom-based array
------------------------------------------------------------------------- */

void FixAveCorrelatePeratom::grow_arrays(int nmax) {
  memory->create(array,nmax,(nvalues )*nsave,"fix_ave/correlate/peratom:array");
  if (variable_flag == VAR_DEPENDENED || variable_flag == DIST_DEPENDENED) memory->create(variable_store,ngroup_glo,nsave*variable_nvalues,"fix_ave/correlate/peratom:variable_store");
  array_atom = array;
  if (array) vector_atom = array[0];
  else vector_atom = NULL;
}

/* ----------------------------------------------------------------------
   write data into restart file:
   - correlation
------------------------------------------------------------------------- */
void FixAveCorrelatePeratom::write_restart(FILE *fp){
  // calculate size of array
  int ncount = corr_length;
  int ncorr = corr_length*npair;
  // count and correlation
  int n = ncount+ 2*ncorr;


  if (mean_flag) n += bins*nvalues + bins;

  //write data
  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    // count and correlation
    fwrite(save_count,sizeof(double),ncount,fp);
    fwrite(&save_corr[0][0],sizeof(double),ncorr,fp);
    fwrite(&save_corr_err[0][0],sizeof(double),ncorr,fp);

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

/* ----------------------------------------------------------------------
   compute the center-of-mass coords of group of atoms, with body_index index
   masstotal = total mass
   return center-of-mass coords in cm[]
   must unwrap atoms to compute center-of-mass correctly
------------------------------------------------------------------------- */

void FixAveCorrelatePeratom::xcm(int index, double masstotal, double *cm)
{

  double **x = atom->x;
  int *mask = atom->mask;
  int *type = atom->type;
  imageint *image = atom->image;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;

  double cmone[3];
  cmone[0] = cmone[1] = cmone[2] = 0.0;

  double massone;
  double unwrap[3];

  if (rmass) {
    for (int i = 0; i < nlocal; i++)
      if (body[i] == index) {
        massone = rmass[i];
        domain->unmap(x[i],image[i],unwrap);
        cmone[0] += unwrap[0] * massone;
        cmone[1] += unwrap[1] * massone;
        cmone[2] += unwrap[2] * massone;
      }
  } else {
    for (int i = 0; i < nlocal; i++)
      if (body[i] == index) {
        massone = mass[type[i]];
        domain->unmap(x[i],image[i],unwrap);
        cmone[0] += unwrap[0] * massone;
        cmone[1] += unwrap[1] * massone;
        cmone[2] += unwrap[2] * massone;
      }
  }

  MPI_Allreduce(cmone,cm,3,MPI_DOUBLE,MPI_SUM,world);
  if (masstotal > 0.0) {
    cm[0] /= masstotal;
    cm[1] /= masstotal;
    cm[2] /= masstotal;
  }
}

