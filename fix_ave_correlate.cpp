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
#include "fix_ave_correlate.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "atom.h"
#include "comm.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{COMPUTE,FIX,VARIABLE};
enum{ONE,RUNNING};
enum{AUTO,UPPER,LOWER,AUTOUPPER,AUTOLOWER,FULL};

#define INVOKED_SCALAR 1
#define INVOKED_VECTOR 2
#define INVOKED_ARRAY 4
#define INVOKED_PERATOM 8

/* ---------------------------------------------------------------------- */

FixAveCorrelate::FixAveCorrelate(LAMMPS * lmp, int narg, char **arg):
  Fix (lmp, narg, arg)
{
  if (narg < 7) error->all(FLERR,"Illegal fix ave/correlate command");

  MPI_Comm_rank(world,&me);

  nevery = force->inumeric(FLERR,arg[3]);
  nrepeat = force->inumeric(FLERR,arg[4]);
  nfreq = force->inumeric(FLERR,arg[5]);

  global_freq = nfreq;
  // parse values until one isn't recognized

  which = new int[narg-6];
  argindex = new int[narg-6];
  ids = new char*[narg-6];
  value2index = new int[narg-6];
  peratom = new int[narg-6]; //indicates which is a peratom quantity
  indices= new int[narg-6]; //indicates where in array or scalar_values the data is
  n_peratom= 0;
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
          error->all(FLERR,"Illegal fix ave/correlate command");
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
  overwrite = 0;
  char *title1 = NULL;
  char *title2 = NULL;
  char *title3 = NULL;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"type") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate command");
      if (strcmp(arg[iarg+1],"auto") == 0) type = AUTO;
      else if (strcmp(arg[iarg+1],"upper") == 0) type = UPPER;
      else if (strcmp(arg[iarg+1],"lower") == 0) type = LOWER;
      else if (strcmp(arg[iarg+1],"auto/upper") == 0) type = AUTOUPPER;
      else if (strcmp(arg[iarg+1],"auto/lower") == 0) type = AUTOLOWER;
      else if (strcmp(arg[iarg+1],"full") == 0) type = FULL;
      else error->all(FLERR,"Illegal fix ave/correlate command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"ave") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate command");
      if (strcmp(arg[iarg+1],"one") == 0) ave = ONE;
      else if (strcmp(arg[iarg+1],"running") == 0) ave = RUNNING;
      else error->all(FLERR,"Illegal fix ave/correlate command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"start") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate command");
      startstep = force->inumeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"prefactor") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate command");
      prefactor = force->numeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"file") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate command");
      if (me == 0) {
        fp = fopen(arg[iarg+1],"w");
        if (fp == NULL) {
          char str[128];
          sprintf(str,"Cannot open fix ave/correlate file %s",arg[iarg+1]);
          error->one(FLERR,str);
        }
      }
      iarg += 2;
    } else if (strcmp(arg[iarg],"overwrite") == 0) {
      overwrite = 1;
      iarg += 1;
    } else if (strcmp(arg[iarg],"title1") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate command");
      delete [] title1;
      int n = strlen(arg[iarg+1]) + 1;
      title1 = new char[n];
      strcpy(title1,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"title2") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate command");
      delete [] title2;
      int n = strlen(arg[iarg+1]) + 1;
      title2 = new char[n];
      strcpy(title2,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"title3") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate command");
      delete [] title3;
      int n = strlen(arg[iarg+1]) + 1;
      title3 = new char[n];
      strcpy(title3,arg[iarg+1]);
      iarg += 2;
    } else error->all(FLERR,"Illegal fix ave/correlate command");
  }

  // setup and error check
  // for fix inputs, check that fix frequency is acceptable

  if (nevery <= 0 || nrepeat <= 0 || nfreq <= 0)
    error->all(FLERR,"Illegal fix ave/correlate command");
  if (nfreq % nevery)
    error->all(FLERR,"Illegal fix ave/correlate command");
  if (ave == ONE && nfreq < (nrepeat-1)*nevery)
    error->all(FLERR,"Illegal fix ave/correlate command");
  if (ave != RUNNING && overwrite)
    error->all(FLERR,"Illegal fix ave/correlate command");
  int n_scalar= 0;
  for (int i = 0; i < nvalues; i++) {
    if (which[i] == COMPUTE) {
      int icompute = modify->find_compute(ids[i]);
      //no such compute
      if (icompute < 0)
        error->all(FLERR,"Compute ID for fix ave/correlate does not exist");

      if (argindex[i] == 0) { //allegedly a scalar
	if (modify->compute[icompute]->peratom_flag == 1) {
	  peratom[i]= 1;
	  indices[i]= n_peratom++;
	} else if (modify->compute[icompute]->scalar_flag == 1) {
	  peratom[i]= 0;
	  indices[i]= n_scalar++;
	} else {
	  error->all(FLERR, "Fix ave/correlate compute does not calculate either a global or peratom scalar");
	}
      } else {
	if (modify->compute[icompute]->peratom_flag == 1) {
	  if (argindex[i] > modify->compute[icompute]->size_peratom_cols) {
	    error->all(FLERR,"Fix ave/correlate compute vector is accessed out-of-range");
	  }
	  peratom[i]= 1;
	  indices[i]= n_peratom++;
	} else if (modify->compute[icompute]->vector_flag == 1) {
	  if (argindex[i] > modify->compute[icompute]->size_vector) {
	    error->all(FLERR,"Fix ave/correlate compute vector is accessed out-of-range");
	  }
	  peratom[i]= 0;
	  indices[i]= n_scalar++;
	} else {
	  error->all(FLERR, "Fix ave/correlate compute does not calculate either a global or peratom vector");
	}
      }
    } else if (which[i] == FIX) {
      int ifix = modify->find_fix(ids[i]);
      if (ifix < 0)
        error->all(FLERR,"Fix ID for fix ave/correlate does not exist");
      if (argindex[i] == 0) { //allegedly a scalar
	if (modify->fix[ifix]->peratom_flag == 1) {
	  peratom[i]= 1;
	  indices[i]= n_peratom++;
	} else if (modify->fix[ifix]->scalar_flag == 1) {
	  peratom[i]= 0;
	  indices[i]= n_scalar++;
	} else {
	  error->all(FLERR, "Fix ave/correlate fix does not calculate either a global or peratom scalar");
	}
      } else {
	if (modify->fix[ifix]->peratom_flag == 1) {
	  if (argindex[i] > modify->fix[ifix]->size_peratom_cols) {
	    error->all(FLERR,"Fix ave/correlate fix vector is accessed out-of-range");
	  }
	  peratom[i]= 1;
	  indices[i]= n_peratom++;
	} else if (modify->fix[ifix]->vector_flag == 1) {
	  if (argindex[i] > modify->fix[ifix]->size_vector) {
	    error->all(FLERR,"Fix ave/correlate fix vector is accessed out-of-range");
	  }
	  peratom[i]= 0;
	  indices[i]= n_scalar++;
	} else {
	  error->all(FLERR, "Fix ave/correlate fix does not calculate either a global or peratom vector");
	}
      }
      if (peratom[i] && nevery % modify->fix[ifix]->peratom_freq)
	error->all(FLERR,"Fix for fix ave/correlate "
                   "not computed at compatible time");
      else if (nevery % modify->fix[ifix]->global_freq)
        error->all(FLERR,"Fix for fix ave/correlate "
                   "not computed at compatible time");

    } else if (which[i] == VARIABLE) {
      int ivariable = input->variable->find(ids[i]);
      if (ivariable < 0)
        error->all(FLERR,"Variable name for fix ave/correlate does not exist");
      if (input->variable->atomstyle(ivariable) == 1) {
	peratom[i]= 1;
	indices[i]= n_peratom++;
      } else if (input->variable->equalstyle(ivariable) == 1) {
	peratom[i]= 0;
	indices[i]= n_scalar++;
      } else {
        error->all(FLERR, "Fix ave/correlate variable is not an equal- or atom-style variable");
      }
    }
  }

  // npair = # of correlation pairs to calculate

  if (type == AUTO) npair = nvalues;
  if (type == UPPER || type == LOWER) npair = nvalues*(nvalues-1)/2;
  if (type == AUTOUPPER || type == AUTOLOWER) npair = nvalues*(nvalues+1)/2;
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
      if (type == AUTO)
        for (int i = 0; i < nvalues; i++)
          fprintf(fp," %s*%s",arg[6+i],arg[6+i]);
      else if (type == UPPER)
        for (int i = 0; i < nvalues; i++)
          for (int j = i+1; j < nvalues; j++)
            fprintf(fp," %s*%s",arg[6+i],arg[6+j]);
      else if (type == LOWER)
        for (int i = 0; i < nvalues; i++)
          for (int j = 0; j <= i-1; j++)
            fprintf(fp," %s*%s",arg[6+i],arg[6+j]);
      else if (type == AUTOUPPER)
        for (int i = 0; i < nvalues; i++)
          for (int j = i; j < nvalues; j++)
            fprintf(fp," %s*%s",arg[6+i],arg[6+j]);
      else if (type == AUTOLOWER)
        for (int i = 0; i < nvalues; i++)
          for (int j = 0; j <= i; j++)
            fprintf(fp," %s*%s",arg[6+i],arg[6+j]);
      else if (type == FULL)
        for (int i = 0; i < nvalues; i++)
          for (int j = 0; j < nvalues; j++)
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

  memory->create(scalar_values, nrepeat, nvalues-n_peratom, "ave/correlate:values");
  memory->create(count,nrepeat,"ave/correlate:count");
  memory->create(save_count,nrepeat,"ave/correlate:save_count");
  memory->create(corr,nrepeat,npair,"ave/correlate:corr");
  memory->create(local_accum,nrepeat,"ave/correlate:local_accum");
  memory->create(global_accum,nrepeat,"ave/correlate:global_accum");
  memory->create(save_corr,nrepeat,npair,"ave/correlate:save_corr");

  int i,j;
  for (i = 0; i < nrepeat; i++) {
    save_count[i] = count[i] = local_accum[i] = global_accum[i] = 0;
    for (j = 0; j < npair; j++)
      save_corr[i][j] = corr[i][j] = 0.0;
  }

  // this fix produces a global array

  array_flag = 1;
  size_array_rows = nrepeat;
  size_array_cols = npair+2;
  extarray = 0;

  // nvalid = next step on which end_of_step does something
  // add nvalid to all computes that store invocation times
  // since don't know a priori which are invoked by this fix
  // once in end_of_step() can set timestep for ones actually invoked

  lastindex = -1;
  firstindex = 0;
  nsample = 0;
  nvalid = nextvalid();
  modify->addstep_compute_all(nvalid);
  
  array= NULL;
  if(n_peratom > 0) {
    grow_arrays(atom->nmax);
    atom->add_callback(0);
  }
}

/* ---------------------------------------------------------------------- */

FixAveCorrelate::~FixAveCorrelate()
{
  delete [] which;
  delete [] argindex;
  delete [] value2index;
  delete [] peratom;
  delete [] indices;
  for (int i = 0; i < nvalues; i++) delete [] ids[i];
  delete [] ids;

  memory->destroy(scalar_values);
  memory->destroy(array);
  memory->destroy(count);
  memory->destroy(save_count);
  memory->destroy(corr);
  memory->destroy(local_accum);
  memory->destroy(global_accum);
  memory->destroy(save_corr);
  
  if (fp && me == 0) fclose(fp);
}

/* ---------------------------------------------------------------------- */

int FixAveCorrelate::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAveCorrelate::init()
{
  // set current indices for all computes,fixes,variables

  for (int i = 0; i < nvalues; i++) {
    if (which[i] == COMPUTE) {
      int icompute = modify->find_compute(ids[i]);
      if (icompute < 0)
        error->all(FLERR,"Compute ID for fix ave/correlate does not exist");
      value2index[i] = icompute;

    } else if (which[i] == FIX) {
      int ifix = modify->find_fix(ids[i]);
      if (ifix < 0)
        error->all(FLERR,"Fix ID for fix ave/correlate does not exist");
      value2index[i] = ifix;

    } else if (which[i] == VARIABLE) {
      int ivariable = input->variable->find(ids[i]);
      if (ivariable < 0)
        error->all(FLERR,"Variable name for fix ave/correlate does not exist");
      value2index[i] = ivariable;
    }
  }

  // need to reset nvalid if nvalid < ntimestep b/c minimize was performed

  if (nvalid < update->ntimestep) {
    lastindex = -1;
    firstindex = 0;
    nsample = 0;
    nvalid = nextvalid();
    modify->addstep_compute_all(nvalid);
  }
}

/* ----------------------------------------------------------------------
   only does something if nvalid = current timestep
------------------------------------------------------------------------- */

void FixAveCorrelate::setup(int vflag)
{
  end_of_step();
}

/* ---------------------------------------------------------------------- */

void FixAveCorrelate::end_of_step()
{
  int i,j,m;
  double scalar;
  double *peratom_data;

  // skip if not step which requires doing something

  bigint ntimestep = update->ntimestep;
  if (ntimestep != nvalid) return;

  // accumulate results of computes,fixes,variables to origin
  // compute/fix/variable may invoke computes so wrap with clear/add

  modify->clearstep_compute();

  // lastindex = index in values ring of latest time sample

  lastindex++;
  if (lastindex == nrepeat) lastindex = 0;

  for (i = 0; i < nvalues; i++) {
    m = value2index[i];

    // invoke compute if not previously invoked

    if (which[i] == COMPUTE) {
      Compute *compute = modify->compute[m];

      if (peratom[i]) {
	if(!(compute->invoked_flag & INVOKED_PERATOM)) {
	  compute->compute_peratom();
	  compute->invoked_flag |= INVOKED_PERATOM;
	}
	if (argindex[i] == 0) {
	  peratom_data= compute->vector_atom;
	} else {
	  peratom_data= compute->array_atom[argindex[i]-1];
	}
      } else if (argindex[i] == 0) {
        if (!(compute->invoked_flag & INVOKED_SCALAR)) {
          compute->compute_scalar();
          compute->invoked_flag |= INVOKED_SCALAR;
        }
        scalar = compute->scalar;
      } else {
        if (!(compute->invoked_flag & INVOKED_VECTOR)) {
          compute->compute_vector();
          compute->invoked_flag |= INVOKED_VECTOR;
        }
        scalar = compute->vector[argindex[i]-1];
      }

    // access fix fields, guaranteed to be ready
    } else if (which[i] == FIX) {
      if (peratom[i] && argindex[i] == 0) 
	peratom_data= modify->fix[m]->vector_atom;
      else if(peratom[i] && argindex[i] > 0)
	peratom_data= modify->fix[m]->array_atom[i-1];
      else if (argindex[i] == 0)
        scalar = modify->fix[m]->compute_scalar();
      else
        scalar = modify->fix[m]->compute_vector(argindex[i]-1);

    // evaluate equal-style variable
    } else if (which[i] == VARIABLE && peratom[i]) {
      memory->create(peratom_data, atom->nlocal, "ave/correlation:peratom_data");
      input->variable->compute_atom(m, igroup, peratom_data, 1, 0);
    } else
      scalar = input->variable->compute_equal(m);
    
    
    if (!peratom[i]) {
      scalar_values[lastindex][indices[i]] = scalar;
    } else {
      int offset= indices[i]*nrepeat + lastindex;
      for(int j= 0; j < atom->nlocal; j++) {
	array[j][offset]= peratom_data[j];
      }
      //if this was done by an atom-style variable, we need to free the mem we allocated
      if (which[i] == VARIABLE && peratom[i]) {
	memory->destroy(peratom_data);
      }
    }
  }

  // fistindex = index in values ring of earliest time sample
  // nsample = number of time samples in values ring

  if (nsample < nrepeat) nsample++;
  else {
    firstindex++;
    if (firstindex == nrepeat) firstindex = 0;
  }

  nvalid += nevery;
  modify->addstep_compute(nvalid);

  // calculate all Cij() enabled by latest values

  accumulate();
  if (ntimestep % nfreq || ntimestep == 0) return;
  // save results in save_count and save_corr

  for (i = 0; i < nrepeat; i++) {
    save_count[i] = count[i];
    if (count[i])
      for (j = 0; j < npair; j++)
        save_corr[i][j] = prefactor*corr[i][j]/count[i];
    else
      for (j = 0; j < npair; j++)
        save_corr[i][j] = 0.0;
  }

  // output result to file

  if (fp && me == 0) {
    if (overwrite) fseek(fp,filepos,SEEK_SET);
    fprintf(fp,BIGINT_FORMAT " %d\n",ntimestep,nrepeat);
    for (i = 0; i < nrepeat; i++) {
      fprintf(fp,"%d %d %d",i+1,i*nevery,count[i]);
      if (count[i])
        for (j = 0; j < npair; j++)
          fprintf(fp," %g",prefactor*corr[i][j]/count[i]);
      else
        for (j = 0; j < npair; j++)
          fprintf(fp," 0.0");
      fprintf(fp,"\n");
    }
    fflush(fp);
    if (overwrite) {
      long fileend = ftell(fp);
      ftruncate(fileno(fp),fileend);
    }
  }

  // zero accumulation if requested
  // recalculate Cij(0)

  if (ave == ONE) {
    for (i = 0; i < nrepeat; i++) {
      count[i] = 0;
      for (j = 0; j < npair; j++)
        corr[i][j] = 0.0;
    }
    nsample = 1;
    accumulate();
  }
}

/* ----------------------------------------------------------------------
   accumulate correlation data using more recently added values
------------------------------------------------------------------------- */

void FixAveCorrelate::accumulate()
{
  int i,j,k,m,n,ipair;

  for (k = 0; k < nsample; k++) count[k]++;
  
  int nlocal= atom->nlocal;
  int *mask= atom->mask;

  if (type == AUTO) {
    m = n = lastindex;
    ipair = 0;
    for (i = 0; i < nvalues; i++) {
      if (peratom[i]) { //time correlation of peratom value (e.g. velocity)
	int peratom_extent= 0;
	for (j= 0; j < nlocal; j++) {
	  if(mask[j] & groupbit) {
	    peratom_extent++;
	    for (k = 0; k < nsample; k++) {
	      local_accum[k]+= array[j][indices[i] * nrepeat + m]*array[j][indices[i] * nrepeat + n];
	      m--;
	      if (m < 0) m = nrepeat-1;
	    }
	  }
	}
	// reduce the results from each proc to calculate the global correlation
	MPI_Allreduce(&peratom_extent, &peratom_extent, 1, MPI_INT, MPI_SUM, world);
	MPI_Allreduce(local_accum, global_accum, nsample, MPI_DOUBLE, MPI_SUM, world);
	for (k = 0; k < nsample; k++) {
	  global_accum[k]/= peratom_extent;
	  corr[k][ipair]+= global_accum[k];
	  local_accum[k] = global_accum[k] = 0;
	}
      }else{ //time correlation of global value (e.g. temperature)
	for (k = 0; k < nsample; k++) {
	  corr[k][ipair]+= scalar_values[m][indices[i]]*scalar_values[n][indices[i]];
	  m--;
	  if (m < 0) m = nrepeat-1;
	}
	
      }
      ipair++;
    }
  } else if (type == UPPER) {
    m = n = lastindex;
    for (k = 0; k < nsample; k++) {
      ipair = 0;
      for (i = 0; i < nvalues; i++)
        for (j = i+1; j < nvalues; j++) {
	  if(peratom[i] && peratom[j]) {
	    double accum= 0.0;
	    int peratom_extent= 0;
	    for (int l= 0; l < nlocal; l++) {
	      if(mask[l] & groupbit) {
		accum+= array[l][indices[i] * nrepeat + m]*array[l][indices[j] * nrepeat + n];
		peratom_extent++;
	      }
	    }
	    MPI_Allreduce(&accum, &accum, 1, MPI_DOUBLE, MPI_SUM, world);
	    MPI_Allreduce(&peratom_extent, &peratom_extent, 1, MPI_INT, MPI_SUM, world);
	    accum/= peratom_extent;
	    corr[k][ipair++]+= accum;
	  } else if (peratom[i]) {
	    double accum= 0.0;
	    int peratom_extent= 0;
	    for (int l= 0; l < nlocal; l++) {
	      if(mask[l] & groupbit) {
		accum+= array[l][indices[i] * nrepeat + m];
		peratom_extent++;
	      }
	    }
	    MPI_Allreduce(&accum, &accum, 1, MPI_DOUBLE, MPI_SUM, world);
	    MPI_Allreduce(&peratom_extent, &peratom_extent, 1, MPI_INT, MPI_SUM, world);
	    accum/= peratom_extent;
	    corr[k][ipair++]+= accum*scalar_values[indices[j]][n];
	  } else if (peratom[j]) {
	    double accum= 0.0;
	    int peratom_extent= 0;
	    for (int l= 0; l < nlocal; l++) {
	      if(mask[l] & groupbit) {
		accum+= array[l][indices[j] * nrepeat + m];
		peratom_extent++;
	      }
	    }
	    MPI_Allreduce(&accum, &accum, 1, MPI_DOUBLE, MPI_SUM, world);
	    MPI_Allreduce(&peratom_extent, &peratom_extent, 1, MPI_INT, MPI_SUM, world);
	    accum/= peratom_extent;
	    corr[k][ipair++]+= accum*scalar_values[indices[i]][n];
	  } else {
	    corr[k][ipair++]+= scalar_values[m][indices[i]]*scalar_values[n][indices[i]];
	  }
	}
      m--;
      if (m < 0) m = nrepeat-1;
    }
  } else if (type == LOWER) {
    m = n = lastindex;
    for (k = 0; k < nsample; k++) {
      ipair = 0;
      for (i = 0; i < nvalues; i++) {
        for (j = 0; j <= i-1; j++) {
	  if(peratom[i] && peratom[j]) {
	    double accum= 0.0;
	    int peratom_extent= 0;
	    for (int l= 0; l < nlocal; l++) {
	      if(mask[l] & groupbit) {
		accum+= array[l][indices[i] * nrepeat + m]*array[l][indices[j] * nrepeat + n];
		peratom_extent++;
	      }
	    }
	    MPI_Allreduce(&accum, &accum, 1, MPI_DOUBLE, MPI_SUM, world);
	    MPI_Allreduce(&peratom_extent, &peratom_extent, 1, MPI_INT, MPI_SUM, world);
	    accum/= peratom_extent;
	    corr[k][ipair++]+= accum;
	  } else if (peratom[i]) {
	    double accum= 0.0;
	    int peratom_extent= 0;
	    for (int l= 0; l < nlocal; l++) {
	      if(mask[l] & groupbit) {
		accum+= array[l][indices[i] * nrepeat + m];
		peratom_extent++;
	      }
	    }
	    MPI_Allreduce(&accum, &accum, 1, MPI_DOUBLE, MPI_SUM, world);
	    MPI_Allreduce(&peratom_extent, &peratom_extent, 1, MPI_INT, MPI_SUM, world);
	    accum/= peratom_extent;
	    corr[k][ipair++]+= accum*scalar_values[indices[j]][n];
	  } else if (peratom[j]) {
	    double accum= 0.0;
	    int peratom_extent= 0;
	    for (int l= 0; l < nlocal; l++) {
	      if(mask[l] & groupbit) {
		accum+= array[l][indices[j] * nrepeat + m];
		peratom_extent++;
	      }
	    }
	    MPI_Allreduce(&accum, &accum, 1, MPI_DOUBLE, MPI_SUM, world);
	    MPI_Allreduce(&peratom_extent, &peratom_extent, 1, MPI_INT, MPI_SUM, world);
	    accum/= peratom_extent;
	    corr[k][ipair++]+= accum*scalar_values[indices[i]][n];
	  } else {
	    corr[k][ipair++]+= scalar_values[m][indices[i]]*scalar_values[n][indices[i]];
	  }
	}
      }
      m--;
      if (m < 0) m = nrepeat-1;
    }
  } else if (type == AUTOUPPER) {
    m = n = lastindex;
    for (k = 0; k < nsample; k++) {
      ipair = 0;
      for (i = 0; i < nvalues; i++)
        for (j = i; j < nvalues; j++) {
	  if(peratom[i] && peratom[j]) {
	    double accum= 0.0;
	    int peratom_extent= 0;
	    for (int l= 0; l < nlocal; l++) {
	      if(mask[l] & groupbit) {
		accum+= array[l][indices[i] * nrepeat + m]*array[l][indices[j] * nrepeat + n];
		peratom_extent++;
	      }
	    }
	    MPI_Allreduce(&accum, &accum, 1, MPI_DOUBLE, MPI_SUM, world);
	    MPI_Allreduce(&peratom_extent, &peratom_extent, 1, MPI_INT, MPI_SUM, world);
	    accum/= peratom_extent;
	    corr[k][ipair++]+= accum;
	  } else if (peratom[i]) {
	    double accum= 0.0;
	    int peratom_extent= 0;
	    for (int l= 0; l < nlocal; l++) {
	      if(mask[l] & groupbit) {
		accum+= array[l][indices[i] * nrepeat + m];
		peratom_extent++;
	      }
	    }
	    MPI_Allreduce(&accum, &accum, 1, MPI_DOUBLE, MPI_SUM, world);
	    MPI_Allreduce(&peratom_extent, &peratom_extent, 1, MPI_INT, MPI_SUM, world);
	    accum/= peratom_extent;
	    corr[k][ipair++]+= accum*scalar_values[indices[j]][n];
	  } else if (peratom[j]) {
	    double accum= 0.0;
	    int peratom_extent= 0;
	    for (int l= 0; l < nlocal; l++) {
	      if(mask[l] & groupbit) {
		accum+= array[l][indices[j] * nrepeat + m];
		peratom_extent++;
	      }
	    }
	    MPI_Allreduce(&accum, &accum, 1, MPI_DOUBLE, MPI_SUM, world);
	    MPI_Allreduce(&peratom_extent, &peratom_extent, 1, MPI_INT, MPI_SUM, world);
	    accum/= peratom_extent;
	    corr[k][ipair++]+= accum*scalar_values[indices[i]][n];
	  } else {
	    corr[k][ipair++]+= scalar_values[m][indices[i]]*scalar_values[n][indices[i]];
	  }
	}
      m--;
      if (m < 0) m = nrepeat-1;
    }
  } else if (type == AUTOLOWER) {
    m = n = lastindex;
    for (k = 0; k < nsample; k++) {
      ipair = 0;
      for (i = 0; i < nvalues; i++)
        for (j = 0; j <= i; j++) {
	  if(peratom[i] && peratom[j]) {
	    double accum= 0.0;
	    int peratom_extent= 0;
	    for (int l= 0; l < nlocal; l++) {
	      if(mask[l] & groupbit) {
		accum+= array[l][indices[i] * nrepeat + m]*array[l][indices[j] * nrepeat + n];
		peratom_extent++;
	      }
	    }
	    MPI_Allreduce(&accum, &accum, 1, MPI_DOUBLE, MPI_SUM, world);
	    MPI_Allreduce(&peratom_extent, &peratom_extent, 1, MPI_INT, MPI_SUM, world);
	    accum/= peratom_extent;
	    corr[k][ipair++]+= accum;
	  } else if (peratom[i]) {
	    double accum= 0.0;
	    int peratom_extent= 0;
	    for (int l= 0; l < nlocal; l++) {
	      if(mask[l] & groupbit) {
		accum+= array[l][indices[i] * nrepeat + m];
		peratom_extent++;
	      }
	    }
	    MPI_Allreduce(&accum, &accum, 1, MPI_DOUBLE, MPI_SUM, world);
	    MPI_Allreduce(&peratom_extent, &peratom_extent, 1, MPI_INT, MPI_SUM, world);
	    accum/= peratom_extent;
	    corr[k][ipair++]+= accum*scalar_values[indices[j]][n];
	  } else if (peratom[j]) {
	    double accum= 0.0;
	    int peratom_extent= 0;
	    for (int l= 0; l < nlocal; l++) {
	      if(mask[l] & groupbit) {
		accum+= array[l][indices[j] * nrepeat + m];
		peratom_extent++;
	      }
	    }
	    MPI_Allreduce(&accum, &accum, 1, MPI_DOUBLE, MPI_SUM, world);
	    MPI_Allreduce(&peratom_extent, &peratom_extent, 1, MPI_INT, MPI_SUM, world);
	    accum/= peratom_extent;
	    corr[k][ipair++]+= accum*scalar_values[indices[i]][n];
	  } else {
	    corr[k][ipair++]+= scalar_values[m][indices[i]]*scalar_values[n][indices[i]];
	  }
	 }
      m--;
      if (m < 0) m = nrepeat-1;
    }
  } else if (type == FULL) {
    m = n = lastindex;
    for (k = 0; k < nsample; k++) {
      ipair = 0;
      for (i = 0; i < nvalues; i++)
        for (j = 0; j < nvalues; j++) {
	  if(peratom[i] && peratom[j]) {
	    double accum= 0.0;
	    int peratom_extent= 0;
	    for (int l= 0; l < nlocal; l++) {
	      if(mask[l] & groupbit) {
		accum+= array[l][indices[i] * nrepeat + m]*array[l][indices[j] * nrepeat + n];
		peratom_extent++;
	      }
	    }
	    MPI_Allreduce(&accum, &accum, 1, MPI_DOUBLE, MPI_SUM, world);
	    MPI_Allreduce(&peratom_extent, &peratom_extent, 1, MPI_INT, MPI_SUM, world);
	    accum/= peratom_extent;
	    corr[k][ipair++]+= accum;
	  } else if (peratom[i]) {
	    double accum= 0.0;
	    int peratom_extent= 0;
	    for (int l= 0; l < nlocal; l++) {
	      if(mask[l] & groupbit) {
		accum+= array[l][indices[i] * nrepeat + m];
		peratom_extent++;
	      }
	    }
	    MPI_Allreduce(&accum, &accum, 1, MPI_DOUBLE, MPI_SUM, world);
	    MPI_Allreduce(&peratom_extent, &peratom_extent, 1, MPI_INT, MPI_SUM, world);
	    accum/= peratom_extent;
	    corr[k][ipair++]+= accum*scalar_values[indices[j]][n];
	  } else if (peratom[j]) {
	    double accum= 0.0;
	    int peratom_extent= 0;
	    for (int l= 0; l < nlocal; l++) {
	      if(mask[l] & groupbit) {
		accum+= array[l][indices[j] * nrepeat + m];
		peratom_extent++;
	      }
	    }
	    MPI_Allreduce(&accum, &accum, 1, MPI_DOUBLE, MPI_SUM, world);
	    MPI_Allreduce(&peratom_extent, &peratom_extent, 1, MPI_INT, MPI_SUM, world);
	    accum/= peratom_extent;
	    corr[k][ipair++]+= accum*scalar_values[indices[i]][n];
	  } else {
	    corr[k][ipair++]+= scalar_values[m][indices[i]]*scalar_values[n][indices[i]];
	  }
	}
      m--;
      if (m < 0) m = nrepeat-1;
    }
  }
}

/* ----------------------------------------------------------------------
   return I,J array value
------------------------------------------------------------------------- */

double FixAveCorrelate::compute_array(int i, int j)
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

bigint FixAveCorrelate::nextvalid()
{
  bigint nvalid = update->ntimestep;
  if (startstep > nvalid) nvalid = startstep;
  if (nvalid % nevery) nvalid = (nvalid/nevery)*nevery + nevery;
  return nvalid;
}

/* ---------------------------------------------------------------------- */

void FixAveCorrelate::reset_timestep(bigint ntimestep)
{
  if (ntimestep > nvalid) error->all(FLERR,"Fix ave/correlate missed timestep");
}

/* --------------------------------------------------------------------- */

int FixAveCorrelate::pack_exchange(int i, double* buf) {
  int offset= 0;
  for (int m= 0; m < n_peratom; m++) {
    for (int k= 0; k < nsample; k++) {
      buf[offset] = array[i][offset];
      offset++;
    }
    for (int k= nsample; k < nrepeat; k++) {
      buf[offset++]= 0.0;
    }
  }
  return offset;
}

/* --------------------------------------------------------------------- */

int FixAveCorrelate::unpack_exchange(int nlocal, double* buf) {
  int offset= 0;
  for (int m= 0; m < n_peratom; m++) {
    for (int k= 0; k < nsample; k++) {
      array[nlocal][offset]= buf[offset];
      offset++;
    }
    for (int k= nsample; k < nrepeat; k++) {
      array[nlocal][offset++]= 0.0;
    }
  }
  return offset;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixAveCorrelate::memory_usage() {
  double bytes;
  bytes = atom->nmax* n_peratom * nrepeat * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate atom-based array
------------------------------------------------------------------------- */

void FixAveCorrelate::grow_arrays(int nmax) {
  memory->grow(array,nmax,n_peratom*nrepeat,"fix_ave/correlate:array");
  array_atom = array;
  if (array) vector_atom = array[0];
  else vector_atom = NULL;
}

/* ----------------------------------------------------------------------
   copy values within local atom-based array
------------------------------------------------------------------------- */

void FixAveCorrelate::copy_arrays(int i, int j, int delflag) {
  int offset= 0;
  for (int m= 0; m < n_peratom; m++) {
    for (int k= 0; k < nsample; k++) {
      array[j][offset] = array[i][offset];
      offset++;
    }
    for (int k= nsample; k < nrepeat; k++) {
      array[j][offset++]= 0.0;
    }
  }
}