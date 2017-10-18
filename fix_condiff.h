/*
 * fix_condiff.h
 *
 *  Created on: Oct 24, 2016
 *      Author: skasper
 */

#ifdef FIX_CLASS

FixStyle(condiff, FixCondiff)

#else

#ifndef LMP_FIX_CONDIFF_H
#define LMP_FIX_CONDIFF_H

#include "lmptype.h"
#include <mpi.h>

#ifdef FFT_SINGLE
typedef float FFT_SCALAR;
#define MPI_FFT_SCALAR MPI_FLOAT
#else
typedef double FFT_SCALAR;
#define MPI_FFT_SCALAR MPI_DOUBLE
#endif

#include "fix.h"

namespace LAMMPS_NS {

class FixCondiff : public Fix {

public:
    FixCondiff(class LAMMPS*, int, char**);
    ~FixCondiff();
    int setmask();
    void post_force(int);
    void final_integrate();
    void kspace_check();
    void pppm_check();

protected:
    void setup();
    virtual void assign_vf();
    void reassign_vf();
    void euler_step();
    void deallocate();
    void allocate();
    void compute_rho1d(const FFT_SCALAR&, const FFT_SCALAR&,
        const FFT_SCALAR&);
    void compute_rho_coeff();
    void compute_drho1d(const FFT_SCALAR&, const FFT_SCALAR&, const FFT_SCALAR&);
    void set_grid_local();
    void setup_grid();

    FFT_SCALAR*** density_brick_velocity_x;
    FFT_SCALAR*** density_brick_velocity_y;
    FFT_SCALAR*** density_brick_velocity_z;
    FFT_SCALAR*** density_brick_counter_x;

    FFT_SCALAR*** density_brick_force_x;
    FFT_SCALAR*** density_brick_force_y;
    FFT_SCALAR*** density_brick_force_z;

    double* rand;

    FFT_SCALAR **rho1d, **rho_coeff, **drho1d, **drho_coeff;

    int order, minorder, order_allocated;

    int jgroup, groupbit_condiff;

    double T;
    double D;
    int seed;

    double* boxlo;
    double shift;

    double dt;

    double wienerConst;

    int me, nprocs;

    double delxinv, delyinv, delzinv, delvolinv; //setup
    double volume;

    double shiftone; //make_rho
    int nlower, nupper;

    int nx_pppm, ny_pppm, nz_pppm;
    int nxlo_in, nylo_in, nzlo_in, nxhi_in, nyhi_in, nzhi_in;
    int nxlo_out, nylo_out, nzlo_out, nxhi_out, nyhi_out, nzhi_out;
    int ngrid;

    class RanMars* random;

    int nmax;

    FILE* file;
};
}

#endif
#endif
