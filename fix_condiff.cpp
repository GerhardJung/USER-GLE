#include "fix_condiff.h"
#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pppm.h"
#include "atom.h"
#include "neighbor.h"
#include "force.h"
#include "pair.h"
#include "domain.h"
#include "memory.h"
#include "error.h"
#include "group.h"
#include "update.h"
#include "random_mars.h"
#include "math_const.h"
#include "math_special.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;
using namespace MathSpecial;

#define OFFSET 16384
#ifdef FFT_SINGLE
#define ZEROF 0.0f
#define ONEF 1.0f
#else
#define ZEROF 0.0
#define ONEF 1.0
#endif

//The class constructor.
FixCondiff::FixCondiff(LAMMPS* lmp, int narg, char** arg)
    : Fix(lmp, narg, arg)
{
    MPI_Comm_rank(world, &me);

    if (narg < 4)
        error->all(FLERR, "Illegal fix condiff command"); //4 mandatory arguments

    density_brick_velocity_x = NULL;
    density_brick_velocity_y = NULL;
    density_brick_velocity_z = NULL;

    density_brick_force_x = NULL;
    density_brick_force_y = NULL;
    density_brick_force_z = NULL;

    density_brick_counter_x = NULL;

    rand = NULL;

    rho1d = rho_coeff = drho1d = drho_coeff = NULL;
    order = force->kspace->order;
    minorder = 2;
    order_allocated = order;

    //Set groupbit_condiff on condiff particles, so fix knows which particles to act on
    jgroup = group->find(arg[3]);
    if (jgroup == -1)
        error->all(FLERR, "Could not find fix condiff group ID");
    groupbit_condiff = group->bitmask[jgroup];

    kspace_check();
    pppm_check();

    //Standard values
    T = 1.0;
    D = 1.0;
    seed = 11111;

    //Optional args
    int iarg = 4;

    while (iarg < narg) {
        if (strcmp(arg[iarg], "temp") == 0) {
            if (iarg + 2 > narg)
                error->all(FLERR, "Illegal fix condiff command");
            T = atof(arg[iarg + 1]);
            if (T <= 0)
                error->all(FLERR, "Illegal fix condiff command");
            iarg += 2;
        }
        else if (strcmp(arg[iarg], "diff") == 0) {
            if (iarg + 2 > narg)
                error->all(FLERR, "Illegal fix condiff command");
            D = atof(arg[iarg + 1]);
            if (D <= 0)
                error->all(FLERR, "Illegal fix condiff command");
            iarg += 2;
        }
        else if (strcmp(arg[iarg], "seed") == 0) {
            if (iarg + 2 > narg)
                error->all(FLERR, "Illegal fix condiff command");
            seed = atoi(arg[iarg + 1]);
            if (seed <= 0)
                error->all(FLERR, "Illegal fix condiff command");
            iarg += 2;
        }
        else
            error->all(FLERR, "Illegal fix condiff command");
    }

    if (me == 0) {
        printf("Temp = %f\n", T);
        printf("Diffusion Coefficient = %f\n", D);
        printf("Seed = %i\n", seed);
    }

    //Random Number Generator
    random = new RanMars(lmp, seed);
}

//The class destructor
FixCondiff::~FixCondiff()
{
    deallocate();
    memory->destroy(density_brick_velocity_x);
    memory->destroy(density_brick_velocity_y);
    memory->destroy(density_brick_velocity_z);
    memory->destroy(density_brick_counter_x);
    memory->destroy(density_brick_force_x);
    memory->destroy(density_brick_force_y);
    memory->destroy(density_brick_force_z);
}

//Where algorithm steps in
int FixCondiff::setmask()
{
    int mask = 0;
    mask |= POST_FORCE;
    mask |= FINAL_INTEGRATE;
    return mask;
}

void FixCondiff::post_force(int vspace)
{
    setup();
    setup_grid();
    assign_vf();
    reassign_vf();
}

void FixCondiff::final_integrate()
{
    euler_step();
}

void FixCondiff::setup()
{
    nx_pppm = force->kspace->nx_pppm;
    ny_pppm = force->kspace->ny_pppm;
    nz_pppm = force->kspace->nz_pppm;

    double* prd;
    prd = domain->prd;

    double xprd = prd[0];
    double yprd = prd[1];
    double zprd = prd[2];
    double zprd_slab = zprd;
    volume = xprd * yprd * zprd_slab;

    delxinv = nx_pppm / xprd;
    delyinv = ny_pppm / yprd;
    delzinv = nz_pppm / zprd_slab;

    delvolinv = delxinv * delyinv * delzinv;
    dt = update->dt;
    wienerConst = sqrt(2 * D * dt);
}

//setup_grid() framework taken from pppm.cpp
void FixCondiff::setup_grid()
{
    //Free all arrays previously allocated
    deallocate();

    //Reset portion of global grid that each proc owns
    set_grid_local();
    allocate();
    compute_rho_coeff();
}

void FixCondiff::assign_vf()
{
    int l, m, n, nx, ny, nz, mx, my, mz;
    FFT_SCALAR dx, dy, dz, x0, y0, z0;

    //Clear 3d density array

    memset(&(density_brick_velocity_x[nzlo_out][nylo_out][nxlo_out]), 0,
        ngrid * sizeof(FFT_SCALAR));
    memset(&(density_brick_velocity_y[nzlo_out][nylo_out][nxlo_out]), 0,
        ngrid * sizeof(FFT_SCALAR));
    memset(&(density_brick_velocity_z[nzlo_out][nylo_out][nxlo_out]), 0,
        ngrid * sizeof(FFT_SCALAR));
    memset(&(density_brick_force_x[nzlo_out][nylo_out][nxlo_out]), 0,
        ngrid * sizeof(FFT_SCALAR));
    memset(&(density_brick_force_y[nzlo_out][nylo_out][nxlo_out]), 0,
        ngrid * sizeof(FFT_SCALAR));
    memset(&(density_brick_force_z[nzlo_out][nylo_out][nxlo_out]), 0,
        ngrid * sizeof(FFT_SCALAR));
    memset(&(density_brick_counter_x[nzlo_out][nylo_out][nxlo_out]), 0,
        ngrid * sizeof(FFT_SCALAR));

    //Loop over my velocities, add their contribution to nearby grid points
    //(nx,ny,nz) = global coords of grid pt to "lower left" of charge
    //(dx,dy,dz) = distance to "lower left" grid pt
    //(mx,my,mz) = global coords of moving stencil pt

    double** v = atom->v;
    double** x = atom->x;
    double** f = atom->f;
    int nlocal = atom->nlocal;
    int* mask = atom->mask;

    for (int i = 0; i < nlocal; i++) {

        //Take velocity of dpd-particles and map them on grid
        if (mask[i] & groupbit) {
            nx = static_cast<int>((x[i][0] - boxlo[0]) * delxinv + shift) - OFFSET;
            ny = static_cast<int>((x[i][1] - boxlo[1]) * delyinv + shift) - OFFSET;
            nz = static_cast<int>((x[i][2] - boxlo[2]) * delzinv + shift) - OFFSET;
            dx = nx + shiftone - (x[i][0] - boxlo[0]) * delxinv;
            dy = ny + shiftone - (x[i][1] - boxlo[1]) * delyinv;
            dz = nz + shiftone - (x[i][2] - boxlo[2]) * delzinv;

            compute_rho1d(dx, dy, dz);

            z0 = delvolinv;
            for (n = nlower; n <= nupper; n++) {
                mz = n + nz;
                y0 = z0 * rho1d[2][n];
                for (m = nlower; m <= nupper; m++) {
                    my = m + ny;
                    x0 = y0 * rho1d[1][m];
                    for (l = nlower; l <= nupper; l++) {
                        mx = l + nx;
                        density_brick_velocity_x[mz][my][mx] += x0 * rho1d[0][l] * v[i][0];
                        density_brick_velocity_y[mz][my][mx] += x0 * rho1d[0][l] * v[i][1];
                        density_brick_velocity_z[mz][my][mx] += x0 * rho1d[0][l] * v[i][2];
                        density_brick_counter_x[mz][my][mx] += x0 * rho1d[0][l];
                    }
                }
            }
        }

        //Take force of condiff-particles and map them on grid
        if (mask[i] & groupbit_condiff) {
            nx = static_cast<int>((x[i][0] - boxlo[0]) * delxinv + shift) - OFFSET;
            ny = static_cast<int>((x[i][1] - boxlo[1]) * delyinv + shift) - OFFSET;
            nz = static_cast<int>((x[i][2] - boxlo[2]) * delzinv + shift) - OFFSET;
            dx = nx + shiftone - (x[i][0] - boxlo[0]) * delxinv;
            dy = ny + shiftone - (x[i][1] - boxlo[1]) * delyinv;
            dz = nz + shiftone - (x[i][2] - boxlo[2]) * delzinv;

            compute_rho1d(dx, dy, dz);

            z0 = delvolinv;
            for (n = nlower; n <= nupper; n++) {
                mz = n + nz;
                y0 = z0 * rho1d[2][n];
                for (m = nlower; m <= nupper; m++) {
                    my = m + ny;
                    x0 = y0 * rho1d[1][m];
                    for (l = nlower; l <= nupper; l++) {
                        mx = l + nx;
                        density_brick_force_x[mz][my][mx] += x0 * rho1d[0][l] * f[i][0];
                        density_brick_force_y[mz][my][mx] += x0 * rho1d[0][l] * f[i][1];
                        density_brick_force_z[mz][my][mx] += x0 * rho1d[0][l] * f[i][2];
                    }
                }
            }
        }
    }
}

void FixCondiff::reassign_vf()
{

    int l, m, n, nx, ny, nz, mx, my, mz;
    FFT_SCALAR dx, dy, dz, x0, y0, z0;

    double** v = atom->v;
    double** x = atom->x;
    double** f = atom->f;
    int nlocal = atom->nlocal;
    int* mask = atom->mask;

    for (int i = 0; i < nlocal; i++) {

        //Remap velocities on condiff-particles (pseudo-ions)
        if (mask[i] & groupbit_condiff) {
            nx = static_cast<int>((x[i][0] - boxlo[0]) * delxinv + shift) - OFFSET;
            ny = static_cast<int>((x[i][1] - boxlo[1]) * delyinv + shift) - OFFSET;
            nz = static_cast<int>((x[i][2] - boxlo[2]) * delzinv + shift) - OFFSET;
            dx = nx + shiftone - (x[i][0] - boxlo[0]) * delxinv;
            dy = ny + shiftone - (x[i][1] - boxlo[1]) * delyinv;
            dz = nz + shiftone - (x[i][2] - boxlo[2]) * delzinv;

            compute_rho1d(dx, dy, dz);

            v[i][0] = 0;
            v[i][1] = 0;
            v[i][2] = 0;

            for (n = nlower; n <= nupper; n++) {
                mz = n + nz;
                z0 = rho1d[2][n];
                for (m = nlower; m <= nupper; m++) {
                    my = m + ny;
                    y0 = z0 * rho1d[1][m];
                    for (l = nlower; l <= nupper; l++) {
                        mx = l + nx;
                        x0 = y0 * rho1d[0][l];
                        if (density_brick_counter_x[mz][my][mx] != 0) {
                            v[i][0] += (density_brick_velocity_x[mz][my][mx] * x0 / density_brick_counter_x[mz][my][mx]);
                            v[i][1] += (density_brick_velocity_y[mz][my][mx] * x0 / density_brick_counter_x[mz][my][mx]);
                            v[i][2] += (density_brick_velocity_z[mz][my][mx] * x0 / density_brick_counter_x[mz][my][mx]);
                        }
                    }
                }
            }
        }

        //Assign (normalized) force of pseudo-ions to dpd-particles
        if (mask[i] & groupbit) {
            nx = static_cast<int>((x[i][0] - boxlo[0]) * delxinv + shift) - OFFSET;
            ny = static_cast<int>((x[i][1] - boxlo[1]) * delyinv + shift) - OFFSET;
            nz = static_cast<int>((x[i][2] - boxlo[2]) * delzinv + shift) - OFFSET;
            dx = nx + shiftone - (x[i][0] - boxlo[0]) * delxinv;
            dy = ny + shiftone - (x[i][1] - boxlo[1]) * delyinv;
            dz = nz + shiftone - (x[i][2] - boxlo[2]) * delzinv;

            compute_rho1d(dx, dy, dz);

            for (n = nlower; n <= nupper; n++) {
                mz = n + nz;
                z0 = rho1d[2][n];
                for (m = nlower; m <= nupper; m++) {
                    my = m + ny;
                    y0 = z0 * rho1d[1][m];
                    for (l = nlower; l <= nupper; l++) {
                        mx = l + nx;
                        x0 = y0 * rho1d[0][l];
                        if (density_brick_counter_x[mz][my][mx] != 0) {
                            f[i][0] += density_brick_force_x[mz][my][mx] * x0 / density_brick_counter_x[mz][my][mx];
                            f[i][1] += density_brick_force_y[mz][my][mx] * x0 / density_brick_counter_x[mz][my][mx];
                            f[i][2] += density_brick_force_z[mz][my][mx] * x0 / density_brick_counter_x[mz][my][mx];
                        }
                    }
                }
            }
        }
    }
}

void FixCondiff::euler_step()
{
    double** v = atom->v;
    double** x = atom->x;
    double** f = atom->f;
    int nlocal = atom->nlocal;
    int* mask = atom->mask;

    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit_condiff) {
            rand[0] = 2*random->uniform()-1;//gaussian()
            rand[1] = 2*random->uniform()-1;//gaussian()
            rand[2] = 2*random->uniform()-1;//gaussian()
	    //file = fopen("charge_density.cor", "a");
            //fprintf(file, "%f\t%f\t%f\n", rand[0], rand[1], rand[2]);
            //fclose(file);
            x[i][0] += v[i][0] * dt + f[i][0] * dt * D / T + wienerConst * rand[0];
            x[i][1] += v[i][1] * dt + f[i][1] * dt * D / T + wienerConst * rand[1];
            x[i][2] += v[i][2] * dt + f[i][2] * dt * D / T + wienerConst * rand[2];
        }
    }
}

//compute_rho1d framework taken from pppm.cpp
void FixCondiff::compute_rho1d(const FFT_SCALAR& dx, const FFT_SCALAR& dy,
    const FFT_SCALAR& dz)
{
    int k, l;
    FFT_SCALAR r1, r2, r3;

    for (k = (1 - order) / 2; k <= order / 2; k++) {
        r1 = r2 = r3 = ZEROF;

        for (l = order - 1; l >= 0; l--) {
            r1 = rho_coeff[l][k] + r1 * dx;
            r2 = rho_coeff[l][k] + r2 * dy;
            r3 = rho_coeff[l][k] + r3 * dz;
        }
        rho1d[0][k] = r1;
        rho1d[1][k] = r2;
        rho1d[2][k] = r3;
    }
}

//compute_rho_coeff framework taken from pppm.cpp
void FixCondiff::compute_rho_coeff()
{
    int j, k, l, m;
    FFT_SCALAR s;

    FFT_SCALAR** a;

    memory->create2d_offset(a, order, -order, order, "condiff:a");

    for (k = -order; k <= order; k++)
        for (l = 0; l < order; l++)
            a[l][k] = 0.0;

    a[0][0] = 1.0;
    for (j = 1; j < order; j++) {
        for (k = -j; k <= j; k += 2) {
            s = 0.0;
            for (l = 0; l < j; l++) {
                a[l + 1][k] = (a[l][k + 1] - a[l][k - 1]) / (l + 1);
#ifdef FFT_SINGLE
                s += powf(0.5, (float)l + 1) * (a[l][k - 1] + powf(-1.0, (float)l) * a[l][k + 1]) / (l + 1);
#else
                s += pow(0.5, (double)l + 1) * (a[l][k - 1] + pow(-1.0, (double)l) * a[l][k + 1]) / (l + 1);
#endif
            }
            a[0][k] = s;
        }
    }

    m = (1 - order) / 2;
    for (k = -(order - 1); k < order; k += 2) {
        for (l = 0; l < order; l++) {
            rho_coeff[l][m] = a[l][k];
        }
        for (l = 1; l < order; l++)
            drho_coeff[l - 1][m] = l * a[l][k];
        m++;
    }

    memory->destroy2d_offset(a, -order);
}

//compute_drho1d framework taken from pppm.cpp
void FixCondiff::compute_drho1d(const FFT_SCALAR& dx, const FFT_SCALAR& dy,
    const FFT_SCALAR& dz)
{
    int k, l;
    FFT_SCALAR r1, r2, r3;

    for (k = (1 - order) / 2; k <= order / 2; k++) {
        r1 = r2 = r3 = ZEROF;

        for (l = order - 2; l >= 0; l--) {
            r1 = drho_coeff[l][k] + r1 * dx;
            r2 = drho_coeff[l][k] + r2 * dy;
            r3 = drho_coeff[l][k] + r3 * dz;
        }
        drho1d[0][k] = r1;
        drho1d[1][k] = r2;
        drho1d[2][k] = r3;
    }
}

//set_grid_local() framework taken from pppm.cpp
void FixCondiff::set_grid_local()
{
    // global indices of PPPM grid range from 0 to N-1
    // nlo_in,nhi_in = lower/upper limits of the 3d sub-brick of
    // global PPPM grid that I own without ghost cells
    // for slab PPPM, assign z grid as if it were not extended

    nxlo_in = static_cast<int>(comm->xsplit[comm->myloc[0]] * nx_pppm);
    nxhi_in = static_cast<int>(comm->xsplit[comm->myloc[0] + 1] * nx_pppm) - 1;

    nylo_in = static_cast<int>(comm->ysplit[comm->myloc[1]] * ny_pppm);
    nyhi_in = static_cast<int>(comm->ysplit[comm->myloc[1] + 1] * ny_pppm) - 1;

    nzlo_in = static_cast<int>(comm->zsplit[comm->myloc[2]] * nz_pppm);
    nzhi_in = static_cast<int>(comm->zsplit[comm->myloc[2] + 1] * nz_pppm) - 1;

    // nlower,nupper = stencil size for mapping particles to PPPM grid

    nlower = -(order - 1) / 2;
    nupper = order / 2;

    // shift values for particle <-> grid mapping
    // add/subtract OFFSET to avoid int(-0.75) = 0 when want it to be -1

    if (order % 2)
        shift = OFFSET + 0.5;
    else
        shift = OFFSET;
    if (order % 2)
        shiftone = 0.0;
    else
        shiftone = 0.5;

    // nlo_out,nhi_out = lower/upper limits of the 3d sub-brick of
    // global PPPM grid that my particles can contribute charge to
    // effectively nlo_in,nhi_in + ghost cells
    // nlo,nhi = global coords of grid pt to "lower left" of smallest/largest
    // position a particle in my box can be at
    // dist[3] = particle position bound = subbox + skin/2.0 + qdist
    // qdist = offset due to TIP4P fictitious charge
    // convert to triclinic if necessary
    // nlo_out,nhi_out = nlo,nhi + stencil size for particle mapping
    // for slab PPPM, assign z grid as if it were not extended

    double *prd, *sublo, *subhi;

    prd = domain->prd;
    boxlo = domain->boxlo;
    sublo = domain->sublo;
    subhi = domain->subhi;

    double xprd = prd[0];
    double yprd = prd[1];
    double zprd = prd[2];
    double zprd_slab = zprd;
    double qdist = 0.0;
    double dist[3];
    double cuthalf = 0.5 * neighbor->skin + qdist;
    dist[0] = dist[1] = dist[2] = cuthalf;

    int nlo, nhi;

    nlo = static_cast<int>((sublo[0] - dist[0] - boxlo[0]) * nx_pppm / xprd + shift) - OFFSET;
    nhi = static_cast<int>((subhi[0] + dist[0] - boxlo[0]) * nx_pppm / xprd + shift) - OFFSET;
    nxlo_out = nlo + nlower;
    nxhi_out = nhi + nupper;

    nlo = static_cast<int>((sublo[1] - dist[1] - boxlo[1]) * ny_pppm / yprd + shift) - OFFSET;
    nhi = static_cast<int>((subhi[1] + dist[1] - boxlo[1]) * ny_pppm / yprd + shift) - OFFSET;
    nylo_out = nlo + nlower;
    nyhi_out = nhi + nupper;

    nlo = static_cast<int>((sublo[2] - dist[2] - boxlo[2]) * nz_pppm / zprd_slab + shift) - OFFSET;
    nhi = static_cast<int>((subhi[2] + dist[2] - boxlo[2]) * nz_pppm / zprd_slab + shift) - OFFSET;
    nzlo_out = nlo + nlower;
    nzhi_out = nhi + nupper;

    if (force->kspace->stagger_flag) {
        nxhi_out++;
        nyhi_out++;
        nzhi_out++;
    }
    // PPPM grid pts owned by this proc, including ghosts

    ngrid = (nxhi_out - nxlo_out + 1) * (nyhi_out - nylo_out + 1) * (nzhi_out - nzlo_out + 1);
}

//deallocate() framework taken from pppm.cpp
void FixCondiff::deallocate()
{
    memory->destroy3d_offset(density_brick_velocity_x, nzlo_out, nylo_out, nxlo_out);
    memory->destroy3d_offset(density_brick_counter_x, nzlo_out, nylo_out, nxlo_out);
    memory->destroy3d_offset(density_brick_force_x, nzlo_out, nylo_out, nxlo_out);
    memory->destroy3d_offset(density_brick_velocity_y, nzlo_out, nylo_out, nxlo_out);

    memory->destroy3d_offset(density_brick_force_y, nzlo_out, nylo_out, nxlo_out);
    memory->destroy3d_offset(density_brick_velocity_z, nzlo_out, nylo_out, nxlo_out);

    memory->destroy3d_offset(density_brick_force_z, nzlo_out, nylo_out, nxlo_out);

    memory->destroy2d_offset(rho1d, -order_allocated / 2);
    memory->destroy2d_offset(drho1d, -order_allocated / 2);
    memory->destroy2d_offset(rho_coeff, (1 - order_allocated) / 2);
    memory->destroy2d_offset(drho_coeff, (1 - order_allocated) / 2);

    memory->destroy(rand);
}

//allocate() framework taken from pppm.cpp
void FixCondiff::allocate()
{
    memory->create3d_offset(density_brick_velocity_x, nzlo_out, nzhi_out, nylo_out, nyhi_out,
        nxlo_out, nxhi_out, "condiff:density_brick_velocity_x");
    memory->create3d_offset(density_brick_counter_x, nzlo_out, nzhi_out, nylo_out, nyhi_out,
        nxlo_out, nxhi_out, "condiff:density_brick_counter_x");
    memory->create3d_offset(density_brick_force_x, nzlo_out, nzhi_out, nylo_out, nyhi_out,
        nxlo_out, nxhi_out, "condiff:density_brick_force_x");
    memory->create3d_offset(density_brick_velocity_y, nzlo_out, nzhi_out, nylo_out, nyhi_out,
        nxlo_out, nxhi_out, "condiff:density_brick_velocity_y");

    memory->create3d_offset(density_brick_force_y, nzlo_out, nzhi_out, nylo_out, nyhi_out,
        nxlo_out, nxhi_out, "condiff:density_brick_force_y");
    memory->create3d_offset(density_brick_velocity_z, nzlo_out, nzhi_out, nylo_out, nyhi_out,
        nxlo_out, nxhi_out, "condiff:density_brick_velocity_z");

    memory->create3d_offset(density_brick_force_z, nzlo_out, nzhi_out, nylo_out, nyhi_out,
        nxlo_out, nxhi_out, "condiff:density_brick_force_z");

    order_allocated = order;
    memory->create2d_offset(rho1d, 3, -order / 2, order / 2, "condiff:rho1d");
    memory->create2d_offset(drho1d, 3, -order / 2, order / 2, "condiff:drho1d");
    memory->create2d_offset(rho_coeff, order, (1 - order) / 2, order / 2, "condiff:rho_coeff");
    memory->create2d_offset(drho_coeff, order, (1 - order) / 2, order / 2,
        "condiff:drho_coeff");

    memory->create(rand, 3, "condiff:rand");

    // create ghost grid object for rho and velocity field communication
    //int (*procneigh)[2] = comm->procneigh;
}

//check if pppm computation is used
void FixCondiff::kspace_check()
{
    if (!force->kspace)
        error->all(FLERR, "Not using kspace computations");
}

//check if pppm computation is used
void FixCondiff::pppm_check()
{
    if (!force->pair->pppmflag)
        error->all(FLERR, "Not using pppm computations");
}
