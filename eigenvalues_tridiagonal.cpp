/* Library from http://cacs.usc.edu/education/phys516/src/TB/ */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NR_END 1
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

#include "eigenvalues_tridiagonal.h"

/******************************************************************************/
void tqli(double d[], double e[], int n, double **z)
/*******************************************************************************
QL algorithm with implicit shifts, to determine the eigenvalues and eigenvectors
of a real, symmetric, tridiagonal matrix, or of a real, symmetric matrix
previously reduced by tred2 sec. 11.2. On input, d[1..n] contains the diagonal
elements of the tridiagonal matrix. On output, it returns the eigenvalues. The
vector e[1..n-1] inputs the subdiagonal elements of the tridiagonal matrix. 
On output e is destroyed. When finding only the eigenvalues,
several lines may be omitted, as noted in the comments. If the eigenvectors of
a tridiagonal matrix are desired, the matrix z[1..n][1..n] is input as the
identity matrix. If the eigenvectors of a matrix that has been reduced by tred2
are required, then z is input as the matrix output by tred2. In either case,
the kth column of z returns the normalized eigenvector corresponding to d[k].
*******************************************************************************/
{
  double pythag(double a, double b);
  int m,l,iter,i,k;
  double s,r,p,g,f,dd,c,b;

  for (l=1;l<=n;l++) {
    iter=0;
    do {
      for (m=l;m<=n-1;m++) { /* Look for a single small subdiagonal element to split the matrix. */
	dd=fabs(d[m])+fabs(d[m+1]);
	if ((double)(fabs(e[m])+dd) == dd) break;
      }
      if (m != l) {
	if (iter++ == 30) printf("Too many iterations in tqli");
	g=(d[l+1]-d[l])/(2.0*e[l]); /* Form shift. */
	r=pythag(g,1.0);
	g=d[m]-d[l]+e[l]/(g+SIGN(r,g)); /* This is dm - ks. */
	s=c=1.0;
	p=0.0;
	for (i=m-1;i>=l;i--) { /* A plane rotation as in the original QL, followed by Givens */
	  f=s*e[i];          /* rotations to restore tridiagonal form.                     */
	  b=c*e[i];
	  e[i+1]=(r=pythag(f,g));
	  if (r == 0.0) { /* Recover from underflow. */
	    d[i+1] -= p;
	    e[m]=0.0;
	    break;
	  }
	  s=f/r;
	  c=g/r;
	  g=d[i+1]-p;
	  r=(d[i]-g)*s+2.0*c*b;
	  d[i+1]=g+(p=s*r);
	  g=c*r-b;
	  /* Next loop can be omitted if eigenvectors not wanted */
	  for (k=1;k<=n;k++) { /* Form eigenvectors. */
	    f=z[k][i+1];
	    z[k][i+1]=s*z[k][i]+c*f;
	    z[k][i]=c*z[k][i]-s*f;
	  }
	}
	if (r == 0.0 && i >= l) continue;
	d[l] -= p;
	e[l]=g;
	e[m]=0.0;
      }
    } while (m != l);
  }
}

/******************************************************************************/
double pythag(double a, double b)
/*******************************************************************************
Computes (a2 + b2)1/2 without destructive underflow or overflow.
*******************************************************************************/
{
	double absa,absb;
	absa=fabs(a);
	absb=fabs(b);
	if (absa > absb) return absa*sqrt(1.0+(absb/absa)*(absb/absa));
	else return (absb == 0.0 ? 0.0 : absb*sqrt(1.0+(absa/absb)*(absa/absb)));
}
