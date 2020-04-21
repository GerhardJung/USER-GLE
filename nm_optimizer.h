/* -*- c++ -*- ----------------------------------------------------------
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
   Contributing author: Sebastian Morr, Gerhard Jung
   Copyright (C) 2013 Sebastian Morr sebastian@morr.cc
------------------------------------------------------------------------- */

// Implements a basic Nelder Mead Optimizer
// see Nelder, John A.; R. Mead (1965). "A simplex method for function minimization". Computer Journal. 7: 308–313

#include <ctime>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <stdio.h>
using namespace std;

// Float vector with standard operations
class Vector {
public:
    Vector () {
    }
    Vector(float* values, int dim) {
      for (int i=0; i<dim; i++) {
	coords.push_back(values[i]);
      }
    }

    float& operator[](int i) {
        return coords[i];
    }
    float at(int i) const {
        return coords[i];
    }
    int dimension() const {
        return coords.size();
    }
    void prepare(int size) {
        for (int i=0; i<size; i++) {
            coords.push_back(0);
        }
    }
    Vector operator+(Vector other) {
        Vector result;
        result.prepare(dimension());
        for (int i=0; i<dimension(); i++) {
            result[i] = coords[i] + other[i];
        }
        return result;
    }
    void operator+=(Vector other) {
        for (int i=0; i<dimension(); i++) {
            coords[i] += other[i];
        }
    }
    Vector operator-(Vector other) {
        Vector result;
        result.prepare(dimension());
        for (int i=0; i<dimension(); i++) {
            result[i] = coords[i] - other[i];
        }
        return result;
    }
    bool operator==(Vector other) {
        if (dimension() != other.dimension()) {
            return false;
        }
        for (int i=0; i<dimension(); i++) {
            if (other[i] != coords[i]) {
                return false;
            }
        }
        return true;
    }
    Vector operator*(float factor) {
        Vector result;
        result.prepare(dimension());
        for (int i=0; i<dimension(); i++) {
            result[i] = coords[i]*factor;
        }
        return result;
    }
    Vector operator/(float factor) {
        Vector result;
        result.prepare(dimension());
        for (int i=0; i<dimension(); i++) {
            result[i] = coords[i]/factor;
        }
        return result;
    }
    void operator/=(float factor) {
        for (int i=0; i<dimension(); i++) {
            coords[i] /= factor;
        }
    }
    bool operator<(const Vector other) const {
        for (int i=0; i<dimension(); i++) {
            if (at(i) < other.at(i))
                return false;
            else if (at(i) > other.at(i))
                return true;
        }
        return false;
    }
    float length() {
        float sum = 0;
        for (int i=0; i<dimension(); i++) {
            sum += coords[i]*coords[i];
        }
        return pow(sum, 0.5f);
    }
private:
    vector<float> coords;
};

// Database class to store Vactors and their Values
class ValueDB {
    public:
        ValueDB() {
	  
        }
        void init(int dimension, double *target_function, int mem_count) {
	  this->dimension = dimension;
          this->target_function = target_function;
          this->mem_count = mem_count;
        }
        float lookup(Vector vec) {
            if (!contains(vec)) {
		float score = calc_score(vec, mem_count);
		insert(vec, score);
		return score;
            } else {
                return values[vec];
            }
        }
        void insert(Vector vec, float value) {
            values[vec] = value;
        }
        
        void reset() {
	  values.clear();
	}
	float calc_score(Vector v, int mem_count){
    
  float sum=0.0;
  int n,s;
    
  for(n=0;n<mem_count;n++){
    double loc_sum = 0.0;
    for(s=0;s<dimension;s++){
      int ind = n+s;
      //if (n+s>dimension-1) ind -= dimension;
      if (n+s>dimension-1) continue;
      loc_sum += v[s]*v[ind];
    }
    loc_sum -= target_function[n];
    sum += loc_sum*loc_sum;
  }
  
  return sum;
	}
    private:
        bool contains(Vector vec) {
            map<Vector, float>::iterator it = values.find(vec); 
            return it != values.end();
        }
        map<Vector, float> values;
	int dimension;
	double *target_function;
  int mem_count;
};

class NelderMeadOptimizer {
    public:
        NelderMeadOptimizer(int dimension, float termination_distance, double *target_function, int mem_count) {
            this->dimension = dimension;
            srand(time(NULL));
            alpha = 1;
            gamma = 2;
            beta = 0.5;
            sigma = 0.5;
            this->termination_distance = termination_distance;
	    this->target_function = target_function;
       this->mem_count = mem_count;
	    //for (int i=0; i<dimension;i++) printf("target: %f\n",target_function[i]);
	    extr_ind = new int[3];
	    extr_values = new float[3];
        }
        ~NelderMeadOptimizer() {
	  delete [] extr_ind;
	  delete [] extr_values;
	}
        // used in `step` to sort the vectors
        bool operator()(const Vector& a, const Vector& b) {
            return f(a) < f(b);
        }
        // termination criteria: each pair of vectors in the simplex has to
        // have a distance of at most `termination_distance`
        bool done() {
            if (vectors.size() < dimension + 1) {
                return false;
            }
            float mean = 0.0;
	    for (int i=0; i<dimension+1; i++) {
	      mean += f(vectors[i]);
	    }
	    mean /= dimension +1;
	    double var = 0.0;
	    for (int i=0; i<dimension+1; i++) {
	      double val = f(vectors[i]);
	      var += (val-mean)*(val-mean);
	    }
	    var /= dimension;
	    var = sqrt(var);

	    if (var < termination_distance) return true;
	    else return false;
        }
        void insert(Vector vec, float score) {
            if (vectors.size() < dimension+1) {
                vectors.push_back(vec);
            }
        }
        void print_comp(Vector best) {
          
// 	    for(int n=0;n<dimension;n++){
// 	    float loc_sum = 0.0;
// 	      for(int s=0;s<dimension;s++){
// 		if(n+s>=dimension) continue;
// 		loc_sum += best[s]*best[s+n];
// 	      }
// 	      printf("%d %f %f\n",n,target_function[n],loc_sum);
// 	    }
	    
	}
        void print_v() {
	  int n,i;
	  for ( i=0; i<vectors.size(); i++ ) {
	    for ( n=0; n<dimension; n++) {
	      printf("%f ",vectors[i][n]);
	    }
	    printf("\n");
	  }
	}
	float f(Vector v){
  float sum=0.0;
  int n,s;
    
  for(n=0;n<mem_count;n++){
    double loc_sum = 0.0;
    for(s=0;s<dimension;s++){
      int ind = n+s;
      //if (n+s>dimension-1) ind -= dimension;
      if (n+s>dimension-1) continue;
      loc_sum += v[s]*v[ind];
    }
    loc_sum -= target_function[n];
    sum += loc_sum*loc_sum;
  }
  
  return sum;
	}
	void find_vectors(){
	  extr_values[0] = 999999.0;
	  extr_values[1] = 0.0;
	  extr_values[2] = 0.0;
	  
	  for (int i = 0; i<dimension+1; i++) {
	    float val = f(vectors[i]);
	    if (val < extr_values[0]) {
	      extr_values[0] = val;
	      extr_ind[0] = i;
	    }
	    if (val > extr_values[1]) {
	      extr_values[1] = val;
	      extr_ind[1]=i;
	    }
	    if (val > extr_values[2]) {
	      extr_values[1] = extr_values[2];
	      extr_ind[1] = extr_ind[2];
	      extr_values[2] = val;
	      extr_ind[2]=i;
	    }
	  }  
	}
        Vector step(Vector vec, float score) {
            try {
                if (vectors.size() < dimension+1) {
                    vectors.push_back(vec);
                }

                // otherwise: optimize!
                if (vectors.size() == dimension+1) {
		  int counter = 0;
                    while(!done()) {
		      //print_comp(vectors[0]);
                        find_vectors();
                        Vector cog; // center of gravity
                        cog.prepare(dimension);
                        for (int i = 0; i<dimension+1; i++) {
			  if (i!=extr_ind[2])
                            cog += vectors[i];
			    //printf("score=%f\n",f(vectors[i]));
                        }
                        cog /= dimension;
                        Vector best = vectors[extr_ind[0]];
			float vbest = extr_values[0];
                        Vector worst = vectors[extr_ind[2]];
			float vworst = extr_values[2];
                        Vector second_worst = vectors[extr_ind[1]];
			float vsworst = extr_values[1];
			//printf("best=%d/%f, sworst=%d/%f, worst=%d/%f\n",extr_ind[0],vbest,extr_ind[1],vsworst,extr_ind[2],vworst);
                        // reflect
                        Vector reflected = cog + (cog - worst)*alpha;
			float vreflected = f(reflected);
			if (vreflected < vbest) {
			  //print_comp(reflected);
			    //expand
                            Vector expanded = cog + (cog - worst)*gamma;
			    float vexpanded = f(expanded);
                            if (vexpanded < vreflected) {
			      //printf("expanded\n");
                                vectors[extr_ind[2]] = expanded;
                            } else {
			      //printf("reflected1\n");
                                vectors[extr_ind[2]] = reflected;
                            }
			} else if (vreflected <= vsworst) {
                            vectors[extr_ind[2]] = reflected;
			    //printf("reflected2\n");
                        } else {
			    if (vreflected < vworst ) {
			      vectors[extr_ind[2]] = reflected;
			      worst = vectors[extr_ind[2]];
			    }
			    // contract
			    Vector h;
			    h.prepare(dimension);
			    Vector contracted = cog + (worst - cog)*beta;
                            if (f(contracted) > vworst) {
			      	//printf("rescaled\n");
                                for (int i=0; i<dimension+1; i++) {
                                    vectors[i] = (vectors[i]+best)/2.0;
                                }
                                
                            } else {
				vectors[extr_ind[2]] = contracted;
				//printf("contracted\n");
                            }
                        }
                        if (counter%100==0)
			  printf("count %d: best value: %f\n",counter,f(best));
			counter++;
                    }

                    // algorithm is terminating, output: simplex' center of gravity
                    Vector cog;
		    cog.prepare(dimension);
                    for (int i = 0; i<=dimension; i++) {
                        cog += vectors[i];
                    }
                    return cog/(dimension+1);
                } else {
                    // as long as we don't have enough vectors, request random ones,
                    // with coordinates between 0 and 1. If you want other start vectors,
                    // simply ignore these and use `step` on the vectors you want.
                    Vector result;
                    result.prepare(dimension);
                    for (int i = 0; i<dimension; ++i) {
                        result[i] = 0.001*(rand()%1000);
                    }
                    return result;
                }
            } catch (Vector v) {
                return v;
            }
        }
        void restart() {
	  vectors.clear();
	  
	}
    private:
        int dimension;
        float alpha, gamma, beta, sigma;
        float termination_distance;
        vector<Vector> vectors;
	int *extr_ind;
	float *extr_values;
	double *target_function;
  int mem_count;
};
