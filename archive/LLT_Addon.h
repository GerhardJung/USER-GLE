#include <Eigen/Dense>

#ifndef LLT_ADDON_H
#define LLT_ADDON_H

using namespace Eigen;

/**
 * \class LLT_Addon
 * 
 * \brief Standard Cholesky decomposition (LL^T) of a matrix and associated features. Addon to class LLT to include line/column update in O(N^2).
 * 
 * 
 * 
 */
template<typename MatrixType, int UpLo = Lower> class LLT_Addon : public LLT<MatrixType, UpLo> {
  // Typedefs
  typedef internal::LLT_Traits<MatrixType,UpLo> Traits;
  typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
      
public:
  LLT_Addon () : LLT<MatrixType, UpLo> () { }
  
  template<typename InputType>
  explicit LLT_Addon(const EigenBase<InputType>& matrix) : LLT<MatrixType,UpLo> (matrix){ }
    
  template<typename InputType>
  LLT_Addon& compute(const EigenBase<InputType>& matrix){
    LLT<MatrixType,UpLo>::compute(matrix);
  }
  
  /** \returns a view of the lower triangular matrix L */
  inline typename Traits::MatrixL matrixL() const
  {
    eigen_assert(this->m_isInitialized && "LLT is not initialized.");
    return Traits::getL(this->m_matrix);
  }
  
  template<typename VectorType>
  LLT_Addon rankUpdate(const VectorType& vec, const RealScalar& sigma = 1) {
    LLT<MatrixType, UpLo>::rankUpdate(vec,sigma);
    return *this;
  }
  
  template<typename VectorType>
    LLT_Addon vectorUpdate(const VectorType& vec, const Index& i);
  
};

template<typename MatrixType, int UpLo>
template<typename VectorType>
LLT_Addon<MatrixType,UpLo> LLT_Addon<MatrixType,UpLo>::vectorUpdate(const VectorType& vec, const Index& i) {
  
  typedef typename MatrixType::RealScalar RealScalar;
  typedef typename VectorType::RealScalar RealScalarV;
  typedef typename MatrixType::ColXpr ColXpr;
  typedef typename MatrixType::RowXpr RowXpr;
  typedef Matrix<RealScalar,Dynamic,1> TempVectorType;
  
  ColXpr col = this->m_matrix.col(i);
  RowXpr row = this->m_matrix.row(i);
  Index n = this->m_matrix.cols();
  
  TempVectorType save1(n);
  for (Index k=0; k<n; k++) {
    if (k<=i) save1(k) = 0.0;
    else save1(k) = col(k);
  }
  
  // calculate M21
  for (Index k=0; k<i; k++ ) {
    RealScalarV y = vec(k);
    RowXpr rowk = this->m_matrix.row(k);
    for (Index l=0; l<k; l++ ) {
      y -= row(l)*rowk(l);
    }
    row(k) = y/rowk(k);
  }
  

  // calculate M22
  RealScalarV y = vec(i);
  for (Index k=0; k<i; k++ ) {
    y -= row(k)*row(k);
  }
  row(i) = std::sqrt(y);
  
  // calculate M32
  for (Index k=i+1; k<n; k++ ) {
    RealScalarV y = vec(k);
    RowXpr rowk = this->m_matrix.row(k);
    for (Index l=0; l<i; l++ ) {
      y -= rowk(l)*row(l);
    }
    col(k)=y/row(i);
  }
  
  TempVectorType save2(n);
  for (Index k=0; k<n; k++) {
    if (k<=i) save2(k) = 0.0;
    else save2(k) = col(k);
  }
  
  // calculate M33 (rankUpdate)
  rankUpdate(save1,1); 
  rankUpdate(save2,-1);

  
  return *this;
}


#endif