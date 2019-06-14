!SUBROUTINE        
      SUBROUTINE ADDSUB(A,B,C,D)  
      IMPLICIT NONE  
      DOUBLE PRECISION A,B,C,D  
!f2py intent(in) :: A,B  
!f2py intent(out) :: C,D  
      C = A + B  
      D = A - B  
      print*, "ADDSUB From Fortran!"  
      print*, "ADD=",C  
      print*, "SUB=",D  
      RETURN  
      END 
