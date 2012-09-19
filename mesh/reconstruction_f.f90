subroutine nolimit(idir, a, lda, qx, qy, ng)

  implicit none

  integer, intent(in) :: idir
  integer, intent(in) :: qx, qy, ng
  double precision, intent(inout) :: a(qx, qy)
  double precision, intent(  out) :: lda(qx, qy)

!f2py intent(in) :: idir
!f2py depend(qx, qy) :: a, lda
!f2py intent(in) :: a
!f2py intent(out) :: lda

! call this as: lda = reconstruction_f.nolimit(1,a,qx,qy,ng)

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny
  integer :: i, j

  ! 1-based indexing in Fortran, instead of 0-based a la Python
  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng+1; ihi = ng+nx; jlo = ng+1; jhi = ng+ny

  lda(:,:) = 0.0

  select case (idir)

  case (1)

     do j = jlo-2, jhi+2
        do i = ilo-2, ihi+2
           lda(i,j) = 0.5d0*(a(i+1,j) - a(i-1,j))
        enddo
     enddo

  case (2)

     do j = jlo-2, jhi+2
        do i = ilo-2, ihi+2
           lda(i,j) = 0.5d0*(a(i,j+1) - a(i,j-1))
        enddo
     enddo

  end select

end subroutine nolimit



  
  
