! -*- f90 -*-
!- 
!- File    : test.f
!- ------------------------
!- 
      subroutine mkivar(fitsfile, flagfile, rmsfile, ivarfile)
      use ImageClass
      use ImageMath

      implicit none
!
!-    nc is the number of columns
!-    nr is the number of rows
!

      integer :: i, j, nc, nr, bitpix, stat
      integer, allocatable :: flag(:,:)
      real, allocatable :: image(:, :), rms(:, :), ivar(:,:)
      real :: sat
      character(len=120) :: fitsfile, flagfile, rmsfile, ivarfile 
      character(len = 72) :: comment

      call get_header(trim(fitsfile),'NAXIS1',nc,comment)
      call get_header(trim(fitsfile),'NAXIS2',nr,comment)
      call get_header(trim(fitsfile),'BITPIX',bitpix,comment)

      call get_header(trim(fitsfile),'SATURATE',sat,comment)

      call deletefile(trim(ivarfile))
      
 
      allocate(image(nc,nr), flag(nc,nr), ivar(nc,nr), rms(nc,nr), stat=stat)
      call get_image_rl(trim(fitsfile),image,nc,nr)
      call get_image_rl(trim(rmsfile),rms,nc,nr)
      call get_image_in(trim(flagfile),flag,nc,nr)

      ivar = 0.0

      where (flag == 0) ivar = 1.0/rms**2 
      where (flag == 2) ivar = 1.0/rms**2 
      where (flag == 2048) ivar = 1.0/rms**2 
      where (flag == 2050) ivar = 1.0/rms**2 

      where (image > 0.9*sat) ivar = 0.0 


      call put_image(trim(ivarfile), ivar, nc, nr)


      end subroutine mkivar
