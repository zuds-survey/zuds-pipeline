! -*- f90 -*-
      module ImageMath
!
!- Some comments and extra code added by RS 2008/06/17
!
      interface get_med_image
       module procedure get_med_image_in
       module procedure get_med_image_rl
      end interface get_med_image

      interface get_med_arr
       module procedure get_med_arr_in
       module procedure get_med_arr_rl
      end interface get_med_arr

      interface maskit
       module procedure maskit_in
       module procedure maskit_rl
      end interface maskit

      interface smoothbox
       module procedure smoothbox_in
       module procedure smoothbox_rl
      end interface smoothbox

      interface rebin
       module procedure rebin_in
       module procedure rebin_rl
      end interface rebin

      contains

      subroutine oversubdes(imageo,image,nc,chip,nctot,nr,nrtot,meda,medb)

!
!     Routine to overscan subtract the DES images, 
!
      
      integer, intent (in) :: nc, nr, nctot, chip, nrtot
      integer, intent (in) :: imageo(nctot,nrtot)
      integer, intent (in out) :: image(nc,nr), meda, medb

      integer :: i,j,k,l
      integer, parameter :: nchp = 62, ncos=50
      integer :: osa(nr), smtha(nr), tmpa(ncos)
      integer :: osb(nr), smthb(nr), tmpb(ncos)
      integer :: image_a(nc/2,nr), image_b(nc/2,nr)

      integer, parameter :: xsbias_a = 2105, xfbias_a = 2154, ysbias_a = 51, yfbias_a = 4146
      integer, parameter :: xsbias_b = 7, xfbias_b = 56, ysbias_b = 51, yfbias_b = 4146
      integer, parameter :: xsdata_a = 1081, xfdata_a = 2104, ysdata_a = 51, yfdata_a = 4146
      integer, parameter :: xsdata_b = 57, xfdata_b = 1080, ysdata_b = 51, yfdata_b = 4146

      osa = 0
      smtha = 0
      osb = 0
      smthb = 0

      do j = 1, nr

         tmpa = imageo(xsbias_a:xfbias_a,j + ysdata_a - 1)
         tmpb = imageo(xsbias_b:xfbias_b,j + ysdata_b - 1)
      
         call get_med_arr(tmpa,ncos,osa(j))
         call get_med_arr(tmpb,ncos,osb(j))
         
      end do


      call get_med_arr(osa,nr,meda)
!      call smootharr(osa,smtha,nr,10)

      call get_med_arr(osb,nr,medb)
!      call smootharr(osb,smthb,nr,10)


!     for now take off a constant 

      image_a = imageo(xsdata_a:xfdata_a, ysdata_a:yfdata_a)
      image_b = imageo(xsdata_b:xfdata_b, ysdata_b:yfdata_b)


      do i = 1, nc/2
         do j = 1, nr

            image_a(i,j) = image_a(i,j) - osa(j)
            image_b(i,j) = image_b(i,j) - osb(j)

         end do
      end do

      image(1:nc/2,:) = image_b
      image((nc/2+1):nc,:) = image_a


      return
      
      end subroutine oversubdes

       subroutine  get_med_image_in( a, n, m, med )
         integer          , intent( in ) :: n, m
         integer          , intent( in ) :: a(n,m)
         integer          , intent(in out ) :: med 

         integer :: mid, k(1)
         integer :: arr(n*m)

         k(1) = n*m
         arr = reshape(a,k)

         call hpsort_in(arr, k(1))

         mid = int((k(1))/2) + 1

         med = arr(mid)
         
         return

      end subroutine get_med_image_in
     
      subroutine  get_med_image_rl( a, n, m, med )
         integer             , intent(    in ) :: n, m
         real          , intent(in ) :: a(n,m)
         real          , intent(in out ) :: med 

         integer :: mid, k(1)
         real :: arr(n*m)

         k(1) = n*m
         arr = reshape(a,k)

         call hpsort_rl(arr, k(1))

         mid = int((k(1))/2) + 1

         med = arr(mid)
         
         return

      end subroutine get_med_image_rl

       subroutine  get_med_arr_in( a, n, med )
         integer          , intent( in ) :: n
         integer          , intent( in ) :: a(n)
         integer          , intent(in out ) :: med 

         integer :: mid
         integer :: arr(n)

         arr = a

         call hpsort_in(arr, n)

         mid = int(n/2) + 1

         med = arr(mid)
         
         return

      end subroutine get_med_arr_in
     
      subroutine  get_med_arr_rl( a, n, med )
         integer             , intent(    in ) :: n
         real          , intent(in ) :: a(n)
         real          , intent(in out ) :: med 

         integer :: mid
         real :: arr(n)

         arr = a

         call hpsort_rl(arr, n)

         mid = int(n/2) + 1

         med = arr(mid)
         
         return

      end subroutine get_med_arr_rl
!- ***
!- *** hpsort :
!- *** Sorts an array arr( : ) into ascending numerical order using the
!- *** heapsort algorithm.  The array is replaced on output by its sorted
!- *** rearrangement.
!- ***
       subroutine hpsort_rl( arr, n )
!-     ------------------------
       integer, intent( in) :: n
       real, intent( in out ) :: arr( n )

        integer    :: i
        real :: b

        do i = n / 2, 1, - 1
         call sift_rl( arr, n, i, n )
        end do

        do i = n, 2, - 1
         b = arr( 1 )
         arr( 1 ) = arr( i )
         arr( i ) = b
         call sift_rl( arr, n, 1, i - 1 )
        end do

       end subroutine hpsort_rl
        
      subroutine sift_rl( arr, n, l, r )
!-    -----------------------
      integer, intent( in ) :: n, l, r
      real, intent (in out) :: arr(n)

      integer    :: j, jold
      real :: a

      a = arr( l )
      jold = l
      j = l + l
      do
         if( j .gt. r ) exit
         if( j .lt. r )then
            if( arr( j ) .lt. arr( j + 1 ) ) j = j + 1
         end if
         if( a .ge. arr( j ) ) exit
         arr( jold ) = arr( j )
         jold = j
         j = j + j
      end do
      arr( jold ) = a

      end subroutine sift_rl

!- ***
!- *** hpsort :
!- *** Sorts an array arr( : ) into ascending numerical order using the
!- *** heapsort algorithm.  The array is replaced on output by its sorted
!- *** rearrangement.
!- ***
       subroutine hpsort_in( arr, n )
!-     ------------------------
       integer, intent( in) :: n
       integer, intent( in out ) :: arr( n )

        integer    :: i
        integer :: b

        do i = n / 2, 1, - 1
         call sift_in( arr, n, i, n )
        end do
        do i = n, 2, - 1
         b = arr( 1 )
         arr( 1 ) = arr( i )
         arr( i ) = b
         call sift_in( arr, n, 1, i - 1 )
        end do

       end subroutine hpsort_in
        
      subroutine sift_in( arr, n, l, r )
!-    -----------------------
      integer, intent( in ) :: n, l, r
      integer, intent (in out) :: arr(n)

      integer    :: j, jold
      integer :: a

      a = arr( l )
      jold = l
      j = l + l
      do
         if( j .gt. r ) exit
         if( j .lt. r )then
            if( arr( j ) .lt. arr( j + 1 ) ) j = j + 1
         end if
         if( a .ge. arr( j ) ) exit
         arr( jold ) = arr( j )
         jold = j
         j = j + j
      end do
      arr( jold ) = a

      end subroutine sift_in



!
!     sigma-clipping least squares fitter
!     y = a*x + b
!

      subroutine lsq(x,y,n,a,b)
      integer, intent (in) :: n
      real, intent(in) :: x(n), y(n)
      real, intent(in out) :: a, b

!
!     if there is a median counts > 500 or the 3-sig
!     spread is < 0.001 and xm > 100 then the pixel is bad
!
      real, parameter :: cmax = 500.0, cmin = 0.001
!
      integer :: i, m, ns, mid
      real :: sxy, sx, sy, sx2
      real :: vh, vl, vm, vc(n), var(n)

!
!     Drop 2.5% of the points off either end of y
!

      m = int(real(n)*0.025)
      mid = int(n/2) + 1

      vc = y
      call hpsort_rl(vc,n)

      vh = vc(n-m)
      vl = vc(m)
      vm = vc(mid)

      if ( vm > cmax ) then
         
         a = 0.0
         b = 0.0

      else if (abs(vh-vl) < 0.001 ) then
         
         a = 0.0
         b = vh

      else
         

         ns = 0
         sx = 0.0
         sy = 0.0
         sxy = 0.0
         sx2 = 0.0

         do i = m, n-m
              ns = ns + 1
              sx = sx + x(i)
              sy = sy + y(i)
              sxy = sxy + x(i)*y(i)
              sx2 = sx2 + x(i)*x(i)
         end do


         a = (real(ns)*sxy - sx*sy)/(real(ns)*sx2 - sx*sx)
         b = (sy*sx2 - sx*sxy)/(real(ns)*sx2 - sx*sx)

         do i = 1, n
            var(i) = y(i) - (a*x(i) + b)
         end do

!
!     Drop 2.5% of the points off either end of var
!

         vc = var

         call hpsort_rl(vc,n)
         vh = vc(n-m)
         vl = vc(m)

         ns = 0
         sx = 0.0
         sy = 0.0
         sxy = 0.0
         sx2 = 0.0

         do i = 1, n
            if (var(i) > vl .and. var(i) < vh) then
               ns = ns + 1
               sx = sx + x(i)
               sy = sy + y(i)
               sxy = sxy + x(i)*y(i)
               sx2 = sx2 + x(i)*x(i)
            end if
         end do


         a = (real(ns)*sxy - sx*sy)/(real(ns)*sx2 - sx*sx)
         b = (sy*sx2 - sx*sxy)/(real(ns)*sx2 - sx*sx)

      end if
    
      return
         
      end subroutine lsq

      subroutine get_var(x,y,n,a,b,var)
      integer, intent(in) :: n
      real, intent (in) :: x(n), y(n), a, b
      real, intent (in out) :: var

      real :: sig(n)
      integer :: ms, nms

      do i = 1, n
         sig(i) = abs(y(i) - (a*x(i) + b))
      end do

      ms = int(real(n)*0.318)

      call hpsort_rl(sig,n)

      var = sig(n-ms)
      
      return
      
      end subroutine get_var

      subroutine get_var_med(y,n,b,var)
      integer, intent(in) :: n
      real, intent (in) :: y(n), b
      real, intent (in out) :: var

      real :: sig(n), var1, var2
      integer :: ms

      do i = 1, n
         sig(i) = y(i) -  b
      end do

      ms = int(real(n)*0.1585)

      call hpsort_rl(sig,n)

      var1 = abs(sig(n/2)-sig(ms))
      var2 = abs(sig(n/2)-sig(n-ms))
     
      var = max(var1,var2)
 
      return
      
      end subroutine get_var_med

      subroutine get_var_med_wdt(y,n,b,var,wdt)
      integer, intent(in) :: n
      real, intent (in) :: y(n), b
      real, intent (in out) :: var, wdt

      real :: sig(n), var1, var2
      integer :: ms

      do i = 1, n
         sig(i) = y(i) -  b
      end do

      ms = int(real(n)*0.1585)

      call hpsort_rl(sig,n)

      var1 = abs(sig(n/2)-sig(ms))
      var2 = abs(sig(n/2)-sig(n-ms))
     
      var = max(var1,var2)

      wdt = abs(var1-var2)
 
      return
      
      end subroutine get_var_med_wdt

      subroutine get_var_med1(y,n,b,var)
      integer, intent(in) :: n
      real, intent (in) :: y(n), b
      real, intent (in out) :: var

      real :: sig(n)
      integer :: ms

      do i = 1, n
         sig(i) = abs(y(i) -  b)
      end do

      ms = int(n*0.6827)

      call hpsort_rl(sig,n)

      var = sig(ms)
     
      return
      
      end subroutine get_var_med1


      subroutine get_star_mask(im,mask,n,m,med,skysig)
      integer, intent(in) :: n, m
      integer, intent (in out) :: mask(n,m)
      real, intent (in) :: im(n,m), med, skysig

      real :: diff(n,m), v(n,m)
      integer :: i,j, emmask(n,m)
      integer :: il, jl, iu, ju
      integer, parameter :: npix=5
      real, parameter :: sig=3.0

      mask = 1
      emmask = 1
      diff = abs(im-med)
      v = sig*skysig
      write (*,*) 'get_star_mask:  sqrt(med) = ', sqrt(med), ', skysig = ', skysig

      where(diff > v) mask = 0

      do i = 1, n 
       do j = 1,  m  

        if (mask(i,j) == 0) then

         il = max((i-npix), 1)
         iu = min((i+npix), n) 

         jl = max((j-npix), 1)
         ju = min((j+npix), m) 

         emmask(il:iu,jl:ju) = 0

        end if

       end do
      end do

      mask = mask*emmask

      return
      
      end subroutine get_star_mask

      subroutine get_mask_flat(var,mask,n,m,per)
      integer, intent(in) :: n, m
      integer, intent (in out) :: mask(n,m)
      real, intent (in) :: var(n,m), per

      real :: v(n,m)
      real, parameter :: sig = 1.0

      v = sig*var

      where(v > per) mask = 0

      return
      
      end subroutine get_mask_flat

      subroutine get_mask_flat1(var,mask,n,m,med,per)
      integer, intent(in) :: n, m
      integer, intent (in out) :: mask(n,m)
      real, intent (in) :: var(n,m), per, med

      real :: v(n,m)
      real, parameter :: sig = 1.0

      v = var-med
      v = abs(v)

      where(v > per) mask = 0

      return
      
      end subroutine get_mask_flat1

      subroutine get_mask_fringe(var,mask,n,m)
      integer, intent(in) :: n, m
      integer, intent (in out) :: mask(n,m)
      real, intent (in) :: var(n,m)

      real :: sig(n*m), sigc
      integer :: ms, k(1)

      k(1) = n*m
      ms = int(real(k(1))*0.318)
      mask = 1
      
      sig = reshape(var,k)

      call hpsort_rl(sig,k(1))

      sigc = 3.0*sig(k(1)-ms)

      where(var > sigc) mask = 0

      where (var < 0.001) mask = 0
      
      return
      
      end subroutine get_mask_fringe

      subroutine get_mask_dark(var,medim,mask,n,m)
      integer, intent(in) :: n, m
      integer, intent (in out) :: mask(n,m)
      real, intent (in) :: var(n,m), medim(n,m)

      real :: sig(n*m), sigc
      integer :: ms, k(1)

      k(1) = n*m
      ms = int(real(k(1))*0.318)
      
      sig = reshape(var,k)

      call hpsort_rl(sig,k(1))

      sigc = 3.0*sig(k(1)-ms)

      sigc = max(sigc,1.0)

      where(var > sigc) mask = 0

      where (var < 0.001 .and. medim > 100.0 ) mask = 0
      
      return
      
      end subroutine get_mask_dark
     
      subroutine maskit_in(image, mask, nc, nr, arr, nmsk)
      integer, intent (in) :: nc, nr, nmsk 
      integer, intent (in) :: image(nc, nr), mask(nc, nr)
      integer, intent (in out) :: arr(nmsk) 
  
      integer :: i,j,k 
      
      arr = 0
      k = 0

      do i = 1, nc
       do j = 1, nr
        if (mask(i,j) == 1) then 
         k = k + 1
         arr(k) = image(i,j)
        end if
       end do
      end do
       
      end subroutine maskit_in 

      subroutine maskit_rl(image, mask, nc, nr, arr, nmsk)
      integer, intent (in) :: nc, nr, nmsk 
      real, intent (in) :: image(nc, nr)
      integer, intent (in) :: mask(nc, nr)
      real, intent (in out) :: arr(nmsk) 
  
      integer :: i,j,k 
      
      arr = 0.0
      k = 0

      do i = 1, nc
       do j = 1, nr
        if (mask(i,j) == 1) then 
         k = k + 1
         arr(k) = image(i,j)
        end if
       end do
      end do
       
      end subroutine maskit_rl 

      subroutine smootharr(image, smooth, nc, nbox)
      integer, intent (in) :: nc, nbox
      integer, intent (in) :: image(nc)
      integer, intent (in out) :: smooth(nc)

      integer :: imsb(nbox)
      integer :: i,j,nb2,ncl,nch
      integer :: med
 
      nb2 = nbox/2
      smooth = 0
      smvar = 0


      do i = 1, nc
            
            ncl = max((i-nb2),1)
            nch = min((i+nb2),nc)

            if (ncl == 1) nch = nbox
            if (nch == nc) ncl = nc - nbox

            imsb = image(ncl:nch)
            
            call get_med_arr(imsb,nbox,med)
          
            smooth(i) = med
            
      end do
            
      
       
      end subroutine smootharr 

!-    RS 2008/06/17:  The following does smoothing on an image by sliding
!-    a square box nbox pixels on a side across the image, taking the
!-    median at each position.

      subroutine smoothbox_in(image, smooth, nc, nr, nbox, smvar)
      integer, intent (in) :: nc, nr, nbox
      integer, intent (in) :: image(nc, nr)
      integer, intent (in out) :: smooth(nc,nr)

      integer :: imsb(nbox,nbox)
      integer :: i,j,nb2,ncl,nch,nrl,nrh
      integer :: med
 
      integer, intent (in out), optional :: smvar(nc,nr)


      nb2 = nbox/2
      smooth = 0
      smvar = 0


      do i = 1, nc
         do j = 1, nr
            
            ncl = max((i-nb2),1)
            nch = min((i+nb2),nc)

            if (ncl == 1) nch = nbox
            if (nch == nc) ncl = nc - nbox

            nrl = max((j-nb2),1)
            nrh = min((j+nb2),nr)

            if (nrl == 1) nrh = nbox
            if (nrh == nr) nrl = nr - nbox

            imsb = image(ncl:nch,nrl:nrh)
            
            call get_med_image(imsb,nbox,nbox,med)
          
            smooth(i,j) = med
            
         end do
      end do
            
      
       
      end subroutine smoothbox_in 


      subroutine smoothbox_rl(image, smooth, nc, nr, nbox, smvar)
      integer, intent (in) :: nc, nr, nbox
      real, intent (in) :: image(nc, nr)
      real, intent (in out) :: smooth(nc,nr)

      real, intent (in out), optional :: smvar(nc,nr)
      

      real :: imsb(nbox,nbox)
      real :: imar(nbox*nbox), medar(nbox*nbox)
      integer :: i,j,nb2,k(1)
      real :: med
      integer :: ncl,nch,nrl,nrh

      nb2 = nbox/2
      k(1) = nbox*nbox
      smooth = 0.0

!$omp parallel do private(ncl,nch,nrl,nrh,imsb,med)

      do i = 1, nc

         ncl = max((i-nb2),1)
         nch = min((i+nb2),nc)

         if (ncl == 1) nch = nbox
         if (nch == nc) ncl = nc - nbox


         do j = 1, nr
            
            nrl = max((j-nb2),1)
            nrh = min((j+nb2),nr)

            if (nrl == 1) nrh = nbox
            if (nrh == nr) nrl = nr - nbox

            imsb = image(ncl:nch,nrl:nrh)
            
            call get_med_image(imsb,nbox,nbox,med)
          
            smooth(i,j) = med

!            if (present (smvar )) then
!              
!              medar = med
!              imar = reshape(imsb,k)
!              call get_var(imar,medar,k(1),1.0,0.0,smvar(i,j))
!
!            end if       
            
         end do
      end do

      end subroutine smoothbox_rl 

      subroutine fixpix(image, smooth, mask, nc, nr, nbox)
      integer, intent (in) :: nc, nr, nbox
      real, intent (in) :: image(nc, nr)
      integer, intent (in) :: mask(nc,nr)
      real, intent (in out) :: smooth(nc,nr)

      real :: imsb(nbox,nbox)
      integer :: msksb(nbox, nbox)
      real :: imar(nbox*nbox), medar(nbox*nbox)
      integer :: i,j,nb2,k(1)
      real :: med
      integer :: ncl,nch,nrl,nrh, nmsk, istat
      real, dimension(:), allocatable :: arr

      nb2 = nbox/2
      k(1) = nbox*nbox

!$omp parallel do private(ncl,nch,nrl,nrh,imsb,med,msksb,arr,nmsk)

      do i = 1, nc

         ncl = max((i-nb2),1)
         nch = min((i+nb2),nc)

         if (ncl == 1) nch = nbox
         if (nch == nc) ncl = nc - nbox


         do j = 1, nr
            
            nrl = max((j-nb2),1)
            nrh = min((j+nb2),nr)

            if (nrl == 1) nrh = nbox
            if (nrh == nr) nrl = nr - nbox


            if (mask(i,j) == 1) then            

               imsb = image(ncl:nch,nrl:nrh)

               msksb = abs(mask(ncl:nch,nrl:nrh) - 1)
               nmsk = sum(msksb)

               if (nmsk > 0 ) then

                  allocate(arr(nmsk), stat=istat)
                  call maskit(imsb, msksb, nbox, nbox, arr, nmsk)
                  call get_med_arr(arr,nmsk,med)
                  
                  smooth(i,j) = med

                  deallocate(arr)
   
               else 

                  call get_med_image(imsb,nbox,nbox,med)
                  
                  smooth(i,j) = med

               end if

            end if

         end do
      end do

      end subroutine fixpix 

      subroutine surface(image,mask,nc,nr,nbox)
      integer, intent (in) :: nc, nr, nbox
      real, intent (in out) :: image(nc, nr)
      integer, intent (in) :: mask(nc,nr)


      real :: imsb(nbox,nbox)
      integer :: mksb(nbox,nbox),ncl,nch,nrl,nrh
      integer :: i,j,k,l, ncstp, nrstp, istat
      integer :: ncint(nc), nrint(nr) 
      real :: med
      
      real, allocatable :: arr(:), smooth(:,:)

      ncstp = int(nc/nbox) 
      nrstp = int(nr/nbox) 
      allocate(smooth(ncstp,nrstp), stat=istat)            
      smooth = 0.0


      if (ncstp <= 1 .or. nrstp <= 1) then
        print *, 'Can Not do Surface, box is same/smaller than frame'
        deallocate(smooth)
        return
      end if
  
      do k = 1, ncstp 
            
            ncl = (k-1)*nbox + 1 
            nch = min(k*nbox,nc) 
            ncint(ncl:nch) = k

         do l = 1, nrstp

            nrl = (l-1)*nbox + 1 
            nrh = min(l*nbox,nr)
            nrint(nrl:nrh) = l


            imsb = image(ncl:nch,nrl:nrh)
            mksb = mask(ncl:nch,nrl:nrh)

            nmsk = sum(mksb)

            if (nmsk > 0) then
              
              allocate(arr(nmsk), stat=istat)            
              call maskit(imsb, mksb, nbox, nbox, arr, nmsk)
              call get_med_arr(arr,nmsk,med)
          
              smooth(k,l) = med

              deallocate(arr)

            else 
 
              call get_med_image(imsb,nbox,nbox,med)
              
              smooth(k,l) = med

            end if
            
         end do
      end do

      do i = 1, nc
       do j = 1, nr

        image(i,j) = image(i,j) - smooth(ncint(i),nrint(j)) 

       end do
      end do

      deallocate(smooth)

      end subroutine surface 

      subroutine surface4(image,mask,nc,nr,nbox)
      integer, intent (in) :: nc, nr, nbox
      real, intent (in out) :: image(nc, nr)
      integer, intent (in) :: mask(nc,nr)


      real :: imsb(nbox,nbox)
      integer :: mksb(nbox,nbox),ncl,nch,nrl,nrh
      integer :: i,j,k,l, ncstp, nrstp, istat, nb2
      integer :: nc1(nc), nc2(nc), nr1(nr), nr2(nr) 
      real :: med, pt11, pt12, pt21, pt22, smtbilin 
      
      real, allocatable :: arr(:), smooth(:,:)
      integer, allocatable :: ncp(:), nrp(:)

      ncstp = int(nc/nbox) + 1 
      nrstp = int(nr/nbox) + 1 
      nb2 = int(nbox/2) 
      allocate(smooth(ncstp,nrstp), ncp(ncstp), nrp(nrstp), stat=istat)            
      smooth = 0.0


      if (ncstp <= 1 .or. nrstp <= 1) then
        print *, 'Can Not do Surface, box is same/smaller than frame'
        deallocate(smooth,ncp,nrp)
        return
      end if
  
      do k = 1, ncstp 

         if (k == 1) then
          ncp(k) = 1 
         else
          ncp(k) = (k-1)*nbox
         end if
         
      end do

      do l = 1, nrstp 

         if (l == 1) then
          nrp(l) = 1 
         else
          nrp(l) = (l-1)*nbox
         end if

      end do

      do k = 1, ncstp - 1 

       nc1(ncp(k):ncp(k+1)) = k
       nc2(ncp(k):ncp(k+1)) = k+1
       
      end do

      do l = 1, nrstp - 1 

       nr1(nrp(l):nrp(l+1)) = l
       nr2(nrp(l):nrp(l+1)) = l+1

      end do

      do k = 1, ncstp
         
         if (k == 1) then

           ncl = 1 
           nch = nbox

         else if (k == ncstp) then

           ncl = nc - nbox + 1
           nch = nc
           
         else

           ncl = ncp(k) - nb2
           nch = ncp(k) + nb2

         end if        

         do l = 1, nrstp

           if (l == 1) then

             nrl = 1 
             nrh = nbox

           else if (l == nrstp) then

             nrl = nr - nbox + 1
             nrh = nr
           
           else

             nrl = nrp(l) - nb2 
             nrh = nrp(l) + nb2 

           end if        


           imsb = image(ncl:nch,nrl:nrh)
           mksb = mask(ncl:nch,nrl:nrh)

           nmsk = sum(mksb)

           if (nmsk > 0) then
              
             allocate(arr(nmsk), stat=istat)            
             call maskit(imsb, mksb, nbox, nbox, arr, nmsk)
             call get_med_arr(arr,nmsk,med)
          
             smooth(k,l) = med

             deallocate(arr)

           else 
 
             call get_med_image(imsb,nbox,nbox,med)
              
             smooth(k,l) = med

           end if
            
         end do
      end do

      do i = 1, nc

       x = real(i)
       x1 = real(ncp(nc1(i)))
       x2 = real(ncp(nc2(i)))
       
       do j = 1, nr

        pt11 = smooth(nc1(i),nr1(j))
        pt12 = smooth(nc1(i),nr2(j))
        pt21 = smooth(nc2(i),nr1(j))
        pt22 = smooth(nc2(i),nr2(j))

        y = real(j)
        y1 = real(nrp(nr1(j)))
        y2 = real(nrp(nr2(j)))

        den = (x2-x1)*(y2-y1)

        smtbilin = pt11/den*(x2-x)*(y2-y) + pt12/den*(x2-x)*(y-y1) + pt21/den*(x-x1)*(y2-y) + pt22/den*(x-x1)*(y-y1)

        image(i,j) = image(i,j) - smtbilin 

       end do
      end do

      deallocate(smooth,ncp,nrp)

      end subroutine surface4 

      subroutine surface2(image,mask,nc,nr,nbox)
      integer, intent (in) :: nc, nr, nbox
      real, intent (in out) :: image(nc, nr)
      integer, intent (in) :: mask(nc,nr)

      real, allocatable :: arr(:)

      real :: imsb(nbox,nbox), smooth(nc,nr)
      integer :: mksb(nbox,nbox),ncl,nch,nrl,nrh
      integer :: i,j,k,l, nb2
      real :: med
    
      nb2 = nbox/2
 
      do i = 1, nc

         ncl = max((i-nb2),1)
         nch = min((i+nb2),nc)

         if (ncl == 1) nch = nbox
         if (nch == nc) ncl = nc - nbox

         do j = 1, nr

            nrl = max((j-nb2),1)
            nrh = min((j+nb2),nr)

            if (nrl == 1) nrh = nbox
            if (nrh == nr) nrl = nr - nbox

            imsb = image(ncl:nch,nrl:nrh)
            mksb = mask(ncl:nch,nrl:nrh)

            nmsk = sum(mksb)

            if (nmsk > 0) then

              allocate(arr(nmsk), stat=istat)
              call maskit(imsb, mksb, nbox, nbox, arr, nmsk)

              smooth(i,j) = sum(arr)/real(nmsk)

              deallocate(arr)

            else

              smooth(i,j) = sum(imsb)/real(nbox*nbox)

            end if


         end do
      end do

       
      do i = 1, nc
       do j = 1, nr

        image(i,j) = image(i,j) - smooth(i,j)

       end do
      end do


      end subroutine surface2 

      subroutine surface3(image,mask,nc,nr,nbox)
      integer, intent (in) :: nc, nr, nbox
      real, intent (in out) :: image(nc, nr)
      integer, intent (in) :: mask(nc,nr)

      real :: imsb(nbox,nbox), smooth(nc,nr)
      integer :: mksb(nbox,nbox),ncl,nch,nrl,nrh
      integer :: i,j,k,l, nb2
      real :: med
    
      nb2 = nbox/2
      smooth = image
 
      do i = 1, nc

         ncl = max((i-nb2),1)
         nch = min((i+nb2),nc)

         if (ncl == 1) nch = nbox
         if (nch == nc) ncl = nc - nbox

         do j = 1, nr

            nrl = max((j-nb2),1)
            nrh = min((j+nb2),nr)

            if (nrl == 1) nrh = nbox
            if (nrh == nr) nrl = nr - nbox

            imsb = smooth(ncl:nch,nrl:nrh)
            mksb = mask(ncl:nch,nrl:nrh)

            nmsk = sum(mksb)

            if (nmsk > 0) then

              image(i,j) = image(i,j) - sum(imsb*real(mask))/real(nmsk)

            else

              image(i,j) = image(i,j) - sum(imsb)/real(nbox*nbox)

            end if


         end do
      end do

      end subroutine surface3 


      subroutine smoothrep(image, smooth, nc, nr, nbox)
      integer, intent (in) :: nc, nr, nbox
      real, intent (in) :: image(nc, nr)
      real, intent (in out) :: smooth(nc,nr)

      real :: imsb(nbox,nbox)
      integer :: i,j,nb2,nbxsq
      real :: med

      nb2 = (nbox-1)/2
      nbxsq = nbox*nbox
      smooth = 0.0


      do i = 1, nc
        do j = 1, nr
            
          ncl = max((i-nb2),1)
          nch = min((i+nb2),nc)

          if (ncl == 1) nch = nbox
          if (nch == nc) ncl = nc - nbox
 
          nrl = max((j-nb2),1)
          nrh = min((j+nb2),nr)
 
          if (nrl == 1) nrh = nbox
          if (nrh == nr) nrl = nr - nbox
 
          imsb = image(ncl:nch,nrl:nrh)
             
          call get_med_image(imsb,nbox,nbox,med)
           
          smooth(i,j) = med
             
       end do
      end do

      end subroutine smoothrep

!- RS 2008/06/17:  A routine to rebin images.  Not a sliding window!
!- The image is divided into bins and all pixels in a given bin are
!- replaced by the sum of pixels within that bin.  Final image is
!- reduced in each dimension by a factor of nbox.

      subroutine rebin_in(image, smooth, mask, nc, nr, nbox)
      integer, intent (in) :: nc, nr, nbox
      integer, intent (in) :: mask(nc,nr)
      integer, intent (in) :: image(nc, nr)
      integer, intent (in out) :: smooth(nc/nbox,nr/nbox)

      integer :: i,j,di,dj,ncl,nch,nrl,nrh,nbxsq

      smooth = 0
      nbxsq = nbox*nbox

      do i = 1, nc, nbox
        do j = 1, nr, nbox

!-        Define the boundaries of the box we're looking at...

          ncl = i
          nch = min((i+nbox-1),nc)
          nrl = j
          nrh = min((j+nbox-1),nr)

!-        If there are no pixels masked within this box, sum all the pixel
!-        values and replace all the relevant pixels in the smoothed image
!-        with the appropriate sum.  If any pixels are masked, set the sum
!-        in that box to zero.

!         write (*,*) 'i = ', i, ', j = ', j, ', nbox = ', nbox
!         write (*,*) 'sum(mask(', ncl, ':', nch,',',nrl,':',nrh,')) = ', sum(mask(ncl:nch,nrl:nrh))
          if (sum(mask(ncl:nch,nrl:nrh)) .eq. nbxsq) then
             smooth(1+i/nbox,1+j/nbox) = sum(image(ncl:nch,nrl:nrh))
!            write (*,*) 'Setting smooth(', 1+i/nbox, ',', 1+j/nbox, ') = ', smooth(1+i/nbox,1+j/nbox)
          end if

       end do
      end do

      end subroutine rebin_in

      subroutine rebin_rl (image, smooth, mask, nc, nr, nbox)
      integer, intent (in) :: nc, nr, nbox
      integer, intent (in) :: mask(nc,nr)
      real, intent (in) :: image(nc, nr)
      real, intent (in out) :: smooth(nc/nbox,nr/nbox)

      integer :: i,j,di,dj,ncl,nch,nrl,nrh,nbxsq

      smooth = 0.0
      nbxsq = nbox*nbox

      do i = 1, nc, nbox
        do j = 1, nr, nbox

!-        Define the boundaries of the box we're looking at...

          ncl = i
          nch = min((i+nbox-1),nc)
          nrl = j
          nrh = min((j+nbox-1),nr)

!-        If there are no pixels masked within this box, sum all the pixel
!-        values and replace all the relevant pixels in the smoothed image
!-        with the appropriate sum.  If any pixels are masked, set the sum
!-        in that box to zero.

!         write (*,*) 'i = ', i, ', j = ', j, ', nbox = ', nbox
!         write (*,*) 'sum(mask(', ncl, ':', nch,',',nrl,':',nrh,')) = ', sum(mask(ncl:nch,nrl:nrh))
          if (sum(mask(ncl:nch,nrl:nrh)) .eq. nbxsq) then
             smooth(1+i/nbox,1+j/nbox) = sum(image(ncl:nch,nrl:nrh))
!            write (*,*) 'Setting smooth(', 1+i/nbox, ',', 1+j/nbox, ') = ', smooth(1+i/nbox,1+j/nbox)
          end if

       end do
      end do

      end subroutine rebin_rl

      subroutine badpix(image,nc,nr,x,y,n,rms,n2sig,n3sig,nbox)

      integer, intent (in) :: nc, nr, n, nbox
      real, intent (in) :: image(nc,nr), rms, x(n), y(n)
      integer, intent (in out) :: n2sig(n), n3sig(n)

      integer :: i,j,xl,xh,yl,yh,xi,yi


      do k = 1, n

        xi = int(x(k) + 0.5) 
        yi = int(y(k) + 0.5) 

        xl = max(1,xi-nbox)
        xh = min(nc,xi+nbox)

        yl = max(1,yi-nbox)
        yh = min(nr,yi+nbox)

        n2sig(k) = 0
        n3sig(k) = 0

        do i = xl, xh
          do j = yl, yh 

            if (image(i,j) < -2.0*rms) n2sig(k) = n2sig(k) + 1
            if (image(i,j) < -3.0*rms) n3sig(k) = n3sig(k) + 1

          end do
        end do

      end do

      end subroutine badpix

      subroutine symm(image,nc,nr,x,y,n,rms,nb,fct)

      integer, intent (in) :: nc, nr, n, nb
      real, intent (in) :: image(nc,nr), rms, x(n), y(n)
      real, intent (in out) :: fct(n)

      real :: imrms(nc,nr)
      real :: imf(2*nb+3, 2*nb+3)
      real :: im1(2*nb+1, 2*nb+1),  im2(2*nb+1, 2*nb+1)
      real :: im3(2*nb+1, 2*nb+1),  im4(2*nb+1, 2*nb+1)
      real :: diff(2*nb+1, 2*nb+1)
      real :: offx, offy

      integer :: i,j,k,l,m,xl,xh,yl,yh,xi,yi

      fct = 0.0

      imrms = image/rms

      do m = 1, n

        xi = int(x(m) + 0.5) 
        yi = int(y(m) + 0.5) 

        offx = x(m) - real(xi)
        offy = y(m) - real(yi)

        xl = xi-nb-1
        xh = xi+nb+1

        yl = yi-nb-1
        yh = yi+nb+1
        
        if (yl < 1 .or. xl < 1 .or. yh > nr .or. xh > nc) then

           fct(m) = -1000.0

        else

           imf = imrms(xl:xh,yl:yh)
           im1 = 0.0
           nbx = 2*nb + 3
           call smthmat(im1,imf,nbx,offx,offy) 

           do i = 1, 2*nb+1
              k = 2*nb+1 - i + 1

              do j = 1, 2*nb+1
                 l =  2*nb+1 - j + 1
                 
                 im2(k,j) = im1(i,j) 
                 im3(i,l) = im1(i,j) 
                 im4(k,l) = im1(i,j) 

              end do

           end do

           diff = 0.0

           diff = abs(im1 - im2) + abs(im1 - im3) + abs(im1-im4) + abs(im2-im3)
         
           fct(m) = sum(diff)/4.0/real((2*nb+1)*(2*nb+1) - 2*nb )/1.352
 
           if (fct(m) > 1000.0) fct(m) = 1000.0
           if (fct(m) < -1000.0) fct(m) = -1000.0

        end if

      end do

      end subroutine symm


      subroutine smthmat(ims, imb, n, offx, offy)

      integer, intent ( in ) :: n
      real, intent ( in ) :: imb(n,n), offx, offy
      real, intent ( in out ) :: ims(n-2,n-2)

      integer :: i,j,k,l,xi,yi 
      real :: x,y,f00,f01,f10,f11

      if ( offx <= 0.0 ) then 
        x = 1.0 + offx
        xi = 0
      else
        x = offx
        xi = 1
      end if

      if ( offy <= 0.0 ) then 
        y = 1.0 + offy
        yi = 0
      else
        y = offy
        yi = 1
      end if

      do i = 1, n - 2

       k = i + xi 

       do j = 1, n - 2

         l = j + yi

         f00 = imb(k,l)
         f01 = imb(k,l+1)
         f10 = imb(k+1,l)
         f11 = imb(k+1,l+1)
         
         ims(i,j) = f00*(1.0-x)*(1.0-y) + f10*x*(1.0-y) + f01*(1.0-x)*y + f11*x*y 

       end do

      end do

      end subroutine smthmat


      subroutine nmaskpix(mask,nc,nr,x,y,n,nmask,nbox)

      integer, intent (in) :: nc, nr, n, nbox
      real, intent (in) :: x(n), y(n)
      integer, intent (in) :: mask(nc,nr)
      integer, intent (in out) :: nmask(n)

      integer :: i,j,xl,xh,yl,yh,xi,yi


      do k = 1, n

        xi = int(x(k) + 0.5) 
        yi = int(y(k) + 0.5) 

        xl = max(1,xi-nbox)
        xh = min(nc,xi+nbox)

        yl = max(1,yi-nbox)
        yh = min(nr,yi+nbox)

        nmask(k) = 0

        nmask(k) = sum(mask(xl:xh,yl:yh))

      end do

      end subroutine nmaskpix

      SUBROUTINE indexx(n,arr,indx)
      integer, intent (in) :: n
      integer, intent (in out) :: indx(n)
      real, intent (in out)  :: arr(n)

      integer, parameter :: M=7,NSTACK=50
      integer :: i,indxt,ir,itemp,j,jstack,k,l,istack(NSTACK)
      real :: a

      do 11 j=1,n
        indx(j)=j
11    continue
      jstack=0
      l=1
      ir=n
1     if(ir-l.lt.M)then
        do 13 j=l+1,ir
          indxt=indx(j)
          a=arr(indxt)
          do 12 i=j-1,l,-1
            if(arr(indx(i)).le.a)goto 2
            indx(i+1)=indx(i)
12        continue
          i=l-1
2         indx(i+1)=indxt
13      continue
        if(jstack.eq.0)return
        ir=istack(jstack)
        l=istack(jstack-1)
        jstack=jstack-2
      else
        k=(l+ir)/2
        itemp=indx(k)
        indx(k)=indx(l+1)
        indx(l+1)=itemp
        if(arr(indx(l)).gt.arr(indx(ir)))then
          itemp=indx(l)
          indx(l)=indx(ir)
          indx(ir)=itemp
        endif
        if(arr(indx(l+1)).gt.arr(indx(ir)))then
          itemp=indx(l+1)
          indx(l+1)=indx(ir)
          indx(ir)=itemp
        endif
        if(arr(indx(l)).gt.arr(indx(l+1)))then
          itemp=indx(l)
          indx(l)=indx(l+1)
          indx(l+1)=itemp
        endif
        i=l+1
        j=ir
        indxt=indx(l+1)
        a=arr(indxt)
3       continue
          i=i+1
        if(arr(indx(i)).lt.a)goto 3
4       continue
          j=j-1
        if(arr(indx(j)).gt.a)goto 4
        if(j.lt.i)goto 5
        itemp=indx(i)
        indx(i)=indx(j)
        indx(j)=itemp
        goto 3
5       indx(l+1)=indx(j)
        indx(j)=indxt
        jstack=jstack+2
        if(jstack.gt.NSTACK)pause 'NSTACK too small in indexx'
        if(ir-i+1.ge.j-l)then
          istack(jstack)=ir
          istack(jstack-1)=i
          ir=j-1
        else
          istack(jstack)=j-1
          istack(jstack-1)=l
          l=i
        endif
      endif
      goto 1

      end subroutine indexx

      subroutine cr_rej(image, clean, mask, gain, read, sat, nc, nr)
      integer, intent (in) :: nc, nr
      real, intent (in) :: image(nc,nr)
      real, intent(in) :: gain, read, sat
      real, intent (in out) :: clean(nc, nr)
      integer, intent (in out) :: mask(nc,nr)

      integer, parameter :: nlp = 3
      real, parameter :: prefact = 4.0
      integer :: i, j, k, l
      integer :: il, ih, jl, jh
      real, dimension(nc,nr) :: imageresamp, smooth
      real, dimension(nc,nr) :: s, ss, sp, f, m3, m5, m7, lof, tmp
      real, dimension (nlp,nlp) :: lptr, frctimage
      real, dimension (2*nc, 2*nr) :: imagesub, imgcn
      integer, dimension(nc,nr) :: maskl, masksp, maskm, masks, maskspl, maskg
      real :: med, meds, spcut, spcutlow
     

      lptr(1,:) = (/ 0.0, -1.0,  0.0/)
      lptr(2,:) = (/-1.0,  4.0, -1.0/)
      lptr(3,:) = (/ 0.0, -1.0,  0.0/)
      
      mask = 0
      maskl = 0
      masksp = 0
      masks = 0
      maskm = 0
      maskspl = 0
      maskg = 0

      s  = 0.0
      sp = 0.0
      m3 = 0.0
      m5 = 0.0
      m7 = 0.0
      imagesub = 0.0

      imagesub(2*nc,2*nr) = image(nc,nr)

!$omp parallel do private(k,l)

      do i = 1, 2*nc
         k = int((i+1)/2)

         do j = 1, 2*nr
            l = int((j+1)/2)

            imagesub(i,j) = image(k,l)

         end do
      end do

      imgcn = 0.0

!$omp parallel do private(il,ih,jl,jh,k,l,frctimage)

      do i = int(nlp/2)+1, 2*nc - int(nlp/2) -1 
         il = i - int(nlp/2)
         ih = i + int(nlp/2)

         do j = int(nlp/2) + 1, 2*nr - int(nlp/2) - 1
            jl = j - int(nlp/2)
            jh = j + int(nlp/2)

            frctimage =  imagesub(il:ih,jl:jh)
            
            do k = 1, nlp
               do l = 1, nlp

                  imgcn(i,j) = imgcn(i,j) + lptr(k,l)*frctimage(k,l)

               end do
            end do
            
            if (imgcn(i,j) < 0.0) imgcn(i,j) = 0.0

         end do

      end do

!$omp parallel do

      do i = 1, nc
         do j = 1, nr

            imageresamp(i,j) = 0.25*(imgcn(2*i-1,2*j-1) + imgcn(2*i-1,2*j) + imgcn(2*i,2*j-1) + imgcn(2*i,2*j))

         end do
      end do


      call get_med_image_rl(image, nc, nr, med)

      call smoothbox_rl(image, m3, nc, nr, 3)
      call smoothbox_rl(image, m5, nc, nr, 5)

      smooth = 1.0/gain*sqrt(m5 + read**2)

      call smoothbox_rl(m3, m7, nc, nr, 7)

      f = m3 - m7
      
!
      m5 = abs(image - med)/sqrt(med)
!
       where (f < 0.01) f = 0.01

      s = imageresamp / smooth / 2.0 

      call smoothbox_rl(s, ss, nc, nr, 5)

      sp = s - ss

      lof = imageresamp / f

      call get_med_image_rl(sp, nc, nr, med)
      tmp = abs(sp - med)
      call get_med_image_rl(tmp, nc, nr, meds)

      spcut = 10.0*(meds * 1.48) + med
      spcutlow = 5.0*(meds * 1.48) + med

      where (lof > 2.0) maskl = 1
      where (sp > spcut) masksp = 1
      where (m5 > 5.0 ) maskm = 1
      where (image < sat) masks = 1

      mask = maskl * masksp * maskm * masks

      maskg = mask

!
!     Grow the mask comparing to spcutlow
!

!$omp parallel do private(i,j,ncl,nch,nrl,nrh)

      do i = 1, nc
       do j = 1, nr

          if (mask(i,j) == 1) then

            ncl = max((i-1),1)
            nch = min((i+1),nc)

            nrl = max((j-1),1)
            nrh = min((j+1),nr)

            maskg(ncl:nch,nrl:nrh) = 1

          end if

         end do
      end do

      where (sp < spcutlow) maskg = 0

      mask = maskl * maskg * maskm * masks

      print *, 'Total pixels masked = ', sum(mask)

      call fixpix(image, clean, mask, nc, nr, 5)

      return

      end subroutine cr_rej

      end module ImageMath

