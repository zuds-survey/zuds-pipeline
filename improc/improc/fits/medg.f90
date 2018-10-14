!-
!- File    : medg.f
!- ------------------------
!- Original by PN
!- Some comments and mods added by RS 2008/06/26
!- Modified for f2py by DG 2018/10/14
!-
program proc
    use ImageClass
    use ImageMath

    implicit none
    !   !
    !   !-    nc+nos is the number of columns
    !   !-    nr is the number of rows
    !   !

    integer :: nc, nr, i, j, stat
    real, allocatable :: image(:, :)
    real :: med, var, zp, see, lmt, pix
    character (len = 120) :: filenm
    character (len = 72) :: comment

    read(*, *) filenm

    call get_header(trim(filenm), 'NAXIS1', nc, comment)
    call get_header(trim(filenm), 'NAXIS2', nr, comment)

    allocate(image(nc, nr), stat = stat)

    call get_image(trim(filenm), image, nc, nr)

    call get_header(trim(filenm), 'MAGZP', zp, comment)
    call get_header(trim(filenm), 'SEEING', see, comment)

    call get_med_image(image, nc, nr, med)
    image = abs(image - med)
    call get_med_image(image, nc, nr, var)

    lmt = -2.5 * log10(3.0 * sqrt(3.14159 * see * see) * var * 1.48) + zp

    print *, lmt, zp, see, pix

    call put_header(trim(filenm), 'MEDSKY', med, 'Median sky in cts')
    call put_header(trim(filenm), 'SKYSIG', var, 'Median skysig in cts')
    call put_header(trim(filenm), 'LMT_MG', lmt, '3-sig limiting mag')

    en