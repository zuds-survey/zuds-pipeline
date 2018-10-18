#include "fitsio.h"
#include "fitsutil.hh"
#include "sethead.hh"


void updateheader(char* fname, char* key, int datatype, void* value){

    fitsfile *fptr;       /* pointer to the FITS file, defined in fitsio.h */

    int status, nkeys, keypos, hdutype, ii, jj;
    char card[FLEN_CARD];   /* standard string lengths defined in fitsioc.h */
    char comment[100];

    status = 0;

    if ( fits_open_file(&fptr, fname, READONLY, &status) )
         printerror( status );

    if ( fits_movabs_hdu(fptr, 1, &hdutype, &status) )
         printerror( status );

    if ( fits_update_key(fptr, datatype, key, value, comment, &status) )
         printerror( status );

    return;
}

