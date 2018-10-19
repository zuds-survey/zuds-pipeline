#include "fitsio.h"
#include "fitsutil.hh"
#include "sethead.hh"


void updateheader(char* fname, char* key, int datatype, void* value){

    fitsfile *fptr;       /* pointer to the FITS file, defined in fitsio.h */

    int status, nkeys, keypos, hdutype, ii, jj;
    char card[FLEN_CARD];   /* standard string lengths defined in fitsioc.h */
    char comment[10];

    # initialize to empty string to pass blank comment
    comment[0] = '\0';


    float* intervalf;
    char** intervalc;
    int*   intervali;

    status = 0;

    if ( fits_open_file(&fptr, fname, READWRITE, &status) )
         printerror( status );

    if ( fits_movabs_hdu(fptr, 1, &hdutype, &status) )
         printerror( status );


    if (datatype == TFLOAT){
        intervalf = (float *)value;
        if ( fits_update_key(fptr, datatype, key, intervalf, comment, &status) ){
            printerror( status );
        }
    } else if ( datatype == TSTRING ) {
        intervalc = (char**)value;
        if ( fits_update_key(fptr, datatype, key, intervalc, comment, &status) ){
            printerror( status );
        }
    } else if (datatype == TINT) {
        intervali = (int *) value;
        if ( fits_update_key(fptr, datatype, key, intervali, comment, &status) ){
            printerror( status );
        }
    }

    if ( fits_close_file(fptr, &status) )
         printerror( status );

    return;
}

