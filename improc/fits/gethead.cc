#include "fitsio.h"
#include "gethead.h"
#include <stdio.h>
#include <stdexcept>
using namespace std;


void printerror( int status )
{
    /*****************************************************/
    /* Print out cfitsio error messages and exit program */
    /*****************************************************/


    if (status)
    {
       char *bp;
       FILE *stream;
       size_t size;

       stream = open_memstream(&bp, &size);

       fits_report_error(stream, status); /* print error report */
       fflush(stream);

       throw std::runtime_error(bp); /* throw c++ exception which is propagated up to python */

    }
    return;
}

void readheader(char* fname, char* key, int datatype, void* value){

    fitsfile *fptr;       /* pointer to the FITS file, defined in fitsio.h */

    int status, nkeys, keypos, hdutype, ii, jj;
    char card[FLEN_CARD];   /* standard string lengths defined in fitsioc.h */
    char comment[100];

    status = 0;

    if ( fits_open_file(&fptr, fname, READONLY, &status) )
         printerror( status );

    if ( fits_movabs_hdu(fptr, 1, &hdutype, &status) )
         printerror( status );

    if ( fits_read_key(fptr, datatype, key, value, comment, &status) )
         printerror( status );

    return;
}
