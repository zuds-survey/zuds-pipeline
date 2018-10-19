#include "fitsio.h"
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
