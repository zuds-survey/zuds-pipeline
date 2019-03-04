function table_to_csv(source, write_header) {
    const columns = ['mjd', 'filter', 'flux', 'fluxerr', 'zp', 'zpsys', 'lim_mag', 'mag', 'magerr', 'instrument'];
    const nrows = source.get_length();
    const lines = [];

    if (write_header) {
        lines.push(columns.join(','));
    }

    for (let i = 0; i < nrows; i++) {
        let row = [];
        for (let j = 0; j < columns.length; j++) {
            const column = columns[j];
            try {
                row.push(source.data[column][i].toString());
            } catch (error) {
                if (column === 'zp'){
                    row.push(25.);
                } else if (column === 'zpsys') {
                    row.push('ab');
                } else {
                    throw error;
                }
            }
        }
        lines.push(row.join(','))
    }
    return lines.join('\n').concat('\n')
}


const filename = 'objname.csv';
filetext = '';
var write_header = true;

for (let i=0; i < toggle.labels.length; i++){
    if (toggle.active.includes(i)) {
        if (slider.value > 0) {
            filetext += table_to_csv(eval('bin' + i).data_source, write_header);
        } else {
            filetext += table_to_csv(eval('obs' + i).data_source, write_header);
        }
        write_header = false;
    }
}
const blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename)
} else {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.target = '_blank';
    link.style.visibility = 'hidden';
    link.dispatchEvent(new MouseEvent('click'));
}

