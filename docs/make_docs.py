""" Read the schema file and make a documentation page """

import json


def write_file_contents(root, direc, filename, outputf):
    inputf = root + "/" + direc + "/" + filename
    with open(inputf) as json_file:
        data = json.load(json_file)

    for row in data['fields']:
        print(row['name'])
        outputf.write('<font size="3" color="black">%s (%s): %s</font>\n'%(
            row['name'],row['type'],row['doc']))
        outputf.write('</br>\n')
        outputf.write('</br>\n')


def write_docs(alert_type):
    """ 
    Parameters: alert_type -- can be "single" or "stack"
    """
    root = "../pipeline/alert_schemas/"
    direc = "schema_%s" %alert_type

    # Make the page title
    outputf = open("./%s.html" %direc, "w")
    outputf.write('<!doctype html>\n')
    outputf.write('<html>\n')
    outputf.write('<head>\n')
    outputf.write('<title>Documentation for the ZUDS Avro Schema</title>\n')
    outputf.write('</head>\n')
    outputf.write('<body>\n')

    outputf.write('<font size="5" color="blue"><b>ZUDS Avro Schema for Single Alerts</font></b>\n')
    outputf.write('</br>\n')
    outputf.write('</br>\n')

    # Read the schema file for ztf.alert
    outputf.write('<font size="4" color="black"><b>ztf.alert</font></b>\n')
    outputf.write('</br>\n')
    outputf.write('</br>\n')
    filename = 'alert.avsc'
    write_file_contents(root, direc, filename, outputf)
    outputf.write('</br>\n')
    outputf.write('</br>\n')

    # Same for ztf.alert.candidate
    outputf.write('<font size="4" color="black"><b>ztf.alert.candidate</font></b>')
    outputf.write('</br>')
    outputf.write('</br>')
    filename = 'candidate.avsc'
    write_file_contents(root, direc, filename, outputf)
    outputf.write('</br>')
    outputf.write('</br>')

    # Same for ztf.alert.light_curve
    outputf.write('<font size="4" color="black"><b>ztf.alert.light_curve</font></b>\n')
    outputf.write('</br>\n')
    outputf.write('</br>\n')
    filename = 'light_curve.avsc'
    write_file_contents(root, direc, filename, outputf)
    outputf.write('</br>\n')
    outputf.write('</br>\n')


if __name__=="__main__":
    write_docs("single")
    write_docs("stack")
