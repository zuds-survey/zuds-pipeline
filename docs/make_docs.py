""" Read the schema file and make a documentation page """

import json

root = "../pipeline/alert_schemas/"
direc = "schema_single"

# Make the page title
outputf = open("./%s.html" %direc, "w")
outputf.write('<!doctype html>')
outputf.write('<html>')
outputf.write('<head>')
outputf.write('<title>Documentation for the ZUDS Avro Schema</title>')
outputf.write('</head>')
outputf.write('<body>')

outputf.write('<font size="5" color="blue"><b>ZUDS Avro Schema for Single Alerts</font></b>')
outputf.write('</br>')
outputf.write('</br>')


def write_file_contents(filename):
    inputf = root + "/" + direc + "/" + filename
    with open(inputf) as json_file:
        data = json.load(json_file)

    for row in data['fields']:
        print(row['name'])
        outputf.write('<font size="3" color="black">%s (%s): %s</font>'%(
            row['name'],row['type'],row['doc']))
        outputf.write('</br>')


# Read the schema file for ztf.alert
outputf.write('<font size="4" color="black"><b>ztf.alert</font></b>')
outputf.write('</br>')
outputf.write('</br>')
filename = 'alert.avsc'
write_file_contents(filename)
outputf.write('</br>')
outputf.write('</br>')

# Same for ztf.alert.candidate
outputf.write('<font size="4" color="black"><b>ztf.alert.candidate</font></b>')
outputf.write('</br>')
outputf.write('</br>')
filename = 'candidate.avsc'
write_file_contents(filename)
outputf.write('</br>')
outputf.write('</br>')

# Same for ztf.alert.light_curve
outputf.write('<font size="4" color="black"><b>ztf.alert.light_curve</font></b>')
outputf.write('</br>')
outputf.write('</br>')
filename = 'light_curve.avsc'
write_file_contents(filename)
outputf.write('</br>')
outputf.write('</br>')

