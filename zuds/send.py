""" This code sends ZUDS alerts to IPAC topics """
import os
import pickle
import json
import confluent_kafka
import fastavro
import avro.schema
from io import BytesIO


__all__ = ['send_alert']


def combine_schemas(schema_files):
    """Combine multiple nested schemas into a single schema.
    Taken from Eric's lsst-dm Github page
    """
    known_schemas = avro.schema.Names()

    for s in schema_files:
        schema = load_single_avsc(s, known_schemas)
    # using schema.to_json() doesn't fully propagate the nested schemas
    # work around as below
    props = dict(schema.props)
    fields_json = [field.to_json() for field in props['fields']]
    props['fields'] = fields_json
    return props


def load_single_avsc(file_path, names):
    """Load a single avsc file.
    Taken from Eric's lsst-dm Github page
    """

    curdir = os.path.dirname(__file__)
    file_path = os.path.join(curdir, '..', 'alert_schemas', file_path)

    with open(file_path) as file_text:
        json_data = json.load(file_text)
    schema = avro.schema.SchemaFromJSONData(json_data, names)
    return schema


def send(topicname, records, schema):
    """ Send an avro "packet" to a particular topic at IPAC
    Parameters
    ----------
    topic: name of the topic, e.g. ztf_20191221_programid2_zuds
    records: a list of dictionaries
    schema: schema definition
    """
    # Parse the schema file
    #schema_definition = fastavro.schema.load_schema(schemafile)

    # Write into an in-memory "file"
    out = BytesIO()
    fastavro.writer(out, schema, records)
    out.seek(0) # go back to the beginning

    # Connect to the IPAC Kafka brokers
    producer = confluent_kafka.Producer({'bootstrap.servers': 'ztfalerts04.ipac.caltech.edu:9092,ztfalerts05.ipac.caltech.edu:9092,ztfalerts06.ipac.caltech.edu:9092'})

    # Send an avro alert
    producer.produce(topic=topicname, value=out.read())
    producer.flush()


def send_alert(alert_object):
    """
    Send an alert to IPAC. Figure out the alert type,
    and write to the relevant topic.
    """
    # Placeholder -- alert creation date UTC
    # Eventually this will come from the alert

    if alert_object.sent:
        raise RuntimeError(f'Refusing to send alert '
                           f'{alert_object.alert["objectId"]},'
                           f' alert has already been sent out.')


    ac = alert_object.created_at
    alert_date = f'{ac.year}{ac.month:02d}{ac.day:02d}'
    alert = alert_object.to_dict()

    imtype = alert['candidate']['alert_type']
    if imtype == 'single':
        schema = combine_schemas(
            ["schema_single/candidate.avsc", "schema_single/light_curve.avsc",
             "schema_single/alert.avsc"])
        topicname = "ztf_%s_programid2_zuds" %alert_date
        send(topicname, [alert], schema)
    elif imtype == 'stack':
        schema = combine_schemas(
            ["schema_stack/candidate.avsc", "schema_stack/light_curve.avsc",
             "schema_stack/alert.avsc"])
        topicname = "ztf_%s_programid2_zuds_stack" %alert_date
        send(topicname, [alert], schema)


if __name__=="__main__":
    alerts = pickle.load(open('zuds_alerts.test2.pkl','rb'))
    for ii,alert in enumerate(alerts):
        print(ii)
        print(alert['candidate']['alert_type'])
        send_alert(alert)
