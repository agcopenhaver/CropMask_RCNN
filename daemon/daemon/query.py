import psycopg2
import json
import pprint

def save_query():

    try:
        conn = psycopg2.connect(
            database='AfricaSandbox', 
            user='postgis', 
            host='sandbox.crowdmapper.org', 
            password='P0stG1S')
    except:
        print("I am unable to connect to the database")

    cur = conn.cursor()

    testkmlid = 'ZA0608542'
    cur.execute("""select gid, ST_AsGeoJSON(geom_clean) from qaqcfields where name='ZA0608542'""")

    rows = cur.fetchall()

    geom_dict = {}

    for i, row in enumerate(rows):
        geom_dict[i]=row[1]

    with open('data/test_geojson.dict', 'w') as f:
        json.dump(geom_dict, f, sort_keys=True, indent=4)

    return 'Dictionary succesfully saved'


def show_json():

    with open('data/test_geojson.dict', 'r') as f:
        data = json.load(f)
        return pprint.pformat(data,indent=4)
