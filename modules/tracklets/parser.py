import sys
import os
import json

from xml.etree.ElementTree import fromstring
from xmljson import Parker

xp = Parker()

def usage():
    print('Converts a tracklets XML file to JSON')
    print('Usage: python parser.py [tracklet labels XML]')

def xml_to_dict(data):
    return xp.data(fromstring(data))

def clean_items_list(data, startTime):
    tracklets = data.get('tracklets', {})
    if type(tracklets.get('item')) is not list:
        item = tracklets.get('item', {})
        cleaned = []
        objID = 0
        objType = item.get('objectType', '')
        start = item.get('first_frame', 0)
        h, w, l = item.get('h', 0), item.get('w', 0), item.get('l', 0)
        for frame, pose in enumerate(item.get('poses', {}).get('item', [])):
            cleaned.append({
                'object_id': objID,
                'object_type': objType,
                'timestamp': startTime + start + frame,
                'tx': pose['tx'],
                'ty': pose['ty'],
                'tz': pose['tz'],
                'rx': pose['rx'],
                'ry': pose['ry'],
                'rz': pose['rz'],
                'width': w,
                'height': h,
                'depth': l,
            })
    else:
        items = tracklets.get('item', [])
        cleaned = []
        for count, item in enumerate(items):
            objID = count
            objType = item.get('objectType', '')
            start = item.get('first_frame', 0)
            h, w, l = item.get('h', 0), item.get('w', 0), item.get('l', 0)
            for frame, pose in enumerate(item.get('poses', {}).get('item', [])):
                cleaned.append({
                    'object_id': objID,
                    'object_type': objType,
                    'timestamp': startTime + start + frame,
                    'tx': pose['tx'],
                    'ty': pose['ty'],
                    'tz': pose['tz'],
                    'rx': pose['rx'],
                    'ry': pose['ry'],
                    'rz': pose['rz'],
                    'width': w,
                    'heigth': h,
                    'depth': l,
                })
    return cleaned

# XXX figure out how to get timestamp
def timestamp_from_msg():
    return 0

if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage()
        sys.exit()

    try:
        f = open(sys.argv[1])
        data = f.read().replace('\n', '')
        f.close()
    except:
        print('Unable to read file: %s' % sys.argv[1])
        sys.exit()

    dataDict = xml_to_dict(data)
    cleaned = clean_items_list(dataDict, timestamp_from_msg())
    dataJson = json.dumps({
        'data': cleaned,
    })
    print(dataJson)
