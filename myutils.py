#from rnn_minibatch import *
import pickle
import csv
import time
from collections import namedtuple
from array import *
import random
import copy
import math

Record = namedtuple('Record', 'trip_id, origin_call, timestamp,coordinates')

def fastDistance(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def HaversineDistance(p1, p2):
    #returns the distance in km
    REarth = 6371
    lat1 = p1[1]
    lat2 = p2[1]
    lat = abs(lat1-lat2)*math.pi/180
    lon = abs(p1[0]-p2[0])*math.pi/180
    lat1 = lat1*math.pi/180
    lat2 = lat2*math.pi/180
    a = math.sin(lat/2)**2 + math.cos(lat1)*math.cos(lat2)*(math.sin(lon/2)**2)
    d = 2*math.atan2(math.sqrt(a),math.sqrt(1-a))
    d = REarth*d
    return d

def load_data(filename='../data/train.csv', max_entries=100):

    data=[]
    first = True
    with open(filename, 'rb') as f:
        input_sequence = []
        n_entries = 0
        reader = csv.reader(f)
        for row in reader:
            if first:
                missing_data_idx = row.index("MISSING_DATA")
                origin_call_idx = row.index("ORIGIN_CALL")
                polyline_idx = row.index("POLYLINE")
                trip_id_idx = row.index("TRIP_ID")
                timestamp_idx = row.index("TIMESTAMP")
                first = False
            else:
                if row[missing_data_idx] == "False":
                    trip_id = row[trip_id_idx]
                    origin_call = row[origin_call_idx]
                    timestamp = eval(row[timestamp_idx])
                    polyline = eval(row[polyline_idx])
                    if len(polyline) > 0:
                        record = Record(trip_id=trip_id, origin_call=origin_call,
                                    timestamp=timestamp, coordinates=polyline)
                        data.append(record)
                        n_entries = n_entries + 1
                        if n_entries % (max_entries/20) == 0:
                            print "%d/%d" % (n_entries,max_entries)
            if n_entries > max_entries:
                break
    return data

def make_test_data(input_data, n_entries=100):
    test_data = []
    ground_truth = []
    data_len = len(input_data)
    for i in xrange(n_entries):
        idx = random.randint(0, data_len - 1)
        input_record = input_data[idx]
        ground_truth.append(input_record.coordinates[-1])
        l = random.randint(1, len(input_record.coordinates))
        test_record = Record(trip_id=input_record.trip_id,
                             origin_call=input_record.origin_call,
                             timestamp=input_record.timestamp,
                             coordinates=input_record.coordinates[0:l])
        test_data.append(test_record)
    return test_data,ground_truth

if __name__ == "__main__":
    n_entries = 1000
    t0 = time.time()
    data = load_data(max_entries = n_entries)
    #pickle.dump(data, open('data_'+str(n_entries)+'.pickle', 'wb'))
    #print str(data)
    print "Elapsed time: %f" % (time.time() - t0)
    input("Press Enter to continue...")
