#from rnn_minibatch import *
import pickle
import csv
import time
from collections import namedtuple
from array import *
import random
import copy
import math
import scipy.sparse
import datetime
import numpy

Record = namedtuple('Record', 'trip_id, origin_call, timestamp,coordinates')

METADATA_LEN = 2

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

def get_n_coordinates(entry):
    max_features=entry.shape[0]
    len = 0
    # find length of this record
    for i in xrange(METADATA_LEN,max_features):
        if entry[i]==0:
            break
        len = len + 1
    # make sure this is an even number
    assert(len%2 == 0)
    return len/2

def get_n_features(n_coordinates):
    return 2*n_coordinates + METADATA_LEN

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


def load_data_sparse(filename='../data/train.csv', max_entries=100, max_features=20):
    data=scipy.sparse.lil_matrix((max_entries, max_features))
    target=numpy.empty([max_entries,2])
    first = True
    ids=[]
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
                    polyline = eval(row[polyline_idx])
                    polyline_len = len(polyline)
                    if polyline_len > 0:
                        # save ids
                        ids.append(row[trip_id_idx])
                        # save minute of day and week of day into feature matrix
                        timestamp = eval(row[timestamp_idx])
                        dt = datetime.datetime.fromtimestamp(timestamp)
                        time = dt.hour*60 + dt.minute
                        weekday = dt.weekday()
                        metadata=[time,weekday]
                        assert METADATA_LEN == len(metadata)
                        data[n_entries,:METADATA_LEN]=metadata
                        # save coordinates (up to max_features-metadata) into feature matrix
                        n_coordinates = min(max_features-METADATA_LEN,polyline_len)
                        data[n_entries,METADATA_LEN:METADATA_LEN+n_coordinates] = numpy.ravel(polyline)[:n_coordinates]
                        # save end destination into target matrix
                        target[n_entries]=polyline[-1]
                        n_entries = n_entries + 1
                        if n_entries % (max_entries/20) == 0:
                            print "%d/%d" % (n_entries,max_entries)
            if n_entries >= max_entries:
                break
    return data.tocsr(),target,ids

def load_data_dense(filename='../data/train.csv', max_entries=100, max_coordinates=20, skip_records = 0, total_records=-1):
    max_features = get_n_features(max_coordinates)
    data=numpy.empty([max_entries,max_features])
    target=numpy.empty([max_entries,2])
    first = True
    ids=[]
    if total_records>0:
        step = int((total_records-skip_records)/max_entries)
    else:
        step = 1
    with open(filename, 'rb') as f:
        input_sequence = []
        n_entries = 0
        reader = csv.reader(f)
        n_parsed = 0
        for row in reader:
            if first:
                missing_data_idx = row.index("MISSING_DATA")
                origin_call_idx = row.index("ORIGIN_CALL")
                polyline_idx = row.index("POLYLINE")
                trip_id_idx = row.index("TRIP_ID")
                timestamp_idx = row.index("TIMESTAMP")
                first = False
            else:
                n_parsed = n_parsed + 1
                if (n_parsed % step != 0) or (n_parsed < skip_records):
                    continue
                if row[missing_data_idx] == "False":
                    polyline = eval(row[polyline_idx])
                    polyline_len = len(polyline)
                    if polyline_len > 0:
                        # save ids
                        ids.append(row[trip_id_idx])
                        # save minute of day and week of day into feature matrix
                        timestamp = eval(row[timestamp_idx])
                        dt = datetime.datetime.fromtimestamp(timestamp)
                        time = dt.hour*60 + dt.minute
                        weekday = dt.weekday()
                        metadata=[time,weekday]
                        assert METADATA_LEN == len(metadata)
                        data[n_entries,:METADATA_LEN]=metadata
                        # save coordinates (up to max_coordinates) into feature matrix
                        n_coordinates = min(max_coordinates,polyline_len)
                        n_features = get_n_features(n_coordinates)
                        data[n_entries,METADATA_LEN:n_features] = numpy.ravel(polyline[:n_coordinates])
                        # save end destination into target matrix
                        target[n_entries]=polyline[-1]
                        n_entries = n_entries + 1
                        if n_entries % (max_entries/20) == 0:
                            print "%d/%d" % (n_entries,max_entries)
            if n_entries >= max_entries:
                break
    return data[0:n_entries],target[0:n_entries],ids[0:n_entries]

def load_data_ncoords(filename='../data/train.csv', max_entries=100, n_coordinates=20, total_records=-1):
    n_features = get_n_features(n_coordinates)
    data=numpy.empty([max_entries,n_features])
    target=numpy.empty([max_entries,2])
    first = True
    ids=[]
    if total_records>0:
        step = int(total_records/max_entries)
    else:
        step = 1
    with open(filename, 'rb') as f:
        input_sequence = []
        n_entries = 0
        reader = csv.reader(f)
        n_parsed = 0
        for row in reader:
            if first:
                missing_data_idx = row.index("MISSING_DATA")
                origin_call_idx = row.index("ORIGIN_CALL")
                polyline_idx = row.index("POLYLINE")
                trip_id_idx = row.index("TRIP_ID")
                timestamp_idx = row.index("TIMESTAMP")
                first = False
            else:
                n_parsed = n_parsed + 1
                if n_parsed % step != 0:
                    continue
                if row[missing_data_idx] == "False":
                    polyline = eval(row[polyline_idx])
                    polyline_len = len(polyline)
                    if polyline_len >= n_coordinates:
                        # save ids
                        ids.append(row[trip_id_idx])
                        # save minute of day and week of day into feature matrix
                        timestamp = eval(row[timestamp_idx])
                        dt = datetime.datetime.fromtimestamp(timestamp)
                        time = dt.hour*60 + dt.minute
                        weekday = dt.weekday()
                        metadata=[time,weekday]
                        assert METADATA_LEN == len(metadata)
                        data[n_entries,:METADATA_LEN]=metadata
                        data[n_entries,METADATA_LEN:n_features] = numpy.ravel(polyline[n_coordinates])
                        # save end destination into target matrix
                        target[n_entries]=polyline[-1]
                        n_entries = n_entries + 1
                        if n_entries % (max_entries/20) == 0:
                            print "%d/%d" % (n_entries,max_entries)
            if n_entries >= max_entries:
                break
    print "loaded %d entries out of %d max" % (n_entries, max_entries)
    return data[0:n_entries],target[0:n_entries],ids[0:n_entries]

def make_test_data_dense(data, target, n_entries=100, required_n_coordinates=-1):
    ground_truth=numpy.empty([n_entries,2])
    data_len = data.shape[0]
    max_features = data.shape[1]
    test_data = numpy.zeros([n_entries, max_features])
    for i in xrange(n_entries):
        while True:
            # pick random index within data
            idx = random.randint(0, data_len - 1)
            n_coordinates = get_n_coordinates(data[idx])
            if required_n_coordinates==-1:
                # pick random number of coordinates
                l = random.randint(1, n_coordinates)
            else:
                l = required_n_coordinates
            if n_coordinates>=l:
                n_features = get_n_features(l)
                test_data[i,0:n_features] = data[idx,0:n_features]
                ground_truth[i] = target[idx]
                break
    return test_data,ground_truth

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
