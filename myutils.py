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
import matplotlib.pyplot as plt
import binascii

Record = namedtuple('Record', 'trip_id, origin_call, timestamp,coordinates')

METADATA_LEN = 3
TARGET_LEN = 3
TIME_STEP = 15

class VisualizeTrip:
    def __init__(self):
        self.color_index = 0

    def __call__(self,polyline,prediction=None,truth=None):
        colors = "bgrcmyk"
        color = colors[self.color_index % len(colors)]
        self.color_index += 1
        x = numpy.array(polyline)
        # start of itinerary
        plt.plot(x[0,0],x[0,1],'>',c=color)
        # rest of itinerary
        plt.plot(x[:,0],x[:,1],'-',c=color)
        # prediction
        if prediction is not None:
            plt.plot(prediction[0],prediction[1],'o',c=color)
        # ground truth
        if truth is not None:
            plt.plot(truth[0],truth[1],'D',c=color)
        # draw dotted line between prediction and ground truth
        if (prediction is not None) and (truth is not None):
            plt.plot([truth[0],prediction[0]],[truth[1],prediction[1]],':',c=color)
        # draw dashed line between prediction and last known coordinate
        if (prediction is not None):
            plt.plot([x[-1,0],prediction[0]],[x[-1,1],prediction[1]],'--',c=color)

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

def get_polyline(entry):
    n_coordinates = get_n_coordinates(entry)
    polyline = []
    for i in xrange(n_coordinates):
        polyline.append([entry[METADATA_LEN+2*i],entry[METADATA_LEN+2*i+1]])
    return polyline

def get_taxi_id(entry):
    assert(METADATA_LEN>2)
    return int(entry[2])

def get_trip_stats(entry):
    polyline = get_polyline(entry)
    poly_len = len(polyline)
    air_distance = HaversineDistance(polyline[0], polyline[-1])
    land_distance = 0
    for i in xrange(1,poly_len):
        land_distance += HaversineDistance(polyline[i], polyline[i-1])
    return air_distance, land_distance

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

def load_data_dense(filename='../data/train.csv', max_entries=100, max_coordinates=20, skip_records = 0, total_records=-1):
    max_features = get_n_features(max_coordinates)
    data=numpy.empty([max_entries,max_features])
    target=numpy.empty([max_entries,TARGET_LEN])
    first = True
    ids=[]
    if total_records>0:
        step = int((total_records-skip_records)/max_entries)
    else:
        step = 1
    if max_entries/20 > 1:
        progress_report_step = int(max_entries/20)
    else:
        progress_report_step = 1
    print "Opening %s..." % filename
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
                taxi_id_idx = row.index("TAXI_ID")
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
                        metadata=[time,weekday,int(eval(row[taxi_id_idx]))]
                        assert METADATA_LEN == len(metadata)
                        data[n_entries,:METADATA_LEN]=metadata
                        # save coordinates (up to max_coordinates) into feature matrix
                        n_coordinates = min(max_coordinates,polyline_len)
                        n_features = get_n_features(n_coordinates)
                        data[n_entries,METADATA_LEN:n_features] = numpy.ravel(polyline[:n_coordinates])
                        # save end destination into target matrix
                        target[n_entries,0:2]=polyline[-1]
                        # save total trip time
                        target[n_entries,2]=polyline_len * TIME_STEP
                        n_entries = n_entries + 1
                        if n_entries % progress_report_step == 0:
                            print "%d/%d" % (n_entries,max_entries)
            if n_entries >= max_entries:
                break
    return data[0:n_entries],target[0:n_entries],ids[0:n_entries]

def load_predictions(destination_file='../out-destination.csv', n_entries=0):
    predictions=numpy.empty([n_entries,2])
    trip_ids = []
    first = True
    n_parsed = 0
    print "Opening %s..." % destination_file
    with open(destination_file, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            if first:
                trip_id_idx = row.index("TRIP_ID")
                latitude_idx = row.index("LATITUDE")
                longitude_idx = row.index("LONGITUDE")
                first = False
            else:
                assert(n_parsed < n_entries)
                prediction = [eval(row[longitude_idx]),eval(row[latitude_idx])]
                trip_id = row[trip_id_idx]
                predictions[n_parsed]=prediction
                trip_ids.append(trip_id)
                n_parsed += 1
    assert(n_parsed == n_entries)
    return predictions, trip_ids

def load_data_ncoords(filename='../data/train.csv', max_entries=100, n_coordinates=20, total_records=-1):
    n_features = get_n_features(n_coordinates)
    data=numpy.empty([max_entries,n_features])
    target=numpy.empty([max_entries,TARGET_LEN])
    first = True
    ids=[]
    step = max(1,int(total_records/max_entries))
    print "Opening %s..." % filename
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
                taxi_id_idx = row.index("TAXI_ID")
                first = False
            else:
                if n_parsed % step != 0:
                    n_parsed += 1
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
                        metadata=[time,weekday,int(eval(row[taxi_id_idx]))]
                        assert METADATA_LEN == len(metadata)
                        data[n_entries,:METADATA_LEN]=metadata
                        data[n_entries,METADATA_LEN:n_features] = numpy.ravel(polyline[:n_coordinates])
                        # save end destination into target matrix
                        target[n_entries,0:2]=polyline[-1]
                        # save remaining trip time
                        target[n_entries,2]=(polyline_len - n_coordinates) * TIME_STEP
                        n_entries = n_entries + 1
                        n_parsed += 1
                        if n_entries % (max_entries/20) == 0:
                            print "%d/%d" % (n_entries,max_entries)
            if n_entries >= max_entries:
                break
    print "loaded %d entries out of %d max" % (n_entries, max_entries)
    return data[0:n_entries],target[0:n_entries],ids[0:n_entries]

def make_test_data_dense(data, target, ids, n_entries=100, required_n_coordinates=-1, randomize=True):
    data_len = data.shape[0]
    max_features = data.shape[1]
    if n_entries>data_len:
        n_entries = data_len
    ground_truth=numpy.empty([n_entries,TARGET_LEN])
    test_data = numpy.zeros([n_entries, max_features])
    test_ids = [""] * n_entries
    for i in xrange(n_entries):
        while True:
            # pick random index within data
            if randomize:
                idx = random.randint(0, data_len - 1)
            else:
                if n_entries > data_len:
                    idx = binascii.crc32(str(i)) % data_len
                else:
                    idx = i
            n_coordinates = get_n_coordinates(data[idx])
            if required_n_coordinates==-1:
                if randomize:
                    # pick random number of coordinates
                    #l = random.randint(1, n_coordinates)
                    l = min(random.randint(1, n_coordinates),random.randint(1, n_coordinates))
                else:
                    if n_coordinates>1:
                        l = 1 + (binascii.crc32(str(i)) % (n_coordinates-1))
                    else:
                        l = 1
            else:
                l = required_n_coordinates
            if n_coordinates>=l:
                n_features = get_n_features(l)
                test_data[i,0:n_features] = data[idx,0:n_features]
                ground_truth[i] = target[idx]
                test_ids[i] = ids[idx]
                break
    return test_data,ground_truth,test_ids

def make_2nd_step_features(data, predictions):
    feature_len = 13
    data_len = data.shape[0]
    assert(data_len == predictions.shape[0])
    data_out = numpy.zeros([data_len, feature_len])
    for i in xrange(data_len):
        entry = data[i]
        t = entry[0]
        weekday = entry[1]
        taxi_id = entry[2]
        start_lng = entry[3]
        start_lat = entry[4]
        n_coordinates = get_n_coordinates(entry)
        n_features = get_n_features(n_coordinates)
        end_lng = entry[n_features - 2]
        end_lat = entry[n_features - 1]
        air_distance, land_distance = get_trip_stats(entry)
        if air_distance>0:
            dist_ratio = land_distance/air_distance
        else:
            dist_ratio = -1
        lng_direction = end_lng - start_lng
        lat_direction = end_lat - start_lat
        lng_prediction = predictions[i][0]
        lat_prediction = predictions[i][1]
        #new_entry = [t, weekday, taxi_id, n_coordinates,
        #             start_lng, start_lat,
        #             end_lng, end_lat,
        #             dist_ratio, lng_direction, lat_direction,
        #             lng_prediction, lat_prediction]
        new_entry = [0, 0, 0, 0,
                     start_lng, start_lat,
                     end_lng, end_lat,
                     dist_ratio, 0, 0,
                     lng_prediction, lat_prediction]
        data_out[i]= new_entry
        assert(feature_len == len(new_entry))
    return data_out

def compute_average_haversine_dist (predictions, ground_truth):
    n_entries = predictions.shape[0]
    assert(n_entries == ground_truth.shape[0])
    total_dist = 0
    for i in xrange(n_entries):
        dist = HaversineDistance(predictions[i], ground_truth[i])
        total_dist += dist
    return (total_dist/n_entries)

if __name__ == "__main__":
    n_entries = 1000
    t0 = time.time()
    data = load_data(max_entries = n_entries)
    #pickle.dump(data, open('data_'+str(n_entries)+'.pickle', 'wb'))
    #print str(data)
    print "Elapsed time: %f" % (time.time() - t0)
    input("Press Enter to continue...")
