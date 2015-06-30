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

def is_cv_match(hour_start, n_coordinates):
    cv_hours = [18, 8.5, 17.75, 4, 14.5]
    margin = 0 # 20./3600 # 20 seconds
    hour_end = hour_start + TIME_STEP*n_coordinates/3600.
    for hour in cv_hours:
        if hour_start<hour and hour_end>hour-margin:
            n_snapshot_coordinates = min(int((hour-hour_start)*3600/TIME_STEP), n_coordinates)
            return n_snapshot_coordinates
    return 0

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
    data=numpy.empty([max_entries,max_features],dtype=numpy.float32)
    target=numpy.empty([max_entries,TARGET_LEN],dtype=numpy.float32)
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
                        dt = datetime.datetime.utcfromtimestamp(timestamp)
                        time = dt.hour*60 + dt.minute + dt.second/60.
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
                        target[n_entries,2]=(polyline_len-1) * TIME_STEP
                        n_entries = n_entries + 1
                        if n_entries % progress_report_step == 0:
                            print "%d/%d" % (n_entries,max_entries)
            if n_entries >= max_entries:
                break
    return data[0:n_entries],target[0:n_entries],ids[0:n_entries]

def load_predictions(destination_file='../out-destination.csv', time_file=None, n_entries=0):
    predictions=numpy.zeros([n_entries,TARGET_LEN])
    trip_ids = [""] * n_entries
    first = True
    n_parsed = 0
    
    if destination_file is not None:
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
                    predictions[n_parsed,0:2]=prediction
                    trip_ids[n_parsed]=trip_id
                    n_parsed += 1
        assert(n_parsed == n_entries)
    if time_file != None:
        n_parsed = 0
        print "Opening %s..." % time_file
        with open(time_file, 'rb') as f:
            first = True
            reader = csv.reader(f)
            for row in reader:
                if first:
                    trip_id_idx = row.index("TRIP_ID")
                    travel_time_idx = row.index("TRAVEL_TIME")
                    first = False
                else:
                    assert(n_parsed < n_entries)
                    prediction = eval(row[travel_time_idx])
                    trip_id = row[trip_id_idx]
                    if destination_file is not None:
                        assert(trip_id == trip_ids[n_parsed])
                    trip_ids[n_parsed]=trip_id
                    predictions[n_parsed,2]=prediction
                    n_parsed += 1
    return predictions, trip_ids

def save_predictions(predictions, ids, dest_filename='out-destination.csv', time_filename='out-time.csv'):
    n_entries = predictions.shape[0]
    print "saving %d predictions into %s and %s..." % (n_entries, dest_filename, time_filename)
    fdest = open(dest_filename,'w')
    ftime = open(time_filename,'w')
    fdest.write("\"TRIP_ID\",\"LATITUDE\",\"LONGITUDE\"\n")
    ftime.write("\"TRIP_ID\",\"TRAVEL_TIME\"\n")
    for i in xrange(n_entries):
        # write result
        fdest.write("\"" + ids[i] + "\",")
        fdest.write(str(predictions[i,1]))
        fdest.write(",")
        fdest.write(str(predictions[i,0]))
        fdest.write("\n")

        # write result
        ftime.write("\"" + ids[i] + "\",")
        ftime.write(str(int(predictions[i,2])))
        ftime.write("\n")
    # close files
    fdest.close()
    ftime.close()    

#def write_cv_set(input_filename, output_filename, max_entries=50000):
    #target=numpy.empty([max_entries,TARGET_LEN])
    #first = True
    #ids=[]
    #print "Opening %s for reading..." % input_filename
    #fin = open(input_filename, 'rb')
    #print "Opening %s for writing..." % output_filename
    #fout = open(output_filename, 'w')
    
    #n_entries = 0
    #reader = csv.reader(fin)
    #n_parsed = 0
    #for row in reader:
        #if first:
            #missing_data_idx = row.index("MISSING_DATA")
            #polyline_idx = row.index("POLYLINE")
            #timestamp_idx = row.index("TIMESTAMP")
            #trip_id_idx = row.index("TRIP_ID")
            #first = False
        #else:
            #if row[missing_data_idx] == "False":
                #polyline = eval(row[polyline_idx])
                #polyline_len = len(polyline)
                #if polyline_len > 0:
                    #print row
                    ## save ids
                    #ids.append(row[trip_id_idx])
                    ## save minute of day and week of day into feature matrix
                    #timestamp = eval(row[timestamp_idx])
                    #dt = datetime.datetime.utcfromtimestamp(timestamp)
                    #time = dt.hour*60 + dt.minute
                    ## save end destination into target matrix
                    #target[n_entries,0:2]=polyline[-1]
                    ## save total trip time
                    #target[n_entries,2]=(polyline_len-1) * TIME_STEP
                    #n_entries = n_entries + 1
                #if n_entries >= max_entries:
                    #break
            
    #fin.close()
    #fout.close()

def load_data_ncoords(filename='../data/train.csv', max_entries=100, n_coordinates=20, total_records=-1):
    n_features = get_n_features(n_coordinates)
    data=numpy.empty([max_entries,n_features],dtype=numpy.float32)
    target=numpy.empty([max_entries,TARGET_LEN],dtype=numpy.float32)
    first = True
    ids=[]
    step = max(1,int(total_records/max_entries))

    #
    n_rejected=0

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

                        #
                        #n_features = get_n_features(polyline_len)
                        #entry = numpy.zeros([n_features])
                        #entry[METADATA_LEN:n_features]=numpy.ravel(polyline)
                        #air_distance, land_distance = get_trip_stats(entry)
                        #if land_distance>10 and air_distance<1:
                        #    n_rejected +=1
                        #    print "reject (%d)" % n_rejected
                        #    continue

                        # save ids
                        ids.append(row[trip_id_idx])
                        # save minute of day and week of day into feature matrix
                        timestamp = eval(row[timestamp_idx])
                        dt = datetime.datetime.utcfromtimestamp(timestamp)
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

# this way of making test vectors is meant to replicate the Kaggle test
# set, i.e. it is taking snapshots at 6pm, 8.30am, 5.45pm, 4am, 2.30pm
def make_test_data_cv(data, target, ids, n_entries=100):
    data_len = data.shape[0]
    max_features = data.shape[1]
    if n_entries>data_len:
        n_entries = data_len
    ground_truth=numpy.empty([n_entries,TARGET_LEN])
    test_data = numpy.zeros([n_entries, max_features])
    test_ids = [""] * n_entries
    n_found_entries = 0
    for i in xrange(data_len):
        entry = data[i]
        n_coordinates = get_n_coordinates(entry)
        hour_start = entry[0]/60.
        assert(hour_start >= 0 and hour_start <24)
        n_snapshot_coordinates = is_cv_match(hour_start, n_coordinates)
        if n_snapshot_coordinates>0:
            n_features = get_n_features(n_snapshot_coordinates)
            test_data[n_found_entries,0:n_features] = entry[0:n_features]
            ground_truth[n_found_entries] = target[i]
            test_ids[n_found_entries] = ids[i]
            n_found_entries += 1
    print "found %d entries for CV / %d total entries" % (n_found_entries,data_len)
    
    #f = open('cvset.txt','w')
    #f.write("\"TRIP_ID\",\"POLYLINE_LENGTH\"\n")
    #for i in xrange(n_found_entries):
    #    f.write("\"%s\",\"%d\"\n" % (test_ids[i], get_n_coordinates(test_data[i])) )
    #f.close()
    
    return test_data[:n_found_entries],ground_truth[:n_found_entries],test_ids[:n_found_entries]


def make_test_data_cv_alt(data, target, ids, n_entries=100):
    data_len = data.shape[0]
    max_features = get_n_features(2) # two coordinates
    if n_entries>data_len:
        n_entries = data_len
    ground_truth=numpy.empty([n_entries,TARGET_LEN])
    test_data = numpy.zeros([n_entries, max_features])
    test_ids = [""] * n_entries
    n_found_entries = 0
    for i in xrange(data_len):
        entry = data[i]
        n_coordinates = get_n_coordinates(entry)
        hour_start = entry[0]/60.
        assert(hour_start >= 0 and hour_start <24)
        n_snapshot_coordinates = is_cv_match(hour_start, n_coordinates)
        if n_snapshot_coordinates>0:
            n_features_in = get_n_features(n_snapshot_coordinates)
            n_features_out = get_n_features(2)
            test_data[n_found_entries,:n_features_out-2] = entry[0:n_features_out-2]
            test_data[n_found_entries,n_features_out-2:n_features_out]= entry[n_features_in-2:n_features_in]
            ground_truth[n_found_entries] = target[i]
            test_ids[n_found_entries] = ids[i]
            n_found_entries += 1
    print "found %d entries for CV / %d total entries" % (n_found_entries,data_len)
    
    #f = open('cvset.txt','w')
    #f.write("\"TRIP_ID\",\"POLYLINE_LENGTH\"\n")
    #for i in xrange(n_found_entries):
    #    f.write("\"%s\",\"%d\"\n" % (test_ids[i], get_n_coordinates(test_data[i])) )
    #f.close()
    
    return test_data[:n_found_entries],ground_truth[:n_found_entries],test_ids[:n_found_entries]


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

def make_alt_features(data, randomize=False):
    feature_len = 8
    data_len = data.shape[0]
    data_out = numpy.zeros([data_len, feature_len],dtype=numpy.float32)
    for i in xrange(data_len):
        entry = data[i]
        t = entry[0]
        weekday = entry[1]
        taxi_id = entry[2]
        start_lng = entry[3]
        start_lat = entry[4]
        n_coordinates = get_n_coordinates(entry)
        if randomize:
            l = min(random.randint(1, n_coordinates),random.randint(1, n_coordinates))
        else:
            l = n_coordinates
        n_features = get_n_features(l)
        end_lng = entry[n_features - 2]
        end_lat = entry[n_features - 1]
        air_distance, land_distance = get_trip_stats(entry)
        if air_distance>0:
            dist_ratio = land_distance/air_distance
        else:
            dist_ratio = -1
        lng_direction = end_lng - start_lng
        lat_direction = end_lat - start_lat
        new_entry = [weekday, t, taxi_id, start_lng, start_lat,
                     end_lng, end_lat,
                     dist_ratio]
        data_out[i]= new_entry
        assert(feature_len == len(new_entry))
    return data_out


def mean_haversine_dist (predictions, ground_truth):
    n_entries = predictions.shape[0]
    assert(n_entries == ground_truth.shape[0])
    total_dist = 0
    for i in xrange(n_entries):
        dist = HaversineDistance(predictions[i], ground_truth[i])
        total_dist += dist
    mean = total_dist/n_entries
    return (mean)

def RMSLE (predictions, ground_truth):
    n_entries = predictions.shape[0]
    assert(n_entries == ground_truth.shape[0])
    total_log_time = 0
    for i in xrange(n_entries):
        t1 = predictions[i,2]
        t2 = ground_truth[i,2]
        assert (t1>=0)
        assert (t2>=0)
        total_log_time += (numpy.log(t1+1) - numpy.log(t2+1))**2
    mean = total_log_time/n_entries
    return math.sqrt(mean)

def adjust_predict_time(data,predictions,ground_truth=None):
    n_entries = data.shape[0]
    assert(n_entries == predictions.shape[0])
    for i in xrange(n_entries):
        entry = data[i]
        air_distance, land_distance = get_trip_stats(entry)
        n_coordinates = get_n_coordinates(entry)
        time_so_far = TIME_STEP * (n_coordinates-1)
        if time_so_far>0:
            bird_speed_so_far = air_distance/time_so_far
        else:
            bird_speed_so_far = 0.01
        bird_speed_so_far = max(0.0025,bird_speed_so_far)
        bird_speed_so_far = min(0.02,bird_speed_so_far)
        print "speed=%f km/s" % bird_speed_so_far
        n_features = get_n_features(n_coordinates)
        end_lng = entry[n_features - 2]
        end_lat = entry[n_features - 1]
        end_point = [end_lng, end_lat]
        prediction_lng = predictions[i][0]
        prediction_lat = predictions[i][1]
        prediction_point = [prediction_lng,prediction_lat]
        remaining_dist_prediction = HaversineDistance(end_point, prediction_point)
        remaining_time_prediction = remaining_dist_prediction/bird_speed_so_far
        prediction = time_so_far+remaining_time_prediction
        print "time_so_far=%f dist_so_far=%f remaining_dist=%f remaining time=%f prediction=%f gt=%f" % (time_so_far,
                                                                                                   air_distance,
                                                                                                   remaining_dist_prediction,
                                                                                                   remaining_time_prediction,
                                                                                                   prediction,
                                                                                                   ground_truth[i,2] if ground_truth is not None else -1)
        predictions[i,2] = prediction

def test_set_stats(data):
    n_entries = data.shape[0]
    total_coordinates = 0.0
    for i in xrange(n_entries):
        total_coordinates += get_n_coordinates(data[i])
    return (total_coordinates/n_entries)

if __name__ == "__main__":
    n_entries = 1000
    t0 = time.time()
    data = load_data(max_entries = n_entries)
    #pickle.dump(data, open('data_'+str(n_entries)+'.pickle', 'wb'))
    #print str(data)
    print "Elapsed time: %f" % (time.time() - t0)
    input("Press Enter to continue...")
