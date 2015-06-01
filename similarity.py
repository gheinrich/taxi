import myutils
import time

MAX_DIST = 1e6

def trip_dist(full_trip, partial_trip):
    distance_metric = myutils.fastDistance
    #distance_metric = myutils.HaversineDistance
    end_weight = 0.5
    full_trip_len = len(full_trip)
    partial_trip_len = len(partial_trip)
    if partial_trip_len > full_trip_len:
        return MAX_DIST
    else:
        d_start = distance_metric(full_trip[0], partial_trip[0])
        d_end = distance_metric(full_trip[partial_trip_len-1], partial_trip[partial_trip_len-1])
        return d_start*(1-end_weight) + d_end*end_weight     

# find closest trip searching only on train data start
def predict_sim1(test_record, data):
    test_trip_len = len(test_record.coordinates)
    data_len = len(data)
    dmin = MAX_DIST
    prediction = []
    for i in xrange(data_len):
        d = trip_dist(data[i].coordinates, test_record.coordinates)
        if d < dmin:
            dmin = d
            prediction = data[i].coordinates[-1]
    return prediction

# return last coordinate of test record
def predict_last(test_record, data):
    return test_record.coordinates[-1]

# find closest trip searching from train data start onwards
def predict_sim2(test_record, data):
    test_trip_len = len(test_record.coordinates)
    data_len = len(data)
    dmin = MAX_DIST
    prediction = []
    for i in xrange(data_len):
        d = MAX_DIST
        train_trip_len = len(data[i].coordinates)
        for j in xrange(train_trip_len):
            x = trip_dist(data[i].coordinates[j:], test_record.coordinates)
            if x > d:
                break
            else:
                d = x
        if d < dmin:
            dmin = d
            prediction = data[i].coordinates[-1]
    return prediction

def main():
    n_entries = 200000
    
    if False:
        # adaptive number of test samples
        train_test_ratio = 0.99
        n_train_entries = int(n_entries * train_test_ratio)
        n_test_entries = int(n_entries * (1-train_test_ratio))
    else:
        # fixed number of test samples
        n_test_entries = 320
        n_train_entries = n_entries - n_test_entries
      
    print "loading training data..."
    data = myutils.load_data(max_entries = n_entries)
    
    print "loading test data..."
    train_data = data[0:n_train_entries]
    test_data, ground_truth = myutils.make_test_data(data[n_train_entries:], n_test_entries)
    
    print "making predictions..."
    # make predictions and work out mean haversine distance to ground truth
    predictions = []
    total_dist = 0
    for i in xrange(n_test_entries):
        prediction = predict_sim1(test_data[i], train_data)
        predictions.append(prediction)
        d = myutils.HaversineDistance( prediction, ground_truth[i])
        total_dist = total_dist + d
        if i % (n_test_entries/20) == 0:
            print "%d/%d" % (i,n_test_entries)
    
    print "Mean haversine distance: %f" % (total_dist / n_test_entries)
    
if __name__ == '__main__':
    t0 = time.time()
    main()    
    print "Elapsed time: %f" % (time.time() - t0)

