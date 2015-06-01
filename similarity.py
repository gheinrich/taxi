import myutils
import time

MAX_DIST = 1e6

MAKE_TEST_SET = True

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

# use weighted average of closest trips
def predict_sim12(test_record, data):
    test_trip_len = len(test_record.coordinates)
    data_len = len(data)
    dmin = MAX_DIST
    prediction = []
    distances = []
    total_dist = 0
    n_dist = 0
    # compute distances to test record
    for i in xrange(data_len):
        d = trip_dist(data[i].coordinates, test_record.coordinates)
        distances.append(d)
        if d<MAX_DIST:
            total_dist = total_dist + d
            n_dist = n_dist + 1
    mean_dist = total_dist/n_dist
    # compute weighted average of final destinations
    prediction = [0,0]
    total_weight = 0
    for i in xrange(data_len):
        if distances[i]<MAX_DIST:
            weight = 1/(1+10*distances[i]/mean_dist)
            prediction = [
                prediction[0] + weight*data[i].coordinates[-1][0],
                prediction[1] + weight*data[i].coordinates[-1][1]]
            total_weight = total_weight + weight
    prediction = [prediction[0]/total_weight, prediction[1]/total_weight]
    return prediction

def main():
    n_entries = 20000

    # open file for writing
    f = open('out.csv','w')
    f.write("\"TRIP_ID\",\"LATITUDE\",\"LONGITUDE\"\n")

    print "loading training data..."
    data = myutils.load_data(max_entries = n_entries)

    print "loading test data..."

    if MAKE_TEST_SET:
        # fixed number of test samples
        n_test_entries = 320
        n_train_entries = n_entries - n_test_entries
        train_data = data[0:n_train_entries]
        test_data, ground_truth = myutils.make_test_data(data[n_train_entries:], n_test_entries)
        #test_data, ground_truth = myutils.make_test_data(data[0:n_train_entries], n_test_entries)
    else:
        train_data = data
        test_data = myutils.load_data(filename='../data/test.csv', max_entries = 1e6)
        n_test_entries = len(test_data)

    print "making predictions..."
    # make predictions and work out mean haversine distance to ground truth
    predictions = []
    total_dist = 0
    for i in xrange(n_test_entries):

        # make prediction
        prediction = predict_sim1(test_data[i], train_data)
        predictions.append(prediction)

        # compare against ground truth
        if MAKE_TEST_SET:
            d = myutils.HaversineDistance( prediction, ground_truth[i])
            total_dist = total_dist + d

        # write result
        f.write("\"" + test_data[i].trip_id + "\",")
        f.write(str(prediction[1]))
        f.write(",")
        f.write(str(prediction[0]))
        f.write("\n")

        #report progress
        if i % (n_test_entries/20) == 0:
            print "%d/%d" % (i,n_test_entries)

    # close file
    f.close()

    # report performace v.s. ground truth
    if MAKE_TEST_SET:
        print "Mean haversine distance: %f" % (total_dist / n_test_entries)

if __name__ == '__main__':
    t0 = time.time()
    main()
    print "Elapsed time: %f" % (time.time() - t0)

