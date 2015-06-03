import myutils
import time
import sklearn
from sklearn.ensemble import RandomForestRegressor

MAX_DIST = 1e6

MAKE_TEST_SET = False


def main():
    n_entries = 800000

    print "loading training data..."
    data,target,dummy_ids = myutils.load_data_sparse(max_entries = n_entries, max_features=50)

    print "loading test data..."
    if MAKE_TEST_SET:
        n_test_entries = 320
        test_data,ground_truth = myutils.make_test_data_sparse(data,target,
                                                               n_entries=n_test_entries)
    else:
        test_data,dummy_target,test_ids = myutils.load_data_sparse(filename='../data/test.csv',
                                                     max_entries = n_entries,
                                                     max_features=50)
        n_test_entries = len(test_ids)

    print "building model..."
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, n_jobs=-1)
    model.fit(data,target)

    print "predicting..."
    predictions = model.predict(test_data)

    if MAKE_TEST_SET:
        print "checking predictions..."
        dist = 0
        for i in xrange(n_test_entries):
            p1 = ground_truth[i]
            p2 = predictions[i]
            dist = dist + myutils.HaversineDistance( p1, p2)
        print "Mean haversine distance: %f" % (dist / n_test_entries)
    else:
        # open file for writing
        f = open('out.csv','w')
        f.write("\"TRIP_ID\",\"LATITUDE\",\"LONGITUDE\"\n")

        for i in xrange(n_test_entries):
            # write result
            f.write("\"" + test_ids[i] + "\",")
            f.write(str(predictions[i,1]))
            f.write(",")
            f.write(str(predictions[i,0]))
            f.write("\n")

        # close file
        f.close()

if __name__ == '__main__':
    t0 = time.time()
    main()
    print "Elapsed time: %f" % (time.time() - t0)

