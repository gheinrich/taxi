import myutils
import time
import sklearn
from sklearn.ensemble import RandomForestRegressor
import affinity
import multiprocessing
from optparse import OptionParser
from sklearn.externals import joblib

MAX_DIST = 1e6

MAKE_TEST_SET = True

def train(options):
    assert options.n_coordinates != 0
    n_entries = 300

    data,target,dummy_ids = myutils.load_data_ncoords(max_entries = n_entries,
                                                      n_coordinates=options.n_coordinates,
                                                      total_records=1e6)

    print "building model with %d coordinates ..." % options.n_coordinates
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, n_jobs=-1)
    model.fit(data,target)

    filename = "./models/model_%d_%d.pkl" % (n_entries, options.n_coordinates)
    print "saving model into %s..." % filename
    joblib.dump(model, filename)

    if MAKE_TEST_SET:
        n_test_entries = 320
        test_data,ground_truth = myutils.make_test_data_sparse(data,target, n_entries=n_test_entries,
                                                               n_features=10)
        predictions = model.predict(test_data)
        print "checking predictions..."
        dist = 0
        for i in xrange(n_test_entries):
            p1 = ground_truth[i]
            p2 = predictions[i]
            dist = dist + myutils.HaversineDistance( p1, p2)
        print "Mean haversine distance: %f" % (dist / n_test_entries)

def train_and_test():

    n_entries = 5000

    print "loading training data..."
    data,target,dummy_ids = myutils.load_data_dense(max_entries = n_entries, max_features=20, total_records=1e6)

    print "loading test data..."
    if MAKE_TEST_SET:
        n_test_entries = 320
        test_data,ground_truth = myutils.make_test_data_sparse(data,target, n_entries=n_test_entries,
                                                               n_features=20)
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

def main():
    affinity.set_process_affinity_mask(0, 2**multiprocessing.cpu_count()-1)

    parser = OptionParser()
    parser.add_option("-n", "--ncoordinates", dest="n_coordinates", type="int",
                  help="specify number of coordinates", default=0)
    parser.add_option("-t", "--train",
                  action="store_true", dest="train", default=False,
                  help="train only")

    (options, args) = parser.parse_args()

    if options.train:
        train(options)
    else:
        train_and_test()

if __name__ == '__main__':
    t0 = time.time()
    main()
    print "Elapsed time: %f" % (time.time() - t0)

