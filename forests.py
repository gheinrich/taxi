import myutils
import time
import sklearn
from sklearn.ensemble import RandomForestRegressor
import affinity
import multiprocessing
from optparse import OptionParser
from sklearn.externals import joblib
import os

MAX_DIST = 1e6

MAKE_TEST_SET = True

def train(options):
    n_coordinates = options.n_coordinates
    assert n_coordinates != 0
    n_entries = options.n_train
    n_estimators = options.n_estimators
    directory = options.dir

    data,target,dummy_ids = myutils.load_data_ncoords(max_entries = n_entries,
                                                      n_coordinates=n_coordinates,
                                                      total_records=1e6)

    print "splitting data into training/test sets..."
    n_test_entries = 320
    data_train,data_test,target_train,target_test = sklearn.cross_validation.train_test_split(data,
                                                                                              target,
                                                                                              test_size=n_test_entries)

    print "building model with %d coordinates ..." % n_coordinates
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1)
    model.fit(data_train,target_train)

    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = "%s/model_%d_%d.pkl" % (directory, n_entries, n_coordinates)
    print "saving model into %s..." % filename
    joblib.dump(model, filename)

    print "computing test set predictions..."
    predictions = model.predict(data_test)
    dist = 0
    for i in xrange(n_test_entries):
        p1 = target_test[i]
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
    parser.add_option("-c", "--ncoordinates", dest="n_coordinates", type="int",
                  help="specify number of coordinates", default=0)
    parser.add_option("-n", "--ntrain", dest="n_train", type="int",
                  help="specify number of coordinates", default=1000)
    parser.add_option("-e", "--nestimators", dest="n_estimators", type="int",
                  help="specify number of RF estimators", default=100)
    parser.add_option("-d", "--dir", dest="dir", type="string",
                  help="input/output directory", default='models')
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

