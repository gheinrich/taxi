import myutils
import time
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
import affinity
import multiprocessing
from optparse import OptionParser
from sklearn.externals import joblib
import numpy
import os

MAX_DIST = 1e6

MAKE_TEST_SET = True

def get_model_id(model_sizes, n_coordinates):
    model = 0
    for j in xrange(len(model_sizes)):
        if model_sizes[j]<=n_coordinates:
            model = model_sizes[j]
    return model

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
    data_train,data_test,target_train,target_test = train_test_split(data,
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

def predict(options):
    directory = options.dir

    model_sizes = [1,2,5,8,10,12,14,17,20,23,26,30,32,34,36,38,40,43,47,50,60,70,
                   80,90,100,110,120,130,160,220]
    #model_sizes = [1,47,50]

    print "loading test data..."
    if MAKE_TEST_SET:
        n_test_entries = 320
        n_entries = 2*n_test_entries
        data,target,ids = myutils.load_data_dense(max_entries = n_entries,
                                                  max_coordinates=200,
                                                  skip_records=1e6,
                                                  total_records=1.5*1e6)
        data_test,ground_truth = myutils.make_test_data_dense(data,target, n_entries=n_test_entries)
        ids_test=['dummy'] * n_test_entries
    else:
        data_test,dummy_target,ids_test = myutils.load_data_dense(filename='../data/test.csv',
                                                  max_entries = 320,
                                                  max_coordinates=400)

    n_test_entries = data_test.shape[0]

    print "predicting %d entries..." % n_test_entries
    n_predicted = 0
    total_dist = 0
    predictions = numpy.zeros([n_test_entries,2])
    for model_size in model_sizes:
        model_name = "%s/model_%d_%d.pkl" % (directory, options.n_train, model_size)
        print "Opening model %s" % model_name
        model = joblib.load(model_name)
        for i in xrange(n_test_entries):
            n_coordinates = myutils.get_n_coordinates(data_test[i])

            model_fit = get_model_id(model_sizes, n_coordinates)

            if model_fit == model_size:

                # build input data
                n_features = myutils.get_n_features(model_size)
                x = numpy.zeros([1,n_features])
                x[0] = data_test[i,0:n_features]

                # predict
                y = model.predict(x)
                predictions[i] = y[0]

                if MAKE_TEST_SET:
                    # compare against ground truth
                    p1 = ground_truth[i]
                    p2 = y[0]
                    dist = myutils.HaversineDistance( p1, p2)
                    dist_string = "haversine_distance=%f" % dist
                    total_dist = total_dist + dist
                else:
                    dist_string = ""

                n_predicted = n_predicted + 1
                print "[%d/%d] Processing TRIP_ID='%s' ncoordinates=%d model=%d %s" % (n_predicted,
                                                                                   n_test_entries,
                                                                                   ids_test[i],
                                                                                   n_coordinates,
                                                                                   model_size,
                                                                                       dist_string)
    if MAKE_TEST_SET:
        print "Average haversine distance=%f" % (total_dist/n_test_entries)
    else:
        print "writing output file..."
        # open file for writing
        f = open('out.csv','w')
        f.write("\"TRIP_ID\",\"LATITUDE\",\"LONGITUDE\"\n")

        for i in xrange(n_test_entries):
            # write result
            f.write("\"" + ids_test[i] + "\",")
            f.write(str(predictions[i,1]))
            f.write(",")
            f.write(str(predictions[i,0]))
            f.write("\n")

        # close file
        f.close()

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
    parser.add_option("-p", "--predict",
                  action="store_true", dest="predict", default=False,
                  help="predict")

    (options, args) = parser.parse_args()

    if options.train:
        train(options)
    elif options.predict:
        predict(options)
    else:
        train_and_test()

if __name__ == '__main__':
    t0 = time.time()
    main()
    print "Elapsed time: %f" % (time.time() - t0)

