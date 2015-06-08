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
import matplotlib.pyplot as plt
import os
import random

MAX_DIST = 1e6

MAKE_TEST_SET = False
VISUALIZE = True

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
        # draw dashed line between prediction and ground truth
        if (prediction is not None) and (truth is not None):
            plt.plot([truth[0],prediction[0]],[truth[1],prediction[1]],'--',c=color)

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

    #model_sizes = [1,2,5,8,10,12,14,17,20,23,26,30,32,34,36,38,40,43,47,50,60,70,
    #               80,90,100,110,120,130,160,220]
    model_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                   19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                   35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51,
                   52, 53, 54, 57, 59, 60, 62, 63, 64, 65, 66, 67, 68, 70, 71, 72,
                   73, 76, 78, 79, 80, 83, 84, 85, 94, 97, 107, 110, 111, 112, 115,
                   134, 137, 138, 152, 155, 157, 163, 164, 192, 215, 220, 225, 238,
                   267, 327, 361, 369, 387, 400]
    #model_sizes = [1,20]

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
                    dist_string = "haversine_distance=%.2f" % dist
                    total_dist = total_dist + dist
                else:
                    dist_string = ""

                n_predicted = n_predicted + 1
                air_distance, land_distance = myutils.get_trip_stats(data_test[i])
                if air_distance>0:
                    ratio = land_distance/air_distance
                else:
                    ratio = 0
                print "[%d/%d] Processing TRIP_ID='%s' ncoordinates=%d dist_ratio=%.2f model=%d %s" % (n_predicted,
                                                                                   n_test_entries,
                                                                                   ids_test[i],
                                                                                   n_coordinates,
                                                                                   ratio,
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

    if VISUALIZE:
        visu = VisualizeTrip()
        max_figures = 12
        trips_per_figure = max(1,int(n_test_entries/max_figures))
        for i in xrange(n_test_entries):
            if i%trips_per_figure ==0:
                plt.figure(i/trips_per_figure+1)
            # visualize
            if MAKE_TEST_SET:
                truth = ground_truth[i]
            else:
                truth = None
            visu(myutils.get_polyline(data_test[i]),predictions[i],truth)
        plt.show()

def train_and_test():
    assert(0)


def gen_commands(options):

    data_test,dummy_target,ids_test = myutils.load_data_dense(filename='../data/test.csv',
                                                  max_entries = 320,
                                                  max_coordinates=400)

    n_test_entries = data_test.shape[0]

    size_list = []
    for i in xrange(n_test_entries):
        s = myutils.get_n_coordinates(data_test[i])
        if not (s in size_list):
            size_list.append(s)
    size_list.sort()

    for size in size_list:
        print "python forests.py --train --dir %s -c %d -n %d -e %d" % (options.dir,
                                                                        size,
                                                                        options.n_train,
                                                                        options.n_estimators)


    return 0

def stats(options):
    n_entries = options.n_train
    data,target,ids = myutils.load_data_dense(max_entries = n_entries,
                                              max_coordinates=200,
                                              total_records=1.5*1e6,
                                              load_taxi_id=True)
    drivers = {}
    for i in xrange(n_entries):
        air_distance, land_distance = myutils.get_trip_stats(data[i])
        if air_distance>5:
            taxi_id = myutils.get_taxi_id(data[i])
            ratio = land_distance/air_distance
            if taxi_id in drivers:
                d = drivers[taxi_id]
                drivers[taxi_id] = [(d[0]*d[1]+ratio)/(d[1]+1), d[1]+1]
            else:
                drivers[taxi_id] = [ratio, 1]

    print drivers


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
    parser.add_option("-g", "--generate",
                  action="store_true", dest="gen_commands", default=False,
                  help="generate train commands")
    parser.add_option("-s", "--stats",
                  action="store_true", dest="stats", default=False,
                  help="generate train commands")

    (options, args) = parser.parse_args()

    if options.train:
        train(options)
    elif options.predict:
        predict(options)
    elif options.gen_commands:
        gen_commands(options)
    elif options.stats:
        stats(options)
    else:
        train_and_test()

if __name__ == '__main__':
    t0 = time.time()
    main()
    print "Elapsed time: %f" % (time.time() - t0)

