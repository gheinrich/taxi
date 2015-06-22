import myutils
import time
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.cluster import MiniBatchKMeans
import affinity
import multiprocessing
from optparse import OptionParser
from sklearn.externals import joblib
import numpy
import matplotlib.pyplot as plt
import os
import random
from operator import itemgetter

MAX_DIST = 1e6

MAKE_TEST_SET = False
VISUALIZE = False
SAVEFIGS = False
DEFAULT_N_TEST_ENTRIES = 150000
DISPLAY_PREDICTION_STATS = False

def get_model_id(model_sizes, n_coordinates):
    model = 0
    for j in xrange(len(model_sizes)):
        if model_sizes[j]<=n_coordinates:
            model = model_sizes[j]
    return model

def report(grid_scores, n_top=10):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              numpy.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

# Plot CV scores of a 2D grid search
def plotGridResults2D(x, y, x_label, y_label, grid_scores):

    scores = [abs(s[1]) for s in grid_scores]
    scores = numpy.array(scores).reshape(len(x), len(y))

    plt.figure()
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.RdYlGn)
    plt.xlabel(y_label)
    plt.ylabel(x_label)
    plt.colorbar()
    plt.xticks(numpy.arange(len(y)), y, rotation=45)
    plt.yticks(numpy.arange(len(x)), x)
    plt.title('Validation accuracy')


def train(options):
    n_coordinates = options.n_coordinates
    assert n_coordinates != 0
    n_entries = options.n_train
    n_estimators = options.n_estimators
    directory = options.dir

    data,target,dummy_ids = myutils.load_data_ncoords(filename = options.input_train,
                                                      max_entries = n_entries,
                                                      n_coordinates=n_coordinates,
                                                      total_records=1e6)

    print "splitting data into training/test sets..."
    n_test_entries = 320
    data_train,data_test,target_train,target_test = train_test_split(data,
                                                                     target,
                                                                     test_size=n_test_entries)

    print "building model with %d coordinates ..." % n_coordinates
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, oob_score=False)
    model.fit(data_train,target_train)

    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = "%s/model_%d_%d.pkl" % (directory, n_entries, n_coordinates)
    print "saving model into %s..." % filename
    joblib.dump(model, filename)

    print "computing test set predictions..."
    predictions = model.predict(data_test)
    dist = 0
    log_time = 0
    for i in xrange(n_test_entries):
        p1 = target_test[i]
        p2 = predictions[i]
        dist += myutils.HaversineDistance( p1, p2)
        t1 = target_test[i,2] + (n_coordinates-1)*myutils.TIME_STEP
        t2 = predictions[i,2] + (n_coordinates-1)*myutils.TIME_STEP
        log_time += (numpy.log(t1+1) - numpy.log(t2+1))**2
    print "Mean haversine distance: %f, RMSLE=%f" % (dist/n_test_entries, numpy.sqrt(log_time/n_test_entries))


def hypertune(options):
    n_coordinates = options.n_coordinates
    assert n_coordinates != 0
    n_entries = options.n_train
    n_estimators = options.n_estimators
    directory = options.dir

    data,target,dummy_ids = myutils.load_data_ncoords(filename = options.input_train,
                                                      max_entries = n_entries,
                                                      n_coordinates=n_coordinates,
                                                      total_records=-1)
    n_entries = data.shape[0]

    # create a scorer function out of our evaluation metric
    scorer = sklearn.metrics.make_scorer(myutils.mean_haversine_dist, greater_is_better=False)

    # range of hyperparameters to try
    n_estimators_range = numpy.array([1,  10,   20, 25, 50, 100, 200])
    max_depth_range = numpy.array([11,  35,  101,  251, 401,  501, 1000, 10000])

    # criss Grid search object
    grid = GridSearchCV(sklearn.ensemble.RandomForestRegressor(),
                    {'max_depth' : max_depth_range,
                    'n_estimators' : n_estimators_range},
                   cv=sklearn.cross_validation.KFold(n_entries, n_folds=10), n_jobs=-1,
                   scoring=scorer)
    grid.fit(data,target)

    report(grid.grid_scores_)

    plotGridResults2D(max_depth_range, n_estimators_range, 'max depth', 'n estimators', grid.grid_scores_)
    plt.show()

######################


def predict(options):
    directory = options.dir

    model_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                   19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                   35, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47, 48, 49, 51,
                   52, 53, 54, 57, 59, 60, 62, 63, 64, 65, 66, 67, 68, 70, 71, 72,
                   73, 76, 78, 79, 80, 83, 84, 85, 94, 97, 107, 110, 111, 112, 115,
                   134, 137, 138, 152, 155, 157, 163, 164, 192, 215, 220, 225, 238,
                   267, 327, 361, 369, 387, 400]
    #model_sizes = [1, 51, 107]

    print "loading test data..."
    if MAKE_TEST_SET:
        n_test_entries = DEFAULT_N_TEST_ENTRIES
        n_entries = min(200000, 100 * n_test_entries)

        data,target,ids = myutils.load_data_dense(filename=options.input_test,
                                                  max_entries = n_entries,
                                                  max_coordinates=500,
                                                  skip_records=0,
                                                  total_records=-1)
        data_test,ground_truth,ids_test = myutils.make_test_data_dense(data,
                                                                       target,
                                                                       ids,
                                                                       n_entries=n_test_entries,
                                                                       randomize = False)
    else:
        data_test,dummy_target,ids_test = myutils.load_data_dense(filename='../data/test.csv',
                                                  max_entries = 320,
                                                  max_coordinates=400)

    n_test_entries = data_test.shape[0]

    print "predicting %d entries..." % n_test_entries
    n_predicted = 0
    total_dist = 0
    total_log_time = 0
    predictions = numpy.zeros([n_test_entries,myutils.TARGET_LEN])
    visu = myutils.VisualizeTrip()
    for model_size in model_sizes:
        model_name = "%s/model_%d_%d.pkl" % (directory, options.n_train, model_size)

        entry_indices = []
        for i in xrange(n_test_entries):
            n_coordinates = myutils.get_n_coordinates(data_test[i])

            model_fit = get_model_id(model_sizes, n_coordinates)

            if model_fit == model_size:
                entry_indices.append(i)

        if len(entry_indices)>0:
            print "Opening model %s" % model_name
            model = joblib.load(model_name)

            # build input data
            n_features = myutils.get_n_features(model_size)

            # create vector to inculde all test samples for this model
            X = numpy.zeros([len(entry_indices),n_features])
            for idx,val in enumerate(entry_indices):
                X[idx] = data_test[val,0:n_features]

            # predict all test samples for this model
            Y = model.predict(X)
            for idx,val in enumerate(entry_indices):
                predictions[val,0:2] = Y[idx,0:2]
                if Y[idx,2]<=0:
                    print "!!!! time prediction=%f" % Y[idx,2]
                if model_size>1:
                    predictions[val,2] = Y[idx,2] + (model_size-1)*myutils.TIME_STEP
                else:
                    predictions[val,2] = 660

                if MAKE_TEST_SET:
                    # compare against ground truth
                    dest_truth = ground_truth[val,0:2]
                    p2 = predictions[val,0:2]
                    dist = myutils.HaversineDistance( dest_truth, p2)

                    dist_string = "haversine_dist=%.2f" % dist
                    total_dist = total_dist + dist
                    t1 = ground_truth[val,2]
                    t2 = predictions[val,2]
                    log_time = (numpy.log(t1+1) - numpy.log(t2+1))**2
                    total_log_time += log_time
                    time_diff_string = "time diff=%ds, gt=%d (root log=%.3f,running=%f)" % (int(t2-t1),
                                                                                            int(t1),
                                                                                            numpy.sqrt(log_time),
                                                                                     numpy.sqrt(total_log_time/(n_predicted+1)))
                    fig_filename = "fig_%05.2f_%05.2f_%d.png" % (dist, log_time, val)
                else:
                    dist_string = ""
                    time_diff_string = ""
                    dest_truth = None
                    fig_filename = "fig_%d_%s.png" % (val,ids_test[val])

                if SAVEFIGS:
                    plt.figure(val)
                    visu(myutils.get_polyline(data_test[val]),predictions[val],dest_truth)
                    plt.savefig(fig_filename)

                n_predicted = n_predicted + 1

                if DISPLAY_PREDICTION_STATS:
                    air_distance, land_distance = myutils.get_trip_stats(data_test[val])
                    if air_distance>0:
                        ratio = land_distance/air_distance
                    else:
                        ratio = 0
                    print "[%d/%d] Processing TRIP_ID='%s' ncoords=%d dist_ratio=%.2f model=%d %s %s" % (n_predicted,
                                                                                           n_test_entries,
                                                                                           ids_test[val],
                                                                                           n_coordinates,
                                                                                           ratio,
                                                                                           model_size,
                                                                                           dist_string,
                                                                                           time_diff_string)
    if MAKE_TEST_SET:
        print "Average haversine distance=%f, RMSLE=%.3f" % (total_dist/n_test_entries, numpy.sqrt(total_log_time/n_test_entries))

    print "writing output file..."
    # open files for writing
    fdest = open('out-destination.csv','w')
    fdest.write("\"TRIP_ID\",\"LATITUDE\",\"LONGITUDE\"\n")

    ftime = open('out-time.csv','w')
    ftime.write("\"TRIP_ID\",\"TRAVEL_TIME\"\n")

    for i in xrange(n_test_entries):
        # write result
        fdest.write("\"" + ids_test[i] + "\",")
        fdest.write(str(predictions[i,1]))
        fdest.write(",")
        fdest.write(str(predictions[i,0]))
        fdest.write("\n")

        # write result
        ftime.write("\"" + ids_test[i] + "\",")
        ftime.write(str(int(predictions[i,2])))
        ftime.write("\n")

    # close files
    fdest.close()
    ftime.close()

    if VISUALIZE:
        plt.close("all")
        max_figures = 4
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

def split(options):
    ratio = 10

    fin = open(options.input_train,'rb')
    ftrain = open('mytrain.csv','w')
    ftest = open('mytest.csv','w')

    n_parsed = 0
    for line in fin:
        if n_parsed == 0:
            # write header to both files
            ftrain.write(line)
            ftest.write(line)
        elif n_parsed % ratio ==0:
            ftest.write(line)
        else:
            ftrain.write(line)
        n_parsed += 1

    fin.close()
    ftrain.close()
    ftest.close()

def train_step2(options):

    print "loading test data..."
    directory = options.dir

    n_test_entries = DEFAULT_N_TEST_ENTRIES
    n_entries = min(200000, 100 * n_test_entries)
    data,target,ids = myutils.load_data_dense(filename=options.input_test,
                                              max_entries = n_entries,
                                              max_coordinates=500,
                                              skip_records=0,
                                              total_records=-1)
    data_made,ground_truth,ids_test = myutils.make_test_data_dense(data,
                                                                   target,
                                                                   ids,
                                                                   n_entries=n_test_entries,
                                                                   randomize = False)

    print "loading previous prediction..."
    predictions, prediction_ids = myutils.load_predictions(destination_file=options.input_destination_prediction,
                                                           time_file=options.input_time_prediction,
                                                           n_entries = n_test_entries)

    print "Average dist=%f, RMSLE=%f" % (myutils.mean_haversine_dist(predictions, ground_truth),
                                         myutils.RMSLE(predictions, ground_truth) )

    print "building new feature vectors..."
    data_nf = myutils.make_2nd_step_features(data_made, predictions)

    print "splitting set..."
    [data_train, data_test,
     data_nf_train, data_nf_test,
     predictions_train, predictions_test,
     target_train, target_test] = train_test_split(data_made,
                                                   data_nf,
                                                   predictions,
                                                   ground_truth,
                                                   test_size=DEFAULT_N_TEST_ENTRIES/4)

    print "Before training: train set average dist=%f, RMSLE=%f" % (myutils.mean_haversine_dist(predictions_train, target_train),
                                                                   myutils.RMSLE(predictions_train, target_train))
    print "Before training: test set average dist=%f RMSLE=%f" % (myutils.mean_haversine_dist(predictions_test, target_test),
                                                                  myutils.RMSLE(predictions_test, target_test))

    print "building model..."
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=500, n_jobs=-1)
    model.fit(data_nf_train,target_train)
    print str(model.feature_importances_)

    model_name = "%s/model_2nd_step.pkl" % directory
    print "saving model into %s..." % model_name
    if not os.path.exists(directory):
        os.makedirs(directory)
    joblib.dump(model, model_name)

    print "computing predictions..."
    predictions_train_2nd_step = model.predict(data_nf_train)
    predictions_test_2nd_step = model.predict(data_nf_test)

    print "After training: train set average dist=%f RMSLE=%f" % (myutils.mean_haversine_dist(predictions_train_2nd_step, target_train),
                                                                myutils.RMSLE(predictions_train_2nd_step, target_train))
    print "After training: test set average dist=%f RMSLE=%f" % (myutils.mean_haversine_dist(predictions_test_2nd_step, target_test),
                                                                 myutils.RMSLE(predictions_test_2nd_step, target_test))

def predict_step2(options):
    directory = options.dir

    n_test_entries = 320
    print "loading test set..."
    data_test,dummy_target,ids_test = myutils.load_data_dense(filename='../data/test.csv',
                                                              max_entries = n_test_entries,
                                                              max_coordinates=400)

    print "loading previous prediction..."
    predictions, prediction_ids = myutils.load_predictions(destination_file=options.input_destination_prediction,
                                                           n_entries = n_test_entries)

    print "building new feature vectors..."
    data = myutils.make_2nd_step_features(data_test, predictions)

    model_name = "%s/model_2nd_step.pkl" % directory
    print "loading 2nd step model '%s'..." % model_name
    model = joblib.load(model_name)

    print "making 2nd step predictions..."
    predictions_test_2nd_step = model.predict(data)
    assert(n_test_entries == predictions_test_2nd_step.shape[0])

    print "making time predictions..."
    myutils.adjust_predict_time(data_test, predictions_test_2nd_step)

    print "saving destination predictions..."
    fdest = open('out-destination-2ndstep.csv','w')
    ftime = open('out-time-2ndstep.csv','w')
    fdest.write("\"TRIP_ID\",\"LATITUDE\",\"LONGITUDE\"\n")
    ftime.write("\"TRIP_ID\",\"TRAVEL_TIME\"\n")
    for i in xrange(n_test_entries):
        # write result
        fdest.write("\"" + ids_test[i] + "\",")
        fdest.write(str(predictions_test_2nd_step[i,1]))
        fdest.write(",")
        fdest.write(str(predictions_test_2nd_step[i,0]))
        fdest.write("\n")

        # write result
        ftime.write("\"" + ids_test[i] + "\",")
        ftime.write(str(int(predictions_test_2nd_step[i,2])))
        ftime.write("\n")
    # close files
    fdest.close()
    ftime.close()


def cluster(options):
    n_coordinates = 1
    n_entries = options.n_train
    directory = options.dir

    print "loading data..."
    dummy_data,target,dummy_ids = myutils.load_data_ncoords(filename = options.input_train,
                                                      max_entries = n_entries,
                                                      n_coordinates=n_coordinates,
                                                      total_records=1e6)

    print "finding clusters..."
    n_clusters = 1000
    km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=5*n_clusters, init='random', n_init=20)
    km.fit(target)
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    model_name = "%s/model_cluster.pkl" % directory
    print "saving model into %s" % model_name
    joblib.dump(km, model_name)

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
    parser.add_option("", "--split",
                  action="store_true", dest="split", default=False,
                  help="split training file into train and test sets (90/10%)")
    parser.add_option("", "--input_train",
                  dest="input_train", default='../data/mytrain.csv',
                  help="input training file")
    parser.add_option("", "--input_test",
                  dest="input_test", default='../data/mytest.csv',
                  help="input test file")
    parser.add_option("", "--train_step2",
                  action="store_true", dest="train_step2", default=False,
                  help="adjust previous prediction")
    parser.add_option("", "--predict_step2",
                  action="store_true", dest="predict_step2", default=False,
                  help="adjust previous prediction")
    parser.add_option("", "--input_dest_prediction",
                  dest="input_destination_prediction", default='out-destination.csv',
                  help="input destination prediction file")
    parser.add_option("", "--input_time_prediction",
                  dest="input_time_prediction", default=None,
                  help="input time prediction file")
    parser.add_option("", "--hypertune",
                  action="store_true", dest="hypertune", default=False,
                  help="hyper parameter tuning")
    parser.add_option("", "--cluster",
                  action="store_true", dest="cluster", default=False,
                  help="final destination clustering")

    (options, args) = parser.parse_args()

    if options.train:
        train(options)
    elif options.predict:
        predict(options)
    elif options.gen_commands:
        gen_commands(options)
    elif options.stats:
        stats(options)
    elif options.split:
        split(options)
    elif options.train_step2:
        train_step2(options)
    elif options.predict_step2:
        predict_step2(options)
    elif options.hypertune:
        hypertune(options)
    elif options.cluster:
        cluster(options)
    else:
        train_and_test()

if __name__ == '__main__':
    t0 = time.time()
    main()
    print "Elapsed time: %f" % (time.time() - t0)

