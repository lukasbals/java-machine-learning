package at.fhv.lu.sparkML;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import scala.Tuple2;

import java.util.HashMap;
import java.util.Map;

/**
 * @author Lukas Bals
 */
public class RandomForestUtil {
    static void train(JavaRDD<LabeledPoint> training, JavaRDD<LabeledPoint> test) {
        long start = System.currentTimeMillis();

        // Train a RandomForest model.
        // Empty categoricalFeaturesInfo indicates all features are continuous.
        int numClasses = 10;
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        int numTrees = 20;
        String featureSubsetStrategy = "auto"; // Let the algorithm choose.
        String impurity = "gini";
        int maxDepth = 10;
        int maxBins = 32;
        int seed = 12345;

        RandomForestModel model = RandomForest.trainClassifier(training, numClasses,
                categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins,
                seed);

        long end = System.currentTimeMillis();
        double trainingTime = (end - start) / 1000.;

        testAndPrintResult(model, test, trainingTime);
    }

    private static void testAndPrintResult(RandomForestModel decisionTreeModel, JavaRDD<LabeledPoint> test, double trainingTime) {
        JavaPairRDD<Double, Double> predictionAndLabel = test.mapToPair(p -> new Tuple2<>(decisionTreeModel.predict(p.features()), p.label()));
        double accuracy = predictionAndLabel.filter(pl -> pl._1().equals(pl._2())).count() / (double) test.count();

        System.out.println("RandomForest accuracy: " + accuracy);
        System.out.println("RandomForest training time: " + trainingTime);
    }
}
