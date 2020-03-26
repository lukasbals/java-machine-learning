package at.fhv.lu.sparkML;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import scala.Tuple2;

import java.util.HashMap;
import java.util.Map;

/**
 * @author Lukas Bals
 */
public class DecisionTreeUtil {

    static void train(JavaRDD<LabeledPoint> training, JavaRDD<LabeledPoint> test) {
        long start = System.currentTimeMillis();

        // Set parameters.
        //  Empty categoricalFeaturesInfo indicates all features are continuous.
        int numClasses = 10;
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        String impurity = "gini";
        int maxDepth = 10;
        int maxBins = 32;

        DecisionTreeModel model = DecisionTree.trainClassifier(training, numClasses,
                categoricalFeaturesInfo, impurity, maxDepth, maxBins);

        long end = System.currentTimeMillis();
        double trainingTime = (end - start) / 1000.;

        testAndPrintResult(model, test, trainingTime);
    }

    private static void testAndPrintResult(DecisionTreeModel decisionTreeModel, JavaRDD<LabeledPoint> test, double trainingTime) {
        JavaPairRDD<Double, Double> predictionAndLabel = test.mapToPair(p -> new Tuple2<>(decisionTreeModel.predict(p.features()), p.label()));
        double accuracy = predictionAndLabel.filter(pl -> pl._1().equals(pl._2())).count() / (double) test.count();

        System.out.println("DecisionTree accuracy: " + accuracy);
        System.out.println("DecisionTree training time: " + trainingTime);
    }
}
