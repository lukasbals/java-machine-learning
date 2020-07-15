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
    static void train(JavaRDD<LabeledPoint> training, JavaRDD<LabeledPoint> test, Printer printer) {

        long start = System.currentTimeMillis();

        int numClasses = 10;
        // Empty categoricalFeaturesInfo indicates all features are continuous.
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        String impurity = "gini";
        int maxDepth = 10;
        int maxBins = 32;

        DecisionTreeModel model = DecisionTree.trainClassifier(training, numClasses,
                categoricalFeaturesInfo, impurity, maxDepth, maxBins);

        long end = System.currentTimeMillis();

        long trainingTime = (end - start);
        printer.printToLogAndResults("DecisionTree training time: " + trainingTime);

        evaluate(model, test, printer);
    }

    private static void evaluate(DecisionTreeModel decisionTreeModel, JavaRDD<LabeledPoint> test, Printer printer) {
        long start = System.currentTimeMillis();

        JavaPairRDD<Double, Double> predictionAndLabel = test.mapToPair(
                p -> new Tuple2<>(decisionTreeModel.predict(p.features()), p.label())
        );

        long numCorrect = predictionAndLabel.filter(pl -> pl._1().equals(pl._2())).count();
        double accuracy = numCorrect / (double) test.count();

        long end = System.currentTimeMillis();


        long evaluationTime = (end - start);
        printer.printToLogAndResults("Evaluation time: " + evaluationTime);
        printer.printToLogAndResults("DecisionTree accuracy: " + accuracy);
    }
}
