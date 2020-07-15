package at.fhv.lu.sparkML;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.tree.model.TreeEnsembleModel;
import scala.Tuple2;

import java.util.HashMap;
import java.util.Map;

/**
 * @author Lukas Bals
 */
public class RandomForestUtil {
    static void train(JavaRDD<LabeledPoint> training, JavaRDD<LabeledPoint> test, Printer printer) {
        long start = System.currentTimeMillis();

        int numClasses = 10;
        // Empty categoricalFeaturesInfo indicates all features are continuous.
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        String featureSubsetStrategy = "auto"; // Let the algorithm choose.
        String impurity = "gini"; // Suggested value
        int maxBins = 32; // Suggested value

        int numTrees = 20;
        int maxDepth = 10;
        int seed = 100;

        RandomForestModel model = RandomForest.trainClassifier(
                training,
                numClasses,
                categoricalFeaturesInfo,
                numTrees,
                featureSubsetStrategy,
                impurity,
                maxDepth,
                maxBins,
                seed
        );

        long end = System.currentTimeMillis();

        long trainingTime = (end - start);
        printer.printToLogAndResults("RandomForest training time: " + trainingTime);

        evaluate(model, test, printer);
    }

    private static void evaluate(TreeEnsembleModel randomForestModel, JavaRDD<LabeledPoint> test, Printer printer) {
        long start = System.currentTimeMillis();

        JavaPairRDD<Double, Double> predictionAndLabel = test.mapToPair(
                p -> new Tuple2<>(randomForestModel.predict(p.features()), p.label())
        );

        long numCorrect = predictionAndLabel.filter(pl -> pl._1().equals(pl._2())).count();
        double accuracy = numCorrect / (double) test.count();

        long end = System.currentTimeMillis();


        long evaluationTime = (end - start);
        printer.printToLogAndResults("Evaluation time: " + evaluationTime);
        printer.printToLogAndResults("RandomForest accuracy: " + accuracy);
    }
}
