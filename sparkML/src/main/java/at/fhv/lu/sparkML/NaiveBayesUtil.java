package at.fhv.lu.sparkML;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

/**
 * @author Lukas Bals
 */
public class NaiveBayesUtil {

    static void train(JavaRDD<LabeledPoint> training, JavaRDD<LabeledPoint> test, Printer printer) {
        long start = System.currentTimeMillis();

        NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);

        long end = System.currentTimeMillis();

        long trainingTime = (end - start);
        printer.printToLogAndResults("NaiveBayes training time: " + trainingTime);

        evaluate(model, test, printer);
    }

    private static void evaluate(NaiveBayesModel naiveBayesModel, JavaRDD<LabeledPoint> test, Printer printer) {
        long start = System.currentTimeMillis();

        JavaPairRDD<Double, Double> predictionAndLabel = test.mapToPair(
                p -> new Tuple2<>(naiveBayesModel.predict(p.features()), p.label())
        );

        long numCorrect = predictionAndLabel.filter(pl -> pl._1().equals(pl._2())).count();
        double accuracy = numCorrect / (double) test.count();

        long end = System.currentTimeMillis();


        long evaluationTime = (end - start);
        printer.printToLogAndResults("Evaluation time: " + evaluationTime);
        printer.printToLogAndResults("NaiveBayes accuracy: " + accuracy);
    }
}
