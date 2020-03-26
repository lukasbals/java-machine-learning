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

    static void train(JavaRDD<LabeledPoint> training, JavaRDD<LabeledPoint> test) {
        long start = System.currentTimeMillis();

        NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);

        long end = System.currentTimeMillis();
        double trainingTime = (end - start) / 1000.;

        testAndPrintResult(model, test, trainingTime);
    }

    private static void testAndPrintResult(NaiveBayesModel naiveBayesModel, JavaRDD<LabeledPoint> test, double trainingTime) {
        JavaPairRDD<Double, Double> predictionAndLabel = test.mapToPair(
                p -> new Tuple2<>(naiveBayesModel.predict(p.features()), p.label())
        );
        double accuracy = predictionAndLabel.filter(pl -> pl._1().equals(pl._2())).count() / (double) test.count();

        System.out.println("NaiveBayes accuracy: " + accuracy);
        System.out.println("NaiveBayes training time: " + trainingTime);
    }
}
