package at.fhv.lu.sparkML;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;

import java.util.ArrayList;

public class SparkMLTester {
    public static void main(String[] args) {
        Printer printer = new Printer();
        printer.printToLogAndResults("=== SparkML Testing ===");

        SparkConf sparkConf = new SparkConf().setMaster("local").setAppName("JavaML");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);

        String path = "../data/mnist";


        for (int i = 0; i < 12; i++) {

            ArrayList<JavaRDD<LabeledPoint>> trainingAndTestData = Loader.loadDataSet(path, jsc);
            JavaRDD<LabeledPoint> training = trainingAndTestData.get(0);
            JavaRDD<LabeledPoint> test = trainingAndTestData.get(1);

//            NaiveBayesUtil.train(training, test, printer);

//            DecisionTreeUtil.train(training, test, printer);

            RandomForestUtil.train(training, test, printer);

            printer.printToLogAndResults("===========================================");

        }

        jsc.stop();
    }
}
