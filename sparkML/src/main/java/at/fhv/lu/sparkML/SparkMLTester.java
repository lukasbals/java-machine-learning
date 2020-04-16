package at.fhv.lu.sparkML;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

public class SparkMLTester {
    public static void main(String[] args) {
        SparkConf sparkConf = new SparkConf().setMaster("local").setAppName("JavaNaiveBayesExample");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);

        String path = "../data/mnist";

        JavaRDD<LabeledPoint> inputData = MLUtils.loadLibSVMFile(jsc.sc(), path).toJavaRDD();
        JavaRDD<LabeledPoint>[] tmp = inputData.randomSplit(new double[]{0.9, 0.1});
        JavaRDD<LabeledPoint> training = tmp[0]; // training set
        JavaRDD<LabeledPoint> test = tmp[1]; // test set


//        NaiveBayesUtil.train(training, test);

//        DecisionTreeUtil.train(training, test);

        RandomForestUtil.train(training, test);

        jsc.stop();
    }
}
