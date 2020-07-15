package at.fhv.lu.sparkML;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

import java.util.ArrayList;


/**
 * @author Lukas Bals
 */
public class Loader {
    public static ArrayList<JavaRDD<LabeledPoint>> loadDataSet(String location, JavaSparkContext jsc) {
        JavaRDD<LabeledPoint> inputData = MLUtils.loadLibSVMFile(jsc.sc(), location).toJavaRDD();
        JavaRDD<LabeledPoint>[] tmp = inputData.randomSplit(new double[]{0.5, 0.5});
        JavaRDD<LabeledPoint>[] tmp2 = tmp[0].randomSplit(new double[]{0.75, 0.25});
        JavaRDD<LabeledPoint> training = tmp2[0];
        JavaRDD<LabeledPoint> test = tmp2[1];

        ArrayList<JavaRDD<LabeledPoint>> trainingAndTestData = new ArrayList<>();
        trainingAndTestData.add(training);
        trainingAndTestData.add(test);
        return trainingAndTestData;
    }
}
