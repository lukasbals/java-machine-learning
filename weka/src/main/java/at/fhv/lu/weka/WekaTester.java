package at.fhv.lu.weka;

import weka.core.Instances;

import java.util.Random;

/**
 * @author Lukas Bals
 */
public class WekaTester {
    public static void main(String[] args) {
        Printer printer = new Printer();

        printer.printToLogAndResults("=== Weka Testing ===");

        // Load data sets (training + test data)
        long start = System.currentTimeMillis();

        Instances dataSet = Loader.loadTrainingSet();
        // Randomize that data so it's not possible to learn the same set twice
        dataSet.randomize(new Random());

        int numInstances = dataSet.numInstances();
        Instances trainingSet = new Instances(dataSet, 0, (int) (numInstances * 0.9));
        Instances testSet = new Instances(dataSet, (int) (numInstances * 0.9), (int) (numInstances * 0.1));

        long end = System.currentTimeMillis();
        printer.printToLogAndResults(String.format("Loaded dataset in %s seconds", (end - start) / 1000.));
        printer.printToLogAndResults(String.format("Instances in Training Set: %s", trainingSet.numInstances()));
        printer.printToLogAndResults(String.format("Instances in Test Set: %s", testSet.numInstances()));

//        NaiveBayesUtil.run(trainingSet, testSet, printer);
//        DecisionTreeUtil.run(trainingSet, testSet, printer);
        RandomForestUtil.run(trainingSet, testSet, printer);

        printer.closeStream();
    }
}
