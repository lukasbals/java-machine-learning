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

        for (int i = 0; i < 12; i++) {
            long start = System.currentTimeMillis();

            Instances dataSet = Loader.loadTrainingSet();

            // Randomize that data so it's not possible to learn the same set twice
            dataSet.randomize(new Random());

            Instances dataSetNew = new Instances(dataSet, 0, (int) (dataSet.numInstances() * 0.5));

            int numInstances = dataSetNew.numInstances();
            Instances trainingSet = new Instances(dataSetNew, 0, (int) (numInstances * 0.75));
            Instances testSet = new Instances(dataSetNew, (int) (numInstances * 0.75), (int) (numInstances * 0.25));


            long end = System.currentTimeMillis();
            printer.printToLogAndResults(String.format(
                    "Loaded dataset in %s seconds", (end - start))
            );
            printer.printToLogAndResults(
                    String.format("Instances in Training Set: %s", trainingSet.numInstances())
            );
            printer.printToLogAndResults(
                    String.format("Instances in Test Set: %s", testSet.numInstances())
            );

//            NaiveBayesUtil.run(trainingSet, testSet, printer);

//            DecisionTreeUtil.run(trainingSet, testSet, printer);

            RandomForestUtil.run(trainingSet, testSet, printer);

            printer.printToLogAndResults("===========================================");

        }
        printer.closeStream();
    }
}
