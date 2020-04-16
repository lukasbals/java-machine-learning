package at.fhv.lu.weka;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

/**
 * @author Lukas Bals
 */
public class Evaluator {
    public static void evaluate(Classifier classifier, Instances trainingSet, Instances testSet, Printer printer) {
        try {
            System.gc();

            // Training with the MNIST dataset
            printer.printToLogAndResults("Training...");
            long start = System.currentTimeMillis();
            classifier.buildClassifier(trainingSet);
            long end = System.currentTimeMillis();
            double trainingTime = (end - start) / 1000.;
            printer.printToLogAndResults(String.format("Training Time: %s seconds", trainingTime));

            System.gc();

            // Testing how many of the 10 000 test data items can be identified correctly
            Evaluation evaluation = new Evaluation(trainingSet);
            System.out.println("Evaluating...");
            start = System.currentTimeMillis();
            evaluation.evaluateModel(classifier, testSet);
            end = System.currentTimeMillis();
            double evaluationTime = (end - start) / 1000.;

            // Printing results
            printer.printToLogAndResults(String.format("Evaluation Time: %ss", evaluationTime));
            printer.printToLogAndResults(String.format("Total Time: %ss", trainingTime + evaluationTime));
            printer.printToLogAndResults(String.format("Correctly Classified Instances: %s", (int) evaluation.correct()));
            printer.printToLogAndResults(String.format("Incorrectly Classified Instances: %s", (int) evaluation.incorrect()));
            printer.printToLogAndResults(String.format("Total Number of Instances: %s", (int) evaluation.numInstances()));
            printer.printToLogAndResults(String.format("Accuracy: %s percent", evaluation.pctCorrect()));

            System.gc();
        } catch (Exception e) {

        }
    }
}
