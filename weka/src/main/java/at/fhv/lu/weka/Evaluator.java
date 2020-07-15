package at.fhv.lu.weka;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

/**
 * @author Lukas Bals
 */
public class Evaluator {
    public static void evaluate(
            Classifier classifier,
            Instances trainingSet,
            Instances testSet,
            Printer printer
    ) {
        try {
            System.gc();

            printer.printToLogAndResults("Training...");
            long start = System.currentTimeMillis();
            classifier.buildClassifier(trainingSet);
            long end = System.currentTimeMillis();
            long trainingTime = (end - start);
            printer.printToLogAndResults(String.format("Training Time: %sms", trainingTime));

            System.gc();

            Evaluation evaluation = new Evaluation(trainingSet);
            System.out.println("Evaluating...");
            start = System.currentTimeMillis();
            evaluation.evaluateModel(classifier, testSet);
            end = System.currentTimeMillis();
            long evaluationTime = (end - start);

            printer.printToLogAndResults(
                    String.format("Evaluation Time: %sms", evaluationTime)
            );
            printer.printToLogAndResults(
                    String.format("Total Time: %sms", trainingTime + evaluationTime)
            );
            printer.printToLogAndResults(
                    String.format("Correctly Classified Instances: %s", (int) evaluation.correct())
            );
            printer.printToLogAndResults(
                    String.format("Incorrectly Classified Instances: %s", (int) evaluation.incorrect())
            );
            printer.printToLogAndResults(
                    String.format("Total Number of Instances: %s", (int) evaluation.numInstances())
            );
            printer.printToLogAndResults(
                    String.format("Accuracy: %s percent", evaluation.pctCorrect())
            );

            System.gc();
        } catch (Exception e) {
            System.out.print(e.getMessage());
        }
    }
}
