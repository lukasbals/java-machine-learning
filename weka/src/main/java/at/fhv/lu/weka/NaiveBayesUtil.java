package at.fhv.lu.weka;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

/**
 * @author Lukas Bals
 */
public class NaiveBayesUtil {
    public static void run(Instances trainingSet, Instances testSet, Printer printer) {
        // Evaluate using the Naive Bayes algorithm
        printer.printToLogAndResults("=== Naive Bayes ===");
        NaiveBayes naiveBayes = new NaiveBayes();
        Evaluator.evaluate(naiveBayes, trainingSet, testSet, printer);
    }
}
