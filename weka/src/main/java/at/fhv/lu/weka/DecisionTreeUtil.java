package at.fhv.lu.weka;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;

/**
 * @author Lukas Bals
 */
public class DecisionTreeUtil {
    public static void run(Instances trainingSet, Instances testSet, Printer printer) {
        printer.printToLogAndResults("=== Decision Tree (C4.5) ===");
        Classifier decisionTree = new J48();

        Evaluator.evaluate(decisionTree, trainingSet, testSet, printer);
    }
}
