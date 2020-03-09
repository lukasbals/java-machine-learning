package at.fhv.lu.weka;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
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

        // Evaluate using the Naive Bayes algorithm
        printer.printToLogAndResults("=== Naive Bayes ===");
        NaiveBayes naiveBayes = new NaiveBayes();
        evaluate(naiveBayes, trainingSet, testSet, printer);

        // Evaluate using the Decision Tree (C4.5) algorithm
        printer.printToLogAndResults("=== Decision Tree (C4.5) ===");
        J48 decisionTree = new J48();
        evaluate(decisionTree, trainingSet, testSet, printer);

        // Evaluate using the Random Forest algorithm (with multiple numbers of trees)
        for (int i = 10; i <= 200; i += 10) {
            printer.printToLogAndResults(String.format("=== Random Forest (with %s Trees) ===", i));
            RandomForest randomForest = new RandomForest();
            randomForest.setNumFeatures((int) Math.floor(Math.sqrt(trainingSet.numAttributes() - 1)));
            randomForest.setNumIterations(i);
            evaluate(randomForest, trainingSet, testSet, printer);
        }

        printer.closeStream();
    }

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
