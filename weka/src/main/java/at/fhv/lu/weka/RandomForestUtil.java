package at.fhv.lu.weka;

import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

/**
 * @author Lukas Bals
 */
public class RandomForestUtil {
    public static void run(Instances trainingSet, Instances testSet, Printer printer) {
        int numTrees = 20;
        int maxDepth = 10;
        int seed = 100;

        printer.printToLogAndResults(String.format("=== Random Forest (with %s Trees) ===", numTrees));
        RandomForest randomForest = new RandomForest();
        randomForest.setNumIterations(numTrees);
        randomForest.setMaxDepth(maxDepth);
        randomForest.setSeed(seed);

        Evaluator.evaluate(randomForest, trainingSet, testSet, printer);
    }
}
