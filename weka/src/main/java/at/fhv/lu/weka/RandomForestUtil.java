package at.fhv.lu.weka;

import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

/**
 * @author Lukas Bals
 */
public class RandomForestUtil {
    public static void run(Instances trainingSet, Instances testSet, Printer printer) {
        // Evaluate using the Random Forest algorithm (with multiple numbers of trees)
//        for (int i = 10; i <= 200; i += 10) {
        int numTrees = 20;
        int maxDepth = 10;
        int seed = 12345;

        printer.printToLogAndResults(String.format("=== Random Forest (with %s Trees) ===", numTrees));
        RandomForest randomForest = new RandomForest();
        randomForest.setNumFeatures((int) Math.floor(Math.sqrt(trainingSet.numAttributes() - 1)));
        randomForest.setNumIterations(numTrees);
        randomForest.setMaxDepth(maxDepth);
        randomForest.setSeed(seed);

//        System.out.println(randomForest.globalInfo());

        Evaluator.evaluate(randomForest, trainingSet, testSet, printer);
//        }
    }
}
