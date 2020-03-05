package at.fhv.lu.weka;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 * @author Lukas Bals
 */
public class Loader {
    public static Instances loadTrainingSet() {
        return loadDataSet("../data/train.arff");
    }

    public static Instances loadTestSet() {
        return loadDataSet("../data/t10k.arff");
    }

    private static Instances loadDataSet(String location) {
        try {
            Instances testSet = ConverterUtils.DataSource.read(location);
            testSet.setClassIndex(testSet.numAttributes() - 1);
            return testSet;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
}
