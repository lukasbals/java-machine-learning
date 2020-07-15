package at.fhv.lu.preProcessor;

import java.io.*;

/**
 * @author Lukas Bals
 */
public class MNISTPreProcessor {
    public static void main(String[] args) throws IOException {
        String dataset = "t10k"; // "train" for the training set and "t10k" for the test set.

        int meanFilterSize = 3; // Size of the mean filter.
        long start = System.currentTimeMillis();
        DataInputStream labels = new DataInputStream(
                new FileInputStream("../data/" + dataset + "-labels.idx1-ubyte")
        );
        DataInputStream images = new DataInputStream(
                new FileInputStream("../data/" + dataset + "-images.idx3-ubyte")
        );
        int magicNumber = labels.readInt();
        if (magicNumber != 2049) {
            System.err.println(
                    "The label file has wrong magic number: " + magicNumber + " (should be 2049)"
            );
            System.exit(0);
        }
        magicNumber = images.readInt();
        if (magicNumber != 2051) {
            System.err.println(
                    "The image file has wrong magic number: " + magicNumber + " (should be 2051)"
            );
            System.exit(0);
        }
        int numLabels = labels.readInt();
        int numImages = images.readInt();
        int numRows = images.readInt();
        int numCols = images.readInt();
        if (numLabels != numImages) {
            System.err.println(
                    "The label file and the image file do not contain the same number of items."
            );
            System.err.println("The label file contains: " + numLabels);
            System.err.println("The image file contains: " + numImages);
            System.exit(0);
        }
        int numLabelsRead = 0;
        int numImagesRead = 0;
        FileOutputStream arff = new FileOutputStream("../data/" + dataset + ".arff");
        PrintStream toArff = new PrintStream(arff);
        toArff.println("@relation " + dataset);
        toArff.println();
        for (int pixel = 1; pixel <= numRows * numCols; pixel++) {
            toArff.println("@attribute pixel" + pixel + " {0,1}");
        }
        toArff.println("@attribute label {0,1,2,3,4,5,6,7,8,9}");
        toArff.println();
        toArff.println("@data");
        while (labels.available() > 0 && numLabelsRead < numLabels) {
            byte label = labels.readByte();
            numLabelsRead++;
            int[][] image = new int[numRows][numCols];
            for (int row = 0; row < image.length; row++) {
                for (int col = 0; col < image[row].length; col++) {
                    image[row][col] = images.readUnsignedByte();
                }
            }
            numImagesRead++;
            int[][] blurred = MeanFilter.blur(image, numRows, numCols, meanFilterSize);
            int[][] binarized = OtsuThresholding.binarize(blurred, numRows, numCols);
            for (int row = 0; row < binarized.length; row++) {
                for (int col = 0; col < binarized[row].length; col++) {
                    toArff.print(binarized[row][col] + ",");
                }
            }
            if (numLabelsRead == numImagesRead) {
                toArff.println(label);
            }
            if (numLabelsRead % 1000 == 0) {
                System.out.print(" " + numLabelsRead + "/" + numLabels);
                long end = System.currentTimeMillis();
                long elapsed = end - start;
                long minutes = elapsed / (1000 * 60);
                long seconds = elapsed / 1000 - minutes * 60;
                System.out.println(" " + minutes + "'" + seconds + "''");
            }
        }
        labels.close();
        images.close();
        arff.close();
        long end = System.currentTimeMillis();
        long elapsed = end - start;
        long minutes = elapsed / (1000 * 60);
        long seconds = elapsed / 1000 - minutes * 60;
        System.out.println(
                "Preprocessed " + numLabelsRead + " items in " + minutes + "'" + seconds + "''."
        );
    }
}
