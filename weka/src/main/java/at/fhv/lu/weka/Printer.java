package at.fhv.lu.weka;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;

/**
 * @author Lukas Bals
 */
public class Printer {
    private PrintStream _printStream;

    public Printer() {
        try {
            FileOutputStream results = new FileOutputStream("results.txt", true);
            _printStream = new PrintStream(results);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public void printToLogAndResults(String text) {
        System.out.println(text);
        _printStream.println(text);
    }

    public void closeStream() {
        _printStream.println();
        _printStream.close();
    }
}
