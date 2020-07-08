package util;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;


public class PrepareResults {

//    public static String RESULTS_FILE = "data/eval/swe/results/results.txt";
    public static String RESULTS_FILE = "data/vecmap/results-eng-i.txt";
//    public static String RESULTS_FILE = "results/default/answer/task2/english.txt";

    public static String TARGETS_FILE = "data/targets/eng/targets.txt";

//    public static String METHOD_NAME = "LDA-100-globalThreshold";
    public static String METHOD_NAME = "default-GOLD";
//    public static String METHOD_NAME = "map-ort-i-globalThreshold";
//    public static String METHOD_NAME = "map-unsup";

    public static String LANGUAGE = "english";

    public static String RESULTS_FOLDER = "data/results-pre/map-ort-i/";

    public static String GOLD_FILE = "data/results-pre/gold/task1/english.txt";

    public static double findThreshold(double[] data)
    {
        double[] copy = Arrays.copyOf(data, data.length);
        Arrays.sort(copy);
        return copy[copy.length / 2];
    }

    public static double findThreshold(List<Double> data)
    {
        List<Double> copy = new ArrayList<>(data);
        copy.sort(Comparator.naturalOrder());
        return copy.get(copy.size() / 2);
    }

    public static double findOptimalThreshold(double[] system, int[] gold, String[] targets)
    {
        Target[] systemTargets = new Target[targets.length];
        Map<String, Integer> goldTargets = new HashMap<>();
        for(int i = 0; i < targets.length; i++)
        {
            systemTargets[i] = new Target(targets[i], system[i]);
            goldTargets.put(targets[i], gold[i]);
        }
        Arrays.sort(systemTargets);
        int correct = Arrays.stream(gold).sum();
        int maxCorrect = correct;
        double threshold = 0;
        for(int i = 0; i < targets.length; i++)
        {
            if(goldTargets.get(systemTargets[i].word) == 0)
                correct++;
            else
                correct--;
            if(correct > maxCorrect)
            {
                maxCorrect = correct;
                threshold = systemTargets[i].change + .000001;
            }
        }
        return threshold;
    }

    public static double findGlobalThreshold(String foler) throws IOException {
        List<Double> results = new ArrayList<>();
        for(String file: new File(foler).list())
        {
            results.addAll(Files.lines(Paths.get(foler + "/" + file)).map(Double::parseDouble).collect(Collectors.toList()));
        }
        return findThreshold(results);
    }

    public static void main(String[] args) throws IOException {

        double[] results = Files.lines(Paths.get(RESULTS_FILE)).mapToDouble(Double::parseDouble).toArray();
        //double[] results = Files.lines(Paths.get(RESULTS_FILE)).sorted().map(s -> s.split("\t")[1]).mapToDouble(Double::parseDouble).toArray();
        String[] targets = Files.lines(Paths.get(GOLD_FILE)).sorted().map(s -> s.split("\t")[0]).collect(Collectors.toList()).toArray(new String[0]);
        //int[] gold = Files.lines(Paths.get(GOLD_FILE)).sorted().map(s -> s.split("\t")[1]).mapToInt(Integer::parseInt).toArray();
        //String[] targets = Files.readAllLines(Paths.get(TARGETS_FILE)).toArray(new String[0]);
        //double threshold = findOptimalThreshold(results, gold, targets);
        //double threshold = findGlobalThreshold(RESULTS_FOLDER);
        double threshold = findThreshold(results);
        System.out.println("Threshold: " + threshold);
        int[] binary = Arrays.stream(results).mapToInt(x -> x > threshold ? 1:0).toArray();
        new File("results/" + METHOD_NAME + "/answer/task1/").mkdirs();
        new File("results/" + METHOD_NAME + "/answer/task2/").mkdirs();
        write("results/" + METHOD_NAME + "/answer/task1/" + LANGUAGE + ".txt", targets, binary);
        write("results/" + METHOD_NAME + "/answer/task2/" + LANGUAGE + ".txt", targets, results);

    }

    public static void write(String file, String[] targets, int[] results) throws IOException {
        PrintWriter pw = new PrintWriter(new FileWriter(file));
        for(int i = 0; i < targets.length; i++)
        {
            pw.println(targets[i] + "\t" + results[i]);
        }
        pw.close();
    }

    public static void write(String file, String[] targets, double[] results) throws IOException {
        PrintWriter pw = new PrintWriter(new FileWriter(file));
        for(int i = 0; i < targets.length; i++)
        {
            pw.println(targets[i] + "\t" + results[i]);
        }
        pw.close();
    }


}
