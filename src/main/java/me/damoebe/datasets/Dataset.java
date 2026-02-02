package me.damoebe.datasets;

import com.google.gson.Gson;
import com.google.gson.stream.JsonReader;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Container class for a Dataset
 */
public class Dataset {
    /**
     * All inputs of the dataset
     */
    private final List<List<Double>> inputs;
    /**
     * All target outputs for the inputs
     */
    private final List<List<Double>> targets;
    /**
     * name of the dataset
     */
    private final String name;

    /**
     * Main constructor for object class
     * @param name name of the dataset
     * @param inputs inputs of the dataset
     * @param targets target activations of the dataset
     */
    public Dataset(String name, List<List<Double>> inputs, List<List<Double>> targets){
        this.name = name;
        this.inputs = inputs;
        this.targets = targets;
    }

    public List<List<Double>> getInputs() {
        return inputs;
    }

    public List<List<Double>> getTargets() {
        return targets;
    }
    public String getName() {
        return name;
    }
}
