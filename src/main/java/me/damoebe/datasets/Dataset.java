package me.damoebe.datasets;

import com.google.gson.Gson;
import com.google.gson.stream.JsonReader;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class Dataset {

    private final List<List<Double>> inputs;
    private final List<List<Double>> targets;
    private final String name;

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
