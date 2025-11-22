package me.damoebe.datasets;

import com.google.gson.Gson;
import com.google.gson.stream.JsonReader;
import me.damoebe.datasets.Dataset;

import java.io.FileReader;

public class DatasetReader {
    public static Dataset readFile(String path){
        Gson gson = new Gson();
        JsonReader reader = null;

        try {
            reader = new JsonReader(new FileReader(path));
        }catch (Exception e){
            e.printStackTrace();
            System.out.println("File " + path + " could not be found!");
        }
        Dataset dataset = gson.fromJson(reader, Dataset.class);
        return dataset;
    }
}
