package me.damoebe.architectures;

import com.google.gson.Gson;
import com.google.gson.stream.JsonReader;
import me.damoebe.architectures.mlp.FFNetwork;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.Writer;
import java.util.List;

public abstract class Network {
    public abstract void train(List<Double> input, List<Double> optimalOutput);
    public abstract List<Double> getOutput();
    public abstract void insertInput(List<Double> input);
    public abstract void updateAllActivations();
    /**
     * Loads a FFNetwork from a JSON file
     * @param path The path to the json file
     * @param targetClass The subclass that is used to construct the object
     * @return A FFNetwork subclass Object
     * @throws Exception If the file is not compatible with the target class
     */
    public static FFNetwork loadNetworkFromJson(String path, Class targetClass) throws Exception{
        Gson gson = new Gson();
        JsonReader reader;
        try {
            reader = new JsonReader(new FileReader(path));
        }catch (Exception e){
            throw new Exception("File " + path + " could not be found, is not a json file or isn't using the right format!");
        }
        return gson.fromJson(reader, targetClass);
    }

    /**
     * Loads the FFNetwork Object to a json file
     * @param file The file which should contain the network
     * @throws Exception If anything goes wrong
     * @return true if the file did not exist, false if the file already existed.
     */
    public boolean loadToJsonFile(File file) throws Exception{
        boolean fileDidNotExist;
        Gson gson = new Gson();
        try (Writer writer = new FileWriter(file.getAbsolutePath())){
            fileDidNotExist = file.createNewFile();
            gson.toJson(this, writer);
        }catch (Exception e){
            throw new Exception("The Object could not be casted to a json file at " + file.getAbsolutePath());
        }
        return fileDidNotExist;
    }
}
