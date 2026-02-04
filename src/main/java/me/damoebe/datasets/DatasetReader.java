package me.damoebe.datasets;

import com.google.gson.Gson;
import com.google.gson.stream.JsonReader;

import java.io.FileReader;

/**
 * Contains all Dataset related static methods
 */
public class DatasetReader {
    /**
     * Loads a Dataset class from a JSON File
     * @param path The absolute path to the File
     * @return A Dataset object containing every set of data from the json file
     * @throws Exception if the file can not be read
     */
    public static Dataset readJson(String path) throws Exception{
        Gson gson = new Gson();
        JsonReader reader;

        try {
            reader = new JsonReader(new FileReader(path));
        }catch (Exception e){
            throw new Exception("File " + path + " could not be found, is not a json file or isn't using the right format!");
        }
        return gson.fromJson(reader, Dataset.class);
    }
}
