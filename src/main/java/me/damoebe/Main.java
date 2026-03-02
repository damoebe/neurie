package me.damoebe;

import me.damoebe.datasets.Dataset;
import me.damoebe.datasets.DatasetReader;
import me.damoebe.architectures.mlp.DeepNetwork;
import me.damoebe.architectures.mlp.EvolutionNetwork;
import me.damoebe.architectures.mlp.FFNetwork;
import me.damoebe.architectures.mlp.test.Trainer;
import me.damoebe.architectures.transformer.mha.Head;
import me.damoebe.architectures.transformer.mha.MultiHeadAttention;

import java.io.File;
import java.util.List;

/**
 * Just a testing class
 */
public class Main {
    public static void main(String[] args) {
        run();
    }

    static void testMHA(){
        MultiHeadAttention<Head> mha = new MultiHeadAttention<>(Head.class, 2, 3 ,1, false);

    }

    static void run(){
        Dataset dataset = null;
        try {
            dataset = DatasetReader.readJson(System.getProperty("user.dir") +
                    "/src/main/java/me/damoebe/datasets/data/xor.json");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        FFNetwork deep = new DeepNetwork(2, 1, 2, 1, 0.1);
        deep.setNoise(0);
        FFNetwork deep2 = new DeepNetwork(2, 1, 2, 1, 0.05);
        deep2.setNoise(0.05);
        FFNetwork evo = new EvolutionNetwork(2, 1, 2, 1);
        evo.setNoise(0.1);


        List<FFNetwork> FFNetworks = new java.util.ArrayList<>(List.of(deep, deep2, evo));
        Trainer.train(true, FFNetworks, dataset, 10000, 100);
    }

    /**
     * Testing the network saving and loading system
     */
    static void testNetworkLoader(){
        Dataset dataset = null;
        try {
            dataset = DatasetReader.readJson(System.getProperty("user.dir") +
                    "/src/main/java/me/damoebe/datasets/data/complex.json");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        FFNetwork deep = null;
        if (new File(System.getProperty("user.home") + "/Downloads/neurie_deepNetwork.json").exists()) {
            try {
                deep = FFNetwork.loadNetworkFromJson(System.getProperty("user.home") + "/Downloads/neurie_deepNetwork.json", DeepNetwork.class);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }else {
            deep = new DeepNetwork(3, 1, 2, 1, 0.1);
        }

        deep.setNoise(0.001);
        List<FFNetwork> FFNetworks = new java.util.ArrayList<>(List.of(deep));
        Trainer.train(true, FFNetworks, dataset, 10000, 100);
        try {
            deep.loadToJsonFile(new File(System.getProperty("user.home") + "/Downloads/neurie_deepNetwork.json"));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}