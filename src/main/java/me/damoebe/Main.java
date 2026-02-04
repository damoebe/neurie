package me.damoebe;

import me.damoebe.datasets.Dataset;
import me.damoebe.datasets.DatasetReader;
import me.damoebe.network.DeepNetwork;
import me.damoebe.network.EvolutionNetwork;
import me.damoebe.network.Network;
import me.damoebe.test.Trainer;

import java.io.File;
import java.util.List;

/**
 * Just a testing class
 */
public class Main {
    public static void main(String[] args) {
        run();
    }

    static void run(){
        Dataset dataset = null;
        try {
            dataset = DatasetReader.readJson(System.getProperty("user.dir") +
                    "/src/main/java/me/damoebe/datasets/data/xor.json");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        Network deep = new DeepNetwork(3, 1, 2, 1, 0.1);
        deep.setNoise(0);
        Network deep2 = new DeepNetwork(3, 1, 2, 1, 0.05);
        deep2.setNoise(0.05);
        Network evo = new EvolutionNetwork(3, 1, 2, 1);
        evo.setNoise(0.1);


        List<Network> networks = new java.util.ArrayList<>(List.of(deep, deep2, evo));
        Trainer.train(true, networks, dataset, 10000, 100);
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
        Network deep = null;
        if (new File(System.getProperty("user.home") + "/Downloads/neurie_deepNetwork.json").exists()) {
            try {
                deep = Network.loadNetworkFromJson(System.getProperty("user.home") + "/Downloads/neurie_deepNetwork.json", DeepNetwork.class);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }else {
            deep = new DeepNetwork(3, 1, 2, 1, 0.1);
        }

        deep.setNoise(0.001);
        List<Network> networks = new java.util.ArrayList<>(List.of(deep));
        Trainer.train(true, networks, dataset, 10000, 100);
        try {
            deep.loadToJsonFile(new File(System.getProperty("user.home") + "/Downloads/neurie_deepNetwork.json"));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}