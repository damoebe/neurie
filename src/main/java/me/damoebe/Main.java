package me.damoebe;

import me.damoebe.datasets.Dataset;
import me.damoebe.datasets.DatasetReader;
import me.damoebe.network.DeepNetwork;
import me.damoebe.network.EvolutionNetwork;
import me.damoebe.network.Network;
import me.damoebe.test.Trainer;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        run();
    }

    static void run(){
        Dataset dataset = DatasetReader.readFile(System.getProperty("user.dir") +
                "/src/main/java/me/damoebe/datasets/data/xor.json");

        List<Network> networks = new java.util.ArrayList<>(List.of(
                new DeepNetwork(2, 1, 3, 1, 0.1),
                new EvolutionNetwork(2, 1, 2, 1)
        ));
        for (Network network : networks) {
            network.setNoise(0.05);
        }
        Trainer.train(true, networks, dataset, 10000);
    }
}