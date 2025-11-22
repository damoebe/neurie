package me.damoebe;

import me.damoebe.datasets.Dataset;
import me.damoebe.datasets.DatasetReader;
import me.damoebe.network.Network;
import me.damoebe.test.Trainer;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        Dataset dataset = DatasetReader.readFile(System.getProperty("user.dir") +
                "/src/main/java/me/damoebe/datasets/data/complex.json");

        List<Network> networks = List.of(
                new Network( 3, 1, 10, 1, 0.1),
                new Network( 3, 1, 10, 1, 0.07),
                new Network( 3, 1, 10, 1, 0.05)
        );

        for (Network network : networks){
            network.setNoise(0.1);
        }

        Trainer.train(true, networks, dataset,100000);
    }
}