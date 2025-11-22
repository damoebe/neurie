package me.damoebe;

import me.damoebe.datasets.Dataset;
import me.damoebe.datasets.DatasetReader;
import me.damoebe.network.LearningType;
import me.damoebe.network.Network;
import me.damoebe.test.Testi;
import me.damoebe.test.Trainer;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        run();
    }

    static void run(){
        Dataset dataset = DatasetReader.readFile(System.getProperty("user.dir") +
                "/src/main/java/me/damoebe/datasets/data/complex.json");

        List<Network> networks = List.of(
                new Network( 3, 1, 5, 1, 0.07),
                new Network( 3, 1, 5, 1, 0.07),
                new Network( 3, 1, 5, 1, 0.07)
        );
        int i = 0;
        for (Network network : networks){
            network.setNoise(0.1);
            if (i == 1){
                network.setNoise(0.0);
            }
            i++;
        }

        Trainer.train(true, networks, dataset,10000, LearningType.BACKPROPAGATION);
    }
}