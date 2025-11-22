package me.damoebe.test;

import me.damoebe.network.Network;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class Testi {
    public static void run() {
        // test code -> delete
        ChartDisplay chart = new ChartDisplay("Testi learning", "epoch", "loss", 0);

        Network[] networks = {
                new Network(2, 1, 1, 1, 0.005),
                new Network(2, 1, 1, 1, 0.005),
                new Network(2, 1, 1, 1, 0.005)
        };

        // XOR training data
        List<List<Double>> inputs = Arrays.asList(
                Arrays.asList(0.0, 1.0),
                Arrays.asList(1.0, 0.0)
        );

        // optimal activations
        List<List<Double>> targets = Arrays.asList(
                Arrays.asList(0.0),
                Arrays.asList(1.0)
        );

        // training
        int epochs =100000;
        for (int epoch = 0; epoch < epochs; epoch++) {
            double[] totalLosses = new double[networks.length];
            int i2 = 0;
            for (Network network : networks) {
                for (int i = 0; i < inputs.size(); i++) {
                    network.evolutionLearning(inputs.get(i), targets.get(i));
                    totalLosses[i2] += network.getLoss();
                }
                i2++;
            }
            for (Network network : networks){
                network.finishEpoch();
            }
            if (epoch % 1000 == 0 || epoch == epochs - 1) {
                System.out.printf("Epoch %d, Loss: " + Double.toString(networks[0].getLowestLoss()) + "\n", epoch);
            }
            // chart updater

            if (epoch % 1000 == 0) {
                int i = 0;
                for (Network network : networks) {
                    chart.update(epoch, network.getLowestLoss(), "network" + i);
                    i++;
                }
                try {
                    TimeUnit.MILLISECONDS.sleep(100);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }


        }
    }
}
