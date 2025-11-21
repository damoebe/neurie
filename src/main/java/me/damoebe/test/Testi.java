package me.damoebe.test;

import me.damoebe.network.Network;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class Testi {
    public static void run() {
        // test code
        ChartDisplay chart = new ChartDisplay("Testi learning", "epoch", "loss", 0);

        Network[] networks = {
                new Network(3, 1, 5, 2, 0.005),
                new Network(3, 1, 5, 2, 0.005),
                new Network(3, 1, 5, 2, 0.005)
        };

        // XOR training data
        List<List<Double>> inputs = Arrays.asList(
                Arrays.asList(0.0, 0.0, 0.0),
                Arrays.asList(0.0, 0.0, 1.0),
                Arrays.asList(0.0, 1.0, 0.0),
                Arrays.asList(0.0, 1.0, 1.0),
                Arrays.asList(1.0, 0.0, 0.0),
                Arrays.asList(1.0, 0.0, 1.0),
                Arrays.asList(1.0, 1.0, 0.0),
                Arrays.asList(1.0, 1.0, 1.0)
        );

        // optimal activations
        List<List<Double>> targets = Arrays.asList(
                Arrays.asList(0.0),
                Arrays.asList(1.0),
                Arrays.asList(1.0),
                Arrays.asList(1.0),
                Arrays.asList(1.0),
                Arrays.asList(0.0),
                Arrays.asList(0.0),
                Arrays.asList(0.0)
        );

        // training
        int epochs = 200000;
        for (int epoch = 0; epoch < epochs; epoch++) {
            double[] totalLosses = new double[networks.length];
            int i2 = 0;
            for (Network network : networks) {
                for (int i = 0; i < inputs.size(); i++) {
                    network.train(inputs.get(i), targets.get(i));
                    totalLosses[i2] += network.getLoss();
                }
                i2++;
            }
            if (epoch % 1000 == 0 || epoch == epochs - 1) {
                System.out.printf("Epoch %d, Loss: " + Arrays.toString(totalLosses) + "\n", epoch);
            }
            // chart updater

            if (epoch % 1000 == 0) {
                for (int i = 0; i != totalLosses.length; i++) {
                    chart.update(epoch, totalLosses[i], "network" + i);
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
