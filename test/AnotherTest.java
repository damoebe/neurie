package test;

import network.Network;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class AnotherTest {
    public static void run() {
        // test code
        ChartDisplay chart = new ChartDisplay("XORTest learning", "epoch", "loss", 0);

        Network[] networks = {
                new Network(1, 1, 3, 1, 0.2)
        };

        // XOR training data
        List<List<Double>> inputs = List.of(
                Arrays.asList(0.1)
        );

        // optimal activations
        List<List<Double>> targets = List.of(
                List.of(1.0)
        );

        // training
        int epochs = 20000;
        for (int epoch = 0; epoch < epochs; epoch++) {
            double[] totalLosses = new double[networks.length];
            int i2 = 0;
            for (Network network : networks) {
                for (int i = 0; i < inputs.size(); i++) {
                    network.evolutionLearning(inputs.get(i), targets.get(i));
                    double prediction = network.getOutput().get(0);
                    double error = prediction - targets.get(i).get(0);
                    totalLosses[i2] += error * error;
                }
                network.finishEpoch(1, epoch);
                i2++;
            }
            if (epoch % 1000 == 0 || epoch == epochs - 1) {
                System.out.printf("Epoch %d, Loss: " + Arrays.toString(totalLosses) + "\n", epoch);
            }
            // chart updater
            if (epoch % 200 == 0) {
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
