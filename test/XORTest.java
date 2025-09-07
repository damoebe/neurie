package test;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

import network.Network;


public class XORTest {
    public static void run() {
        // test code
        ChartDisplay chart = new ChartDisplay("XORTest learning", "epoch", "loss", 0);

        Network[] networks = {
                new Network(2, 1, 2, 1, 0.2),
                new Network(2, 1, 2, 1, 0.2),
                new Network(2, 1, 2, 1, 0.2)
        };

        // XOR training data
        List<List<Double>> inputs = Arrays.asList(
                Arrays.asList(0.0, 0.0), // 0, 0
                Arrays.asList(0.0, 1.0), // 0, 1
                Arrays.asList(1.0, 0.0), // 1, 0
                Arrays.asList(1.0, 1.0) // 1, 1
        );

        // optimal activations
        List<List<Double>> targets = Arrays.asList(
                List.of(0.0), // 0
                List.of(1.0), // 1
                List.of(1.0), // 1
                List.of(0.0) // 0
        );

        // training
        int epochs = 5000;
        for (int epoch = 0; epoch < epochs; epoch++) {
            double[] totalLosses = new double[networks.length];
            int i2 = 0;
            for (Network network : networks) {
                for (int i = 0; i < inputs.size(); i++) {
                    network.train(inputs.get(i), targets.get(i));
                    double prediction = network.getOutput().get(0);
                    double error = prediction - targets.get(i).get(0);
                    totalLosses[i2] += error * error;
                }
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

