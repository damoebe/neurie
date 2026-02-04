package me.damoebe.test;

import me.damoebe.datasets.Dataset;
import me.damoebe.network.DeepNetwork;
import me.damoebe.network.EvolutionNetwork;
import me.damoebe.network.Network;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * A class containing all network training related static methods
 */
public class Trainer {
    /**
     * Main training method
     * @param showDiagram whether a diagram should be shown or not
     * @param networks list of networks that should be trained
     * @param dataset A dataset object which is used to train the networks
     * @param epochs The amount of epochs
     * @param spaceBetweenPoints The space between each graphical point (only for diagram)
     */
    public static void train(boolean showDiagram, @NotNull List<Network> networks, @NotNull Dataset dataset, int epochs, int spaceBetweenPoints){

        ChartDisplay chart = null;

        if (showDiagram) {
            chart = new ChartDisplay("neurie-Training (" + dataset.getName() + ")", "epoch", "loss", 0);
        }

        List<List<Double>> inputs = dataset.getInputs();
        List<List<Double>> targets = dataset.getTargets();

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (Network network : networks) {
                for (int i = 0; i < inputs.size(); i++) {
                    network.train(inputs.get(i), targets.get(i));
                }
            }

            for (Network network : networks){
                if (network instanceof EvolutionNetwork evolutionNetwork) {
                    evolutionNetwork.finishEpoch();
                }
            }

            if (epoch % 1000 == 0 || epoch == epochs - 1) {
                double[] networkLosses = new double[networks.size()];
                int i = 0;
                for (Network network : networks){
                    networkLosses[i] = network.getNetworkLoss();
                    i++;
                }
                System.out.printf("Epoch %d, Network-Losses: " + Arrays.toString(networkLosses) + "\n", epoch);
            }
            // chart updater
            if (showDiagram) {

                // chart period points distance
                if (epoch % spaceBetweenPoints == 0) {

                    for (int i = 0; i != networks.size(); i++) {
                        chart.update(epoch, networks.get(i).getNetworkLoss(), "network" + i);
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
}
