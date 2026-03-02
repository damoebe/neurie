package me.damoebe.architectures.mlp.test;

import me.damoebe.datasets.Dataset;
import me.damoebe.architectures.mlp.EvolutionNetwork;
import me.damoebe.architectures.mlp.FFNetwork;
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
     * @param FFNetworks list of FFNetworks that should be trained
     * @param dataset A dataset object which is used to train the FFNetworks
     * @param epochs The amount of epochs
     * @param spaceBetweenPoints The space between each graphical point (only for diagram)
     */
    public static void train(boolean showDiagram, @NotNull List<FFNetwork> FFNetworks, @NotNull Dataset dataset, int epochs, int spaceBetweenPoints){

        ChartDisplay chart = null;

        if (showDiagram) {
            chart = new ChartDisplay("neurie-Training (" + dataset.getName() + ")", "epoch", "loss", 0);
        }

        List<List<Double>> inputs = dataset.getInputs();
        List<List<Double>> targets = dataset.getTargets();

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (FFNetwork FFNetwork : FFNetworks) {
                for (int i = 0; i < inputs.size(); i++) {
                    FFNetwork.train(inputs.get(i), targets.get(i));
                }
            }

            for (FFNetwork FFNetwork : FFNetworks){
                if (FFNetwork instanceof EvolutionNetwork evolutionNetwork) {
                    evolutionNetwork.finishEpoch();
                }
            }

            if (epoch % 1000 == 0 || epoch == epochs - 1) {
                double[] networkLosses = new double[FFNetworks.size()];
                int i = 0;
                for (FFNetwork FFNetwork : FFNetworks){
                    networkLosses[i] = FFNetwork.getNetworkLoss();
                    i++;
                }
                System.out.printf("Epoch %d, FFNetwork-Losses: " + Arrays.toString(networkLosses) + "\n", epoch);
            }
            // chart updater
            if (showDiagram) {

                // chart period points distance
                if (epoch % spaceBetweenPoints == 0) {

                    for (int i = 0; i != FFNetworks.size(); i++) {
                        chart.update(epoch, FFNetworks.get(i).getNetworkLoss(), "network" + i);
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
