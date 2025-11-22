package me.damoebe.test;

import me.damoebe.datasets.Dataset;
import me.damoebe.network.LearningType;
import me.damoebe.network.Network;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class Trainer {
    public static void train(boolean showDiagram, List<Network> networks, Dataset dataset, int epochs, LearningType learningType){

        ChartDisplay chart = null;

        if (showDiagram) {
            chart = new ChartDisplay("neurie-Training (" + dataset.getName() + ")", "epoch", "loss", 0);
        }

        List<List<Double>> inputs = dataset.getInputs();
        List<List<Double>> targets = dataset.getTargets();

        for (int epoch = 0; epoch < epochs; epoch++) {
            double[] totalLosses = new double[networks.size()];
            int i2 = 0;
            for (Network network : networks) {
                for (int i = 0; i < inputs.size(); i++) {
                    if (learningType.equals(LearningType.BACKPROPAGATION)) {
                        network.train(inputs.get(i), targets.get(i));
                    }else{
                        network.evolutionLearning(inputs.get(i), targets.get(i));
                    }
                    totalLosses[i2] += network.getLowestLoss();
                }
                i2++;
            }
            if (learningType.equals(LearningType.EVOLUTION)){
                for (Network network : networks){
                    network.finishEpoch();
                }
            }
            if (epoch % 1000 == 0 || epoch == epochs - 1) {
                System.out.printf("Epoch %d, Loss: " + Arrays.toString(totalLosses) + "\n", epoch);
            }
            // chart updater
            if (showDiagram) {

                // chart period points distance
                if (epoch % 100 == 0) {

                    for (int i = 0; i != totalLosses.length; i++) {
                        if (learningType.equals(LearningType.BACKPROPAGATION)) {
                            chart.update(epoch, networks.get(i).getLoss(), "network" + i);
                        }else {
                            chart.update(epoch, networks.get(i).getLowestLoss(), "network" + i);
                        }
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
