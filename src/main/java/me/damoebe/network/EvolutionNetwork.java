package me.damoebe.network;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * An object class for EvolutionNetworks
 */
public class EvolutionNetwork extends Network {
    /**
     * Main constructor for EvolutionNetwork accessing the super constructor of Network abstract class
     * @param inputSize The size of the first layer
     * @param outputSize The size of the last layer
     * @param hiddenLayerSize The size of the hiddenlayers
     * @param hiddenLayerAmount The amount of hiddenlayers that should be generated
     */
    public EvolutionNetwork(int inputSize, int outputSize, int hiddenLayerSize, int hiddenLayerAmount) {
        super(inputSize, outputSize, hiddenLayerSize, hiddenLayerAmount);
    }

    /**
     * The lowest loss the network has ever reached
     */
    private double lowestLoss = 10;
    /**
     * A list of all losses in the current epoch
     */
    private List<Double> epochLosses = new ArrayList<>();
    /**
     * A copy of the best known version of the network
     */
    private List<Connection> connections = new ArrayList<>();

    @Override
    public void train(List<Double> input, List<Double> optimalOutput){
        this.insertInput(input);
        this.updateAllActivations();
        this.updateLoss(optimalOutput);
    }

    /**
     * Updates the current network loss and adds loss to epoch loss list
     * @param optimalActivations The target activations for the current repetition
     */
    @Override
    public void updateLoss(List<Double> optimalActivations){
        super.updateLoss(optimalActivations);
        epochLosses.add(currentLoss);
    }

    /**
     * Gets the average epoch loss using the epochLosses list
     * @return average loss for current epoch
     */
    private double getAverageEpochLoss(){
        double avgLoss = 0;
        for (Double epochLoss : epochLosses){
            avgLoss += epochLoss;
        }
        return avgLoss / epochLosses.size();
    }

    /**
     * Finishes an epoch by saving potential success and mutating connection weights
     */
    public void finishEpoch(){
        double averageEpochLoss = getAverageEpochLoss();
        if (averageEpochLoss < this.lowestLoss){
            // save success
            this.lowestLoss = averageEpochLoss;
            connections.clear();
            for (Layer layer: layers){
                for (Neuron neuron: layer.neurons()){
                    connections.addAll(neuron.getConnections());
                }
            }
        } else {
            // reset to best know version
            resetEvoNetwork();
        }
        // calculating change
        Random random = new Random();

        for (Layer layer : layers){
            for (Neuron neuron : layer.neurons()){
                for (Connection connection : neuron.getConnections()){
                    double change = (random.nextDouble() * 2 - 1) * noise;
                    connection.setWeight(connection.getWeight() + change);
                }
            }
        }

        // reset epochLoss list
        epochLosses.clear();
    }

    /**
     * Resets the network to the best know version of itself
     */
    private void resetEvoNetwork(){
        if (!connections.isEmpty()) {
            int i = 0;
            for (Layer layer : layers) {

                for (Neuron neuron : layer.neurons()) {
                    for (Connection connection : neuron.getConnections()) {
                        connection.setWeight(connections.get(i).getWeight());
                        i++;
                    }
                }

            }
        }

    }

    @Override
    public double getNetworkLoss(){
        return this.lowestLoss;
    }
}
