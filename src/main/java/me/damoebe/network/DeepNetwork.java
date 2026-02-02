package me.damoebe.network;

import java.util.List;

/**
 * An object class for DeepNetwork objects
 */
public class DeepNetwork extends Network{

    /**
     * Main constructor for DeepNetwork calling the super constructor
     * @param inputSize The size of the first layer
     * @param outputSize The size of the last layer
     * @param hiddenLayerSize The size of the hiddenlayers
     * @param hiddenLayerAmount The amount of hiddenlayers that should be generated
     * @param leaningRate The learningRate that should be used for this DeepNetwork
     */
    public DeepNetwork(int inputSize, int outputSize, int hiddenLayerSize, int hiddenLayerAmount, double leaningRate) {
        super(inputSize, outputSize, hiddenLayerSize, hiddenLayerAmount, leaningRate);
    }

    /**
     * Trains the network with one set of data using backpropagation.
     * @param input The inputs for this repetition
     * @param optimalOutput The optimalOutputs for this repetition
     */
    @Override
    public void train(List<Double> input, List<Double> optimalOutput){
         // forward
         insertInput(input);
         updateAllActivations();
         updateLoss(optimalOutput);
         // backward
         updateOutputDeltas(optimalOutput);
         updateHiddenDeltas();
         updateAllWeightsAndBiases();

    }

    /**
     * Gets the network loss
     * @return The network loss
     */
    @Override
    public double getNetworkLoss() {
        return currentLoss;
    }

    /**
     * Updates all hidden deltas in the network. Warning: The output deltas are updated in updateOutputDeltas
     * Also this method should only be called when the updateOutputDeltas method has been called before
     */
    private void updateHiddenDeltas() {
        for (int l = layers.size() - 2; l > 0; l--) { // skip output and input layer
            for (Neuron neuron : layers.get(l).neurons()) {
                double output = neuron.getActivation();
                // calculate the sum of all next deltas compared with the weight of the connection to this neuron
                double sum = 0;
                for (Neuron next : layers.get(l + 1).neurons()) {
                    for (Connection conn : next.getConnections()) {
                        if (conn.getSourceNeuron() == neuron) {
                            sum += conn.getWeight() * next.getDelta();
                        }
                    }
                }
                double delta = output * (1 - output) * sum;
                neuron.setDelta(delta);
            }
        }
    }

    /**
     * Update all OUTPUT hidden delta values for each neuron
     * @param optimalOutputs The list of all optimal outputs
     */
    private void updateOutputDeltas(List<Double> optimalOutputs) {
        List<Neuron> outputNeurons = layers.get(layers.size()-1).neurons();
        for (int i = 0; i < outputNeurons.size(); i++) {
            Neuron neuron = outputNeurons.get(i);
            double output = neuron.getActivation();
            double error = output - optimalOutputs.get(i); // calc error
            double delta = error * output * (1 - output); // delta learning rule
            neuron.setDelta(delta);
        }
    }

    /**
     * Updates all weights and biases in the network
     */
    private void updateAllWeightsAndBiases() {
        for (Layer layer : layers) {
            for (Neuron neuron : layer.neurons()) {
                neuron.updateWeights(currentLoss, noise, learningRate);
                neuron.updateBias(learningRate);
            }
        }
    }

}