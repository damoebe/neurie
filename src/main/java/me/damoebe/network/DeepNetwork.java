package me.damoebe.network;

import java.util.ArrayList;
import java.util.List;

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

    @Override
    public double getNetworkLoss() {
        return loss;
    }

    // update all deltas in the network
    private void updateHiddenDeltas() {
        for (int l = layers.size() - 2; l > 0; l--) { // skip output layer
            for (Neuron neuron : layers.get(l).neurons()) {
                double output = neuron.getActivation();
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

    // update all output deltas
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

    // updates all connection-weights and biases within the network
    private void updateAllWeightsAndBiases() {
        for (Layer layer : layers) {
            for (Neuron neuron : layer.neurons()) {
                neuron.updateWeights(loss, noise);
                neuron.updateBias();
            }
        }
    }

}