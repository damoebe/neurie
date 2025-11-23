package me.damoebe.network;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class EvolutionNetwork extends Network {
    public EvolutionNetwork(int inputSize, int outputSize, int hiddenLayerSize, int hiddenLayerAmount) {
        super(inputSize, outputSize, hiddenLayerSize, hiddenLayerAmount);
    }

    private double lowestLoss = 10;
    private List<Connection> connections = new ArrayList<>();

    @Override
    public void train(List<Double> input, List<Double> optimalOutput){
        this.insertInput(input);
        this.updateAllActivations();
        this.updateLoss(optimalOutput);
    }

    public void finishEpoch(){

        if (this.loss < this.lowestLoss){
            // save success
            System.out.println(loss + lowestLoss);
            this.lowestLoss = this.loss;
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
    }

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
