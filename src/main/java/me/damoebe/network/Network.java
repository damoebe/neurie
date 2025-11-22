package me.damoebe.network;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Network {

    private final List<Layer> layers = new ArrayList<>(); // the whole network in a list

    private final double learningRate;

    private double loss = 0;
    private double noise = 1;

    public Network(int inputSize, int outputSize, int hiddenLayerSize, int hiddenLayerAmount, double learningRate){ // adjustable values
        // generate network
        this.learningRate = learningRate;
        Layer inputLayer = generateLayer(inputSize, null);
        layers.add(inputLayer);
        Layer prevLayer = inputLayer;
        for (int i = 0; i != hiddenLayerAmount; i++){
            Layer hiddenLayer = generateLayer(hiddenLayerSize, prevLayer);
            layers.add(hiddenLayer);
            prevLayer = hiddenLayer;
        }
        Layer outputLayer = generateLayer(outputSize, prevLayer);
        layers.add(outputLayer);
    }

    // generates a layer for the network
    private Layer generateLayer(int size, Layer prevLayer){
        List<Connection> connections = new ArrayList<>(); // neurons from one layer have same dependencies
        if (prevLayer != null){ // if not input layer
            for (Neuron neuron : prevLayer.neurons()){
                connections.add(new Connection(neuron));
            }
        }
        List<Neuron> neurons = new ArrayList<>();
        for (int i = 0; i != size; i++){
            neurons.add(new Neuron(new ArrayList<>(connections), learningRate));
        }
        return new Layer(neurons); // returns record
    }

    // updates all activations
    public void updateAllActivations(){
        for (Layer layer : layers){
            for (Neuron neuron : layer.neurons()){
                neuron.updateActivation();
            }
        }
    }

    // sets the input neurons activations
    public void insertInput(List<Double> input){
        for (int i = 0; i != input.size(); i++){
            layers.get(0).neurons().get(i).setActivation(input.get(i));
        }
    }

    // returns the activations of the last layer
    public List<Double> getOutput(){
        List<Double> output = new ArrayList<>();
        for (Neuron neuron : layers.get(layers.size()-1).neurons()){
            output.add(neuron.getActivation());
        }
        return output;
    }

    // uses backpropagation to update the weights with ONE set of date
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

    private void updateLoss(List<Double> optimalOutput){
        // update network loss
        double output = getOutput().get(0);
        double error = output - optimalOutput.get(0);
        this.loss = error * error;
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

    public void setNoise(double noise){
        this.noise = noise;
    }

    // test ( can be removed )

    private double lowestLoss = 10;
    private List<Connection> connections = new ArrayList<>();

    public void evolutionLearning(List<Double> input, List<Double> optimalOutput){
        insertInput(input);
        updateAllActivations();
        updateLoss(optimalOutput);
    }

    public void finishEpoch(){

        if (loss < lowestLoss){
            // save success
            lowestLoss = loss;
            connections.clear();
            for (Layer layer: layers){
                for (Neuron neuron: layer.neurons()){
                    connections.addAll(neuron.getConnections());
                }
            }
        } else {
            // reset to best know version
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
        // calculating change
        double mutationRate = 0.1*(loss);
        Random random = new Random();

        for (Layer layer : layers){
            for (Neuron neuron : layer.neurons()){
                for (Connection connection : neuron.getConnections()){
                    double change = (random.nextDouble() * 2 - 1) * mutationRate;
                    connection.setWeight(connection.getWeight() + change);
                }
            }
        }
    }

    public double getLowestLoss(){
        return lowestLoss;
    }
    public double getLoss() {
        return this.loss;
    }

}